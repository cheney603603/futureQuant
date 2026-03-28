"""
因子有效性评估模块 - IC分析、分层回测、因子统计

功能：
- IC分析：计算因子与收益率的相关系数（IC）和信息比率（ICIR）
- 分层回测：按因子值分层，评估各层收益表现
- 因子统计：覆盖率、缺失率、自相关性、换手率
- 可视化准备：生成IC序列、分层收益等数据
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from scipy import stats

from ..core.logger import get_logger
from ..core.exceptions import FactorError

logger = get_logger('factor.evaluator')


class FactorEvaluator:
    """因子有效性评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.results: Dict[str, pd.DataFrame] = {}
    
    def calculate_ic(
        self, 
        factor_df: Union[pd.DataFrame, pd.Series], 
        returns: pd.Series,
        method: str = 'spearman'
    ) -> pd.Series:
        """
        计算因子IC（信息系数）- 时间序列版本
        
        计算因子值与未来收益率的时间序列相关系数。
        注意：此方法适用于单品种时间序列分析，计算的是因子值序列与收益率序列的整体相关性。
        如需横截面IC分析（每天多个品种），请使用 calculate_ic_panel。
        
        Args:
            factor_df: 因子值DataFrame或Series。DataFrame时列为因子名，Series时为单个因子
            returns: 未来收益率序列，index为日期
            method: 相关系数方法，'spearman'(秩相关) 或 'pearson'(线性相关)
            
        Returns:
            IC序列（单值Series），index为因子名，value为IC值
            
        Raises:
            FactorError: 输入数据无效
        """
        if factor_df.empty:
            raise FactorError("factor_df is empty")
        
        if returns.empty:
            raise FactorError("returns is empty")
        
        # 如果是Series，转换为DataFrame
        if isinstance(factor_df, pd.Series):
            factor_df = factor_df.to_frame(name=factor_df.name or 'factor')
        
        # 对齐数据
        common_index = factor_df.index.intersection(returns.index)
        if len(common_index) == 0:
            raise FactorError("No common dates between factor and returns")
        
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # 计算时间序列IC（每个因子与收益率的相关系数）
        ic_results = {}
        
        for col in factor_aligned.columns:
            f_vals = factor_aligned[col]
            r_vals = returns_aligned
            
            # 去除缺失值
            valid_mask = f_vals.notna() & r_vals.notna()
            
            if valid_mask.sum() < 3:
                ic_results[col] = np.nan
                continue
            
            try:
                if method == 'spearman':
                    ic, _ = stats.spearmanr(f_vals[valid_mask], r_vals[valid_mask])
                elif method == 'pearson':
                    ic, _ = stats.pearsonr(f_vals[valid_mask], r_vals[valid_mask])
                else:
                    raise FactorError(f"Unknown correlation method: {method}")
                
                ic_results[col] = ic
            except Exception as e:
                logger.warning(f"Failed to calculate IC for {col}: {e}")
                ic_results[col] = np.nan
        
        ic_series = pd.Series(ic_results)
        
        logger.info(f"Calculated IC: {len(ic_series)} factors, "
                   f"valid: {ic_series.notna().sum()}")
        
        return ic_series
    
    def calculate_ic_panel(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        计算横截面IC（多因子多品种）
        
        Args:
            factor_df: 因子值DataFrame，列为因子名，行为(日期, 品种)MultiIndex
            returns_df: 收益率DataFrame，列为品种，行为日期
            method: 相关系数方法
            
        Returns:
            IC序列DataFrame，行为日期，列为因子名
        """
        if not isinstance(factor_df.index, pd.MultiIndex):
            raise FactorError("factor_df must have MultiIndex (date, symbol)")
        
        ic_results = {}
        
        for factor_name in factor_df.columns:
            ic_list = []
            dates = factor_df.index.get_level_values(0).unique()
            
            for date in dates:
                # 获取当日因子值
                factor_slice = factor_df.loc[date, factor_name]
                if isinstance(factor_slice, pd.Series):
                    factor_slice = factor_slice.to_frame().T
                
                # 获取次日收益率
                date_idx = returns_df.index.get_loc(date)
                if date_idx + 1 >= len(returns_df.index):
                    continue
                next_date = returns_df.index[date_idx + 1]
                
                returns_slice = returns_df.loc[next_date]
                
                # 对齐
                common_symbols = factor_slice.columns.intersection(returns_slice.index)
                if len(common_symbols) < 3:
                    continue
                
                f_vals = factor_slice[common_symbols].iloc[0]
                r_vals = returns_slice[common_symbols]
                
                # 去除缺失值
                valid_mask = f_vals.notna() & r_vals.notna()
                if valid_mask.sum() < 3:
                    continue
                
                try:
                    if method == 'spearman':
                        ic, _ = stats.spearmanr(f_vals[valid_mask], r_vals[valid_mask])
                    else:
                        ic, _ = stats.pearsonr(f_vals[valid_mask], r_vals[valid_mask])
                    ic_list.append((date, ic))
                except Exception:
                    continue
            
            if ic_list:
                dates_list, ic_values = zip(*ic_list)
                ic_results[factor_name] = pd.Series(ic_values, index=dates_list)
        
        return pd.DataFrame(ic_results)
    
    def calculate_icir(
        self, 
        ic_series: pd.Series, 
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        计算ICIR（信息比率）
        
        ICIR = IC均值 / IC标准差
        衡量因子预测能力的稳定性和强度
        
        Args:
            ic_series: IC序列
            window: 滚动窗口，为None时计算全样本
            
        Returns:
            包含ICIR、IC均值、IC标准差的字典
        """
        if ic_series.empty:
            raise FactorError("IC series is empty")
        
        valid_ic = ic_series.dropna()
        
        if len(valid_ic) < 2:
            raise FactorError("Not enough valid IC values")
        
        if window is not None and len(valid_ic) > window:
            # 滚动计算
            rolling_ic = valid_ic.rolling(window=window)
            ic_mean = rolling_ic.mean().iloc[-1]
            ic_std = rolling_ic.std().iloc[-1]
        else:
            # 全样本计算
            ic_mean = valid_ic.mean()
            ic_std = valid_ic.std()
        
        icir = ic_mean / ic_std if ic_std != 0 else np.nan
        
        # 计算年化ICIR（假设252个交易日）
        annual_icir = icir * np.sqrt(252) if not np.isnan(icir) else np.nan
        
        # 计算IC胜率
        ic_win_rate = (valid_ic > 0).mean()
        
        return {
            'icir': icir,
            'annual_icir': annual_icir,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_win_rate': ic_win_rate,
            'n_samples': len(valid_ic)
        }
    
    def calculate_ic_decay(
        self,
        factor_df: pd.DataFrame,
        returns: pd.Series,
        max_lag: int = 10,
        method: str = 'spearman'
    ) -> pd.Series:
        """
        计算IC衰减（不同滞后期的IC）
        
        用于评估因子预测能力的持续性
        
        Args:
            factor_df: 因子值DataFrame
            returns: 收益率序列
            max_lag: 最大滞后期
            method: 相关系数方法
            
        Returns:
            IC衰减序列，index为滞后期，value为对应IC均值
        """
        decay_results = {}
        
        for lag in range(1, max_lag + 1):
            # 将收益率向前移动lag期
            shifted_returns = returns.shift(-lag)
            
            try:
                ic_series = self.calculate_ic(factor_df, shifted_returns, method)
                decay_results[lag] = ic_series.mean()
            except Exception as e:
                logger.warning(f"Failed to calculate IC for lag {lag}: {e}")
                decay_results[lag] = np.nan
        
        return pd.Series(decay_results)
    
    def quantile_backtest(
        self, 
        factor_df: pd.DataFrame, 
        returns: pd.Series,
        n_quantiles: int = 5,
        long_short: bool = True
    ) -> pd.DataFrame:
        """
        分层回测（Quantile Backtest）
        
        按因子值将标的分为n层，计算每层收益，用于评估因子单调性
        
        Args:
            factor_df: 因子值DataFrame，列为因子名，行为日期
            returns: 未来收益率序列
            n_quantiles: 分位数数量（默认5分位）
            long_short: 是否计算多空组合收益（第1层 - 第n层）
            
        Returns:
            分层收益DataFrame，行为日期，列为各层及多空组合
        """
        if factor_df.empty or returns.empty:
            raise FactorError("Empty input data")
        
        # 对齐数据
        common_index = factor_df.index.intersection(returns.index)
        if len(common_index) == 0:
            raise FactorError("No common dates")
        
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        quantile_returns = []
        
        for date in common_index:
            factor_values = factor_aligned.loc[date]
            ret_value = returns_aligned.loc[date]
            
            # 去除缺失值
            valid_mask = factor_values.notna()
            if isinstance(ret_value, pd.Series):
                valid_mask &= ret_value.notna()
            
            if valid_mask.sum() < n_quantiles:
                continue
            
            f_vals = factor_values[valid_mask]
            r_vals = ret_value[valid_mask] if isinstance(ret_value, pd.Series) else ret_value
            
            try:
                # 按因子值分层
                labels = range(1, n_quantiles + 1)
                quantiles = pd.qcut(f_vals, q=n_quantiles, labels=labels, duplicates='drop')
                
                # 计算每层平均收益
                layer_returns = {}
                for q in labels:
                    mask = quantiles == q
                    if mask.sum() > 0:
                        if isinstance(r_vals, pd.Series):
                            layer_returns[f'Q{q}'] = r_vals[mask].mean()
                        else:
                            layer_returns[f'Q{q}'] = r_vals
                
                # 多空组合收益（最高层 - 最低层，假设因子与收益正相关）
                if long_short and 'Q1' in layer_returns and f'Q{n_quantiles}' in layer_returns:
                    layer_returns['long_short'] = layer_returns[f'Q{n_quantiles}'] - layer_returns['Q1']
                
                quantile_returns.append(pd.Series(layer_returns, name=date))
                
            except Exception as e:
                logger.warning(f"Failed to calculate quantile returns for {date}: {e}")
                continue
        
        if not quantile_returns:
            raise FactorError("No valid quantile returns calculated")
        
        result_df = pd.DataFrame(quantile_returns)
        
        logger.info(f"Quantile backtest completed: {len(result_df)} periods, "
                   f"{n_quantiles} quantiles")
        
        return result_df
    
    def quantile_backtest_panel(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        n_quantiles: int = 5,
        long_short: bool = True
    ) -> pd.DataFrame:
        """
        横截面分层回测（多品种）
        
        Args:
            factor_df: 因子值DataFrame，MultiIndex(日期, 品种)
            returns_df: 收益率DataFrame，列为品种，行为日期
            n_quantiles: 分位数数量
            long_short: 是否计算多空组合
            
        Returns:
            分层收益DataFrame
        """
        if not isinstance(factor_df.index, pd.MultiIndex):
            raise FactorError("factor_df must have MultiIndex (date, symbol)")
        
        quantile_returns = []
        dates = factor_df.index.get_level_values(0).unique()
        
        for date in dates:
            # 获取当日因子值
            factor_slice = factor_df.loc[date]
            
            # 获取次日收益率
            date_idx = returns_df.index.get_loc(date)
            if date_idx + 1 >= len(returns_df.index):
                continue
            next_date = returns_df.index[date_idx + 1]
            returns_slice = returns_df.loc[next_date]
            
            # 处理单因子情况：factor_slice是DataFrame，index是symbol，columns是factor_name
            # 处理多因子情况：factor_slice是DataFrame，index是symbol，columns是factor_names
            if isinstance(factor_slice, pd.DataFrame):
                # 取第一个因子的值
                factor_name = factor_df.columns[0]
                f_vals = factor_slice[factor_name]
            else:
                # 单因子单值情况（不应该发生）
                continue
            
            # 对齐
            common_symbols = f_vals.index.intersection(returns_slice.index)
            if len(common_symbols) < n_quantiles:
                continue
            
            f_vals = f_vals[common_symbols]
            r_vals = returns_slice[common_symbols]
            
            # 去除缺失值
            valid_mask = f_vals.notna() & r_vals.notna()
            if valid_mask.sum() < n_quantiles:
                continue
            
            f_vals = f_vals[valid_mask]
            r_vals = r_vals[valid_mask]
            
            try:
                # 分层
                labels = range(1, n_quantiles + 1)
                quantiles = pd.qcut(f_vals, q=n_quantiles, labels=labels, duplicates='drop')
                
                layer_returns = {}
                for q in labels:
                    mask = quantiles == q
                    if mask.sum() > 0:
                        layer_returns[f'Q{q}'] = r_vals[mask].mean()
                
                if long_short and 'Q1' in layer_returns and f'Q{n_quantiles}' in layer_returns:
                    layer_returns['long_short'] = layer_returns[f'Q{n_quantiles}'] - layer_returns['Q1']
                
                quantile_returns.append(pd.Series(layer_returns, name=date))
                
            except Exception as e:
                logger.warning(f"Failed for {date}: {e}")
                continue
        
        return pd.DataFrame(quantile_returns)
    
    def calculate_factor_stats(
        self, 
        factor_df: pd.DataFrame
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        计算因子统计指标
        
        Args:
            factor_df: 因子值DataFrame
            
        Returns:
            统计指标字典，包含：
            - coverage: 覆盖率（非缺失值比例）
            - missing_rate: 缺失率
            - autocorr: 自相关性序列（1阶）
            - mean: 均值
            - std: 标准差
            - skewness: 偏度
            - kurtosis: 峰度
        """
        if factor_df.empty:
            raise FactorError("factor_df is empty")
        
        stats_dict = {}
        
        for col in factor_df.columns:
            series = factor_df[col]
            
            # 覆盖率和缺失率
            coverage = series.notna().mean()
            missing_rate = 1 - coverage
            
            # 基本统计量
            valid_series = series.dropna()
            
            if len(valid_series) > 0:
                mean_val = valid_series.mean()
                std_val = valid_series.std()
                skew_val = valid_series.skew()
                kurt_val = valid_series.kurtosis()
                
                # 自相关性（1阶）
                autocorr = valid_series.autocorr(lag=1)
            else:
                mean_val = std_val = skew_val = kurt_val = autocorr = np.nan
            
            stats_dict[col] = {
                'coverage': coverage,
                'missing_rate': missing_rate,
                'autocorr': autocorr,
                'mean': mean_val,
                'std': std_val,
                'skewness': skew_val,
                'kurtosis': kurt_val,
            }
        
        return stats_dict
    
    def calculate_turnover(
        self, 
        factor_df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        计算因子换手率
        
        衡量因子值变化的频率，换手率过高可能增加交易成本
        
        Args:
            factor_df: 因子值DataFrame
            threshold: 变化阈值，为None时计算排名变化率
            
        Returns:
            换手率序列
        """
        if factor_df.empty:
            raise FactorError("factor_df is empty")
        
        turnover_series = pd.Series(index=factor_df.index[1:], dtype=float)
        
        for col in factor_df.columns:
            series = factor_df[col]
            
            for i in range(1, len(series)):
                prev_val = series.iloc[i-1]
                curr_val = series.iloc[i]
                
                # 去除缺失值
                prev_valid = prev_val.dropna() if isinstance(prev_val, pd.Series) else prev_val
                curr_valid = curr_val.dropna() if isinstance(curr_val, pd.Series) else curr_val
                
                if isinstance(prev_valid, pd.Series) and isinstance(curr_valid, pd.Series):
                    # 计算排名变化
                    common_index = prev_valid.index.intersection(curr_valid.index)
                    if len(common_index) < 2:
                        turnover_series.iloc[i-1] = np.nan
                        continue
                    
                    prev_rank = prev_valid[common_index].rank()
                    curr_rank = curr_valid[common_index].rank()
                    
                    # 换手率 = 排名变化比例
                    rank_change = (prev_rank - curr_rank).abs() / len(common_index)
                    turnover_series.iloc[i-1] = rank_change.mean()
                else:
                    # 单值情况
                    if threshold is not None:
                        turnover_series.iloc[i-1] = 1.0 if abs(curr_valid - prev_valid) > threshold else 0.0
                    else:
                        turnover_series.iloc[i-1] = 0.0 if curr_valid == prev_valid else 1.0
        
        return turnover_series
    
    def calculate_turnover_panel(
        self,
        factor_df: pd.DataFrame
    ) -> pd.Series:
        """
        计算横截面因子换手率（多品种）
        
        Args:
            factor_df: 因子值DataFrame，MultiIndex(日期, 品种)
            
        Returns:
            每日换手率序列
        """
        if not isinstance(factor_df.index, pd.MultiIndex):
            raise FactorError("factor_df must have MultiIndex (date, symbol)")
        
        dates = factor_df.index.get_level_values(0).unique()
        turnover_list = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            # 获取两日的因子值
            prev_factors = factor_df.loc[prev_date]
            curr_factors = factor_df.loc[curr_date]
            
            if isinstance(prev_factors, pd.Series):
                prev_factors = prev_factors.to_frame().T
            if isinstance(curr_factors, pd.Series):
                curr_factors = curr_factors.to_frame().T
            
            # 计算每个因子的换手率
            date_turnover = {}
            for col in factor_df.columns:
                prev_vals = prev_factors[col] if col in prev_factors.columns else pd.Series()
                curr_vals = curr_factors[col] if col in curr_factors.columns else pd.Series()
                
                common_symbols = prev_vals.index.intersection(curr_vals.index)
                if len(common_symbols) < 2:
                    date_turnover[col] = np.nan
                    continue
                
                # 排名变化率
                prev_rank = prev_vals[common_symbols].rank()
                curr_rank = curr_vals[common_symbols].rank()
                
                rank_change = (prev_rank - curr_rank).abs() / len(common_symbols)
                date_turnover[col] = rank_change.mean()
            
            turnover_list.append(pd.Series(date_turnover, name=curr_date))
        
        return pd.DataFrame(turnover_list)
    
    def full_evaluation(
        self,
        factor_df: pd.DataFrame,
        returns: pd.Series,
        n_quantiles: int = 5,
        method: str = 'spearman'
    ) -> Dict[str, Union[pd.DataFrame, pd.Series, Dict]]:
        """
        完整因子评估
        
        执行所有评估指标计算
        
        Args:
            factor_df: 因子值DataFrame
            returns: 收益率序列
            n_quantiles: 分层数量
            method: IC计算方法
            
        Returns:
            包含所有评估结果的字典
        """
        logger.info("Starting full factor evaluation...")
        
        results = {}
        
        # 1. IC分析
        logger.info("Calculating IC...")
        ic_series = self.calculate_ic(factor_df, returns, method)
        results['ic_series'] = ic_series
        results['ic_stats'] = self.calculate_icir(ic_series)
        
        # 2. IC衰减
        logger.info("Calculating IC decay...")
        results['ic_decay'] = self.calculate_ic_decay(factor_df, returns, method=method)
        
        # 3. 分层回测（仅适用于横截面数据）
        logger.info("Running quantile backtest...")
        try:
            results['quantile_returns'] = self.quantile_backtest(
                factor_df, returns, n_quantiles
            )
        except FactorError as e:
            logger.warning(f"Quantile backtest skipped: {e}")
            results['quantile_returns'] = pd.DataFrame()
        
        # 4. 因子统计
        logger.info("Calculating factor statistics...")
        results['factor_stats'] = self.calculate_factor_stats(factor_df)
        
        # 5. 换手率
        logger.info("Calculating turnover...")
        results['turnover'] = self.calculate_turnover(factor_df)
        
        logger.info("Full evaluation completed")
        
        return results
    
    def generate_report_data(
        self,
        evaluation_results: Dict
    ) -> Dict[str, pd.DataFrame]:
        """
        生成报告数据（用于可视化）
        
        Args:
            evaluation_results: full_evaluation返回的结果
            
        Returns:
            适合可视化的数据字典
        """
        report_data = {}
        
        # IC序列
        if 'ic_series' in evaluation_results:
            report_data['ic_series'] = evaluation_results['ic_series'].to_frame('IC')
        
        # 分层收益累计
        if 'quantile_returns' in evaluation_results:
            quantile_df = evaluation_results['quantile_returns']
            report_data['quantile_cumret'] = (1 + quantile_df).cumprod()
        
        # IC统计摘要
        if 'ic_stats' in evaluation_results:
            stats = evaluation_results['ic_stats']
            report_data['ic_summary'] = pd.DataFrame([stats])
        
        # IC衰减
        if 'ic_decay' in evaluation_results:
            report_data['ic_decay'] = evaluation_results['ic_decay'].to_frame('IC_Decay')
        
        return report_data
    
    def get_summary(
        self,
        evaluation_results: Dict,
        factor_name: str = 'factor'
    ) -> pd.DataFrame:
        """
        获取评估摘要
        
        Args:
            evaluation_results: full_evaluation返回的结果
            factor_name: 因子名称
            
        Returns:
            摘要DataFrame
        """
        summary = {'factor': factor_name}
        
        # IC统计
        if 'ic_stats' in evaluation_results:
            stats = evaluation_results['ic_stats']
            summary.update({
                'IC Mean': stats.get('ic_mean', np.nan),
                'IC Std': stats.get('ic_std', np.nan),
                'ICIR': stats.get('icir', np.nan),
                'Annual ICIR': stats.get('annual_icir', np.nan),
                'IC Win Rate': stats.get('ic_win_rate', np.nan),
            })
        
        # 分层收益
        if 'quantile_returns' in evaluation_results:
            qr = evaluation_results['quantile_returns']
            if 'long_short' in qr.columns:
                summary['Long-Short Return'] = qr['long_short'].mean()
                summary['Long-Short Sharpe'] = (
                    qr['long_short'].mean() / qr['long_short'].std() 
                    if qr['long_short'].std() != 0 else np.nan
                )
        
        # 因子统计
        if 'factor_stats' in evaluation_results:
            fs = evaluation_results['factor_stats']
            if isinstance(fs, dict) and len(fs) > 0:
                first_factor = list(fs.keys())[0]
                summary['Coverage'] = fs[first_factor].get('coverage', np.nan)
                summary['Autocorr'] = fs[first_factor].get('autocorr', np.nan)
        
        # 换手率
        if 'turnover' in evaluation_results:
            to = evaluation_results['turnover']
            summary['Turnover'] = to.mean() if hasattr(to, 'mean') else np.nan
        
        return pd.DataFrame([summary])
