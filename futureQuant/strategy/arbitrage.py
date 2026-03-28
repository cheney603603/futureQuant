"""
套利策略模块

包含多种套利策略实现：
- 套利策略基类（ArbitrageStrategy）
- 跨期套利策略（SpreadArbitrageStrategy）
- 跨品种套利策略（CrossVarietyArbitrageStrategy）
- 期现套利策略（FuturesSpotArbitrageStrategy）
"""

from typing import Optional, List, Dict, Any, Tuple
from abc import abstractmethod
import pandas as pd
import numpy as np
from scipy import stats

from .base import BaseStrategy
from ..core.logger import get_logger
from ..core.exceptions import StrategyError

logger = get_logger('strategy.arbitrage')


class ArbitrageStrategy(BaseStrategy):
    """
    套利策略基类
    
    为各类套利策略提供基础功能：
    - 价差计算
    - 价比计算
    - 均值回归判断
    - 信号生成框架
    
    参数：
        lookback: 回看周期，默认60
        entry_threshold: 入场阈值，默认2.0
        exit_threshold: 出场阈值，默认0.5
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(
            name=name or 'Arbitrage',
            symbols=symbols,
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            **kwargs
        )
        
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def calculate_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: float = 1.0
    ) -> pd.Series:
        """
        计算价差
        
        Args:
            price1: 价格序列1
            price2: 价格序列2
            hedge_ratio: 对冲比例，默认为1.0
            
        Returns:
            价差序列
        """
        return price1 - hedge_ratio * price2
    
    def calculate_price_ratio(
        self,
        price1: pd.Series,
        price2: pd.Series
    ) -> pd.Series:
        """
        计算价比
        
        Args:
            price1: 价格序列1
            price2: 价格序列2
            
        Returns:
            价比序列
        """
        return price1 / price2
    
    def calculate_zscore(
        self,
        series: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        计算Z-score（标准化值）
        
        Args:
            series: 输入序列
            lookback: 回看周期，默认使用self.lookback
            
        Returns:
            Z-score序列
        """
        window = lookback or self.lookback
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std
    
    def calculate_percentile(
        self,
        series: pd.Series,
        lookback: Optional[int] = None
    ) -> pd.Series:
        """
        计算当前值在历史分布中的百分位
        
        Args:
            series: 输入序列
            lookback: 回看周期，默认使用self.lookback
            
        Returns:
            百分位序列（0-1）
        """
        window = lookback or self.lookback
        
        def get_percentile(x):
            if len(x) < window:
                return np.nan
            return stats.percentileofscore(x, x.iloc[-1]) / 100
        
        return series.rolling(window=window).apply(get_percentile, raw=False)
    
    def is_mean_reverting(
        self,
        series: pd.Series,
        lookback: Optional[int] = None,
        significance: float = 0.05
    ) -> bool:
        """
        检验序列是否具有均值回归特性（ADF检验）
        
        Args:
            series: 输入序列
            lookback: 检验窗口
            significance: 显著性水平
            
        Returns:
            是否具有均值回归特性
        """
        try:
            window = lookback or self.lookback
            data = series.dropna().iloc[-window:]
            if len(data) < window // 2:
                return False
            
            adf_result = stats.adfuller(data)
            return adf_result[1] < significance
        except Exception as e:
            logger.warning(f"ADF检验失败: {e}")
            return False
    
    def generate_mean_reversion_signals(
        self,
        zscore: pd.Series,
        entry_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        基于Z-score生成均值回归信号
        
        Args:
            zscore: Z-score序列
            entry_threshold: 入场阈值
            exit_threshold: 出场阈值
            
        Returns:
            信号DataFrame
        """
        entry = entry_threshold or self.entry_threshold
        exit_th = exit_threshold or self.exit_threshold
        
        signals = pd.DataFrame(index=zscore.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        signals['confidence'] = 0.0
        
        # 入场信号
        # Z-score < -entry: 做多（预期回归均值）
        # Z-score > entry: 做空（预期回归均值）
        long_entry = zscore < -entry
        short_entry = zscore > entry
        
        signals.loc[long_entry, 'signal'] = 1
        signals.loc[short_entry, 'signal'] = -1
        
        # 出场信号（回归均值附近）
        exit_zone = zscore.abs() < exit_th
        signals.loc[exit_zone, 'signal'] = 0
        
        # 权重：根据偏离程度
        deviation = zscore.abs() / entry
        signals['weight'] = deviation.clip(0.3, 1.0)
        signals.loc[exit_zone, 'weight'] = 0.0
        
        # 置信度
        signals['confidence'] = signals['weight']
        signals['zscore'] = zscore
        
        return signals
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（子类必须实现）
        
        Args:
            data: 输入数据
            
        Returns:
            信号DataFrame
        """
        pass
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        """
        获取参数优化边界
        
        Returns:
            参数边界字典
        """
        return {
            'lookback': (20, 120),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.2, 1.0),
        }


class SpreadArbitrageStrategy(ArbitrageStrategy):
    """
    跨期套利策略
    
    同一品种不同月份合约间的套利，利用远近月合约价差的均值回归特性。
    
    适用场景：
    - 同一商品期货的不同交割月份合约
    - 价差受持仓成本、季节性、供需预期等因素影响
    
    信号规则：
    - 价差 > 均值 + entry_threshold * 标准差：做空价差（卖近月买远月）
    - 价差 < 均值 - entry_threshold * 标准差：做多价差（买近月卖远月）
    - 价差回归均值附近：平仓
    
    参数：
        near_contract: 近月合约代码
        far_contract: 远月合约代码
        entry_threshold: 入场阈值（Z-score），默认2.0
        exit_threshold: 出场阈值（Z-score），默认0.5
        lookback: 回看周期，默认60
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        near_contract: Optional[str] = None,
        far_contract: Optional[str] = None,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        lookback: int = 60,
        **kwargs
    ):
        # 从symbols提取合约代码
        if near_contract is None and symbols is not None and len(symbols) >= 2:
            near_contract = symbols[0]
            far_contract = symbols[1]
        
        super().__init__(
            name=name or 'SpreadArbitrage',
            symbols=symbols or ([near_contract, far_contract] if near_contract and far_contract else []),
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            **kwargs
        )
        
        self.near_contract = near_contract
        self.far_contract = far_contract
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成跨期套利信号
        
        Args:
            data: 包含近月和远月合约价格的数据
                  需要列：'close_near' 和 'close_far' 或根据合约名映射
                  
        Returns:
            信号DataFrame，包含：
            - signal: 综合信号
            - signal_near: 近月合约信号
            - signal_far: 远月合约信号
            - weight: 权重
            - confidence: 置信度
            - spread: 价差
            - zscore: 标准化价差
        """
        if data.empty:
            logger.warning("输入数据为空")
            return pd.DataFrame()
        
        try:
            data = data.copy()
            
            # 确定价格列
            if 'close_near' in data.columns and 'close_far' in data.columns:
                near_col = 'close_near'
                far_col = 'close_far'
            elif self.near_contract and self.far_contract:
                # 尝试根据合约名查找列
                near_col = f'close_{self.near_contract}'
                far_col = f'close_{self.far_contract}'
                if near_col not in data.columns or far_col not in data.columns:
                    # 尝试使用前两个close列
                    close_cols = [c for c in data.columns if c.startswith('close')]
                    if len(close_cols) >= 2:
                        near_col, far_col = close_cols[0], close_cols[1]
                    else:
                        raise StrategyError(
                            f"无法找到价格列，需要 '{near_col}' 和 '{far_col}'"
                        )
            else:
                raise StrategyError("必须指定 near_contract 和 far_contract 或提供 close_near/far 列")
            
            # 计算价差（近月 - 远月）
            data['spread'] = data[near_col] - data[far_col]
            
            # 计算价差的统计特征
            data['spread_mean'] = data['spread'].rolling(window=self.lookback).mean()
            data['spread_std'] = data['spread'].rolling(window=self.lookback).std()
            
            # 标准化价差（Z-score）
            data['zscore'] = (data['spread'] - data['spread_mean']) / data['spread_std']
            
            # 生成均值回归信号
            signals = self.generate_mean_reversion_signals(data['zscore'])
            
            # 转换为跨期套利信号
            # signal = 1: 做多价差（买近月卖远月）
            # signal = -1: 做空价差（卖近月买远月）
            signals['signal_near'] = signals['signal']  # 近月合约信号
            signals['signal_far'] = -signals['signal']   # 远月合约信号（反向）
            
            # 添加价差信息
            signals['spread'] = data['spread']
            signals['spread_mean'] = data['spread_mean']
            signals['spread_std'] = data['spread_std']
            
            # 添加合约信息
            signals['near_contract'] = self.near_contract or near_col
            signals['far_contract'] = self.far_contract or far_col
            
            # 信号原因
            signals['reason'] = signals.apply(
                lambda x: (
                    'spread_arbitrage_long' if x['signal'] == 1 
                    else 'spread_arbitrage_short' if x['signal'] == -1 
                    else 'spread_arbitrage_exit'
                ),
                axis=1
            )
            
            logger.info(
                f"生成跨期套利信号: {self.near_contract}-{self.far_contract}, "
                f"信号数量: {(signals['signal'] != 0).sum()}"
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"生成跨期套利信号失败: {e}")
            raise StrategyError(f"生成跨期套利信号失败: {e}")
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        """获取参数优化边界"""
        return {
            'lookback': (20, 120),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.2, 1.0),
        }


class CrossVarietyArbitrageStrategy(ArbitrageStrategy):
    """
    跨品种套利策略
    
    相关品种间的套利，利用品种间价差的均值回归特性。
    
    适用场景：
    - 产业链上下游品种（如铁矿石-螺纹钢）
    - 替代品种（如豆粕-菜粕）
    - 相关品种（如螺纹钢-热卷）
    
    信号规则：
    - 价比 > 均值 + entry_threshold * 标准差：做空价比（卖品种1买品种2）
    - 价比 < 均值 - entry_threshold * 标准差：做多价比（买品种1卖品种2）
    - 价比回归均值附近：平仓
    
    参数：
        symbol1: 品种1代码
        symbol2: 品种2代码
        hedge_ratio: 对冲比例，默认1.0
        entry_threshold: 入场阈值（Z-score），默认2.0
        exit_threshold: 出场阈值（Z-score），默认0.5
        lookback: 回看周期，默认60
        use_price_ratio: 是否使用价比而非价差，默认True
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        symbol1: Optional[str] = None,
        symbol2: Optional[str] = None,
        hedge_ratio: float = 1.0,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        lookback: int = 60,
        use_price_ratio: bool = True,
        **kwargs
    ):
        # 从symbols提取品种代码
        if symbol1 is None and symbols is not None and len(symbols) >= 2:
            symbol1 = symbols[0]
            symbol2 = symbols[1]
        
        super().__init__(
            name=name or 'CrossVarietyArbitrage',
            symbols=symbols or ([symbol1, symbol2] if symbol1 and symbol2 else []),
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            **kwargs
        )
        
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.hedge_ratio = hedge_ratio
        self.use_price_ratio = use_price_ratio
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成跨品种套利信号
        
        Args:
            data: 包含两个品种价格的数据
                  需要列：'close_1' 和 'close_2' 或根据品种名映射
                  
        Returns:
            信号DataFrame，包含：
            - signal: 综合信号
            - signal_1: 品种1信号
            - signal_2: 品种2信号
            - weight: 权重
            - confidence: 置信度
            - price_ratio: 价比（如使用价比）
            - spread: 价差（如使用价差）
            - zscore: 标准化值
        """
        if data.empty:
            logger.warning("输入数据为空")
            return pd.DataFrame()
        
        try:
            data = data.copy()
            
            # 确定价格列
            if 'close_1' in data.columns and 'close_2' in data.columns:
                col1 = 'close_1'
                col2 = 'close_2'
            elif self.symbol1 and self.symbol2:
                col1 = f'close_{self.symbol1}'
                col2 = f'close_{self.symbol2}'
                if col1 not in data.columns or col2 not in data.columns:
                    close_cols = [c for c in data.columns if c.startswith('close')]
                    if len(close_cols) >= 2:
                        col1, col2 = close_cols[0], close_cols[1]
                    else:
                        raise StrategyError(
                            f"无法找到价格列，需要 '{col1}' 和 '{col2}'"
                        )
            else:
                raise StrategyError("必须指定 symbol1 和 symbol2 或提供 close_1/close_2 列")
            
            # 计算价比或价差
            if self.use_price_ratio:
                data['price_ratio'] = data[col1] / data[col2]
                series = data['price_ratio']
                series_name = 'price_ratio'
            else:
                data['spread'] = self.calculate_spread(
                    data[col1], data[col2], self.hedge_ratio
                )
                series = data['spread']
                series_name = 'spread'
            
            # 计算统计特征
            data[f'{series_name}_mean'] = series.rolling(window=self.lookback).mean()
            data[f'{series_name}_std'] = series.rolling(window=self.lookback).std()
            
            # 标准化
            data['zscore'] = (series - data[f'{series_name}_mean']) / data[f'{series_name}_std']
            
            # 生成均值回归信号
            signals = self.generate_mean_reversion_signals(data['zscore'])
            
            # 转换为跨品种套利信号
            # signal = 1: 做多价比/价差（买品种1卖品种2）
            # signal = -1: 做空价比/价差（卖品种1买品种2）
            signals['signal_1'] = signals['signal']   # 品种1信号
            signals['signal_2'] = -signals['signal']  # 品种2信号（反向）
            
            # 添加价格和比率信息
            if self.use_price_ratio:
                signals['price_ratio'] = data['price_ratio']
                signals['price_ratio_mean'] = data['price_ratio_mean']
                signals['price_ratio_std'] = data['price_ratio_std']
            else:
                signals['spread'] = data['spread']
                signals['spread_mean'] = data['spread_mean']
                signals['spread_std'] = data['spread_std']
                signals['hedge_ratio'] = self.hedge_ratio
            
            # 添加品种信息
            signals['symbol1'] = self.symbol1 or col1
            signals['symbol2'] = self.symbol2 or col2
            
            # 信号原因
            signals['reason'] = signals.apply(
                lambda x: (
                    'cross_variety_long_ratio' if x['signal'] == 1 and self.use_price_ratio
                    else 'cross_variety_short_ratio' if x['signal'] == -1 and self.use_price_ratio
                    else 'cross_variety_long_spread' if x['signal'] == 1
                    else 'cross_variety_short_spread' if x['signal'] == -1
                    else 'cross_variety_exit'
                ),
                axis=1
            )
            
            logger.info(
                f"生成跨品种套利信号: {self.symbol1}-{self.symbol2}, "
                f"使用{'价比' if self.use_price_ratio else '价差'}, "
                f"信号数量: {(signals['signal'] != 0).sum()}"
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"生成跨品种套利信号失败: {e}")
            raise StrategyError(f"生成跨品种套利信号失败: {e}")
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        """获取参数优化边界"""
        return {
            'lookback': (20, 120),
            'hedge_ratio': (0.5, 2.0),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.2, 1.0),
        }


class FuturesSpotArbitrageStrategy(ArbitrageStrategy):
    """
    期现套利策略
    
    利用期货价格与现货价格之间的偏离进行套利。
    
    理论基础：
    - 期货理论价格 = 现货价格 * (1 + 无风险利率 * 期限) + 持有成本
    - 当期货价格偏离理论价格时，存在套利机会
    
    套利类型：
    - 正向套利（Contango）：期货升水，卖期货买现货
    - 反向套利（Backwardation）：期货贴水，买期货卖现货
    
    参数：
        futures_symbol: 期货合约代码
        spot_symbol: 现货品种代码
        risk_free_rate: 无风险利率，默认0.03
        holding_cost: 持有成本率，默认0.02
        days_to_maturity: 到期天数，默认90
        entry_threshold: 入场阈值（年化收益率），默认0.05
        exit_threshold: 出场阈值，默认0.01
        lookback: 回看周期，默认60
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        futures_symbol: Optional[str] = None,
        spot_symbol: Optional[str] = None,
        risk_free_rate: float = 0.03,
        holding_cost: float = 0.02,
        days_to_maturity: int = 90,
        entry_threshold: float = 0.05,
        exit_threshold: float = 0.01,
        lookback: int = 60,
        **kwargs
    ):
        super().__init__(
            name=name or 'FuturesSpotArbitrage',
            symbols=symbols or ([futures_symbol, spot_symbol] if futures_symbol and spot_symbol else []),
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            **kwargs
        )
        
        self.futures_symbol = futures_symbol
        self.spot_symbol = spot_symbol
        self.risk_free_rate = risk_free_rate
        self.holding_cost = holding_cost
        self.days_to_maturity = days_to_maturity
    
    def calculate_theoretical_basis(
        self,
        spot_price: pd.Series,
        risk_free_rate: Optional[float] = None,
        holding_cost: Optional[float] = None,
        days_to_maturity: Optional[int] = None
    ) -> pd.Series:
        """
        计算理论基差
        
        Args:
            spot_price: 现货价格序列
            risk_free_rate: 无风险利率
            holding_cost: 持有成本率
            days_to_maturity: 到期天数
            
        Returns:
            理论基差序列
        """
        rf = risk_free_rate or self.risk_free_rate
        hc = holding_cost or self.holding_cost
        days = days_to_maturity or self.days_to_maturity
        
        # 理论期货价格
        theoretical_futures = spot_price * (1 + (rf + hc) * days / 365)
        
        # 理论基差 = 理论期货 - 现货
        return theoretical_futures - spot_price
    
    def calculate_arbitrage_return(
        self,
        futures_price: pd.Series,
        spot_price: pd.Series,
        days_to_maturity: Optional[int] = None
    ) -> pd.Series:
        """
        计算套利年化收益率
        
        Args:
            futures_price: 期货价格序列
            spot_price: 现货价格序列
            days_to_maturity: 到期天数
            
        Returns:
            年化收益率序列
        """
        days = days_to_maturity or self.days_to_maturity
        
        # 基差 = 期货 - 现货
        basis = futures_price - spot_price
        
        # 理论持有成本
        theoretical_cost = spot_price * (self.risk_free_rate + self.holding_cost) * days / 365
        
        # 超额基差
        excess_basis = basis - theoretical_cost
        
        # 年化收益率（考虑资金占用）
        # 正向套利：卖出期货，买入现货，资金占用约为现货价格
        # 反向套利：买入期货，卖出现货，资金占用约为期货保证金（简化按现货计算）
        annual_return = excess_basis / spot_price * (365 / days)
        
        return annual_return
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成期现套利信号
        
        Args:
            data: 包含期货和现货价格的数据
                  需要列：'close_futures' 和 'close_spot' 或根据品种名映射
                  
        Returns:
            信号DataFrame，包含：
            - signal: 信号（1=正向套利，-1=反向套利）
            - weight: 权重
            - confidence: 置信度
            - basis: 基差
            - theoretical_basis: 理论基差
            - arbitrage_return: 套利年化收益率
        """
        if data.empty:
            logger.warning("输入数据为空")
            return pd.DataFrame()
        
        try:
            data = data.copy()
            
            # 确定价格列
            if 'close_futures' in data.columns and 'close_spot' in data.columns:
                futures_col = 'close_futures'
                spot_col = 'close_spot'
            elif self.futures_symbol and self.spot_symbol:
                futures_col = f'close_{self.futures_symbol}'
                spot_col = f'close_{self.spot_symbol}'
                if futures_col not in data.columns or spot_col not in data.columns:
                    close_cols = [c for c in data.columns if c.startswith('close')]
                    if len(close_cols) >= 2:
                        futures_col, spot_col = close_cols[0], close_cols[1]
                    else:
                        raise StrategyError(
                            f"无法找到价格列，需要 '{futures_col}' 和 '{spot_col}'"
                        )
            else:
                raise StrategyError(
                    "必须指定 futures_symbol 和 spot_symbol 或提供 close_futures/close_spot 列"
                )
            
            # 计算基差
            data['basis'] = data[futures_col] - data[spot_col]
            
            # 计算理论基差
            data['theoretical_basis'] = self.calculate_theoretical_basis(data[spot_col])
            
            # 计算套利年化收益率
            data['arbitrage_return'] = self.calculate_arbitrage_return(
                data[futures_col], data[spot_col]
            )
            
            # 计算历史统计特征
            data['return_mean'] = data['arbitrage_return'].rolling(window=self.lookback).mean()
            data['return_std'] = data['arbitrage_return'].rolling(window=self.lookback).std()
            
            # 生成信号
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            signals['weight'] = 0.0
            signals['confidence'] = 0.0
            
            # 正向套利机会：期货升水过高（卖期货买现货）
            contango_opportunity = data['arbitrage_return'] > self.entry_threshold
            
            # 反向套利机会：期货贴水（买期货卖现货）
            backwardation_opportunity = data['arbitrage_return'] < -self.entry_threshold
            
            signals.loc[contango_opportunity, 'signal'] = 1   # 正向套利
            signals.loc[backwardation_opportunity, 'signal'] = -1  # 反向套利
            
            # 出场信号（收益回归）
            exit_zone = data['arbitrage_return'].abs() < self.exit_threshold
            signals.loc[exit_zone, 'signal'] = 0
            
            # 权重：根据收益偏离程度
            deviation = data['arbitrage_return'].abs() / self.entry_threshold
            signals['weight'] = deviation.clip(0.3, 1.0)
            signals.loc[exit_zone, 'weight'] = 0.0
            
            # 置信度：考虑基差的稳定性
            signals['confidence'] = signals['weight'] * (
                1 - (data['return_std'] / data['return_std'].rolling(window=self.lookback).mean()).fillna(1).clip(0, 1)
            )
            
            # 添加套利信息
            signals['basis'] = data['basis']
            signals['theoretical_basis'] = data['theoretical_basis']
            signals['arbitrage_return'] = data['arbitrage_return']
            signals['return_mean'] = data['return_mean']
            
            # 添加品种信息
            signals['futures_symbol'] = self.futures_symbol or futures_col
            signals['spot_symbol'] = self.spot_symbol or spot_col
            
            # 信号原因
            signals['reason'] = signals.apply(
                lambda x: (
                    'futures_spot_contango' if x['signal'] == 1
                    else 'futures_spot_backwardation' if x['signal'] == -1
                    else 'futures_spot_exit'
                ),
                axis=1
            )
            
            logger.info(
                f"生成期现套利信号: 期货={self.futures_symbol}, 现货={self.spot_symbol}, "
                f"正向套利: {contango_opportunity.sum()}, 反向套利: {backwardation_opportunity.sum()}"
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"生成期现套利信号失败: {e}")
            raise StrategyError(f"生成期现套利信号失败: {e}")
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        """获取参数优化边界"""
        return {
            'lookback': (20, 120),
            'risk_free_rate': (0.01, 0.08),
            'holding_cost': (0.01, 0.05),
            'entry_threshold': (0.02, 0.1),
            'exit_threshold': (0.005, 0.03),
        }
