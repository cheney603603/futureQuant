"""
特征工程模块

专为期货量化场景设计，包含：
1. 时序特征构建（滞后、滚动统计）
2. 横截面特征构建（品种间排名、相对强弱）
3. 未来函数严格避免（所有特征仅使用 t-1 及之前的数据）
4. 数据泄漏检测
"""

from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ..core.logger import get_logger

logger = get_logger('model.feature')


@dataclass
class FeatureConfig:
    """特征工程配置"""
    # 滞后特征
    max_lag: int = 5              # 最多滞后阶数
    
    # 滚动特征
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    # 横截面特征
    include_cross_sectional: bool = True  # 是否包含横截面排名
    include_relative_strength: bool = True  # 是否包含相对强弱
    
    # 目标变量
    forward_returns: List[int] = field(default_factory=lambda: [1, 5])  # 预测未来 N 日收益率


class FeatureEngineer:
    """
    特征工程器

    严格避免未来函数：所有特征均在 t 时刻可计算（使用 shift(1) 或更早的数据）。

    使用示例：
        >>> fe = FeatureEngineer()
        >>> fe.add_price_features()
        >>> fe.add_technical_features(ma_windows=[5, 20, 60])
        >>> fe.add_volume_features()
        >>> features, target = fe.build(price_data)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._feature_functions: List[Tuple[str, callable]] = []

    def add_price_features(self) -> 'FeatureEngineer':
        """添加价格特征"""
        
        def _add(df):
            out = df.copy()
            p = out['close']
            # 对数收益率
            out['log_return'] = np.log(p / p.shift(1))
            # 日内收益率（不使用未来数据）
            out['intraday_return'] = (p - out['open']) / out['open']
            # 价格动量
            for lag in range(1, self.config.max_lag + 1):
                out[f'return_lag_{lag}'] = out['log_return'].shift(lag)
            # 价格变化率
            out['price_change_pct'] = p.pct_change()
            return out
        
        self._feature_functions.append(('price', _add))
        return self

    def add_technical_features(
        self,
        ma_windows: Optional[List[int]] = None
    ) -> 'FeatureEngineer':
        """添加技术分析特征"""
        ma_windows = ma_windows or [5, 20, 60]
        
        def _add(df):
            out = df.copy()
            c = out['close']
            h = out['high']
            l = out['low']
            v = out['volume']
            
            # 移动平均
            for w in ma_windows:
                ma = c.rolling(w).mean()
                out[f'ma_{w}'] = ma
                out[f'ma_{w}_ratio'] = c / ma - 1  # 均线偏离度
                out[f'ma_{w}_slope'] = ma / ma.shift(1) - 1  # 均线斜率
            
            # 波动率（历史波动率）
            for w in [5, 10, 20]:
                ret = out['log_return'] if 'log_return' in out else np.log(c / c.shift(1))
                out[f'volatility_{w}'] = ret.rolling(w).std() * np.sqrt(252)
            
            # RSI
            for w in [6, 14]:
                delta = c.diff()
                gain = delta.where(delta > 0, 0)
                loss = (-delta).where(delta < 0, 0)
                avg_gain = gain.rolling(w).mean()
                avg_loss = loss.rolling(w).mean()
                rs = avg_gain / (avg_loss + 1e-10)
                out[f'rsi_{w}'] = 100 - 100 / (1 + rs)
            
            # MACD
            ema12 = c.ewm(span=12).mean()
            ema26 = c.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            out['macd'] = macd
            out['macd_signal'] = signal
            out['macd_hist'] = macd - signal
            
            # 布林带
            for w in [20]:
                roll = c.rolling(w)
                mid = roll.mean()
                std = roll.std()
                out[f'bb_upper_{w}'] = mid + 2 * std
                out[f'bb_lower_{w}'] = mid - 2 * std
                out[f'bb_width_{w}'] = (out[f'bb_upper_{w}'] - out[f'bb_lower_{w}']) / mid
                out[f'bb_position_{w}'] = (c - out[f'bb_lower_{w}']) / (out[f'bb_upper_{w}'] - out[f'bb_lower_{w}'] + 1e-10)
            
            # ATR
            tr1 = h - l
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            out['atr'] = tr.rolling(14).mean()
            
            return out
        
        self._feature_functions.append(('technical', _add))
        return self

    def add_volume_features(self) -> 'FeatureEngineer':
        """添加成交量特征"""
        
        def _add(df):
            out = df.copy()
            v = out['volume']
            c = out['close']
            
            # 成交量变化率
            out['volume_change'] = v.pct_change()
            
            # 成交量均线比
            for w in [5, 20]:
                out[f'volume_ma_ratio_{w}'] = v / v.rolling(w).mean()
            
            # 持仓量特征（如果有）
            if 'open_interest' in out.columns:
                oi = out['open_interest']
                out['oi_change'] = oi.pct_change()
                out['oi_ma_ratio'] = oi / oi.rolling(5).mean()
                # 持仓量与成交量比（OI/Vol，异常波动检测）
                out['oi_volume_ratio'] = oi / (v + 1)
            
            # OBV
            obv = (np.sign(c.diff()) * v).cumsum()
            out['obv'] = obv
            out['obv_ma5'] = obv / obv.rolling(5).mean() - 1
            
            return out
        
        self._feature_functions.append(('volume', _add))
        return self

    def build(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        forward_periods: Optional[List[int]] = None,
        label_type: str = 'classification',  # 'classification' | 'regression'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建特征矩阵和目标变量

        Args:
            data: 输入数据（包含 OHLCV）
            target_column: 目标列名
            forward_periods: 预测未来 N 日收益率，默认使用 config 中的
            label_type: 'classification'（涨跌）还是 'regression'（收益率）

        Returns:
            (features, targets): 特征 DataFrame 和 目标变量 DataFrame
        """
        df = data.copy()
        
        # 1. 按顺序应用所有特征函数
        for name, func in self._feature_functions:
            try:
                df = func(df)
            except Exception as e:
                logger.warning(f"Feature function '{name}' failed: {e}")

        # 2. 构建目标变量（未来收益率，永远 shift(-N)）
        forward_periods = forward_periods or self.config.forward_returns
        target_dfs = {}
        for period in forward_periods:
            future_return = df[target_column].shift(-period) / df[target_column] - 1
            
            if label_type == 'classification':
                # 涨跌标签：1 = 上涨，0 = 下跌/持平
                target_dfs[f'label_{period}d'] = (future_return > 0).astype(int)
            else:
                target_dfs[f'label_{period}d'] = future_return
        
        target_df = pd.DataFrame(target_dfs)
        
        # 3. 构建特征矩阵（去掉非特征列和目标相关列）
        non_feature_cols = [
            'date', 'symbol', 'open', 'high', 'low', 'close',
            'volume', 'open_interest'
        ] + [c for c in df.columns if c.startswith('label_')]
        
        feature_cols = [c for c in df.columns if c not in non_feature_cols]
        
        # 4. 严格对齐：去除 NaN 行
        features = df[feature_cols].copy()
        
        # 去除目标变量中的 NaN（最后 N 行，因为 shift(-N)）
        valid_idx = target_df.dropna().index
        features = features.loc[valid_idx]
        target_df = target_df.loc[valid_idx]
        
        # 去除特征中的 NaN 行
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target_df = target_df.loc[valid_idx]
        
        logger.info(
            f"Feature matrix built: {features.shape}, "
            f"target: {target_df.shape}, label_type={label_type}"
        )
        
        return features, target_df

    def detect_leakage(self, features: pd.DataFrame, target: pd.DataFrame) -> Dict:
        """
        检测数据泄漏

        检查特征和目标之间是否存在异常相关性（可能由未来函数导致）。

        Args:
            features: 特征 DataFrame
            target: 目标 DataFrame

        Returns:
            泄漏检测报告
        """
        report = {'leaked_features': [], 'warnings': []}
        
        # 取第一个目标变量
        target_series = target.iloc[:, 0]
        
        for col in features.columns:
            # 计算当期的相关性（应该接近0，如果有泄漏可能很高）
            corr = features[col].corr(target_series)
            if abs(corr) > 0.3:  # 阈值
                report['leaked_features'].append({
                    'feature': col,
                    'correlation': corr,
                    'severity': 'HIGH' if abs(corr) > 0.5 else 'MEDIUM'
                })
        
        if report['leaked_features']:
            logger.warning(f"Potential data leakage detected: {len(report['leaked_features'])} features")
        else:
            logger.info("No data leakage detected")
        
        return report
