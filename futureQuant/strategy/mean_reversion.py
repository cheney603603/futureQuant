"""
均值回归策略模块

包含多种均值回归策略实现：
- 基础均值回归策略（布林带、RSI）
- RSI策略
- 布林带策略
- 协整策略（配对交易）
"""

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from .base import BaseStrategy
from ..core.logger import get_logger
from ..core.exceptions import StrategyError

logger = get_logger('strategy.mean_reversion')


class MeanReversionStrategy(BaseStrategy):
    """
    基础均值回归策略
    
    基于价格偏离均值程度产生反向交易信号。
    
    信号规则：
    - 价格低于均值-N倍标准差：做多（超卖回归）
    - 价格高于均值+N倍标准差：做空（超买回归）
    
    参数：
        lookback: 回看周期，默认20
        std_dev: 标准差倍数，默认2.0
        use_zscore: 是否使用Z-score，默认True
        entry_threshold: 入场阈值，默认2.0
        exit_threshold: 出场阈值，默认0.5
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        lookback: int = 20,
        std_dev: float = 2.0,
        use_zscore: bool = True,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(
            name=name or 'MeanReversion',
            symbols=symbols,
            lookback=lookback,
            std_dev=std_dev,
            use_zscore=use_zscore,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            **kwargs
        )
        
        self.lookback = lookback
        self.std_dev = std_dev
        self.use_zscore = use_zscore
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成均值回归信号"""
        if data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        
        # 计算移动平均和标准差
        data['mean'] = data['close'].rolling(window=self.lookback).mean()
        data['std'] = data['close'].rolling(window=self.lookback).std()
        
        # 计算偏离度
        if self.use_zscore:
            data['deviation'] = (data['close'] - data['mean']) / data['std']
        else:
            data['deviation'] = (data['close'] - data['mean']) / data['mean'] * 100
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        
        # 超卖：价格低于均值-N倍标准差
        oversold = data['deviation'] < -self.entry_threshold
        
        # 超买：价格高于均值+N倍标准差
        overbought = data['deviation'] > self.entry_threshold
        
        # 回归区域：接近均值
        near_mean = data['deviation'].abs() < self.exit_threshold
        
        signals.loc[oversold, 'signal'] = 1    # 做多
        signals.loc[overbought, 'signal'] = -1  # 做空
        
        # 权重：根据偏离程度
        signals['weight'] = (data['deviation'].abs() / self.entry_threshold).clip(0.3, 1.0)
        
        # 置信度
        signals['confidence'] = signals['weight']
        
        # 偏离度
        signals['deviation'] = data['deviation']
        signals['mean'] = data['mean']
        signals['upper_band'] = data['mean'] + self.std_dev * data['std']
        signals['lower_band'] = data['mean'] - self.std_dev * data['std']
        
        signals['reason'] = 'mean_reversion'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'lookback': (10, 40),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.2, 1.0),
        }


class RSIStrategy(BaseStrategy):
    """
    RSI均值回归策略
    
    基于RSI指标的超买超卖信号进行均值回归交易。
    
    信号规则：
    - RSI < 超卖阈值：做多
    - RSI > 超买阈值：做空
    - RSI回到中性区域：平仓
    
    参数：
        rsi_period: RSI计算周期，默认14
        oversold: 超卖阈值，默认30
        overbought: 超买阈值，默认70
        use_divergence: 是否使用背离信号，默认False
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        use_divergence: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name or 'RSI',
            symbols=symbols,
            rsi_period=rsi_period,
            oversold=oversold,
            overbought=overbought,
            use_divergence=use_divergence,
            **kwargs
        )
        
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.use_divergence = use_divergence
    
    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """计算RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _detect_divergence(
        self, 
        price: pd.Series, 
        rsi: pd.Series,
        lookback: int = 14
    ) -> pd.Series:
        """
        检测RSI背离
        
        Returns:
            背离信号序列：1=底背离, -1=顶背离, 0=无背离
        """
        divergence = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            # 底背离：价格创新低但RSI未创新低
            price_low = price.iloc[i-lookback:i+1].min()
            rsi_low = rsi.iloc[i-lookback:i+1].min()
            
            if price.iloc[i] <= price_low and rsi.iloc[i] > rsi_low:
                divergence.iloc[i] = 1
            
            # 顶背离：价格创新高但RSI未创新高
            price_high = price.iloc[i-lookback:i+1].max()
            rsi_high = rsi.iloc[i-lookback:i+1].max()
            
            if price.iloc[i] >= price_high and rsi.iloc[i] < rsi_high:
                divergence.iloc[i] = -1
        
        return divergence
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成RSI信号"""
        if data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        
        # 计算RSI
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        
        # 超卖信号
        oversold = data['rsi'] < self.oversold
        
        # 超买信号
        overbought = data['rsi'] > self.overbought
        
        signals.loc[oversold, 'signal'] = 1
        signals.loc[overbought, 'signal'] = -1
        
        # 背离信号
        if self.use_divergence:
            signals['divergence'] = self._detect_divergence(
                data['close'], data['rsi']
            )
        
        # 权重：根据RSI偏离程度
        rsi_deviation = (
            (data['rsi'] - 50).abs() / 50
        )
        signals['weight'] = rsi_deviation.clip(0.3, 1.0)
        
        # 置信度
        signals['confidence'] = signals['weight']
        
        # RSI值
        signals['rsi'] = data['rsi']
        
        signals['reason'] = 'rsi'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'rsi_period': (7, 21),
            'oversold': (20, 35),
            'overbought': (65, 80),
        }


class BollingerBandsStrategy(BaseStrategy):
    """
    布林带策略
    
    使用布林带进行均值回归交易。
    
    布林带由三条线组成：
    - 中轨：N日移动平均
    - 上轨：中轨 + K倍标准差
    - 下轨：中轨 - K倍标准差
    
    信号规则：
    - 价格触及下轨：做多
    - 价格触及上轨：做空
    - 价格回到中轨：平仓
    
    参数：
        period: 计算周期，默认20
        std_dev: 标准差倍数，默认2.0
        use_band_width: 是否使用带宽过滤，默认False
        bandwidth_threshold: 带宽阈值，默认0.03
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        period: int = 20,
        std_dev: float = 2.0,
        use_band_width: bool = False,
        bandwidth_threshold: float = 0.03,
        **kwargs
    ):
        super().__init__(
            name=name or 'BollingerBands',
            symbols=symbols,
            period=period,
            std_dev=std_dev,
            use_band_width=use_band_width,
            bandwidth_threshold=bandwidth_threshold,
            **kwargs
        )
        
        self.period = period
        self.std_dev = std_dev
        self.use_band_width = use_band_width
        self.bandwidth_threshold = bandwidth_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成布林带信号"""
        if data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        
        # 中轨
        data['mid'] = data['close'].rolling(window=self.period).mean()
        
        # 标准差
        data['std'] = data['close'].rolling(window=self.period).std()
        
        # 布林带
        data['upper'] = data['mid'] + self.std_dev * data['std']
        data['lower'] = data['mid'] - self.std_dev * data['std']
        
        # 布林带宽度
        data['bandwidth'] = (data['upper'] - data['lower']) / data['mid']
        
        # 布林带位置 (% B)
        data['bb_position'] = (data['close'] - data['lower']) / (data['upper'] - data['lower'])
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        
        # 带宽过滤：布林带带宽过窄时不交易
        if self.use_band_width:
            valid_bandwidth = data['bandwidth'] >= self.bandwidth_threshold
        else:
            valid_bandwidth = pd.Series(True, index=data.index)
        
        # 触及下轨：做多
        touch_lower = (data['bb_position'] < 0.05) & valid_bandwidth
        
        # 触及上轨：做空
        touch_upper = (data['bb_position'] > 0.95) & valid_bandwidth
        
        # 回归中轨：平仓信号
        near_mid = data['bb_position'].between(0.4, 0.6)
        
        signals.loc[touch_lower, 'signal'] = 1
        signals.loc[touch_upper, 'signal'] = -1
        signals.loc[near_mid, 'signal'] = 0
        
        # 权重：根据偏离程度
        signals['weight'] = data['bb_position'].apply(
            lambda x: max(0.3, min(1.0, abs(x - 0.5) * 2)) if not pd.isna(x) else 0
        )
        
        # 置信度
        signals['confidence'] = signals['weight']
        
        # 布林带值
        signals['upper'] = data['upper']
        signals['lower'] = data['lower']
        signals['mid'] = data['mid']
        signals['bb_position'] = data['bb_position']
        
        signals['reason'] = 'bollinger_bands'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'period': (10, 30),
            'std_dev': (1.5, 2.5),
        }


class CointegrationStrategy(BaseStrategy):
    """
    协整策略（配对交易）
    
    基于两个品种的协整关系进行统计套利。
    
    原理：
    - 找到两个价格序列存在长期均衡关系（协整）的品种
    - 当价差偏离均衡时，做空强品种、做多弱品种
    - 当价差回归时平仓获利
    
    参数：
        pair: 交易对，如 ('RB', 'HC')
        lookback: 协整检验窗口，默认120
        entry_zscore: 入场Z-score阈值，默认2.0
        exit_zscore: 出场Z-score阈值，默认0.5
        hedge_ratio_method: 对冲比例计算方法，'ols'/'kalman'
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        pair: Optional[Tuple[str, str]] = None,
        lookback: int = 120,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        hedge_ratio_method: str = 'ols',
        **kwargs
    ):
        if pair is None and symbols is not None and len(symbols) >= 2:
            pair = (symbols[0], symbols[1])
        
        super().__init__(
            name=name or 'Cointegration',
            symbols=symbols or ([pair[0], pair[1]] if pair else []),
            lookback=lookback,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            hedge_ratio_method=hedge_ratio_method,
            **kwargs
        )
        
        self.pair = pair
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.hedge_ratio_method = hedge_ratio_method
        
        # 缓存
        self._hedge_ratio: Optional[float] = None
        self._spread: Optional[pd.Series] = None
    
    def calculate_hedge_ratio(
        self, 
        price1: pd.Series, 
        price2: pd.Series
    ) -> float:
        """
        计算对冲比例
        
        使用OLS回归：price1 = hedge_ratio * price2 + c
        """
        if self.hedge_ratio_method == 'ols':
            # OLS回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                price2, price1
            )
            return slope
        
        elif self.hedge_ratio_method == 'kalman':
            # 卡尔曼滤波（动态对冲比例）
            # 简化实现，实际应使用pykalman库
            window = min(30, len(price1) // 4)
            return price1.rolling(window=window).corr(price2).iloc[-1]
        
        else:
            raise StrategyError(f"Unknown hedge ratio method: {self.hedge_ratio_method}")
    
    def test_cointegration(
        self, 
        price1: pd.Series, 
        price2: pd.Series
    ) -> Dict[str, float]:
        """
        检验协整关系
        
        使用Engle-Granger两步法
        """
        # 计算对冲比例
        hedge_ratio = self.calculate_hedge_ratio(price1, price2)
        
        # 计算价差
        spread = price1 - hedge_ratio * price2
        
        # ADF检验
        adf_result = stats.adfuller(spread.dropna())
        
        return {
            'hedge_ratio': hedge_ratio,
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_cointegrated': adf_result[1] < 0.05,
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成配对交易信号
        
        Args:
            data: 包含两个品种价格的数据，需要列 'close_1' 和 'close_2'
        """
        if data.empty:
            return pd.DataFrame()
        
        if 'close_1' not in data.columns or 'close_2' not in data.columns:
            raise StrategyError("Data must contain 'close_1' and 'close_2' columns")
        
        data = data.copy()
        
        # 滚动计算对冲比例
        hedge_ratios = []
        for i in range(self.lookback, len(data) + 1):
            p1 = data['close_1'].iloc[i-self.lookback:i]
            p2 = data['close_2'].iloc[i-self.lookback:i]
            hedge_ratios.append(self.calculate_hedge_ratio(p1, p2))
        
        data.loc[data.index[self.lookback-1:], 'hedge_ratio'] = hedge_ratios
        
        # 计算价差
        data['spread'] = data['close_1'] - data['hedge_ratio'] * data['close_2']
        
        # 标准化价差（Z-score）
        data['spread_mean'] = data['spread'].rolling(window=self.lookback).mean()
        data['spread_std'] = data['spread'].rolling(window=self.lookback).std()
        data['zscore'] = (data['spread'] - data['spread_mean']) / data['spread_std']
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['signal_1'] = 0  # 品种1信号
        signals['signal_2'] = 0  # 品种2信号
        signals['weight'] = 0.0
        
        # 价差过大：做空价差（卖1买2）
        spread_high = data['zscore'] > self.entry_zscore
        signals.loc[spread_high, 'signal_1'] = -1
        signals.loc[spread_high, 'signal_2'] = 1
        
        # 价差过小：做多价差（买1卖2）
        spread_low = data['zscore'] < -self.entry_zscore
        signals.loc[spread_low, 'signal_1'] = 1
        signals.loc[spread_low, 'signal_2'] = -1
        
        # 综合信号（用于回测引擎）
        signals['signal'] = signals['signal_1']
        
        # 权重
        signals['weight'] = (data['zscore'].abs() / self.entry_zscore).clip(0.3, 1.0)
        
        # 置信度
        signals['confidence'] = signals['weight']
        
        # 记录价差信息
        signals['spread'] = data['spread']
        signals['zscore'] = data['zscore']
        signals['hedge_ratio'] = data['hedge_ratio']
        
        signals['reason'] = 'cointegration'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'lookback': (60, 180),
            'entry_zscore': (1.5, 3.0),
            'exit_zscore': (0.3, 1.0),
        }
