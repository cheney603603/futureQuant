"""
趋势跟踪策略模块

包含多种趋势跟踪策略实现：
- 基础趋势策略（均线、动量）
- 双均线策略
- 突破策略
- 唐奇安通道策略
"""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np

from .base import BaseStrategy, SignalType
from ..core.logger import get_logger
from ..core.exceptions import StrategyError
from ..core.base import Factor

logger = get_logger('strategy.trend_following')


class TrendFollowingStrategy(BaseStrategy):
    """
    基础趋势跟踪策略
    
    结合均线和动量指标的趋势跟踪策略。
    当价格在均线上方且动量为正时做多，反之做空。
    
    参数：
        ma_period: 移动平均周期，默认20
        momentum_period: 动量计算周期，默认10
        use_atr_filter: 是否使用ATR过滤信号，默认False
        atr_period: ATR计算周期，默认14
        atr_threshold: ATR阈值倍数，默认1.0
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        ma_period: int = 20,
        momentum_period: int = 10,
        use_atr_filter: bool = False,
        atr_period: int = 14,
        atr_threshold: float = 1.0,
        **kwargs
    ):
        super().__init__(
            name=name or 'TrendFollowing',
            symbols=symbols,
            ma_period=ma_period,
            momentum_period=momentum_period,
            use_atr_filter=use_atr_filter,
            atr_period=atr_period,
            atr_threshold=atr_threshold,
            **kwargs
        )
        
        self.ma_period = ma_period
        self.momentum_period = momentum_period
        self.use_atr_filter = use_atr_filter
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成趋势跟踪信号
        
        Args:
            data: OHLCV数据
            
        Returns:
            信号DataFrame
        """
        if data.empty:
            return pd.DataFrame()
        
        # 计算均线
        data = data.copy()
        data['ma'] = data['close'].rolling(window=self.ma_period).mean()
        
        # 计算动量
        data['momentum'] = (
            data['close'] - data['close'].shift(self.momentum_period)
        ) / data['close'].shift(self.momentum_period)
        
        # 计算ATR（如果启用）
        if self.use_atr_filter:
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            data['atr'] = tr.rolling(window=self.atr_period).mean()
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        
        # 趋势判断
        above_ma = data['close'] > data['ma']
        momentum_positive = data['momentum'] > 0
        
        # 做多信号：价格在均线上方 + 动量为正
        long_signal = above_ma & momentum_positive
        
        # 做空信号：价格在均线下方 + 动量为负
        short_signal = ~above_ma & ~momentum_positive
        
        signals.loc[long_signal, 'signal'] = 1
        signals.loc[short_signal, 'signal'] = -1
        
        # ATR过滤：波动率过低时不交易
        if self.use_atr_filter:
            atr_mean = data['atr'].rolling(window=self.atr_period * 2).mean()
            low_volatility = data['atr'] < atr_mean * self.atr_threshold
            signals.loc[low_volatility, 'signal'] = 0
        
        # 设置权重（根据动量强度）
        momentum_abs = data['momentum'].abs()
        signals['weight'] = momentum_abs / momentum_abs.max() if momentum_abs.max() > 0 else 0
        signals['weight'] = signals['weight'].clip(0.2, 1.0)
        
        # 添加置信度
        signals['confidence'] = signals['weight']
        
        # 添加信号原因
        signals['reason'] = 'trend_following'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        """获取参数优化边界"""
        return {
            'ma_period': (5, 60),
            'momentum_period': (5, 30),
            'atr_threshold': (0.5, 2.0),
        }


class DualMAStrategy(BaseStrategy):
    """
    双均线策略
    
    经典的趋势跟踪策略，使用快慢均线交叉产生信号。
    
    信号规则：
    - 金叉（快线上穿慢线）：做多
    - 死叉（快线下穿慢线）：做空
    
    参数：
        fast_period: 快线周期，默认10
        slow_period: 慢线周期，默认30
        ma_type: 均线类型，'sma'/'ema'/'wma'，默认'sma'
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        fast_period: int = 10,
        slow_period: int = 30,
        ma_type: str = 'sma',
        **kwargs
    ):
        if fast_period >= slow_period:
            raise StrategyError("fast_period must be less than slow_period")
        
        super().__init__(
            name=name or 'DualMA',
            symbols=symbols,
            fast_period=fast_period,
            slow_period=slow_period,
            ma_type=ma_type,
            **kwargs
        )
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
    
    def _calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """计算移动平均线"""
        if self.ma_type == 'sma':
            return series.rolling(window=period).mean()
        elif self.ma_type == 'ema':
            return series.ewm(span=period, adjust=False).mean()
        elif self.ma_type == 'wma':
            weights = np.arange(1, period + 1)
            return series.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        else:
            raise StrategyError(f"Unknown MA type: {self.ma_type}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成双均线信号"""
        if data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        
        # 计算快慢均线
        data['fast_ma'] = self._calculate_ma(data['close'], self.fast_period)
        data['slow_ma'] = self._calculate_ma(data['close'], self.slow_period)
        
        # 均线差值
        data['ma_diff'] = data['fast_ma'] - data['slow_ma']
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        
        # 金叉：快线上穿慢线
        golden_cross = (
            (data['ma_diff'] > 0) & 
            (data['ma_diff'].shift(1) <= 0)
        )
        
        # 死叉：快线下穿慢线
        death_cross = (
            (data['ma_diff'] < 0) & 
            (data['ma_diff'].shift(1) >= 0)
        )
        
        # 持仓状态
        long_position = data['ma_diff'] > 0
        short_position = data['ma_diff'] < 0
        
        # 设置信号
        signals.loc[golden_cross, 'signal'] = 1
        signals.loc[death_cross, 'signal'] = -1
        
        # 设置持仓权重
        signals.loc[long_position, 'weight'] = 1.0
        signals.loc[short_position, 'weight'] = 1.0
        
        # 置信度：基于均线差值大小
        ma_diff_norm = data['ma_diff'].abs() / data['close']
        signals['confidence'] = (ma_diff_norm / ma_diff_norm.max()).fillna(0)
        
        signals['reason'] = 'dual_ma_cross'
        signals['fast_ma'] = data['fast_ma']
        signals['slow_ma'] = data['slow_ma']
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'fast_period': (5, 20),
            'slow_period': (20, 60),
        }


class BreakoutStrategy(BaseStrategy):
    """
    突破策略
    
    基于价格突破历史高低点产生信号的经典趋势跟踪策略。
    
    信号规则：
    - 价格突破N日高点：做多
    - 价格跌破N日低点：做空
    
    参数：
        lookback: 回看周期，默认20
        breakout_threshold: 突破阈值（百分比），默认0
        use_trailing_stop: 是否使用移动止损，默认False
        trailing_percent: 移动止损比例，默认0.05
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        lookback: int = 20,
        breakout_threshold: float = 0,
        use_trailing_stop: bool = False,
        trailing_percent: float = 0.05,
        **kwargs
    ):
        super().__init__(
            name=name or 'Breakout',
            symbols=symbols,
            lookback=lookback,
            breakout_threshold=breakout_threshold,
            use_trailing_stop=use_trailing_stop,
            trailing_percent=trailing_percent,
            **kwargs
        )
        
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        self.use_trailing_stop = use_trailing_stop
        self.trailing_percent = trailing_percent
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成突破信号"""
        if data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        
        # 计算N日高低点
        data['high_n'] = data['high'].rolling(window=self.lookback).max().shift(1)
        data['low_n'] = data['low'].rolling(window=self.lookback).min().shift(1)
        
        # 应用突破阈值
        if self.breakout_threshold > 0:
            data['breakout_high'] = data['high_n'] * (1 + self.breakout_threshold)
            data['breakout_low'] = data['low_n'] * (1 - self.breakout_threshold)
        else:
            data['breakout_high'] = data['high_n']
            data['breakout_low'] = data['low_n']
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        
        # 向上突破
        upward_breakout = data['close'] > data['breakout_high']
        
        # 向下突破
        downward_breakout = data['close'] < data['breakout_low']
        
        signals.loc[upward_breakout, 'signal'] = 1
        signals.loc[downward_breakout, 'signal'] = -1
        
        # 权重：突破幅度
        breakout_pct = (
            (data['close'] - data['breakout_high']).abs() / data['close']
        )
        signals['weight'] = (breakout_pct / breakout_pct.max()).fillna(0)
        signals['weight'] = signals['weight'].clip(0.3, 1.0)
        
        # 置信度
        signals['confidence'] = signals['weight']
        
        # 记录突破价格
        signals['breakout_high'] = data['breakout_high']
        signals['breakout_low'] = data['breakout_low']
        
        signals['reason'] = 'breakout'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'lookback': (10, 50),
            'breakout_threshold': (0, 0.02),
            'trailing_percent': (0.02, 0.1),
        }


class DonchianChannelStrategy(BaseStrategy):
    """
    唐奇安通道策略
    
    经典的海龟交易策略核心，使用唐奇安通道进行趋势跟踪。
    
    唐奇安通道由三条线组成：
    - 上轨：N日最高价
    - 下轨：N日最低价
    - 中轨：(上轨 + 下轨) / 2
    
    信号规则：
    - 价格突破上轨：做多
    - 价格跌破下轨：做空
    - 价格回到中轨：平仓
    
    参数：
        entry_period: 入场通道周期，默认20
        exit_period: 出场通道周期，默认10
        use_pyramid: 是否使用金字塔加仓，默认False
        pyramid_count: 最大加仓次数，默认4
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        entry_period: int = 20,
        exit_period: int = 10,
        use_pyramid: bool = False,
        pyramid_count: int = 4,
        **kwargs
    ):
        super().__init__(
            name=name or 'DonchianChannel',
            symbols=symbols,
            entry_period=entry_period,
            exit_period=exit_period,
            use_pyramid=use_pyramid,
            pyramid_count=pyramid_count,
            **kwargs
        )
        
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.use_pyramid = use_pyramid
        self.pyramid_count = pyramid_count
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成唐奇安通道信号"""
        if data.empty:
            return pd.DataFrame()
        
        data = data.copy()
        
        # 计算入场通道（唐奇安通道）
        data['entry_high'] = data['high'].rolling(window=self.entry_period).max().shift(1)
        data['entry_low'] = data['low'].rolling(window=self.entry_period).min().shift(1)
        data['entry_mid'] = (data['entry_high'] + data['entry_low']) / 2
        
        # 计算出场通道
        data['exit_high'] = data['high'].rolling(window=self.exit_period).max().shift(1)
        data['exit_low'] = data['low'].rolling(window=self.exit_period).min().shift(1)
        
        # 生成信号
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['weight'] = 0.0
        signals['position_state'] = 0  # 用于跟踪持仓状态
        
        # 跟踪持仓状态
        position = 0
        for i, (idx, row) in enumerate(data.iterrows()):
            if pd.isna(row['entry_high']) or pd.isna(row['entry_low']):
                continue
            
            close = row['close']
            
            if position == 0:
                # 空仓状态：检查入场信号
                if close > row['entry_high']:
                    position = 1  # 做多
                    signals.loc[idx, 'signal'] = 1
                elif close < row['entry_low']:
                    position = -1  # 做空
                    signals.loc[idx, 'signal'] = -1
            
            elif position == 1:
                # 持有多头：检查出场信号
                if close < row['exit_low']:
                    position = 0  # 平仓
                    signals.loc[idx, 'signal'] = 0
                # 金字塔加仓逻辑
                elif self.use_pyramid and close > data.loc[:idx, 'close'].iloc[-1]:
                    signals.loc[idx, 'signal'] = 1
            
            elif position == -1:
                # 持有空头：检查出场信号
                if close > row['exit_high']:
                    position = 0  # 平仓
                    signals.loc[idx, 'signal'] = 0
                # 金字塔加仓逻辑
                elif self.use_pyramid and close < data.loc[:idx, 'close'].iloc[-1]:
                    signals.loc[idx, 'signal'] = -1
            
            signals.loc[idx, 'position_state'] = position
        
        # 设置权重
        signals['weight'] = signals['signal'].abs().astype(float)
        signals.loc[signals['weight'] == 0, 'weight'] = signals.loc[signals['signal'] == 0].index.map(
            lambda x: 1.0 if signals.loc[x, 'position_state'] != 0 else 0.0
        )
        
        # 置信度
        channel_width = (data['entry_high'] - data['entry_low']) / data['entry_mid']
        signals['confidence'] = (1 - channel_width / channel_width.max()).fillna(0.5)
        
        # 通道值
        signals['channel_high'] = data['entry_high']
        signals['channel_low'] = data['entry_low']
        signals['channel_mid'] = data['entry_mid']
        
        signals['reason'] = 'donchian_channel'
        
        return signals
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        return {
            'entry_period': (10, 40),
            'exit_period': (5, 20),
            'pyramid_count': (1, 6),
        }
