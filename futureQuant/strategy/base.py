"""
策略基类模块

定义所有策略的公共接口和基础功能
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

from ..core.base import Strategy, Factor
from ..core.logger import get_logger
from ..core.exceptions import StrategyError

logger = get_logger('strategy.base')


class SignalType(Enum):
    """信号类型枚举"""
    LONG = 1       # 做多
    SHORT = -1     # 做空
    FLAT = 0       # 空仓
    CLOSE_LONG = 2   # 平多
    CLOSE_SHORT = -2  # 平空


class PositionSide(Enum):
    """持仓方向枚举"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@pd.api.extensions.register_dataframe_accessor("signals")
class SignalsAccessor:
    """
    信号DataFrame扩展方法
    
    为包含信号的DataFrame提供便捷方法
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def to_positions(self, initial_position: int = 0) -> pd.Series:
        """
        将信号转换为持仓
        
        Args:
            initial_position: 初始持仓
            
        Returns:
            持仓序列
        """
        positions = self._obj['signal'].copy()
        # 累积计算持仓（简化版本，实际需要更复杂的逻辑）
        return positions
    
    def filter_by_confidence(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        按置信度过滤信号
        
        Args:
            threshold: 置信度阈值
            
        Returns:
            过滤后的信号DataFrame
        """
        if 'confidence' in self._obj.columns:
            mask = self._obj['confidence'].abs() >= threshold
            return self._obj[mask]
        return self._obj


class BaseStrategy(Strategy):
    """
    策略基类
    
    继承自core.base.Strategy，提供更丰富的功能：
    - 信号生成、过滤和转换
    - 风险管理（止损、止盈、仓位控制）
    - 回测参数管理
    - 性能统计
    
    子类需要实现：
    - generate_signals(): 核心信号生成逻辑
    """
    
    def __init__(
        self, 
        name: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        **params
    ):
        """
        初始化策略
        
        Args:
            name: 策略名称
            symbols: 交易品种列表
            **params: 策略参数
        """
        super().__init__(name=name, **params)
        self.symbols = symbols or []
        
        # 风险管理参数
        self.stop_loss: Optional[float] = params.get('stop_loss', None)  # 止损比例
        self.take_profit: Optional[float] = params.get('take_profit', None)  # 止盈比例
        self.max_position: float = params.get('max_position', 1.0)  # 最大仓位
        self.risk_per_trade: float = params.get('risk_per_trade', 0.02)  # 单笔风险
        
        # 信号过滤参数
        self.signal_threshold: float = params.get('signal_threshold', 0.0)  # 信号阈值
        self.min_holding_period: int = params.get('min_holding_period', 1)  # 最小持仓周期
        
        # 状态变量
        self._signals: Optional[pd.DataFrame] = None
        self._positions: Optional[pd.DataFrame] = None
        self._last_signal_time: Optional[datetime] = None
        
        # 统计变量
        self._trade_count: int = 0
        self._win_count: int = 0
        self._loss_count: int = 0
    
    @property
    def signals(self) -> Optional[pd.DataFrame]:
        """获取最新生成的信号"""
        return self._signals
    
    @property
    def positions(self) -> Optional[pd.DataFrame]:
        """获取持仓记录"""
        return self._positions
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号（必须由子类实现）
        
        Args:
            data: 输入数据，包含OHLCV等
            
        Returns:
            DataFrame with columns:
            - date: 日期
            - symbol: 品种代码（可选，单品种时省略）
            - signal: 信号值 (1:做多, -1:做空, 0:空仓)
            - weight: 权重 (0-1)
            - confidence: 置信度 (可选，0-1)
            - reason: 信号原因 (可选)
        """
        pass
    
    def apply_risk_management(
        self, 
        signals: pd.DataFrame,
        positions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        应用风险管理规则
        
        Args:
            signals: 原始信号
            positions: 当前持仓
            
        Returns:
            调整后的信号
        """
        if signals.empty:
            return signals
        
        result = signals.copy()
        
        # 信号阈值过滤
        if 'signal' in result.columns and self.signal_threshold != 0:
            mask = result['signal'].abs() >= self.signal_threshold
            result.loc[~mask, 'signal'] = 0
            result.loc[~mask, 'weight'] = 0
        
        # 权重限制
        if 'weight' in result.columns:
            result['weight'] = result['weight'].clip(0, self.max_position)
        
        return result
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: Optional[float] = None,
        atr: Optional[float] = None
    ) -> int:
        """
        计算仓位大小
        
        支持多种仓位计算方法：
        1. 固定比例法：position = capital * risk_per_trade / (price * stop_loss)
        2. 波动率法：position = capital * risk_per_trade / volatility
        3. ATR法：position = capital * risk_per_trade / (atr * multiplier)
        
        Args:
            capital: 可用资金
            price: 当前价格
            volatility: 波动率（可选）
            atr: ATR值（可选）
            
        Returns:
            建议仓位（手数）
        """
        if self.stop_loss is not None and self.stop_loss > 0:
            # 固定比例法
            risk_amount = capital * self.risk_per_trade
            position_value = risk_amount / self.stop_loss
            position = int(position_value / price)
        elif atr is not None:
            # ATR法
            risk_amount = capital * self.risk_per_trade
            position = int(risk_amount / (atr * 2))  # 2倍ATR作为止损距离
        elif volatility is not None:
            # 波动率法
            risk_amount = capital * self.risk_per_trade
            position = int(risk_amount / (price * volatility))
        else:
            # 默认：使用最大仓位
            position = int(capital * self.max_position / price)
        
        return max(0, position)
    
    def on_bar(self, bar: pd.Series, context: Dict[str, Any]) -> Optional[Dict]:
        """
        事件驱动接口（用于实盘/仿真）
        
        Args:
            bar: 当前bar数据
            context: 上下文信息，包含：
                - capital: 当前资金
                - positions: 当前持仓
                - open_orders: 未成交订单
                - datetime: 当前时间
                
        Returns:
            交易指令字典或None
        """
        # 默认实现：不做任何操作
        # 子类可以重写此方法实现实时交易逻辑
        return None
    
    def on_tick(self, tick: Dict, context: Dict[str, Any]) -> Optional[Dict]:
        """
        Tick数据驱动接口（高频交易）
        
        Args:
            tick: Tick数据
            context: 上下文信息
            
        Returns:
            交易指令或None
        """
        return None
    
    def on_order_filled(self, order: Dict, context: Dict[str, Any]):
        """
        订单成交回调
        
        Args:
            order: 成交订单信息
            context: 上下文信息
        """
        self._trade_count += 1
    
    def update_stats(self, trade_result: Dict):
        """
        更新交易统计
        
        Args:
            trade_result: 交易结果，包含pnl等
        """
        if 'pnl' in trade_result:
            if trade_result['pnl'] > 0:
                self._win_count += 1
            elif trade_result['pnl'] < 0:
                self._loss_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取策略统计信息
        
        Returns:
            统计信息字典
        """
        total = self._win_count + self._loss_count
        win_rate = self._win_count / total if total > 0 else 0
        
        return {
            'name': self.name,
            'trade_count': self._trade_count,
            'win_count': self._win_count,
            'loss_count': self._loss_count,
            'win_rate': win_rate,
            'params': self.params,
            'factors': [f.name for f in self.factors],
        }
    
    def reset(self):
        """重置策略状态"""
        self._signals = None
        self._positions = None
        self._last_signal_time = None
        self._trade_count = 0
        self._win_count = 0
        self._loss_count = 0
        logger.info(f"Strategy {self.name} reset")
    
    def validate_params(self) -> bool:
        """
        验证参数有效性
        
        Returns:
            参数是否有效
        """
        if self.stop_loss is not None and self.stop_loss <= 0:
            logger.error("stop_loss must be positive")
            return False
        
        if self.take_profit is not None and self.take_profit <= 0:
            logger.error("take_profit must be positive")
            return False
        
        if self.max_position <= 0 or self.max_position > 1:
            logger.error("max_position must be between 0 and 1")
            return False
        
        if self.risk_per_trade <= 0 or self.risk_per_trade > 1:
            logger.error("risk_per_trade must be between 0 and 1")
            return False
        
        return True
    
    def get_param_bounds(self) -> Dict[str, tuple]:
        """
        获取参数优化边界
        
        子类可以重写此方法定义自己的参数空间
        
        Returns:
            参数边界字典，格式：{param_name: (lower, upper)}
        """
        return {
            'stop_loss': (0.01, 0.1),
            'take_profit': (0.02, 0.2),
            'max_position': (0.1, 1.0),
        }
    
    def to_dict(self) -> Dict:
        """
        将策略转换为字典（用于序列化）
        
        Returns:
            策略配置字典
        """
        return {
            'class': self.__class__.__name__,
            'name': self.name,
            'params': self.params,
            'symbols': self.symbols,
            'factors': [
                {'class': f.__class__.__name__, 'name': f.name, 'params': f.params}
                for f in self.factors
            ],
        }
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'BaseStrategy':
        """
        从字典创建策略实例
        
        Args:
            config: 策略配置字典
            
        Returns:
            策略实例
        """
        # 基类不实现具体逻辑，由子类实现
        raise NotImplementedError("Subclasses should implement from_dict")
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"symbols={self.symbols}, params={self.params})")
