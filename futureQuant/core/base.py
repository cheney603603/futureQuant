"""
抽象基类定义 - 所有模块的接口规范

定义了数据获取、因子、策略、模型、回测引擎的抽象接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np


class DataFetcher(ABC):
    """数据获取器抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """获取器名称"""
        pass
    
    @abstractmethod
    def fetch_daily(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            symbol: 合约代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            **kwargs: 额外参数
            
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, open_interest]
        """
        pass
    
    @abstractmethod
    def fetch_symbols(self, variety: Optional[str] = None) -> List[str]:
        """
        获取可交易的合约列表
        
        Args:
            variety: 品种代码，如 'RB'，为None时返回所有
            
        Returns:
            合约代码列表
        """
        pass


class Factor(ABC):
    """因子抽象基类"""
    
    def __init__(self, name: Optional[str] = None, **params):
        """
        初始化因子
        
        Args:
            name: 因子名称，默认为类名
            **params: 因子参数
        """
        self._name = name or self.__class__.__name__
        self.params = params
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        """因子名称"""
        return self._name
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值
        
        Args:
            data: 输入数据，包含OHLCV等
            
        Returns:
            因子值序列，索引与data对齐
        """
        pass
    
    def compute_panel(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        横截面计算（多品种）
        
        Args:
            data_dict: {symbol: DataFrame}
            
        Returns:
            DataFrame，列为品种，行为日期，值为因子值
        """
        results = {}
        for symbol, df in data_dict.items():
            results[symbol] = self.compute(df)
        return pd.DataFrame(results)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class Strategy(ABC):
    """策略抽象基类"""
    
    def __init__(self, name: Optional[str] = None, **params):
        """
        初始化策略
        
        Args:
            name: 策略名称
            **params: 策略参数
        """
        self._name = name or self.__class__.__name__
        self.params = params
        self.factors: List[Factor] = []
        self.symbols: List[str] = []
    
    @property
    def name(self) -> str:
        """策略名称"""
        return self._name
    
    def add_factor(self, factor: Factor):
        """添加因子"""
        self.factors.append(factor)
    
    def set_symbols(self, symbols: List[str]):
        """设置交易品种"""
        self.symbols = symbols
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 输入数据（可能包含多品种）
            
        Returns:
            DataFrame with columns: [date, symbol, signal, weight]
            signal: -1(做空), 0(空仓), 1(做多)
            weight: 仓位权重 0-1
        """
        pass
    
    def on_bar(self, bar: pd.Series, context: Dict[str, Any]) -> Optional[Dict]:
        """
        事件驱动接口（用于实盘/仿真）
        
        Args:
            bar: 当前bar数据
            context: 上下文信息（持仓、资金等）
            
        Returns:
            交易指令或None
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', factors={len(self.factors)})"


class Model(ABC):
    """机器学习模型抽象基类"""
    
    def __init__(self, name: Optional[str] = None, **params):
        """
        初始化模型
        
        Args:
            name: 模型名称
            **params: 模型参数
        """
        self._name = name or self.__class__.__name__
        self.params = params
        self._is_trained = False
        self.model = None
    
    @property
    def name(self) -> str:
        """模型名称"""
        return self._name
    
    @property
    def is_trained(self) -> bool:
        """是否已训练"""
        return self._is_trained
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], **kwargs):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            **kwargs: 额外参数
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测值
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        预测概率（分类模型）
        
        Args:
            X: 特征矩阵
            
        Returns:
            概率矩阵或None（不支持概率预测时）
        """
        return None
    
    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


class BacktestEngine(ABC):
    """回测引擎抽象基类"""
    
    def __init__(self, 
                 initial_capital: float = 1_000_000,
                 commission: float = 0.0001,
                 slippage: float = 0.0,
                 margin_rate: float = 0.1):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率（双边）
            slippage: 滑点（跳数或比例）
            margin_rate: 保证金率
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.margin_rate = margin_rate
        
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.positions: Dict[str, Dict] = {}
        self.current_capital = initial_capital
    
    @abstractmethod
    def run(self, 
            data: pd.DataFrame, 
            strategy: Strategy,
            **kwargs) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            data: 回测数据
            strategy: 策略实例
            **kwargs: 额外参数
            
        Returns:
            回测结果字典
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置回测状态"""
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        self.current_capital = self.initial_capital
