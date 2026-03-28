"""
挖掘上下文模块

定义 MiningContext 数据类，用于在 Agent 之间传递数据和中间结果。
上下文包含价格数据、收益率、配置以及各阶段的挖掘结果。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class MiningContext:
    """
    因子挖掘上下文

    作为数据载体在挖掘流程中传递，包含：
    - 输入数据（价格、收益率、品种列表）
    - 配置参数
    - 中间结果（发现的因子、验证通过的因子等）

    Attributes:
        data: 价格数据 DataFrame，包含 OHLCV 列
        returns: 未来收益率序列（已 shift(-1)）
        symbols: 品种代码列表
        start_date: 开始日期字符串（YYYY-MM-DD）
        end_date: 结束日期字符串（YYYY-MM-DD）
        config: 全局配置字典
        discovered_factors: 发现的因子列表（各挖掘 Agent 写入）
        validated_factors: 验证通过的因子列表
        factor_scores: 因子评分字典 {因子名: 分数}
        backtest_results: 回测结果字典
    """

    # 输入数据
    data: pd.DataFrame
    returns: pd.Series
    symbols: List[str]
    start_date: str
    end_date: str

    # 配置
    config: Dict[str, Any] = field(default_factory=dict)

    # 中间结果（各 Agent 写入）
    discovered_factors: List[Any] = field(default_factory=list)
    validated_factors: List[Any] = field(default_factory=list)
    factor_scores: Dict[str, float] = field(default_factory=dict)
    backtest_results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后的验证和设置"""
        # 确保 data 有正确的索引
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'date' in self.data.columns:
                self.data = self.data.set_index('date')
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except (ValueError, TypeError):
                pass

        # 确保 returns 有正确的索引
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            try:
                self.returns.index = pd.to_datetime(self.returns.index)
            except (ValueError, TypeError):
                pass

    def add_discovered_factors(self, factors: List[Any]):
        """
        添加发现的因子

        Args:
            factors: 因子实例列表
        """
        self.discovered_factors.extend(factors)

    def add_validated_factors(self, factors: List[Any]):
        """
        添加验证通过的因子

        Args:
            factors: 因子实例列表
        """
        self.validated_factors.extend(factors)

    def set_factor_scores(self, scores: Dict[str, float]):
        """
        设置因子评分

        Args:
            scores: {因子名: 分数} 字典
        """
        self.factor_scores.update(scores)

    def set_backtest_results(self, results: Dict[str, Any]):
        """
        设置回测结果

        Args:
            results: 回测结果字典
        """
        self.backtest_results.update(results)

    @property
    def n_discovered(self) -> int:
        """发现的因子数量"""
        return len(self.discovered_factors)

    @property
    def n_validated(self) -> int:
        """验证通过的因子数量"""
        return len(self.validated_factors)

    def __repr__(self) -> str:
        return (
            f"MiningContext("
            f"symbols={self.symbols}, "
            f"date_range=[{self.start_date}, {self.end_date}], "
            f"discovered={self.n_discovered}, "
            f"validated={self.n_validated})"
        )
