# 多智能体因子挖掘与策略自动回测系统 - 技术架构

## 1. 技术选型

### 1.1 整体技术栈

| 层次 | 技术方案 | 选型理由 |
|------|---------|---------|
| **编程语言** | Python 3.10+ | 与 futureQuant 主项目保持一致 |
| **多智能体框架** | 自研轻量级框架 | 避免引入 LangChain 等重依赖，保持可控性 |
| **任务编排** | 状态机 + 规则引擎 | 支持复杂的多 Agent 协作流程 |
| **并行计算** | joblib / concurrent.futures | 简单高效的并行计算方案 |
| **因子存储** | Parquet + SQLite | Parquet 高效列存，SQLite 元数据管理 |
| **配置管理** | pydantic-settings | 与主项目一致，类型安全 |
| **日志系统** | 复用 core.logger | 统一日志格式，便于追踪 |
| **测试框架** | pytest | 与主项目测试框架一致 |

### 1.2 新增依赖

```toml
# requirements-agent.txt
pydantic>=2.0
joblib>=1.3
scipy>=1.10
statsmodels>=0.14  # 用于时间序列分析
```

---

## 2. 系统架构

### 2.1 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent Factor Mining System                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Orchestrator Agent                        │   │
│  │  (任务编排：接收指令、分解任务、协调 Agent、汇总结果)           │   │
│  └───────────────────────────┬──────────────────────────────────┘   │
│                              │                                       │
│  ┌───────────────────────────┼──────────────────────────────────┐   │
│  │                    Mining Agents Layer                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │   │
│  │  │  Technical  │ │ Fundamental │ │   Macro     │              │   │
│  │  │   Agent     │ │   Agent     │ │   Agent     │              │   │
│  │  │ (技术因子)   │ │ (基本面因子) │ │ (宏观因子)   │              │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘              │   │
│  │         │               │               │                      │   │
│  │         └───────────────┴───────────────┘                      │   │
│  │                         │                                      │   │
│  │  ┌──────────────────────┴──────────────────────┐              │   │
│  │  │            Fusion & Selection Agent           │              │   │
│  │  │  (因子融合、去相关、筛选、评分)                 │              │   │
│  │  └──────────────────────┬───────────────────────┘              │   │
│  └──────────────────────────┼─────────────────────────────────────┘   │
│                             │                                         │
│  ┌──────────────────────────┴─────────────────────────────────────┐   │
│  │                   Validation Layer                              │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│  │  │ Look-Ahead   │ │   TimeSeries │ │    Sample    │            │   │
│  │  │   Detector   │ │   CrossVal   │ │   Weighting  │            │   │
│  │  │ (未来函数检测) │ │ (时序交叉验证) │ │ (样本权重)    │            │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │   │
│  │                         │                                      │   │
│  │  ┌──────────────────────┴──────────────────────┐              │   │
│  │  │            Multi-Dimensional Scorer           │              │   │
│  │  │  (预测能力、稳定性、单调性、交易成本、风险)      │              │   │
│  │  └──────────────────────────────────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                             │                                         │
│  ┌──────────────────────────┴─────────────────────────────────────┐   │
│  │                   Backtest Layer                                │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│  │  │   Strategy   │ │    Risk      │ │    Report    │            │   │
│  │  │   Generator  │ │   Controller │ │   Generator  │            │   │
│  │  │ (策略生成)    │ │ (风险控制)    │ │ (报告生成)    │            │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                             │                                         │
│  ┌──────────────────────────┴─────────────────────────────────────┐   │
│  │                   Factor Repository Layer                       │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │   │
│  │  │   Metadata   │ │   Version    │ │ Performance  │            │   │
│  │  │   Storage    │ │   Control    │ │   Tracker    │            │   │
│  │  │ (元数据存储)  │ │ (版本管理)    │ │ (性能追踪)    │            │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        futureQuant Core Modules                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│  │   data/    │ │  factor/   │ │ backtest/  │ │  strategy/ │           │
│  │ DataManager │ │ FactorEngine│ │BacktestEng │ │ BaseStrategy│          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块划分

```
futureQuant/
├── agent/                      # 新增：多智能体模块
│   ├── __init__.py
│   ├── base.py                 # Agent 抽象基类
│   ├── orchestrator.py         # 编排 Agent
│   ├── miners/                 # 挖掘 Agent
│   │   ├── __init__.py
│   │   ├── technical_agent.py  # 技术因子挖掘
│   │   ├── fundamental_agent.py # 基本面因子挖掘
│   │   ├── macro_agent.py      # 宏观因子挖掘
│   │   └── fusion_agent.py     # 因子融合 Agent
│   ├── validators/             # 验证 Agent
│   │   ├── __init__.py
│   │   ├── lookahead_detector.py  # 未来函数检测
│   │   ├── cross_validator.py     # 时序交叉验证
│   │   ├── sample_weighter.py     # 样本权重
│   │   └── scorer.py              # 多维度评分
│   ├── backtest/               # 回测 Agent
│   │   ├── __init__.py
│   │   ├── strategy_generator.py  # 策略生成
│   │   ├── risk_controller.py     # 风险控制
│   │   └── report_generator.py    # 报告生成
│   └── repository/             # 因子库管理
│       ├── __init__.py
│       ├── factor_store.py     # 因子存储
│       ├── version_control.py  # 版本管理
│       └── performance_tracker.py # 性能追踪
├── core/                       # 现有：核心模块（扩展）
│   ├── base.py                 # 新增 Agent 基类
│   └── ...
├── factor/                     # 现有：因子模块（扩展）
│   ├── engine.py               # 扩展：支持 Agent 注册因子
│   ├── evaluator.py            # 扩展：支持加权评估
│   └── ...
└── backtest/                   # 现有：回测模块（扩展）
    └── ...
```

---

## 3. 核心模块设计

### 3.1 Agent 抽象基类

```python
# futureQuant/agent/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..core.base import Factor
from ..core.logger import get_logger


class AgentStatus(Enum):
    """Agent 状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AgentResult:
    """Agent 执行结果"""
    agent_name: str
    status: AgentStatus
    data: Optional[pd.DataFrame] = None  # 因子值
    factors: Optional[List[Factor]] = None  # 因子实例
    metrics: Optional[Dict[str, float]] = None  # 评估指标
    errors: Optional[List[str]] = None
    logs: Optional[List[str]] = None


class BaseAgent(ABC):
    """
    Agent 抽象基类
    
    所有智能体（挖掘、验证、回测）都继承此类。
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict] = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.logger = get_logger(f'agent.{self.name.lower()}')
        
        # 执行历史
        self._history: List[AgentResult] = []
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行 Agent 任务
        
        Args:
            context: 执行上下文，包含数据、参数等
            
        Returns:
            AgentResult: 执行结果
        """
        pass
    
    def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        运行 Agent（带状态管理）
        
        Args:
            context: 执行上下文
            
        Returns:
            AgentResult: 执行结果
        """
        self.status = AgentStatus.RUNNING
        self.logger.info(f"Agent {self.name} starting execution")
        
        try:
            result = self.execute(context)
            self.status = result.status
            self._history.append(result)
            
            self.logger.info(
                f"Agent {self.name} finished: status={result.status.value}"
            )
            return result
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            error_result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(e)],
            )
            self._history.append(error_result)
            self.logger.error(f"Agent {self.name} failed: {e}")
            return error_result
    
    def get_history(self) -> List[AgentResult]:
        """获取执行历史"""
        return self._history
    
    def reset(self):
        """重置 Agent 状态"""
        self.status = AgentStatus.IDLE
        self._history = []
```

### 3.2 技术因子挖掘 Agent

```python
# futureQuant/agent/miners/technical_agent.py

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ..base import BaseAgent, AgentResult, AgentStatus
from ...core.base import Factor
from ...factor.technical import (
    MomentumFactor, RSIFactor, VolatilityFactor,
    ATRFactor, VolumeRatioFactor, OpenInterestChangeFactor
)


class TechnicalMiningAgent(BaseAgent):
    """
    技术因子挖掘 Agent
    
    自动发现技术面有效因子：
    - 动量类：N 日收益率、动量加速度
    - 波动率类：ATR 比率、波动率变化率
    - 成交量类：成交量异动、持仓量变化
    - 形态类：突破、反转形态识别
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="TechnicalMiningAgent", config=config)
        
        # 默认挖掘参数
        self.momentum_windows = config.get('momentum_windows', [5, 10, 20, 60])
        self.volatility_windows = config.get('volatility_windows', [10, 20, 60])
        self.volume_windows = config.get('volume_windows', [5, 10, 20])
        self.ic_threshold = config.get('ic_threshold', 0.02)  # IC 阈值
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行技术因子挖掘
        
        Args:
            context: 包含 'data' (价格数据), 'returns' (收益率) 等
            
        Returns:
            AgentResult: 包含挖掘到的有效因子
        """
        data = context.get('data')
        returns = context.get('returns')
        
        if data is None or returns is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Missing 'data' or 'returns' in context"]
            )
        
        discovered_factors: List[Factor] = []
        factor_results: List[pd.DataFrame] = []
        
        # 1. 动量因子挖掘
        self.logger.info("Mining momentum factors...")
        momentum_factors = self._mine_momentum_factors(data, returns)
        discovered_factors.extend(momentum_factors)
        
        # 2. 波动率因子挖掘
        self.logger.info("Mining volatility factors...")
        volatility_factors = self._mine_volatility_factors(data, returns)
        discovered_factors.extend(volatility_factors)
        
        # 3. 成交量因子挖掘
        self.logger.info("Mining volume factors...")
        volume_factors = self._mine_volume_factors(data, returns)
        discovered_factors.extend(volume_factors)
        
        # 4. 计算所有因子值
        factor_df = self._compute_all_factors(data, discovered_factors)
        
        self.logger.info(
            f"Discovered {len(discovered_factors)} technical factors"
        )
        
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=factor_df,
            factors=discovered_factors,
            metrics={'n_factors': len(discovered_factors)},
        )
    
    def _mine_momentum_factors(
        self, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[Factor]:
        """挖掘动量类因子"""
        factors = []
        
        for window in self.momentum_windows:
            factor = MomentumFactor(window=window)
            ic = self._quick_evaluate(factor, data, returns)
            
            if abs(ic) > self.ic_threshold:
                factors.append(factor)
                self.logger.debug(
                    f"MomentumFactor(window={window}): IC={ic:.4f}"
                )
        
        return factors
    
    def _mine_volatility_factors(
        self, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[Factor]:
        """挖掘波动率类因子"""
        factors = []
        
        for window in self.volatility_windows:
            factor = VolatilityFactor(window=window)
            ic = self._quick_evaluate(factor, data, returns)
            
            if abs(ic) > self.ic_threshold:
                factors.append(factor)
        
        # ATR 因子
        atr_factor = ATRFactor(window=14)
        ic = self._quick_evaluate(atr_factor, data, returns)
        if abs(ic) > self.ic_threshold:
            factors.append(atr_factor)
        
        return factors
    
    def _mine_volume_factors(
        self, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[Factor]:
        """挖掘成交量类因子"""
        factors = []
        
        for window in self.volume_windows:
            volume_factor = VolumeRatioFactor(window=window)
            ic = self._quick_evaluate(volume_factor, data, returns)
            
            if abs(ic) > self.ic_threshold:
                factors.append(volume_factor)
        
        # 持仓量变化因子
        oi_factor = OpenInterestChangeFactor(window=5)
        ic = self._quick_evaluate(oi_factor, data, returns)
        if abs(ic) > self.ic_threshold:
            factors.append(oi_factor)
        
        return factors
    
    def _quick_evaluate(
        self, 
        factor: Factor, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> float:
        """快速评估因子 IC"""
        try:
            factor_values = factor.compute(data)
            
            # 对齐数据
            common_idx = factor_values.dropna().index.intersection(
                returns.dropna().index
            )
            
            if len(common_idx) < 20:
                return 0.0
            
            # 计算 Spearman IC
            from scipy.stats import spearmanr
            ic, _ = spearmanr(
                factor_values.loc[common_idx],
                returns.loc[common_idx]
            )
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0
    
    def _compute_all_factors(
        self, 
        data: pd.DataFrame, 
        factors: List[Factor]
    ) -> pd.DataFrame:
        """计算所有因子值"""
        from ...factor.engine import FactorEngine
        
        engine = FactorEngine()
        engine.register_many(factors)
        
        return engine.compute_all(data)
```

### 3.3 未来函数检测器

```python
# futureQuant/agent/validators/lookahead_detector.py

from typing import Dict, Any, List, Tuple, Optional
import ast
import inspect
import pandas as pd
import numpy as np

from ..base import BaseAgent, AgentResult, AgentStatus
from ...core.base import Factor


class LookAheadDetector(BaseAgent):
    """
    未来函数检测 Agent
    
    通过静态分析和动态测试检测因子是否存在未来函数问题。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(name="LookAheadDetector", config=config)
        
        # 静态检测规则
        self.dangerous_patterns = [
            'shift(-',    # 负向 shift（取未来数据）
            'shift(-1)',  # 常见错误写法
            'shift(-n)',  # 变量形式
            'future',     # 包含 future 关键字
            'lookahead',  # lookahead 关键字
        ]
        
        # 基本面数据发布延迟（天数）
        self.data_lag = {
            'inventory': 3,     # 库存数据通常滞后 3 天
            'warehouse': 2,     # 仓单数据滞后 2 天
            'basis': 1,         # 基差数据滞后 1 天
            'macro': 30,        # 宏观数据滞后约 1 个月
        }
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行未来函数检测
        
        Args:
            context: 包含 'factors' (因子列表), 'data' (价格数据)
            
        Returns:
            AgentResult: 包含检测结果
        """
        factors = context.get('factors', [])
        data = context.get('data')
        
        if not factors:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["No factors provided for detection"]
            )
        
        results = {}
        clean_factors = []
        dirty_factors = []
        
        for factor in factors:
            is_clean, issues = self._detect_factor(factor, data)
            
            results[factor.name] = {
                'is_clean': is_clean,
                'issues': issues,
            }
            
            if is_clean:
                clean_factors.append(factor)
            else:
                dirty_factors.append(factor)
        
        self.logger.info(
            f"Detection complete: {len(clean_factors)} clean, "
            f"{len(dirty_factors)} with look-ahead bias"
        )
        
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            factors=clean_factors,  # 只返回干净的因子
            metrics={
                'total': len(factors),
                'clean': len(clean_factors),
                'dirty': len(dirty_factors),
            },
            logs=[f"{name}: {result}" for name, result in results.items()],
        )
    
    def _detect_factor(
        self, 
        factor: Factor, 
        data: Optional[pd.DataFrame]
    ) -> Tuple[bool, List[str]]:
        """
        检测单个因子
        
        Returns:
            (is_clean, issues): 是否干净，问题列表
        """
        issues = []
        
        # 1. 静态代码分析
        code_issues = self._static_analyze(factor)
        issues.extend(code_issues)
        
        # 2. 动态测试（如果提供了数据）
        if data is not None:
            dynamic_issues = self._dynamic_test(factor, data)
            issues.extend(dynamic_issues)
        
        # 3. 基本面数据延迟检查
        lag_issues = self._check_data_lag(factor)
        issues.extend(lag_issues)
        
        is_clean = len(issues) == 0
        return is_clean, issues
    
    def _static_analyze(self, factor: Factor) -> List[str]:
        """静态代码分析"""
        issues = []
        
        try:
            # 获取因子计算代码
            source = inspect.getsource(factor.compute)
            
            # 检查危险模式
            for pattern in self.dangerous_patterns:
                if pattern in source:
                    issues.append(f"Potential look-ahead: '{pattern}' found in code")
            
            # AST 分析
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # 检查 shift 调用
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr') and node.func.attr == 'shift':
                        # 检查参数是否为负数
                        if node.args:
                            arg = node.args[0]
                            if isinstance(arg, ast.Constant) and arg.value < 0:
                                issues.append(
                                    f"Negative shift({arg.value}) detected"
                                )
                            elif isinstance(arg, ast.UnaryOp) and isinstance(
                                arg.op, ast.USub
                            ):
                                issues.append("Negative shift detected")
        
        except Exception as e:
            issues.append(f"Static analysis failed: {str(e)}")
        
        return issues
    
    def _dynamic_test(
        self, 
        factor: Factor, 
        data: pd.DataFrame
    ) -> List[str]:
        """
        动态测试：验证信号延迟后因子是否失效
        
        原理：如果因子使用了未来数据，
        延迟一期执行后 IC 应该显著下降。
        """
        issues = []
        
        try:
            # 计算因子值
            factor_values = factor.compute(data)
            
            # 计算未来收益率
            returns = data['close'].pct_change().shift(-1)
            
            # 对齐数据
            common_idx = factor_values.dropna().index.intersection(
                returns.dropna().index
            )
            
            if len(common_idx) < 30:
                return issues
            
            # 原始 IC
            from scipy.stats import spearmanr
            ic_original, _ = spearmanr(
                factor_values.loc[common_idx],
                returns.loc[common_idx]
            )
            
            # 延迟一期执行后的 IC
            returns_delayed = data['close'].pct_change().shift(-2)  # 再延迟一期
            common_idx_delayed = factor_values.dropna().index.intersection(
                returns_delayed.dropna().index
            )
            
            ic_delayed, _ = spearmanr(
                factor_values.loc[common_idx_delayed],
                returns_delayed.loc[common_idx_delayed]
            )
            
            # IC 显著下降说明可能存在未来函数
            if abs(ic_delayed) < abs(ic_original) * 0.5:
                issues.append(
                    f"IC dropped significantly after delay: "
                    f"{ic_original:.4f} -> {ic_delayed:.4f}"
                )
        
        except Exception as e:
            issues.append(f"Dynamic test failed: {str(e)}")
        
        return issues
    
    def _check_data_lag(self, factor: Factor) -> List[str]:
        """检查基本面数据延迟"""
        issues = []
        
        # 获取因子依赖的数据类型
        data_deps = getattr(factor, 'data_dependencies', [])
        
        for dep in data_deps:
            if dep in self.data_lag:
                lag = self.data_lag[dep]
                # 检查因子是否处理了延迟
                factor_lag = getattr(factor, 'lag', 0)
                
                if factor_lag < lag:
                    issues.append(
                        f"Data lag not handled: {dep} has {lag} days lag, "
                        f"but factor lag is {factor_lag}"
                    )
        
        return issues
```

### 3.4 时序交叉验证器

```python
# futureQuant/agent/validators/cross_validator.py

from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..base import BaseAgent, AgentResult, AgentStatus
from ...core.base import Factor


@dataclass
class CVResult:
    """交叉验证结果"""
    fold: int
    train_period: tuple
    test_period: tuple
    train_ic: float
    test_ic: float
    train_icir: float
    test_icir: float


class TimeSeriesCrossValidator(BaseAgent):
    """
    时序交叉验证 Agent
    
    支持：
    - Walk-Forward 交叉验证
    - Expanding Window 交叉验证
    - Purged K-Fold 交叉验证
    """
    
    def __init__(
        self,
        method: Literal['walk_forward', 'expanding', 'purged_kfold'] = 'walk_forward',
        n_splits: int = 5,
        train_size: int = 252,  # 训练窗口大小
        test_size: int = 63,    # 测试窗口大小
        purge_gap: int = 5,     # 清洗期（避免数据泄露）
        config: Optional[Dict] = None
    ):
        super().__init__(name="TimeSeriesCrossValidator", config=config)
        
        self.method = method
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行时序交叉验证
        
        Args:
            context: 包含 'factors', 'data', 'returns'
            
        Returns:
            AgentResult: 包含各因子的交叉验证结果
        """
        factors = context.get('factors', [])
        data = context.get('data')
        returns = context.get('returns')
        
        if not factors or data is None or returns is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Missing required context: factors, data, returns"]
            )
        
        all_results = {}
        stable_factors = []
        
        for factor in factors:
            cv_results = self._validate_factor(factor, data, returns)
            all_results[factor.name] = cv_results
            
            # 判断因子是否稳定
            if self._is_stable(cv_results):
                stable_factors.append(factor)
        
        self.logger.info(
            f"Cross-validation complete: {len(stable_factors)}/{len(factors)} "
            f"factors passed stability test"
        )
        
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            factors=stable_factors,
            metrics={
                'n_factors': len(factors),
                'n_stable': len(stable_factors),
            },
            logs=[self._format_cv_result(name, results) 
                  for name, results in all_results.items()],
        )
    
    def _validate_factor(
        self, 
        factor: Factor, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[CVResult]:
        """对单个因子执行交叉验证"""
        
        if self.method == 'walk_forward':
            return self._walk_forward_cv(factor, data, returns)
        elif self.method == 'expanding':
            return self._expanding_window_cv(factor, data, returns)
        else:
            return self._purged_kfold_cv(factor, data, returns)
    
    def _walk_forward_cv(
        self, 
        factor: Factor, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[CVResult]:
        """
        Walk-Forward 交叉验证
        
        滚动训练/测试窗口，模拟实盘部署场景。
        """
        factor_values = factor.compute(data)
        results = []
        
        n = len(data)
        step = self.test_size
        
        fold = 0
        for i in range(self.train_size, n - self.test_size, step):
            train_start = i - self.train_size
            train_end = i - self.purge_gap  # 清洗期
            test_start = i
            test_end = min(i + self.test_size, n)
            
            # 获取训练/测试数据
            train_factor = factor_values.iloc[train_start:train_end]
            train_returns = returns.iloc[train_start:train_end]
            test_factor = factor_values.iloc[test_start:test_end]
            test_returns = returns.iloc[test_start:test_end]
            
            # 计算 IC
            train_ic, train_icir = self._calculate_ic_metrics(
                train_factor, train_returns
            )
            test_ic, test_icir = self._calculate_ic_metrics(
                test_factor, test_returns
            )
            
            results.append(CVResult(
                fold=fold,
                train_period=(data.index[train_start], data.index[train_end-1]),
                test_period=(data.index[test_start], data.index[min(test_end-1, n-1)]),
                train_ic=train_ic,
                test_ic=test_ic,
                train_icir=train_icir,
                test_icir=test_icir,
            ))
            
            fold += 1
            if fold >= self.n_splits:
                break
        
        return results
    
    def _expanding_window_cv(
        self, 
        factor: Factor, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[CVResult]:
        """
        Expanding Window 交叉验证
        
        训练窗口不断扩大，测试窗口固定大小。
        """
        factor_values = factor.compute(data)
        results = []
        
        n = len(data)
        step = self.test_size
        min_train = self.train_size
        
        fold = 0
        for i in range(min_train, n - self.test_size, step):
            train_start = 0  # 从头开始
            train_end = i - self.purge_gap
            test_start = i
            test_end = min(i + self.test_size, n)
            
            train_factor = factor_values.iloc[train_start:train_end]
            train_returns = returns.iloc[train_start:train_end]
            test_factor = factor_values.iloc[test_start:test_end]
            test_returns = returns.iloc[test_start:test_end]
            
            train_ic, train_icir = self._calculate_ic_metrics(
                train_factor, train_returns
            )
            test_ic, test_icir = self._calculate_ic_metrics(
                test_factor, test_returns
            )
            
            results.append(CVResult(
                fold=fold,
                train_period=(data.index[train_start], data.index[train_end-1]),
                test_period=(data.index[test_start], data.index[min(test_end-1, n-1)]),
                train_ic=train_ic,
                test_ic=test_ic,
                train_icir=train_icir,
                test_icir=test_icir,
            ))
            
            fold += 1
            if fold >= self.n_splits:
                break
        
        return results
    
    def _purged_kfold_cv(
        self, 
        factor: Factor, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> List[CVResult]:
        """
        Purged K-Fold 交叉验证
        
        将数据分为 K 折，每折之间有清洗期，避免数据泄露。
        适用于非时序数据或有横截面数据的情况。
        """
        factor_values = factor.compute(data)
        n = len(data)
        fold_size = n // self.n_splits
        
        results = []
        
        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n
            
            # 清洗期
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n, test_end + self.purge_gap)
            
            # 训练集：排除测试集和清洗期
            train_idx = list(range(0, purge_start)) + list(range(purge_end, n))
            
            train_factor = factor_values.iloc[train_idx]
            train_returns = returns.iloc[train_idx]
            test_factor = factor_values.iloc[test_start:test_end]
            test_returns = returns.iloc[test_start:test_end]
            
            train_ic, train_icir = self._calculate_ic_metrics(
                train_factor, train_returns
            )
            test_ic, test_icir = self._calculate_ic_metrics(
                test_factor, test_returns
            )
            
            results.append(CVResult(
                fold=fold,
                train_period=(data.index[train_idx[0]], data.index[train_idx[-1]]),
                test_period=(data.index[test_start], data.index[test_end-1]),
                train_ic=train_ic,
                test_ic=test_ic,
                train_icir=train_icir,
                test_icir=test_icir,
            ))
        
        return results
    
    def _calculate_ic_metrics(
        self, 
        factor_values: pd.Series, 
        returns: pd.Series
    ) -> tuple:
        """计算 IC 和 ICIR"""
        from scipy.stats import spearmanr
        
        common_idx = factor_values.dropna().index.intersection(
            returns.dropna().index
        )
        
        if len(common_idx) < 10:
            return 0.0, 0.0
        
        # 计算每期的 IC
        ic_series = []
        # 简化：计算整体 IC
        ic, _ = spearmanr(
            factor_values.loc[common_idx],
            returns.loc[common_idx]
        )
        
        # ICIR（简化版本，实际应计算滚动 IC）
        icir = ic if not np.isnan(ic) else 0.0  # 简化
        
        return ic if not np.isnan(ic) else 0.0, icir
    
    def _is_stable(self, cv_results: List[CVResult]) -> bool:
        """判断因子是否稳定"""
        if not cv_results:
            return False
        
        # 条件1：测试集 IC 方向一致（同号比例 > 70%）
        test_ics = [r.test_ic for r in cv_results]
        positive_ratio = sum(ic > 0 for ic in test_ics) / len(test_ics)
        
        # 条件2：测试集 IC 均值 > 0.02
        mean_test_ic = np.mean(test_ics)
        
        # 条件3：训练集与测试集 IC 差异不大
        train_ics = [r.train_ic for r in cv_results]
        ic_diff = abs(np.mean(train_ics) - np.mean(test_ics))
        
        return (
            (positive_ratio > 0.7 or positive_ratio < 0.3) and  # 方向一致
            abs(mean_test_ic) > 0.02 and  # IC 显著
            ic_diff < 0.05  # 不过拟合
        )
    
    def _format_cv_result(self, name: str, results: List[CVResult]) -> str:
        """格式化 CV 结果"""
        if not results:
            return f"{name}: No CV results"
        
        test_ics = [r.test_ic for r in results]
        return (
            f"{name}: mean_test_ic={np.mean(test_ics):.4f}, "
            f"std_test_ic={np.std(test_ics):.4f}, "
            f"n_folds={len(results)}"
        )
```

### 3.5 多维度因子评分器

```python
# futureQuant/agent/validators/scorer.py

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..base import BaseAgent, AgentResult, AgentStatus
from ...core.base import Factor
from ...factor.evaluator import FactorEvaluator


@dataclass
class FactorScore:
    """因子综合评分"""
    factor_name: str
    overall_score: float
    prediction_score: float  # 预测能力
    stability_score: float   # 稳定性
    monotonicity_score: float # 单调性
    turnover_score: float    # 换手率
    risk_score: float        # 风险
    
    details: Dict[str, float]


class MultiDimensionalScorer(BaseAgent):
    """
    多维度因子评分 Agent
    
    从多个维度对因子进行综合评分：
    - 预测能力：IC 均值、ICIR、IC 胜率
    - 稳定性：IC 月度稳定性、年度稳定性
    - 单调性：分层回测单调性
    - 交易成本：换手率
    - 风险：最大回撤、下行波动率
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(name="MultiDimensionalScorer", config=config)
        
        # 各维度权重
        self.weights = weights or {
            'prediction': 0.35,
            'stability': 0.25,
            'monotonicity': 0.20,
            'turnover': 0.10,
            'risk': 0.10,
        }
        
        self.evaluator = FactorEvaluator()
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行多维度评分
        
        Args:
            context: 包含 'factors', 'data', 'returns'
            
        Returns:
            AgentResult: 包含评分结果
        """
        factors = context.get('factors', [])
        data = context.get('data')
        returns = context.get('returns')
        
        if not factors or data is None or returns is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Missing required context"]
            )
        
        scores = []
        
        for factor in factors:
            score = self._score_factor(factor, data, returns)
            scores.append(score)
        
        # 按综合评分排序
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        # 筛选高分因子
        threshold = 0.6  # 综合评分阈值
        selected_factors = [
            factors[scores.index(s)] 
            for s in scores 
            if s.overall_score >= threshold
        ]
        
        self.logger.info(
            f"Scoring complete: {len(selected_factors)}/{len(factors)} "
            f"factors passed threshold"
        )
        
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            factors=selected_factors,
            metrics={
                'n_factors': len(factors),
                'n_selected': len(selected_factors),
                'best_score': scores[0].overall_score if scores else 0,
            },
            logs=[self._format_score(s) for s in scores[:10]],  # Top 10
        )
    
    def _score_factor(
        self, 
        factor: Factor, 
        data: pd.DataFrame, 
        returns: pd.Series
    ) -> FactorScore:
        """计算单个因子的综合评分"""
        
        factor_values = factor.compute(data)
        
        # 1. 预测能力评分
        prediction_score, prediction_details = self._score_prediction(
            factor_values, returns
        )
        
        # 2. 稳定性评分
        stability_score, stability_details = self._score_stability(
            factor_values, returns
        )
        
        # 3. 单调性评分
        monotonicity_score, monotonicity_details = self._score_monotonicity(
            factor_values, returns
        )
        
        # 4. 换手率评分
        turnover_score, turnover_details = self._score_turnover(
            factor_values
        )
        
        # 5. 风险评分
        risk_score, risk_details = self._score_risk(
            factor_values, returns
        )
        
        # 综合评分
        overall_score = (
            self.weights['prediction'] * prediction_score +
            self.weights['stability'] * stability_score +
            self.weights['monotonicity'] * monotonicity_score +
            self.weights['turnover'] * turnover_score +
            self.weights['risk'] * risk_score
        )
        
        details = {
            **prediction_details,
            **stability_details,
            **monotonicity_details,
            **turnover_details,
            **risk_details,
        }
        
        return FactorScore(
            factor_name=factor.name,
            overall_score=overall_score,
            prediction_score=prediction_score,
            stability_score=stability_score,
            monotonicity_score=monotonicity_score,
            turnover_score=turnover_score,
            risk_score=risk_score,
            details=details,
        )
    
    def _score_prediction(
        self, 
        factor_values: pd.Series, 
        returns: pd.Series
    ) -> tuple:
        """预测能力评分"""
        
        # IC 分析
        ic_series = self.evaluator.calculate_ic(
            factor_values.to_frame('factor'), returns
        )
        ic_stats = self.evaluator.calculate_icir(ic_series)
        
        ic_mean = ic_stats.get('ic_mean', 0)
        icir = ic_stats.get('icir', 0)
        ic_win_rate = ic_stats.get('ic_win_rate', 0)
        
        # 评分映射
        # IC 均值：|IC| > 0.05 → 1.0, |IC| > 0.03 → 0.8, |IC| > 0.02 → 0.6
        ic_score = min(1.0, abs(ic_mean) / 0.05)
        
        # ICIR：|ICIR| > 2.0 → 1.0, |ICIR| > 1.0 → 0.8, |ICIR| > 0.5 → 0.6
        icir_score = min(1.0, abs(icir) / 2.0)
        
        # IC 胜率：> 60% → 1.0, > 55% → 0.8, > 50% → 0.6
        win_rate_score = max(0, min(1.0, (ic_win_rate - 0.4) / 0.2))
        
        # 综合预测能力评分
        prediction_score = 0.4 * ic_score + 0.4 * icir_score + 0.2 * win_rate_score
        
        details = {
            'ic_mean': ic_mean,
            'icir': icir,
            'ic_win_rate': ic_win_rate,
        }
        
        return prediction_score, details
    
    def _score_stability(
        self, 
        factor_values: pd.Series, 
        returns: pd.Series
    ) -> tuple:
        """稳定性评分"""
        
        # 按月分组计算 IC
        factor_df = factor_values.to_frame('factor')
        factor_df['returns'] = returns
        factor_df['month'] = factor_df.index.to_period('M')
        
        monthly_ics = factor_df.groupby('month').apply(
            lambda x: x['factor'].corr(x['returns'], method='spearman')
        )
        
        # 月度 IC 稳定性
        ic_std = monthly_ics.std()
        stability_score = max(0, min(1.0, 1 - ic_std / 0.2))
        
        details = {
            'monthly_ic_std': ic_std,
            'monthly_ic_mean': monthly_ics.mean(),
        }
        
        return stability_score, details
    
    def _score_monotonicity(
        self, 
        factor_values: pd.Series, 
        returns: pd.Series
    ) -> tuple:
        """单调性评分"""
        
        try:
            # 分层回测
            quantile_returns = self.evaluator.quantile_backtest(
                factor_values.to_frame('factor'), returns, n_quantiles=5
            )
            
            if quantile_returns.empty:
                return 0.0, {'error': 'quantile backtest failed'}
            
            # 检查单调性：Q1 < Q2 < Q3 < Q4 < Q5 或反之
            mean_returns = [
                quantile_returns[f'Q{i}'].mean() for i in range(1, 6)
            ]
            
            # 计算单调性得分（Spearman 秩相关）
            from scipy.stats import spearmanr
            ranks = [1, 2, 3, 4, 5]
            monotonicity, _ = spearmanr(ranks, mean_returns)
            
            # 归一化到 0-1
            monotonicity_score = (abs(monotonicity) + 1) / 2
            
            # 多空组合收益
            long_short = quantile_returns['long_short'].mean()
            
            details = {
                'monotonicity': monotonicity,
                'long_short_return': long_short,
                'quantile_returns': mean_returns,
            }
            
            return monotonicity_score, details
            
        except Exception as e:
            return 0.0, {'error': str(e)}
    
    def _score_turnover(self, factor_values: pd.Series) -> tuple:
        """换手率评分"""
        
        # 计算因子值变化
        factor_change = factor_values.diff().abs()
        factor_range = factor_values.abs().rolling(20).max()
        
        # 相对变化率
        relative_change = (factor_change / factor_range).dropna()
        avg_turnover = relative_change.mean()
        
        # 换手率越低越好：< 0.1 → 1.0, < 0.2 → 0.8, < 0.3 → 0.6
        turnover_score = max(0, min(1.0, 1 - avg_turnover / 0.3))
        
        details = {
            'avg_turnover': avg_turnover,
        }
        
        return turnover_score, details
    
    def _score_risk(
        self, 
        factor_values: pd.Series, 
        returns: pd.Series
    ) -> tuple:
        """风险评分"""
        
        # 基于因子的策略收益
        positions = factor_values.apply(np.sign)
        strategy_returns = positions.shift(1) * returns
        
        # 最大回撤
        cumulative = (1 + strategy_returns.dropna()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # 下行波动率
        negative_returns = strategy_returns[strategy_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # 风险评分：最大回撤 < 10% → 1.0, < 20% → 0.8, < 30% → 0.6
        dd_score = max(0, min(1.0, 1 - max_drawdown / 0.3))
        
        # 下行波动率评分：< 10% → 1.0, < 15% → 0.8, < 20% → 0.6
        dv_score = max(0, min(1.0, 1 - downside_vol / 0.2))
        
        risk_score = 0.6 * dd_score + 0.4 * dv_score
        
        details = {
            'max_drawdown': max_drawdown,
            'downside_volatility': downside_vol,
        }
        
        return risk_score, details
    
    def _format_score(self, score: FactorScore) -> str:
        """格式化评分结果"""
        return (
            f"{score.factor_name}: overall={score.overall_score:.3f} "
            f"(pred={score.prediction_score:.2f}, "
            f"stab={score.stability_score:.2f}, "
            f"mono={score.monotonicity_score:.2f}, "
            f"turn={score.turnover_score:.2f}, "
            f"risk={score.risk_score:.2f})"
        )
```

---

## 4. 数据模型

### 4.1 核心实体关系

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│    Factor    │───────│ FactorVersion│───────│FactorMetadata│
│  (因子实例)   │ 1:N   │  (版本管理)   │ 1:1   │  (元数据)     │
└──────────────┘       └──────────────┘       └──────────────┘
       │
       │ 1:N
       ▼
┌──────────────┐       ┌──────────────┐
│ FactorValue  │───────│FactorPerfTrack│
│  (因子值)     │ 1:N   │  (性能追踪)   │
└──────────────┘       └──────────────┘
```

### 4.2 数据表结构

#### factor_metadata（因子元数据）

| 字段 | 类型 | 说明 |
|------|------|------|
| factor_id | VARCHAR(64) | 因子唯一 ID |
| name | VARCHAR(128) | 因子名称 |
| category | VARCHAR(32) | 因子类别（technical/fundamental/macro） |
| sub_category | VARCHAR(32) | 子类别（momentum/volatility/...） |
| description | TEXT | 因子描述 |
| formula | TEXT | 计算公式 |
| parameters | JSON | 参数配置 |
| data_dependencies | JSON | 数据依赖 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |
| status | VARCHAR(16) | 状态（active/inactive/observed） |

#### factor_version（因子版本）

| 字段 | 类型 | 说明 |
|------|------|------|
| version_id | VARCHAR(64) | 版本 ID |
| factor_id | VARCHAR(64) | 因子 ID |
| version | VARCHAR(16) | 版本号 |
| parameters | JSON | 参数配置 |
| code | TEXT | 计算代码 |
| change_reason | TEXT | 变更原因 |
| created_at | TIMESTAMP | 创建时间 |

#### factor_value（因子值）

| 字段 | 类型 | 说明 |
|------|------|------|
| factor_id | VARCHAR(64) | 因子 ID |
| version_id | VARCHAR(64) | 版本 ID |
| symbol | VARCHAR(16) | 品种代码 |
| date | DATE | 日期 |
| value | DOUBLE | 因子值 |

#### factor_performance（因子性能）

| 字段 | 类型 | 说明 |
|------|------|------|
| factor_id | VARCHAR(64) | 因子 ID |
| version_id | VARCHAR(64) | 版本 ID |
| period | VARCHAR(16) | 统计周期 |
| start_date | DATE | 开始日期 |
| end_date | DATE | 结束日期 |
| ic_mean | DOUBLE | IC 均值 |
| icir | DOUBLE | ICIR |
| ic_win_rate | DOUBLE | IC 胜率 |
| monotonicity | DOUBLE | 单调性 |
| turnover | DOUBLE | 换手率 |
| max_drawdown | DOUBLE | 最大回撤 |
| overall_score | DOUBLE | 综合评分 |
| created_at | TIMESTAMP | 计算时间 |

---

## 5. 接口设计

### 5.1 核心类接口

```python
# futureQuant/agent/__init__.py

from .orchestrator import MultiAgentFactorMiner

# 主要入口类
class MultiAgentFactorMiner:
    """
    多智能体因子挖掘系统主入口
    
    使用示例：
        >>> miner = MultiAgentFactorMiner(
        ...     symbols=['RB', 'I', 'HC'],
        ...     start_date='2020-01-01',
        ...     end_date='2024-12-31',
        ... )
        >>> result = miner.run()
        >>> print(f"发现有效因子: {len(result.factors)}")
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: Optional[Dict] = None
    ):
        """初始化挖掘器"""
        pass
    
    def run(
        self,
        mining_agents: Optional[List[str]] = None,
        validation_methods: Optional[List[str]] = None,
        n_workers: int = 4,
    ) -> MiningResult:
        """
        运行因子挖掘
        
        Args:
            mining_agents: 启用的挖掘 Agent 列表
            validation_methods: 启用的验证方法列表
            n_workers: 并行工作进程数
            
        Returns:
            MiningResult: 挖掘结果
        """
        pass
    
    def run_backtest(
        self,
        factors: List[Factor],
        risk_config: Optional[Dict] = None,
    ) -> BacktestResult:
        """
        运行策略回测
        
        Args:
            factors: 因子列表
            risk_config: 风险控制配置
            
        Returns:
            BacktestResult: 回测结果
        """
        pass
    
    def get_factor_repository(self) -> FactorRepository:
        """获取因子库实例"""
        pass
```

### 5.2 配置文件格式

```yaml
# config/agent_settings.yaml

multi_agent_factor_mining:
  # 数据配置
  data:
    cache_dir: "./data_cache"
    lookback_days: 1260  # 5 年
  
  # 挖掘 Agent 配置
  mining:
    technical:
      enabled: true
      momentum_windows: [5, 10, 20, 60]
      volatility_windows: [10, 20, 60]
      volume_windows: [5, 10, 20]
      ic_threshold: 0.02
    
    fundamental:
      enabled: true
      data_lag:
        inventory: 3
        warehouse: 2
        basis: 1
    
    macro:
      enabled: true
      indicators: ['gdp', 'cpi', 'pmi', 'm2']
  
  # 验证配置
  validation:
    lookahead_detection:
      enabled: true
      static_analysis: true
      dynamic_test: true
    
    cross_validation:
      method: 'walk_forward'
      n_splits: 5
      train_size: 252
      test_size: 63
      purge_gap: 5
    
    sample_weighting:
      enabled: true
      method: 'volatility'  # volatility / liquidity / market_regime
    
    scoring:
      weights:
        prediction: 0.35
        stability: 0.25
        monotonicity: 0.20
        turnover: 0.10
        risk: 0.10
      threshold: 0.6
  
  # 回测配置
  backtest:
    initial_capital: 1000000
    commission: 0.0001
    slippage: 1
    margin_rate: 0.1
    
    risk_control:
      stop_loss: 0.05
      take_profit: 0.10
      max_position: 0.3
      max_drawdown: 0.15
  
  # 因子库配置
  repository:
    storage_dir: "./factor_repo"
    metadata_db: "./factor_repo/metadata.db"
    value_format: "parquet"
    
  # 并行配置
  parallel:
    n_workers: 4
    backend: "loky"  # joblib backend
```

---

## 6. 技术风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| **多 Agent 协调复杂** | 开发难度大 | 采用状态机模式，明确定义 Agent 间通信协议 |
| **因子计算性能** | 计算耗时长 | 使用 joblib 并行计算，Parquet 列存优化 |
| **未来函数检测漏报** | 回测结果失真 | 静态分析 + 动态测试双重检测，持续更新规则库 |
| **过拟合风险** | 实盘表现差 | Walk-Forward 验证，多市场环境测试 |
| **数据质量问题** | 因子评估偏差 | 数据预处理校验，异常值处理 |

---

**文档版本**: v1.0
**创建日期**: 2026-03-26
**最后更新**: 2026-03-26
