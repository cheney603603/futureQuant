# 多智能体因子挖掘与策略回测系统 - 技术架构文档

> **版本**: v1.0  
> **日期**: 2026-03-26  
> **状态**: 架构设计阶段

---

## 1. 系统架构总览

### 1.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              futureQuant v2.0                                    │
│                    多智能体因子挖掘与策略回测系统                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          API Layer (入口层)                                 │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │ │
│  │  │  CLI Interface  │  │  Python SDK     │  │  REST API       │            │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       Orchestrator Layer (调度层)                           │ │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Orchestrator Agent                                  │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │ │ │
│  │  │  │Task Queue   │ │Agent Router │ │Result Agg.  │ │State Manager│     │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │ │ │
│  │  └───────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Agent Layer (智能体层)                               │ │
│  │                                                                             │ │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐      │ │
│  │  │  Mining Agent     │  │  Validation Agent │  │  Risk Control Agent│      │ │
│  │  ├───────────────────┤  ├───────────────────┤  ├───────────────────┤      │ │
│  │  │ • GP Engine       │  │ • Scorer          │  │ • Monitor         │      │ │
│  │  │ • Expression Gen  │  │ • Purged CV       │  │ • Alert System    │      │ │
│  │  │ • Combiner        │  │ • Walk-Forward    │  │ • Position Mgr    │      │ │
│  │  └───────────────────┘  └───────────────────┘  └───────────────────┘      │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        Core Layer (核心层)                                  │ │
│  │                                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │ │
│  │  │ Factor Core │ │ Backtest    │ │ ML Core     │ │ Risk Core   │         │ │
│  │  │ • Engine    │ │ • Engine    │ │ • Pipeline  │ │ • Calculator│         │ │
│  │  │ • Evaluator │ │ • Portfolio │ │ • Features  │ │ • Metrics   │         │ │
│  │  │ • Registry  │ │ • Broker    │ │ • Models    │ │ • Limits    │         │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                     Infrastructure Layer (基础设施层)                       │ │
│  │                                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │ │
│  │  │ Data Store  │ │ Cache       │ │ Message Bus │ │ Logging     │         │ │
│  │  │ • DuckDB    │ │ • Redis     │ │ • Queue     │ │ • Structured│         │ │
│  │  │ • Parquet   │ │ • Memory    │ │ • Pub/Sub   │ │ • File      │         │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 模块依赖关系

```
                    ┌──────────────────┐
                    │   API Layer      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Orchestrator    │
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │Mining Agent │   │Valid Agent  │   │ Risk Agent  │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼─────────┐
                    │    Core Layer    │
                    │  Factor │ Backtest│
                    │  ML     │ Risk    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Infrastructure   │
                    │  Data  │ Cache   │
                    └──────────────────┘
```

---

## 2. 技术选型

### 2.1 核心技术栈

| 层级 | 组件 | 技术选型 | 选型理由 |
|------|------|----------|----------|
| **语言** | 主语言 | Python 3.10+ | 量化生态、AI Agent 支持 |
| **数据存储** | 时序数据 | DuckDB + Parquet | 高效列存储、SQL 支持 |
| **缓存** | 热数据 | Redis / 内存缓存 | 低延迟、支持 TTL |
| **计算** | 向量化 | NumPy / Pandas | 成熟稳定 |
| **计算** | 并行化 | Joblib / Ray | 多核并行 |
| **ML** | 树模型 | XGBoost / LightGBM | 高效、可解释 |
| **ML** | 深度学习 | PyTorch | 灵活、社区活跃 |
| **Agent** | 框架 | LangChain / 自研 | 多智能体协作 |
| **优化** | 遗传规划 | DEAP / gplearn | 因子表达式进化 |
| **优化** | 贝叶斯 | Optuna | 超参优化 |
| **测试** | 单元测试 | pytest | Python 标准 |
| **日志** | 结构化 | structlog | JSON 日志、易解析 |

### 2.2 新增依赖

```toml
# requirements-agent.txt

# Agent 框架
langchain>=0.1.0
langchain-openai>=0.0.5  # 可选，用于 LLM 增强

# 遗传规划
deap>=1.4.0
gplearn>=0.4.2           # 遗传规划因子挖掘

# 并行计算
joblib>=1.3.0
ray>=2.8.0               # 分布式计算（可选）

# 数据存储
duckdb>=0.9.0
pyarrow>=14.0.0          # Parquet 支持

# 缓存
redis>=5.0.0             # 可选

# 可视化
plotly>=5.18.0           # 交互式图表

# 风险模型
scipy>=1.11.0            # 优化、统计
cvxpy>=1.4.0             # 凸优化（组合优化）

# 监控
prometheus-client>=0.19.0  # 可选，指标暴露
```

### 2.3 架构模式

采用 **分层架构 + 智能体模式**：

1. **分层架构**：清晰的职责边界，便于维护和测试
2. **智能体模式**：独立的智能体负责特定任务，通过消息通信
3. **管道模式**：因子挖掘 → 验证 → 组合 → 回测 形成数据处理管道
4. **策略模式**：因子评分、风控规则可插拔替换

---

## 3. 模块详细设计

### 3.1 Agent Layer（智能体层）

#### 3.1.1 Orchestrator Agent（调度智能体）

**职责**：
- 接收用户请求，分解任务
- 路由任务到对应智能体
- 聚合结果，生成最终输出
- 维护全局状态

**接口设计**：

```python
# futureQuant/agent/orchestrator.py

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class TaskType(Enum):
    FACTOR_MINING = "factor_mining"
    FACTOR_VALIDATION = "factor_validation"
    STRATEGY_BACKTEST = "strategy_backtest"
    RISK_MONITORING = "risk_monitoring"
    FULL_PIPELINE = "full_pipeline"

@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: TaskType
    params: Dict[str, Any]
    priority: int = 0
    dependencies: List[str] = None
    
@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: str  # 'success', 'failed', 'partial'
    data: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[str]

class OrchestratorAgent:
    """
    调度智能体
    
    负责任务分解、路由、聚合。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.mining_agent = FactorMiningAgent()
        self.validation_agent = ValidationAgent()
        self.risk_agent = RiskControlAgent()
        self.task_queue = asyncio.Queue()
        self.state_manager = StateManager()
        
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        await self.task_queue.put(task)
        return task.task_id
    
    async def execute(self, task: Task) -> TaskResult:
        """执行任务"""
        if task.task_type == TaskType.FACTOR_MINING:
            return await self.mining_agent.execute(task)
        elif task.task_type == TaskType.FACTOR_VALIDATION:
            return await self.validation_agent.execute(task)
        elif task.task_type == TaskType.RISK_MONITORING:
            return await self.risk_agent.execute(task)
        elif task.task_type == TaskType.FULL_PIPELINE:
            return await self._run_full_pipeline(task)
    
    async def _run_full_pipeline(self, task: Task) -> TaskResult:
        """运行完整流水线"""
        results = {}
        
        # 1. 因子挖掘
        mining_result = await self.mining_agent.execute(task)
        results['mining'] = mining_result
        
        # 2. 因子验证
        validation_task = Task(
            task_id=f"{task.task_id}_validation",
            task_type=TaskType.FACTOR_VALIDATION,
            params={'factors': mining_result.data['factors']}
        )
        validation_result = await self.validation_agent.execute(validation_task)
        results['validation'] = validation_result
        
        # 3. 风控检查
        risk_task = Task(
            task_id=f"{task.task_id}_risk",
            task_type=TaskType.RISK_MONITORING,
            params={'factors': validation_result.data['validated_factors']}
        )
        risk_result = await self.risk_agent.execute(risk_task)
        results['risk'] = risk_result
        
        # 4. 聚合结果
        return TaskResult(
            task_id=task.task_id,
            status='success' if all(r.status == 'success' for r in results.values()) else 'partial',
            data={
                'factors': results['validation'].data['validated_factors'],
                'risk_report': results['risk'].data,
            },
            metrics=self._aggregate_metrics(results),
            logs=[]
        )
```

#### 3.1.2 Factor Mining Agent（因子挖掘智能体）

**职责**：
- 遗传规划生成因子表达式
- 模板搜索生成候选因子
- 因子组合优化
- 因子去重和过滤

**核心算法**：

```python
# futureQuant/agent/mining.py

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random

@dataclass
class FactorExpression:
    """因子表达式"""
    expression: str           # 字符串表达式
    ast: Any                  # 抽象语法树
    params: Dict[str, Any]    # 参数
    
class GeneticProgrammingEngine:
    """
    遗传规划引擎
    
    用于自动搜索因子表达式。
    """
    
    # 基础算子
    PRIMITIVES = {
        # 二元算子
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / (y + 1e-10),
        'max': np.maximum,
        'min': np.minimum,
        
        # 一元算子
        'abs': np.abs,
        'log': lambda x: np.log(np.abs(x) + 1e-10),
        'sign': np.sign,
        'rank': lambda x: pd.Series(x).rank(pct=True).values,
        
        # 时序算子
        'ts_mean': lambda x, w: pd.Series(x).rolling(w).mean().values,
        'ts_std': lambda x, w: pd.Series(x).rolling(w).std().values,
        'ts_max': lambda x, w: pd.Series(x).rolling(w).max().values,
        'ts_min': lambda x, w: pd.Series(x).rolling(w).min().values,
        'delta': lambda x, w: pd.Series(x).diff(w).values,
        'delay': lambda x, w: pd.Series(x).shift(w).values,
        
        # 统计算子
        'ts_corr': lambda x, y, w: pd.Series(x).rolling(w).corr(pd.Series(y)).values,
        'ts_cov': lambda x, y, w: pd.Series(x).rolling(w).cov(pd.Series(y)).values,
    }
    
    # 基础数据
    TERMINALS = ['close', 'open', 'high', 'low', 'volume', 'open_interest', 'vwap']
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 50,
                 tournament_size: int = 3,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # 初始化 DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """配置遗传规划"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # 定义个体生成
        self.toolbox.register("expr", self._generate_expression, max_depth=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                             self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # 遗传操作
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=self.tournament_size)
    
    def _generate_expression(self, max_depth: int) -> str:
        """生成随机表达式"""
        if max_depth == 0:
            return random.choice(self.TERMINALS)
        
        if random.random() < 0.3:
            # 选择终端
            return random.choice(self.TERMINALS)
        
        # 选择算子
        op = random.choice(list(self.PRIMITIVES.keys()))
        
        if op in ['ts_mean', 'ts_std', 'ts_max', 'ts_min', 'delta', 'delay']:
            # 时序算子需要窗口参数
            w = random.choice([5, 10, 20, 60])
            arg = self._generate_expression(max_depth - 1)
            return f"{op}({arg}, {w})"
        elif op in ['ts_corr', 'ts_cov']:
            # 二元时序算子
            w = random.choice([5, 10, 20])
            arg1 = self._generate_expression(max_depth - 1)
            arg2 = self._generate_expression(max_depth - 1)
            return f"{op}({arg1}, {arg2}, {w})"
        elif op in ['add', 'sub', 'mul', 'div', 'max', 'min']:
            # 二元算子
            arg1 = self._generate_expression(max_depth - 1)
            arg2 = self._generate_expression(max_depth - 1)
            return f"{op}({arg1}, {arg2})"
        else:
            # 一元算子
            arg = self._generate_expression(max_depth - 1)
            return f"{op}({arg})"
    
    def _evaluate(self, individual) -> Tuple[float]:
        """评估个体适应度"""
        expression = individual[0]
        try:
            factor_values = self._compute_factor(expression)
            ic = self._calculate_ic(factor_values)
            return (ic,)
        except:
            return (-999,)
    
    def evolve(self, 
               data: pd.DataFrame,
               returns: pd.Series,
               n_factors: int = 10) -> List[FactorExpression]:
        """
        进化搜索
        
        Args:
            data: OHLCV 数据
            returns: 目标收益序列
            n_factors: 返回因子数量
        
        Returns:
            最优因子表达式列表
        """
        self.data = data
        self.returns = returns
        
        # 初始种群
        pop = self.toolbox.population(n=self.population_size)
        
        # 进化
        for gen in range(self.generations):
            offspring = algorithms.varAnd(pop, self.toolbox, 
                                         self.crossover_prob, 
                                         self.mutation_prob)
            
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            pop = self.toolbox.select(offspring, k=len(pop))
            
            # 记录最优
            best = tools.selBest(pop, k=1)[0]
            print(f"Gen {gen}: best IC = {best.fitness.values[0]:.4f}")
        
        # 返回最优因子
        best_individuals = tools.selBest(pop, k=n_factors)
        return [FactorExpression(expression=ind[0], ast=None, params={}) 
                for ind in best_individuals]


class FactorMiningAgent:
    """
    因子挖掘智能体
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.gp_engine = GeneticProgrammingEngine()
        self.template_searcher = TemplateSearcher()
        self.combiner = FactorCombiner()
        
    async def execute(self, task: Task) -> TaskResult:
        """执行因子挖掘任务"""
        data = task.params['data']
        returns = task.params.get('returns')
        n_factors = task.params.get('n_factors', 20)
        
        # 计算目标收益
        if returns is None:
            forward_period = task.params.get('forward_period', 5)
            returns = data['close'].pct_change(forward_period).shift(-forward_period)
        
        all_factors = []
        
        # 1. 遗传规划搜索
        gp_factors = self.gp_engine.evolve(data, returns, n_factors=n_factors // 2)
        all_factors.extend(gp_factors)
        
        # 2. 模板搜索
        template_factors = self.template_searcher.search(data, returns, n_templates=20)
        all_factors.extend(template_factors)
        
        # 3. 因子组合
        combined_factors = self.combiner.combine(all_factors, data, returns)
        
        return TaskResult(
            task_id=task.task_id,
            status='success',
            data={
                'factors': all_factors,
                'combined_factors': combined_factors,
                'n_total': len(all_factors),
            },
            metrics={
                'gp_factors': len(gp_factors),
                'template_factors': len(template_factors),
            },
            logs=[]
        )
```

#### 3.1.3 Validation Agent（验证评估智能体）

**职责**：
- 多维度因子评分
- Purged K-Fold 交叉验证
- Walk-Forward 验证
- 样本权重计算
- 防泄漏检测

```python
# futureQuant/agent/validation.py

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class FactorScoreCard:
    """因子评分卡"""
    factor_name: str
    ic_mean: float
    ic_std: float
    icir: float
    ic_win_rate: float
    monotonicity: float
    turnover: float
    max_correlation: float
    composite_score: float
    grade: str  # S/A/B/C/D

class PurgedKFoldValidator:
    """Purged K-Fold 验证器"""
    
    def __init__(self, 
                 n_splits: int = 5,
                 purge_days: int = 5,
                 embargo_days: int = 2):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def validate(self,
                 factor_values: pd.Series,
                 returns: pd.Series) -> Dict:
        """执行验证"""
        n = len(factor_values)
        fold_size = n // self.n_splits
        
        ic_scores = []
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            
            # 训练集（排除 purge）
            train_end = test_start - self.purge_days
            train_start = 0 if i == 0 else (i - 1) * fold_size + self.embargo_days
            
            if train_start >= train_end:
                continue
            
            # 计算 IC
            test_factor = factor_values.iloc[test_start:test_end]
            test_returns = returns.iloc[test_start:test_end]
            
            ic = test_factor.corr(test_returns, method='spearman')
            ic_scores.append(ic)
        
        return {
            'ic_scores': ic_scores,
            'ic_mean': np.mean(ic_scores),
            'ic_std': np.std(ic_scores),
            'icir': np.mean(ic_scores) / (np.std(ic_scores) + 1e-10),
        }

class SampleWeightCalculator:
    """样本权重计算器"""
    
    def __init__(self,
                 time_decay_rate: float = 0.02,
                 vol_window: int = 20):
        self.time_decay_rate = time_decay_rate
        self.vol_window = vol_window
    
    def calculate(self,
                  returns: pd.Series,
                  data: pd.DataFrame) -> pd.Series:
        """计算样本权重"""
        n = len(returns)
        
        # 时间衰减权重
        t = np.arange(n)
        time_weight = np.exp(-self.time_decay_rate * (n - 1 - t))
        
        # 波动率调整权重
        rolling_vol = returns.rolling(self.vol_window).std()
        vol_weight = 1 / (rolling_vol + 1e-6)
        
        # 综合权重
        combined_weight = time_weight * vol_weight.values
        combined_weight = combined_weight / combined_weight.mean()
        
        return pd.Series(combined_weight, index=returns.index)

class ValidationAgent:
    """验证评估智能体"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scorer = FactorScorer()
        self.validator = PurgedKFoldValidator()
        self.weight_calculator = SampleWeightCalculator()
        self.leak_detector = FutureFunctionDetector()
    
    async def execute(self, task: Task) -> TaskResult:
        """执行验证任务"""
        factors = task.params['factors']
        data = task.params['data']
        returns = task.params.get('returns')
        
        if returns is None:
            forward_period = task.params.get('forward_period', 5)
            returns = data['close'].pct_change(forward_period).shift(-forward_period)
        
        # 计算样本权重
        sample_weights = self.weight_calculator.calculate(returns, data)
        
        validated_factors = []
        score_cards = []
        
        for factor_expr in factors:
            # 1. 计算因子值
            factor_values = self._compute_factor(factor_expr, data)
            
            # 2. 防泄漏检测
            leak_warnings = self.leak_detector.detect(factor_expr.expression)
            if leak_warnings:
                continue  # 跳过有问题的因子
            
            # 3. 多维度评分
            score_card = self.scorer.score(factor_values, returns, sample_weights)
            
            # 4. Purged CV 验证
            cv_result = self.validator.validate(factor_values, returns)
            
            # 5. 综合评估
            if score_card.composite_score >= 0.40:  # D 级以上
                validated_factors.append(factor_expr)
                score_cards.append(score_card)
        
        # 排序
        score_cards.sort(key=lambda x: x.composite_score, reverse=True)
        
        return TaskResult(
            task_id=task.task_id,
            status='success',
            data={
                'validated_factors': validated_factors,
                'score_cards': score_cards,
                'n_validated': len(validated_factors),
            },
            metrics={
                'pass_rate': len(validated_factors) / len(factors),
                'avg_score': np.mean([s.composite_score for s in score_cards]) if score_cards else 0,
            },
            logs=[]
        )
```

#### 3.1.4 Risk Control Agent（风控监控智能体）

**职责**：
- 因子衰减监控
- 相关性监控
- 仓位管理
- 预警通知

```python
# futureQuant/agent/risk.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskAlert:
    """风险预警"""
    alert_id: str
    risk_type: str
    level: RiskLevel
    message: str
    factor_name: Optional[str]
    current_value: float
    threshold: float
    action: str

class FactorDecayMonitor:
    """因子衰减监控器"""
    
    def __init__(self, 
                 ic_threshold: float = 0.01,
                 decay_window: int = 20):
        self.ic_threshold = ic_threshold
        self.decay_window = decay_window
        self.ic_history: Dict[str, List[float]] = {}
    
    def update(self, factor_name: str, ic: float) -> Optional[RiskAlert]:
        """更新 IC 并检测衰减"""
        if factor_name not in self.ic_history:
            self.ic_history[factor_name] = []
        
        self.ic_history[factor_name].append(ic)
        history = self.ic_history[factor_name]
        
        if len(history) < self.decay_window:
            return None
        
        recent_ic = np.mean(history[-self.decay_window:])
        trend = np.polyfit(range(self.decay_window), 
                          history[-self.decay_window:], 1)[0]
        
        # 判断风险等级
        if recent_ic < 0:
            level = RiskLevel.CRITICAL
            action = "建议立即剔除该因子"
        elif recent_ic < self.ic_threshold:
            level = RiskLevel.HIGH
            action = "建议降低该因子权重"
        elif trend < -0.001:
            level = RiskLevel.MEDIUM
            action = "监控因子表现"
        else:
            return None
        
        return RiskAlert(
            alert_id=f"{factor_name}_decay_{len(history)}",
            risk_type="factor_decay",
            level=level,
            message=f"因子 {factor_name} IC 衰减: recent_ic={recent_ic:.4f}, trend={trend:.4f}",
            factor_name=factor_name,
            current_value=recent_ic,
            threshold=self.ic_threshold,
            action=action
        )

class CorrelationMonitor:
    """相关性监控器"""
    
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
    
    def check(self, 
              factor_values: pd.DataFrame) -> Optional[RiskAlert]:
        """检查因子相关性"""
        corr_matrix = factor_values.corr(method='spearman')
        
        # 找到最大相关性（排除对角线）
        max_corr = 0
        max_pair = None
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > max_corr:
                    max_corr = corr
                    max_pair = (corr_matrix.index[i], corr_matrix.columns[j])
        
        if max_corr > self.correlation_threshold:
            return RiskAlert(
                alert_id=f"corr_{max_pair[0]}_{max_pair[1]}",
                risk_type="high_correlation",
                level=RiskLevel.HIGH,
                message=f"因子 {max_pair[0]} 与 {max_pair[1]} 相关性过高: {max_corr:.2f}",
                factor_name=None,
                current_value=max_corr,
                threshold=self.correlation_threshold,
                action="建议进行正交化处理或剔除其中一个因子"
            )
        
        return None

class RiskControlAgent:
    """风控监控智能体"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.decay_monitor = FactorDecayMonitor()
        self.corr_monitor = CorrelationMonitor()
        self.position_manager = PositionManager()
        self.alerts: List[RiskAlert] = []
    
    async def execute(self, task: Task) -> TaskResult:
        """执行风控监控任务"""
        factors = task.params.get('factors', [])
        factor_values = task.params.get('factor_values')
        current_positions = task.params.get('positions', {})
        
        alerts = []
        
        # 1. 因子衰减监控
        if 'ic_values' in task.params:
            for factor_name, ic in task.params['ic_values'].items():
                alert = self.decay_monitor.update(factor_name, ic)
                if alert:
                    alerts.append(alert)
        
        # 2. 相关性监控
        if factor_values is not None and len(factor_values.columns) > 1:
            alert = self.corr_monitor.check(factor_values)
            if alert:
                alerts.append(alert)
        
        # 3. 仓位管理建议
        position_advice = self.position_manager.get_position_advice(
            factors=factors,
            current_positions=current_positions
        )
        
        # 记录预警
        self.alerts.extend(alerts)
        
        return TaskResult(
            task_id=task.task_id,
            status='success',
            data={
                'alerts': alerts,
                'position_advice': position_advice,
                'risk_level': self._aggregate_risk_level(alerts),
            },
            metrics={
                'n_alerts': len(alerts),
                'critical_count': sum(1 for a in alerts if a.level == RiskLevel.CRITICAL),
            },
            logs=[]
        )
    
    def _aggregate_risk_level(self, alerts: List[RiskAlert]) -> RiskLevel:
        """汇总风险等级"""
        if not alerts:
            return RiskLevel.LOW
        
        levels = [a.level for a in alerts]
        if RiskLevel.CRITICAL in levels:
            return RiskLevel.CRITICAL
        elif RiskLevel.HIGH in levels:
            return RiskLevel.HIGH
        elif RiskLevel.MEDIUM in levels:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
```

---

### 3.2 Core Layer（核心层）

#### 3.2.1 Factor Scorer（因子评分器）

```python
# futureQuant/agent/scoring.py

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class FactorScore:
    """因子评分"""
    # 预测能力
    ic_mean: float
    ic_std: float
    icir: float
    ic_win_rate: float
    
    # 分层表现
    quantile_monotonicity: float
    long_short_return: float
    
    # 交易成本
    turnover: float
    
    # 正交性
    max_correlation: float
    
    # 综合得分
    composite_score: float
    grade: str

class FactorScorer:
    """多维度因子评分器"""
    
    # 评分权重
    WEIGHTS = {
        'ic': 0.30,
        'icir': 0.25,
        'monotonicity': 0.15,
        'turnover': 0.10,
        'orthogonality': 0.10,
        'stability': 0.10,
    }
    
    # 评分阈值
    THRESHOLDS = {
        'ic_good': 0.02,
        'ic_excellent': 0.04,
        'icir_good': 0.5,
        'icir_excellent': 1.0,
        'turnover_max': 0.5,
        'correlation_max': 0.7,
    }
    
    def score(self,
              factor_values: pd.Series,
              returns: pd.Series,
              sample_weights: Optional[pd.Series] = None) -> FactorScore:
        """计算因子综合评分"""
        
        # 1. IC 评分
        ic_scores = self._calculate_rolling_ic(factor_values, returns, sample_weights)
        ic_mean = np.mean(ic_scores)
        ic_std = np.std(ic_scores)
        icir = ic_mean / (ic_std + 1e-10)
        ic_win_rate = np.mean(np.array(ic_scores) > 0)
        
        # 2. 分层评分
        quantile_returns = self._calculate_quantile_returns(factor_values, returns)
        monotonicity = self._calculate_monotonicity(quantile_returns)
        long_short = quantile_returns[-1] - quantile_returns[0] if len(quantile_returns) >= 2 else 0
        
        # 3. 换手率
        turnover = self._calculate_turnover(factor_values)
        
        # 4. 正交性（需要外部提供相关性）
        max_correlation = 0.0  # 默认值，由外部更新
        
        # 5. 综合评分
        ic_score = min(1.0, abs(ic_mean) / self.THRESHOLDS['ic_good'])
        icir_score = min(1.0, icir / self.THRESHOLDS['icir_good'])
        mono_score = max(0, monotonicity)
        turnover_score = max(0, 1 - turnover / self.THRESHOLDS['turnover_max'])
        orth_score = 1.0  # 默认值
        stab_score = ic_win_rate
        
        composite = (
            self.WEIGHTS['ic'] * ic_score +
            self.WEIGHTS['icir'] * icir_score +
            self.WEIGHTS['monotonicity'] * mono_score +
            self.WEIGHTS['turnover'] * turnover_score +
            self.WEIGHTS['orthogonality'] * orth_score +
            self.WEIGHTS['stability'] * stab_score
        )
        
        # 等级判定
        if composite >= 0.85:
            grade = 'S'
        elif composite >= 0.70:
            grade = 'A'
        elif composite >= 0.55:
            grade = 'B'
        elif composite >= 0.40:
            grade = 'C'
        else:
            grade = 'D'
        
        return FactorScore(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_win_rate=ic_win_rate,
            quantile_monotonicity=monotonicity,
            long_short_return=long_short,
            turnover=turnover,
            max_correlation=max_correlation,
            composite_score=composite,
            grade=grade
        )
    
    def _calculate_rolling_ic(self,
                              factor_values: pd.Series,
                              returns: pd.Series,
                              sample_weights: Optional[pd.Series] = None,
                              window: int = 20) -> List[float]:
        """计算滚动 IC"""
        ic_scores = []
        n = len(factor_values)
        
        for i in range(window, n):
            f = factor_values.iloc[i-window:i]
            r = returns.iloc[i-window:i]
            
            if sample_weights is not None:
                w = sample_weights.iloc[i-window:i]
                # 加权秩相关
                ic = self._weighted_spearman(f, r, w)
            else:
                ic = f.corr(r, method='spearman')
            
            ic_scores.append(ic)
        
        return ic_scores
    
    def _weighted_spearman(self, x, y, w):
        """加权 Spearman 相关系数"""
        x_rank = x.rank()
        y_rank = y.rank()
        
        # 加权协方差
        mean_x = (x_rank * w).sum() / w.sum()
        mean_y = (y_rank * w).sum() / w.sum()
        
        cov = ((x_rank - mean_x) * (y_rank - mean_y) * w).sum() / w.sum()
        std_x = np.sqrt(((x_rank - mean_x) ** 2 * w).sum() / w.sum())
        std_y = np.sqrt(((y_rank - mean_y) ** 2 * w).sum() / w.sum())
        
        return cov / (std_x * std_y + 1e-10)
    
    def _calculate_quantile_returns(self,
                                    factor_values: pd.Series,
                                    returns: pd.Series,
                                    n_quantiles: int = 5) -> np.ndarray:
        """计算分层收益"""
        quantiles = pd.qcut(factor_values, n_quantiles, labels=False, duplicates='drop')
        quantile_returns = returns.groupby(quantiles).mean().values
        return quantile_returns
    
    def _calculate_monotonicity(self, quantile_returns: np.ndarray) -> float:
        """计算单调性"""
        if len(quantile_returns) < 2:
            return 0.0
        
        # Spearman 相关系数
        ranks = np.arange(len(quantile_returns))
        from scipy.stats import spearmanr
        corr, _ = spearmanr(ranks, quantile_returns)
        return corr
    
    def _calculate_turnover(self, factor_values: pd.Series) -> float:
        """计算换手率"""
        # 因子值变化比例
        factor_rank = factor_values.rank(pct=True)
        rank_change = factor_rank.diff().abs()
        turnover = rank_change.mean()
        return turnover
```

---

### 3.3 Data Flow（数据流）

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流架构                                │
└─────────────────────────────────────────────────────────────────┘

用户请求
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Data Layer  │────▶│ Mining Agent│────▶│ Valid Agent │
│ (OHLCV)     │     │ (因子表达式) │     │ (评分卡)    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Risk Agent  │
                                        │ (预警/仓位) │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Backtest    │
                                        │ (策略验证)  │
                                        └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Report      │
                                        │ (输出报告)  │
                                        └─────────────┘
```

---

## 4. 数据库设计

### 4.1 因子库表结构

```sql
-- 因子定义表
CREATE TABLE factor_definitions (
    factor_id UUID PRIMARY KEY,
    factor_name VARCHAR(100) NOT NULL,
    expression TEXT NOT NULL,
    category VARCHAR(50),          -- technical/fundamental/macro
    parameters JSON,               -- 参数配置
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 因子评分表
CREATE TABLE factor_scores (
    score_id UUID PRIMARY KEY,
    factor_id UUID REFERENCES factor_definitions(factor_id),
    evaluation_date DATE,
    ic_mean FLOAT,
    ic_std FLOAT,
    icir FLOAT,
    ic_win_rate FLOAT,
    monotonicity FLOAT,
    turnover FLOAT,
    max_correlation FLOAT,
    composite_score FLOAT,
    grade CHAR(1),
    sample_weights JSON,
    validation_method VARCHAR(50),
    created_at TIMESTAMP
);

-- 因子相关性矩阵
CREATE TABLE factor_correlations (
    correlation_id UUID PRIMARY KEY,
    factor_id_1 UUID REFERENCES factor_definitions(factor_id),
    factor_id_2 UUID REFERENCES factor_definitions(factor_id),
    correlation_date DATE,
    spearman_corr FLOAT,
    pearson_corr FLOAT,
    created_at TIMESTAMP
);

-- 预警记录表
CREATE TABLE risk_alerts (
    alert_id UUID PRIMARY KEY,
    alert_type VARCHAR(50),
    risk_level VARCHAR(20),
    factor_id UUID REFERENCES factor_definitions(factor_id),
    message TEXT,
    current_value FLOAT,
    threshold FLOAT,
    action_suggested TEXT,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100)
);
```

---

## 5. 部署架构

### 5.1 单机部署

```
┌─────────────────────────────────────────────────────────────────┐
│                        单机部署架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    futureQuant Process                      ││
│  │                                                             ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  ││
│  │  │Orchestrator│ │Mining Agt │ │Valid Agt  │ │Risk Agt   │  ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘  ││
│  │                                                             ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐                ││
│  │  │Factor Core│ │Backtest   │ │ML Core    │                ││
│  │  └───────────┘ └───────────┘ └───────────┘                ││
│  │                                                             ││
│  │  ┌─────────────────────────────────────────────────────┐  ││
│  │  │         DuckDB (嵌入式数据库)                         │  ││
│  │  └─────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      File System                            ││
│  │                                                             ││
│  │  data/          logs/           cache/         reports/    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 分布式部署（可选）

```
┌─────────────────────────────────────────────────────────────────┐
│                        分布式部署架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │ API Gateway │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────▼──────┐                                                │
│  │ Orchestrator│                                                │
│  │   Service   │                                                │
│  └──────┬──────┘                                                │
│         │                                                        │
│    ┌────┴────┬────────────┬────────────┐                       │
│    │         │            │            │                        │
│ ┌──▼──┐   ┌──▼──┐     ┌──▼──┐     ┌──▼──┐                     │
│ │Mining│   │Valid│     │Risk │     │Backtest│                   │
│ │Worker│   │Worker│    │Worker│    │Worker│                    │
│ └──┬──┘   └──┬──┘     └──┬──┘     └──┬──┘                     │
│    │         │            │            │                        │
│    └─────────┴────────────┴────────────┘                       │
│                  │                                              │
│         ┌────────▼────────┐                                    │
│         │  Message Queue  │                                    │
│         │    (Redis)      │                                    │
│         └────────┬────────┘                                    │
│                  │                                              │
│    ┌─────────────┴─────────────┐                              │
│    │                           │                              │
│ ┌──▼──┐                   ┌───▼───┐                          │
│ │Redis│                   │DuckDB │                          │
│ │Cache│                   │Cluster│                          │
│ └─────┘                   └───────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 性能优化策略

### 6.1 计算优化

| 优化点 | 策略 | 预期提升 |
|--------|------|----------|
| 因子计算 | 向量化 + Numba JIT | 10-50x |
| 并行计算 | Joblib 多进程 | 线性扩展 |
| 内存管理 | 分块处理 + 内存映射 | 50% 内存节省 |
| 缓存策略 | Redis 热数据缓存 | 90% 命中率 |
| 增量更新 | 只计算新增数据 | 80% 时间节省 |

### 6.2 存储优化

| 优化点 | 策略 |
|--------|------|
| 时序数据 | Parquet 列存储 + 压缩 |
| 索引优化 | 按日期分区 + 品种索引 |
| 冷热分离 | 热数据 Redis + 冷数据 Parquet |

---

## 7. 监控与运维

### 7.1 监控指标

```python
# 监控指标定义
METRICS = {
    # 系统指标
    'cpu_usage': Gauge('system_cpu_usage', 'CPU 使用率'),
    'memory_usage': Gauge('system_memory_usage', '内存使用率'),
    
    # 业务指标
    'factors_mined': Counter('factors_mined_total', '挖掘因子总数'),
    'factors_validated': Counter('factors_validated_total', '验证因子总数'),
    'backtests_run': Counter('backtests_run_total', '回测运行总数'),
    
    # 性能指标
    'factor_compute_time': Histogram('factor_compute_seconds', '因子计算耗时'),
    'validation_time': Histogram('validation_seconds', '验证耗时'),
    'backtest_time': Histogram('backtest_seconds', '回测耗时'),
    
    # 质量指标
    'avg_factor_ic': Gauge('avg_factor_ic', '平均因子 IC'),
    'avg_factor_score': Gauge('avg_factor_score', '平均因子评分'),
    'risk_alert_count': Counter('risk_alert_total', '风险预警总数'),
}
```

### 7.2 日志规范

```python
# 结构化日志示例
{
    "timestamp": "2026-03-26T10:30:00Z",
    "level": "INFO",
    "agent": "mining",
    "task_id": "task_001",
    "event": "factor_mined",
    "data": {
        "factor_name": "rank(delta(close, 5) / ts_std(close, 20))",
        "ic": 0.032,
        "score": 0.85
    },
    "duration_ms": 150
}
```

---

## 8. 安全与权限

### 8.1 数据安全

- 因子表达式加密存储
- 敏感数据脱敏
- 访问日志审计

### 8.2 操作权限

| 角色 | 权限 |
|------|------|
| 观察者 | 查看报告 |
| 分析师 | 因子挖掘、回测 |
| 管理员 | 全部权限 |

---

## 9. 扩展性设计

### 9.1 插件系统

```python
# 因子算子插件
class FactorOperatorPlugin:
    """因子算子插件接口"""
    
    name: str
    description: str
    
    def compute(self, data: pd.DataFrame, **params) -> pd.Series:
        raise NotImplementedError

# 注册插件
registry.register_operator('my_custom_op', MyCustomOperator())
```

### 9.2 自定义智能体

```python
class CustomAgent(BaseAgent):
    """自定义智能体"""
    
    def execute(self, task: Task) -> TaskResult:
        # 自定义逻辑
        pass

# 注册智能体
orchestrator.register_agent('custom', CustomAgent())
```

---

## 10. 技术风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 遗传规划搜索效率低 | 中 | 高 | 限制搜索空间、并行计算 |
| 内存溢出 | 中 | 高 | 分块处理、内存监控 |
| 相关性计算慢 | 低 | 中 | 近似计算、增量更新 |
| DuckDB 并发限制 | 低 | 中 | 连接池、读写分离 |

---

## 11. 实施路径

详见 `MULTI_AGENT_REFACTOR_PLAN.md`
