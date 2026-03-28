# 多智能体因子挖掘与策略回测系统 - 技术重构计划

> **版本**: v1.0  
> **日期**: 2026-03-26  
> **状态**: 计划阶段  
> **预计工期**: 4 周

---

## 1. 重构目标与范围

### 1.1 目标

将 futureQuant 从单体应用重构为多智能体协作架构，实现：
- 自动化因子挖掘（遗传规划 + 模板搜索）
- 多维度因子评分体系
- Purged CV + Walk-forward 验证
- 智能样本权重
- 实时风控监控

### 1.2 重构范围

| 模块 | 变更类型 | 复杂度 |
|------|----------|--------|
| 新增 `agent/` 模块 | 新增 | 高 |
| 扩展 `factor/` 模块 | 扩展 | 中 |
| 扩展 `model/` 模块 | 扩展 | 中 |
| 扩展 `backtest/` 模块 | 扩展 | 低 |
| 新增 `risk/` 模块 | 新增 | 中 |
| 新增 `data/` 存储 | 扩展 | 低 |
| 测试覆盖 | 新增 | 中 |

### 1.3 非目标

- 不修改现有 API 接口（保持向后兼容）
- 不重构数据库连接逻辑
- 不涉及前端界面开发

---

## 2. 目录结构规划

```
futureQuant/
├── futureQuant/
│   ├── core/                    # 现有，保持不变
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── exceptions.py
│   │
│   ├── data/                    # 现有，扩展
│   │   ├── manager.py
│   │   ├── processor/
│   │   └── storage/             # 新增：DuckDB 存储
│   │       ├── __init__.py
│   │       ├── duckdb_store.py
│   │       └── factor_store.py
│   │
│   ├── factor/                  # 现有，扩展
│   │   ├── engine.py            # 扩展：支持自定义算子
│   │   ├── evaluator.py         # 扩展：多维度评分
│   │   ├── technical/
│   │   ├── fundamental/
│   │   ├── macro/
│   │   └── expression/          # 新增：表达式解析
│   │       ├── __init__.py
│   │       ├── parser.py
│   │       ├── safe_compute.py
│   │       └── operators.py
│   │
│   ├── model/                   # 现有，扩展
│   │   ├── feature_engineering.py  # 扩展：样本权重
│   │   ├── pipeline.py             # 扩展：Purged CV
│   │   ├── supervised/
│   │   ├── time_series/
│   │   └── validation/          # 新增：验证框架
│   │       ├── __init__.py
│   │       ├── purged_cv.py
│   │       ├── walk_forward.py
│   │       └── sample_weight.py
│   │
│   ├── strategy/                # 现有，保持不变
│   │
│   ├── backtest/                # 现有，保持不变
│   │
│   ├── analysis/                # 现有，保持不变
│   │
│   ├── risk/                    # 新增：风控模块
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   ├── monitor.py
│   │   ├── limits.py
│   │   └── alert.py
│   │
│   ├── agent/                   # 新增：智能体模块
│   │   ├── __init__.py
│   │   ├── base.py              # 智能体基类
│   │   ├── orchestrator.py      # 调度智能体
│   │   ├── mining.py            # 因子挖掘智能体
│   │   ├── validation.py        # 验证评估智能体
│   │   ├── risk.py              # 风控监控智能体
│   │   ├── scoring.py           # 因子评分器
│   │   ├── state.py             # 状态管理
│   │   └── gp/                  # 遗传规划引擎
│   │       ├── __init__.py
│   │       ├── engine.py
│   │       ├── primitives.py
│   │       └── fitness.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── unit/
│   │   ├── test_agent.py
│   │   ├── test_scoring.py
│   │   ├── test_validation.py
│   │   ├── test_risk.py
│   │   └── test_gp.py
│   └── integration/
│       └── test_full_pipeline.py
│
├── docs/
│   ├── MULTI_AGENT_REQUIREMENTS.md   # 需求文档
│   ├── MULTI_AGENT_ARCHITECTURE.md   # 架构文档
│   └── MULTI_AGENT_REFACTOR_PLAN.md  # 本文档
│
├── requirements.txt
├── requirements-agent.txt        # 新增：Agent 相关依赖
└── setup.py
```

---

## 3. 分阶段实施计划

### 3.1 阶段一：基础设施（第 1 周）

**目标**：搭建智能体框架和存储层

#### Day 1-2：智能体基础框架

| 任务 | 文件 | 产出 |
|------|------|------|
| 智能体基类 | `agent/base.py` | `BaseAgent`, `Task`, `TaskResult` |
| 状态管理器 | `agent/state.py` | `StateManager` |
| 调度器框架 | `agent/orchestrator.py` | `OrchestratorAgent` |

```python
# agent/base.py 骨架
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Task:
    task_id: str
    task_type: str
    params: Dict[str, Any]
    priority: int = 0

@dataclass
class TaskResult:
    task_id: str
    status: str
    data: Dict[str, Any]
    metrics: Dict[str, float]
    logs: list

class BaseAgent(ABC):
    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        pass
```

#### Day 3-4：存储层扩展

| 任务 | 文件 | 产出 |
|------|------|------|
| DuckDB 存储 | `data/storage/duckdb_store.py` | `DuckDBStore` |
| 因子存储 | `data/storage/factor_store.py` | `FactorStore` |

#### Day 5：因子表达式解析

| 任务 | 文件 | 产出 |
|------|------|------|
| 表达式解析器 | `factor/expression/parser.py` | `ExpressionParser` |
| 安全计算器 | `factor/expression/safe_compute.py` | `SafeFactorComputer` |
| 未来函数检测 | `factor/expression/operators.py` | `FutureFunctionDetector` |

**阶段一验收标准**：
- [ ] `BaseAgent` 可创建并执行任务
- [ ] `DuckDBStore` 可存取因子数据
- [ ] `SafeFactorComputer` 可安全计算因子并检测未来函数

---

### 3.2 阶段二：因子挖掘智能体（第 2 周）

**目标**：实现遗传规划因子挖掘

#### Day 6-7：遗传规划引擎

| 任务 | 文件 | 产出 |
|------|------|------|
| 基础算子 | `agent/gp/primitives.py` | 算子定义 |
| 适应度函数 | `agent/gp/fitness.py` | `ICFitness`, `CompositeFitness` |
| GP 引擎 | `agent/gp/engine.py` | `GeneticProgrammingEngine` |

```python
# agent/gp/engine.py 骨架
class GeneticProgrammingEngine:
    def __init__(self, config: GPConfig):
        self.primitives = PRIMITIVE_SET
        self.terminals = TERMINAL_SET
        self.fitness_func = IC Fitness()
    
    def evolve(self, data, returns, n_factors) -> List[FactorExpression]:
        # 初始化种群
        # 迭代进化
        # 返回最优因子
        pass
```

#### Day 8-9：因子挖掘智能体

| 任务 | 文件 | 产出 |
|------|------|------|
| 模板搜索 | `agent/mining.py` | `TemplateSearcher` |
| 因子组合 | `agent/mining.py` | `FactorCombiner` |
| 挖掘智能体 | `agent/mining.py` | `FactorMiningAgent` |

#### Day 10：集成测试

| 任务 | 文件 | 产出 |
|------|------|------|
| 单元测试 | `tests/unit/test_gp.py` | GP 测试 |
| 集成测试 | `tests/integration/test_mining.py` | 挖掘流水线测试 |

**阶段二验收标准**：
- [ ] GP 引擎可进化生成因子表达式
- [ ] 生成的因子表达式可被安全计算
- [ ] `FactorMiningAgent` 可完成完整挖掘任务

---

### 3.3 阶段三：验证评估智能体（第 3 周）

**目标**：实现多维度评分和交叉验证

#### Day 11-12：验证框架

| 任务 | 文件 | 产出 |
|------|------|------|
| Purged K-Fold | `model/validation/purged_cv.py` | `PurgedKFoldValidator` |
| Walk-Forward | `model/validation/walk_forward.py` | `WalkForwardValidator` |
| 样本权重 | `model/validation/sample_weight.py` | `SampleWeightCalculator` |

```python
# model/validation/purged_cv.py 骨架
class PurgedKFoldValidator:
    def __init__(self, n_splits=5, purge_days=5, embargo_days=2):
        ...
    
    def validate(self, factor_values, returns) -> Dict:
        # 生成分割索引
        # 计算各折 IC
        # 返回验证结果
        pass
```

#### Day 13-14：因子评分器

| 任务 | 文件 | 产出 |
|------|------|------|
| 多维度评分 | `agent/scoring.py` | `FactorScorer`, `FactorScore` |
| 评分卡生成 | `agent/scoring.py` | `FactorScoreCard` |

#### Day 15：验证智能体

| 任务 | 文件 | 产出 |
|------|------|------|
| 验证智能体 | `agent/validation.py` | `ValidationAgent` |
| 单元测试 | `tests/unit/test_validation.py` | 验证测试 |

**阶段三验收标准**：
- [ ] Purged K-Fold 正确执行交叉验证
- [ ] 因子评分卡包含所有维度评分
- [ ] `ValidationAgent` 可完成完整验证任务

---

### 3.4 阶段四：风控监控智能体（第 4 周）

**目标**：实现风险监控和预警

#### Day 16-17：风控模块

| 任务 | 文件 | 产出 |
|------|------|------|
| 风险计算 | `risk/calculator.py` | `RiskCalculator` |
| 监控器 | `risk/monitor.py` | `FactorDecayMonitor`, `CorrelationMonitor` |
| 限制规则 | `risk/limits.py` | `PositionLimiter`, `LeverageLimiter` |
| 预警系统 | `risk/alert.py` | `AlertManager`, `RiskAlert` |

#### Day 18：风控智能体

| 任务 | 文件 | 产出 |
|------|------|------|
| 风控智能体 | `agent/risk.py` | `RiskControlAgent` |
| 仓位管理 | `agent/risk.py` | `PositionManager` |

#### Day 19：完整流水线集成

| 任务 | 文件 | 产出 |
|------|------|------|
| 调度器完善 | `agent/orchestrator.py` | 完整调度逻辑 |
| 集成测试 | `tests/integration/test_full_pipeline.py` | 端到端测试 |

#### Day 20：文档和清理

| 任务 | 文件 | 产出 |
|------|------|------|
| API 文档更新 | `docs/API.md` | 更新 API 文档 |
| 代码清理 | - | 移除临时文件、优化导入 |
| 发布准备 | - | 更新版本号、CHANGELOG |

**阶段四验收标准**：
- [ ] 风控预警正确触发
- [ ] 完整流水线端到端运行成功
- [ ] 文档更新完成

---

## 4. 详细任务清单

### 4.1 阶段一任务

| ID | 任务 | 优先级 | 预估工时 | 依赖 |
|----|------|--------|----------|------|
| 1.1 | 创建 agent/base.py | P0 | 4h | - |
| 1.2 | 创建 agent/state.py | P0 | 2h | 1.1 |
| 1.3 | 创建 agent/orchestrator.py | P0 | 6h | 1.1, 1.2 |
| 1.4 | 创建 data/storage/duckdb_store.py | P1 | 4h | - |
| 1.5 | 创建 data/storage/factor_store.py | P1 | 3h | 1.4 |
| 1.6 | 创建 factor/expression/parser.py | P0 | 6h | - |
| 1.7 | 创建 factor/expression/safe_compute.py | P0 | 4h | 1.6 |
| 1.8 | 创建 factor/expression/operators.py | P0 | 3h | - |
| 1.9 | 单元测试 | P1 | 4h | 1.1-1.8 |

### 4.2 阶段二任务

| ID | 任务 | 优先级 | 预估工时 | 依赖 |
|----|------|--------|----------|------|
| 2.1 | 创建 agent/gp/primitives.py | P0 | 4h | - |
| 2.2 | 创建 agent/gp/fitness.py | P0 | 3h | 2.1 |
| 2.3 | 创建 agent/gp/engine.py | P0 | 8h | 2.1, 2.2 |
| 2.4 | 创建 agent/mining.py (TemplateSearcher) | P1 | 4h | - |
| 2.5 | 创建 agent/mining.py (FactorCombiner) | P1 | 4h | - |
| 2.6 | 创建 agent/mining.py (FactorMiningAgent) | P0 | 4h | 2.3, 2.4, 2.5 |
| 2.7 | 单元测试 | P1 | 4h | 2.1-2.6 |
| 2.8 | 集成测试 | P1 | 4h | 2.7 |

### 4.3 阶段三任务

| ID | 任务 | 优先级 | 预估工时 | 依赖 |
|----|------|--------|----------|------|
| 3.1 | 创建 model/validation/purged_cv.py | P0 | 4h | - |
| 3.2 | 创建 model/validation/walk_forward.py | P0 | 3h | - |
| 3.3 | 创建 model/validation/sample_weight.py | P0 | 3h | - |
| 3.4 | 创建 agent/scoring.py (FactorScorer) | P0 | 6h | - |
| 3.5 | 创建 agent/scoring.py (FactorScoreCard) | P1 | 2h | 3.4 |
| 3.6 | 创建 agent/validation.py | P0 | 4h | 3.1-3.5 |
| 3.7 | 单元测试 | P1 | 4h | 3.1-3.6 |

### 4.4 阶段四任务

| ID | 任务 | 优先级 | 预估工时 | 依赖 |
|----|------|--------|----------|------|
| 4.1 | 创建 risk/calculator.py | P1 | 3h | - |
| 4.2 | 创建 risk/monitor.py | P0 | 4h | - |
| 4.3 | 创建 risk/limits.py | P1 | 3h | - |
| 4.4 | 创建 risk/alert.py | P0 | 3h | - |
| 4.5 | 创建 agent/risk.py | P0 | 4h | 4.1-4.4 |
| 4.6 | 完善 agent/orchestrator.py | P0 | 4h | 4.5 |
| 4.7 | 集成测试 | P0 | 4h | 4.6 |
| 4.8 | 文档更新 | P1 | 3h | 4.7 |
| 4.9 | 代码清理 | P1 | 2h | 4.8 |

---

## 5. 依赖管理

### 5.1 新增依赖

```toml
# requirements-agent.txt

# 遗传规划
deap>=1.4.0

# 数据存储
duckdb>=0.9.0
pyarrow>=14.0.0

# 并行计算
joblib>=1.3.0

# 科学计算
scipy>=1.11.0

# 可视化
plotly>=5.18.0
```

### 5.2 安装命令

```bash
pip install -r requirements.txt
pip install -r requirements-agent.txt
```

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 | 应急计划 |
|------|------|------|----------|----------|
| GP 搜索效率低 | 中 | 高 | 限制搜索空间、并行计算 | 减少种群规模 |
| 内存不足 | 中 | 高 | 分块处理、内存监控 | 限制并发数 |
| DuckDB 并发问题 | 低 | 中 | 连接池管理 | 降级到文件存储 |
| 依赖冲突 | 低 | 中 | 虚拟环境隔离 | 版本锁定 |
| 测试覆盖不足 | 中 | 中 | 边开发边测试 | 延长测试阶段 |

---

## 7. 验收标准

### 7.1 功能验收

| 功能 | 验收标准 | 测试方法 |
|------|----------|----------|
| 因子挖掘 | GP 可进化出 IC > 0.02 的因子 | 单元测试 |
| 因子评分 | 评分卡包含 7 个维度 | 单元测试 |
| Purged CV | 5 折验证无数据泄漏 | 单元测试 |
| 样本权重 | 权重计算正确 | 单元测试 |
| 风控预警 | 超阈值触发预警 | 单元测试 |
| 完整流水线 | 端到端运行成功 | 集成测试 |

### 7.2 性能验收

| 指标 | 目标 | 测试方法 |
|------|------|----------|
| 因子计算 | < 100ms/因子 | 基准测试 |
| GP 进化 | < 10min/100代 | 基准测试 |
| 评分计算 | < 5s/因子 | 基准测试 |
| 内存占用 | < 4GB | 监控 |

### 7.3 质量验收

| 指标 | 目标 |
|------|------|
| 单元测试覆盖率 | > 80% |
| 集成测试通过率 | 100% |
| 代码审查 | 无 Critical 问题 |
| 文档完整度 | API 100% 覆盖 |

---

## 8. 发布计划

### 8.1 版本规划

| 版本 | 发布日期 | 内容 |
|------|----------|------|
| v0.5.0-alpha | 第 2 周末 | 因子挖掘智能体 |
| v0.6.0-alpha | 第 3 周末 | 验证评估智能体 |
| v0.7.0-alpha | 第 4 周末 | 风控监控智能体 |
| v1.0.0 | 第 5 周 | 正式发布 |

### 8.2 发布检查清单

- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 文档更新完成
- [ ] CHANGELOG 更新
- [ ] 版本号更新
- [ ] 依赖版本锁定

---

## 9. 附录

### 9.1 开发环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-agent.txt

# 安装开发工具
pip install pytest pytest-cov black isort mypy
```

### 9.2 测试命令

```bash
# 运行所有测试
pytest tests/

# 运行带覆盖率
pytest tests/ --cov=futureQuant --cov-report=html

# 运行特定测试
pytest tests/unit/test_gp.py -v
```

### 9.3 代码规范

```bash
# 格式化
black futureQuant/
isort futureQuant/

# 类型检查
mypy futureQuant/
```

---

## 10. 联系与支持

**项目负责人**: AI 助手  
**文档维护**: futureQuant 团队  
**问题反馈**: GitHub Issues  
**更新日期**: 2026-03-26
