# 多智能体因子挖掘与策略自动回测系统

## 项目概述

为 futureQuant 量化框架新增**基于多智能体分析的日频因子挖掘及策略自动回测功能**。

### 核心目标

构建一个自动化的因子研究平台，通过多个专业智能体协作，实现：
1. **因子自动发现**：从技术面、基本面、宏观面多维度挖掘有效因子
2. **因子质量保障**：防未来函数、时序交叉验证、样本权重优化
3. **策略自动回测**：生成因子策略并自动评估回测表现
4. **多维度评估体系**：IC/ICIR、分层回测、风险指标、稳定性分析

### 目标用户

- 量化研究员：快速验证因子想法
- 策略开发者：自动化因子挖掘与回测
- 投资经理：评估因子策略的有效性和稳定性

### 核心功能（MVP）

1. **多智能体因子挖掘模块**
   - 技术因子挖掘 Agent
   - 基本面因子挖掘 Agent  
   - 宏观因子挖掘 Agent
   - 因子融合与筛选 Agent

2. **因子评估与验证模块**
   - 防未来函数检测
   - 时序交叉验证（Walk-Forward CV）
   - 样本权重计算
   - 多维度因子评分

3. **策略自动回测模块**
   - 因子策略生成
   - 风险控制规则
   - 回测评估报告

4. **因子库管理模块**
   - 因子持久化存储
   - 因子版本管理
   - 因子性能追踪

### 技术约束

- **集成现有框架**：基于 futureQuant 现有架构扩展
- **Python 3.10+**：保持与主项目一致
- **日频数据**：专注于日频因子研究与回测

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [需求文档 (PRD.md)](./PRD.md) | 详细功能需求、用户故事、验收标准 |
| [技术架构 (ARCHITECTURE.md)](./ARCHITECTURE.md) | 系统架构、模块设计、数据模型 |
| [实现计划 (IMPLEMENTATION.md)](./IMPLEMENTATION.md) | 开发里程碑、任务分解、风险控制 |

## 快速开始

```python
# 示例：使用多智能体因子挖掘系统
from futureQuant.agent import MultiAgentFactorMiner

# 初始化挖掘器
miner = MultiAgentFactorMiner(
    symbols=['RB', 'I', 'HC'],  # 螺纹钢、铁矿石、热卷
    start_date='2020-01-01',
    end_date='2024-12-31',
)

# 运行因子挖掘
result = miner.run()

# 查看挖掘结果
print(f"发现有效因子: {len(result.factors)}")
print(f"最佳因子ICIR: {result.best_factor_icir:.3f}")

# 自动回测
backtest_result = miner.run_backtest(result.best_factors)
print(f"策略夏普比率: {backtest_result.sharpe_ratio:.3f}")
```

## 项目状态

🚧 **规划中** - 需求分析与架构设计阶段

---

**创建日期**: 2026-03-26
**最后更新**: 2026-03-26
