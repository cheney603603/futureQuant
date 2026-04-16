# Agentic AI 因子挖掘改进计划

> 基于 Agentic AI 理念重构因子生成 Agent  
> 任务开始时间: 2026-04-07

## 背景

当前 `FactorMiningAgent` 采用预设的 50+ 候选因子池，属于"静态规则"模式。
Agentic AI 核心理念：让 Agent 具备**自主思考、探索、迭代**能力，而非执行预设流程。

## 目标

将因子挖掘 Agent 从"预设规则驱动"升级为"AI Agent 驱动"，实现：
1. **自主因子探索** - LLM 辅助生成新因子表达式
2. **遗传规划进化** - GP/GE 自动进化优质因子
3. **自我反思迭代** - 基于评估结果自主调整搜索策略

## 修改计划

### Phase 1: 基础架构升级
- [ ] 1.1 扩展 `FactorCandidatePool` 支持动态因子
- [ ] 1.2 引入 gplearn 作为因子表达式引擎
- [ ] 1.3 添加 LLM 因子生成器（GPT/Claude API）

### Phase 2: Agent 核心能力
- [ ] 2.1 实现 `FactorGenerationAgent` - 负责因子生成
- [ ] 2.2 实现 `FactorEvolutionAgent` - 负责因子进化
- [ ] 2.3 实现 `SelfReflection` - 评估结果反思调整

### Phase 3: 集成与测试
- [ ] 3.1 与现有 Agent 体系集成
- [ ] 3.2 端到端测试
- [ ] 3.3 性能优化

## 进度记录

### 2026-04-07
- [x] 分析现有代码结构
- [x] 阅读 factor_mining_agent.py
- [x] 阅读 factor_candidate_pool.py (50+ 候选因子)
- [x] 理解当前架构设计
- [ ] 创建任务文档

## 待确认
- [ ] 是否需要接入 LLM API (OpenAI/Claude/本地模型)?
- [ ] 是否需要 GPU 加速 GP 进化?
- [ ] 项目优先级和截止时间?