# 改进计划：参考 quant_react_interview 架构增强 futureQuant

**创建时间**: 2026-04-20  
**参考来源**: `D:\310Programm\quant_react_interview-main`  
**目标**: 为因子研究工作流引入更健壮的代理控制模型和丰富的节点目录

---

## 一、核心改进点

### 1. 增强型 ReAct 循环 (`react_base.py`)

**参考**: `quant_react_interview-main/agent/react_loop.py` 的 `_LoopCoordinator`

**改进内容**:
- 增加 `consecutive_empty` / `consecutive_errors` 失败计数器
- 增加 `max_empty_turns=3` / `max_error_turns=3` 配置
- 增加 `_inject_recovery()` 机制，失败时注入恢复提示而非直接终止
- 增加 `FINAL_ANSWER` + `pipeline_ready` 双终止条件
- 增加 `ReActMetrics` 统计（tool_calls, errors, iterations）

**文件**: `futureQuant/agent/react_base.py` (覆盖现有)

---

### 2. 因子研究目录 (`engine/nodes/factor_catalog.py`)

**参考**: `quant_react_interview-main/agent/catalog.py` 的 `_StepDescriptor`

**改进内容**:
- 建立完整的因子研究步骤目录（data → factor_mining → evaluation → fusion → backtest）
- 每个步骤包含：
  - `required_fields` / `optional_fields`
  - `field_descriptions`（语义说明，不仅是类型）
  - `output_shape`（如何访问输出）
  - `common_mistakes`（预判 LLM 错误）
  - `sample`（配置示例）
- 提供 `get_catalog()` / `get_details()` API

**文件**: `futureQuant/engine/nodes/factor_catalog.py` (新建)

---

### 3. 因子研究流水线构建器 (`engine/nodes/pipeline_builder.py`)

**参考**: `quant_react_interview-main/engine/core/builder.py`

**改进内容**:
- `FactorPipelineBuilder` 类，管理因子研究流水线
- `add_step()` / `update_step()` / `connect_steps()`
- `_materialize_inputs()` 支持 `$step_id['field']` 引用语法
- `ExecutionContext` 管理步骤间数据传递
- 立即执行验证（`add_step` 后尝试执行，错误立即返回）

**文件**: `futureQuant/engine/nodes/pipeline_builder.py` (新建)

---

### 4. 因子研究流程编排器 (`agent/factor_research_flow.py`)

**参考**: `quant_react_interview-main/engine/core/engine.py` + `orchestrator.py`

**改进内容**:
- 协调完整因子研究流程：data → mining → evaluation → fusion → backtest → report
- 支持流水线 YAML 配置加载
- 提供 `run_pipeline()` 入口
- 整合 BlackboardController，支持 Plan 驱动的执行

**文件**: `futureQuant/agent/factor_research_flow.py` (新建)

---

### 5. 通用工具表面 (`agent/tools/research_tools.py`)

**参考**: `quant_react_interview-main/agent/tools.py`

**改进内容**:
- 6 个通用工具：`add_step`, `update_step`, `connect_steps`, `get_catalog`, `get_details`, `get_pipeline`
- 丰富的 tool description 指导 LLM 正确调用
- 错误时提供 `hint` 引导修复

**文件**: `futureQuant/agent/tools/research_tools.py` (新建)

---

## 二、实施顺序

| 顺序 | 文件 | 依赖 | 状态 |
|------|------|------|------|
| 1 | `engine/nodes/__init__.py` | - | 新建 |
| 2 | `engine/nodes/factor_catalog.py` | - | 新建 |
| 3 | `engine/nodes/pipeline_builder.py` | factor_catalog | 新建 |
| 4 | `agent/tools/research_tools.py` | factor_catalog | 新建 |
| 5 | `agent/react_base.py` | - | 覆盖增强 |
| 6 | `agent/factor_research_flow.py` | pipeline_builder, research_tools | 新建 |

---

## 三、关键设计决策

### 决策1: 失败恢复 vs 直接终止

**quant_react_interview 方案**（采用）:
```python
# 工具错误 → 注入修复提示 → 继续循环
if had_error:
    self.consecutive_errors += 1
    self._inject_recovery("tool_error")  # 提示模型修复而非放弃
```

**futureQuant 现有方案**:
```python
# LLM 没有工具调用 → 简单提示 → 继续
# 没有错误计数器，没有恢复注入
```

### 决策2: 节点目录的信息密度

**quant_react_interview 方案**（采用）:
- `field_descriptions` 详细说明语义
- `common_mistakes` 预判 LLM 常见错误
- `output_shape` 明确指出访问路径

**替代方案**（不用）:
- 简单列出字段名和类型（信息不足）
- 仅靠 tool description 指导（LLM 仍会犯错）

### 决策3: PipelineBuilder 引用语法

**采用**: `$step_id['field']` 语法（与 quant_react_interview 一致）

示例:
```yaml
- step_id: market_data
  kind: data.market_bars
  config:
    symbols: $trigger['universe']
    lookback_days: 60

- step_id: momentum
  kind: factor.momentum
  config:
    bars: $market_data['bars']
    window: 20
```

---

## 四、向后兼容性

所有改进均保持向后兼容：
- 现有 `BaseAgent` / `ReActAgent` 继续工作
- 现有 `BlackboardController` 继续工作
- 新功能作为可选扩展提供

---

## 五、验证计划

1. 运行现有单元测试（确保不破坏已有功能）
2. 新增 `test_react_base.py` 测试增强的 ReAct 循环
3. 新增 `test_factor_catalog.py` 测试目录系统
4. 新增 `test_pipeline_builder.py` 测试流水线构建器
5. 端到端测试：使用新系统执行因子研究流程

---

## 六、经验总结

### 通用工具在 LLM 规划中的局限性

1. **配置格式模糊**：需要 catalog 提供具体示例
2. **引用语法不直观**：`$step_id['field']` 需要显式说明
3. **错误不可见**：不执行就不知道配置错误
4. **信息过载**：catalog 太大时 LLM 会忽略

### 什么使节点目录真正有用

1. **引用语法示例**：给出 `$trigger['universe']` 这样具体的写法
2. **字段语义说明**：`field_descriptions` 说明"为什么需要"而非"是什么类型"
3. **output_shape 访问路径**：模型需要知道确切的访问路径
4. **common_mistakes 预判**：模型会犯特定错误，提前告知避免
5. **即时执行验证**：`add_step` 立即执行验证，错误立即返回

---

*最后更新: 2026-04-20*
