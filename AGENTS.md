# futureQuant Agent 开发规范

> 本文档定义 futureQuant 多 Agent 系统的开发规范、ReAct 架构使用指南和 Tool 注册方式。

---

## 1. 架构总览

futureQuant 采用 **ReAct (Reasoning + Acting)** 架构驱动所有 Agent：

```
User Query -> BlackboardAgent -> ExecutionPlan -> Blackboard
                              -> BlackboardController -> ReAct Agents (Tools) -> Blackboard
```

- **BlackboardAgent**：唯一负责解析自然语言需求并生成 `ExecutionPlan` 的 Agent。
- **BlackboardController**：读取 Blackboard 上的 `execution_plan`，按 `depends_on` 拓扑调度执行。
- **ReActAgent**：所有业务 Agent 的基类，通过 `Thought -> Action (Tool) -> Observation` 循环完成任务。
- **ToolRegistry**：统一工具层，所有 Agent 共享同一套工具（记忆、进度、搜索、代码执行、数据库、黑板读写）。

---

## 2. 快速开始：创建一个 ReAct Agent

```python
from futureQuant.agent.react_base import ReActAgent
from futureQuant.agent.tools import WebSearchTool, CodeExecutionTool

class MyAgent(ReActAgent):
    def __init__(self, llm_client=None):
        super().__init__(name="my_agent", llm_client=llm_client)
        self.register_tools(
            WebSearchTool(),
            CodeExecutionTool(),
        )

    @property
    def system_prompt(self) -> str:
        return (
            "你是一个测试 Agent。"
            "你可以使用 web_search 和 code_execution 工具。"
            "完成任务后请输出 FINAL_ANSWER: <结论>"
        )

# 运行
result = MyAgent().run({"task": "搜索 Python 最新版本"})
print(result.data["final_answer"])
```

---

## 3. 工具层（Tool Layer）

### 3.1 内置工具列表

| 工具名 | 类 | 用途 |
|--------|-----|------|
| `web_search` | `WebSearchTool` | DuckDuckGo 搜索，支持 `time_range` |
| `code_execution` | `CodeExecutionTool` | 受限 Python 沙箱执行 |
| `database` | `DatabaseTool` | SQL 查询 / DataFrame 写入（MySQL/SQLite 自适应） |
| `memory` | `MemoryTool` | 读写 AgentMemoryBank |
| `progress_tracker` | `ProgressTool` | 任务进度追踪与报告生成 |
| `blackboard_read` | `BlackboardReadTool` | 读取中央黑板数据 |
| `blackboard_write` | `BlackboardWriteTool` | 写入中央黑板数据 |

### 3.2 注册工具

```python
self.register_tool(WebSearchTool())
self.register_tools(WebSearchTool(), CodeExecutionTool())
```

### 3.3 使用 `@tool` 装饰器自定义工具

```python
from futureQuant.agent.tools import tool

@tool(name="add", description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b

agent.register_tool(add())
```

---

## 4. LLM Client 配置

通过环境变量或 `settings.yaml` 配置 LLM：

```yaml
llm:
  provider: "openai"      # openai | ollama
  model: "gpt-4o-mini"
  openai_api_key: ""      # 或设置环境变量 FQ_LLM__OPENAI_API_KEY
  ollama_base_url: "http://localhost:11434/v1"
```

环境变量优先级高于配置文件。

---

## 5. 代码执行安全策略

`CodeExecutionTool` 采用多层安全限制：

1. **AST 白名单**：仅允许 `import pandas`, `import numpy` 等安全模块；禁止 `from os import ...`。
2. **文本黑名单**：拦截 `eval`, `exec`, `compile`, `__import__` 字符串模式。
3. **受限 builtins**：移除 `open`, `eval`, `exit` 等危险内置函数。
4. **安全 `__import__`**：保留 `__import__`，但只允许白名单中的顶层模块。
5. **执行超时**：默认 30 秒，通过独立线程实现。

---

## 6. ExecutionPlan 规范

BlackboardAgent 生成的计划必须符合以下 JSON Schema：

```json
{
  "goal": "用户原始需求摘要",
  "steps": [
    {
      "step_id": 1,
      "agent": "data_collector",
      "task": "获取螺纹钢最近一周日线数据",
      "inputs": {},
      "outputs": ["RB_daily"],
      "depends_on": []
    },
    {
      "step_id": 2,
      "agent": "factor_mining",
      "task": "挖掘对 RB 有效的 Alpha101 因子",
      "inputs": {"price_data": "RB_daily"},
      "outputs": ["alpha_factors"],
      "depends_on": [1]
    }
  ]
}
```

- `step_id` 从 1 开始递增。
- `agent` 必须与注册到 `BlackboardController` 的 Agent 名称一致。
- `depends_on` 填写依赖的 `step_id` 列表，控制器会自动拓扑排序。
- `outputs` 中的键名会被预写到 Blackboard，供后续 Agent 读取。

---

## 7. 数据库适配说明

`DatabaseTool` 和 `FactorMySQLStore` 均支持 **MySQL 优先 + SQLite 兜底**：

- 若 `settings.yaml` 中 `data.mysql.enabled=true` 且账号密码正确，自动连接 MySQL。
- 若 MySQL 连接失败，自动降级到 SQLite（`data.db_path`）。
- 所有 SQL 语句均使用 SQLAlchemy `text()`，兼容两种方言。

---

## 8. 测试规范

所有新增 Agent 和 Tool 必须附带单元测试：

- `tests/unit/test_react_agent.py`：ReAct 循环行为测试
- `tests/unit/test_tools.py`：工具功能与安全测试
- `tests/unit/test_blackboard_agent.py`：计划生成测试
- `tests/integration/test_nl_task_flow.py`：端到端自然语言任务流测试（Mock LLM）

测试原则：
- 网络调用必须 Mock（`duckduckgo-search` / `httpx`）。
- LLM 调用使用 `MockLLMClient` 注入固定响应。
- 代码执行工具的异常路径和拦截路径必须覆盖。

---

## 9. 文件结构

```
futureQuant/agent/
├── react_base.py                  # ReAct Agent 基类
├── tools/                         # 统一工具层
│   ├── base.py
│   ├── web_search_tool.py
│   ├── code_execution_tool.py
│   ├── database_tool.py
│   ├── memory_tool.py
│   ├── progress_tool.py
│   └── blackboard_tool.py
├── blackboard/
│   ├── blackboard_agent.py        # 中央规划 Agent
│   └── blackboard_controller.py   # 计划执行调度器
├── data_collector/
│   ├── react_data_collector_agent.py
│   └── llm_path_discovery.py
├── factor_mining/
│   ├── react_factor_mining_agent.py
│   ├── alpha101_pool.py
│   ├── alpha101_generator.py
│   ├── alpha101_miner.py
│   └── factor_mysql_store.py
└── fundamental/
    └── web_fundamental_agent.py
```

---

## 10. 提交规范

- 每个 Agent 文件必须包含模块级 docstring，说明职责和输入输出。
- 所有公共方法必须带类型注解。
- 日志使用 `from ...core.logger import get_logger`。
- 禁止在 Agent 中直接打印（`print`），应使用日志。
- 错误必须捕获并返回 `AgentResult(status=AgentStatus.FAILED, errors=[...])`。
