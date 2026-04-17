"""
Tests for Agent Tool Layer
"""

import pytest

from futureQuant.agent.tools import (
    CodeExecutionTool,
    DatabaseTool,
    ToolRegistry,
    WebSearchTool,
    tool,
)
from futureQuant.agent.tools.base import ToolResult


class TestToolRegistry:
    def test_register_and_schema(self):
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        assert registry.has("web_search")
        schema = registry.to_openai_schema()
        assert len(schema) == 1
        assert schema[0]["function"]["name"] == "web_search"

    def test_execute_tool_call(self):
        @tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        registry = ToolRegistry()
        registry.register(add())

        result = registry.execute_tool_call(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
            }
        )
        assert result.success
        assert result.data == 5

    def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("unknown")
        assert not result.success
        assert "not found" in result.error


class TestCodeExecutionTool:
    def test_safe_execution(self):
        tool = CodeExecutionTool()
        result = tool.execute(code="result = 2 + 3\n")
        assert result.success
        assert result.data == 5

    def test_pandas_execution(self):
        tool = CodeExecutionTool()
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n"
            "result = df['a'].sum()\n"
        )
        result = tool.execute(code=code)
        assert result.success
        assert result.data == 3

    def test_forbidden_import(self):
        tool = CodeExecutionTool()
        result = tool.execute(code="import os\nos.system('ls')")
        assert not result.success
        assert "Forbidden module import" in result.error

    def test_forbidden_from_import(self):
        tool = CodeExecutionTool()
        result = tool.execute(code="from os import system")
        assert not result.success
        assert "Forbidden 'from ... import ...'" in result.error

    def test_forbidden_eval(self):
        tool = CodeExecutionTool()
        result = tool.execute(code="eval('1+1')")
        assert not result.success
        assert "Forbidden function call" in result.error

    def test_timeout(self):
        tool = CodeExecutionTool()
        code = "import time\ntime.sleep(10)\nresult = 1"
        result = tool.execute(code=code, timeout=1)
        assert not result.success
        assert "timed out" in result.error


class TestWebSearchTool:
    def test_schema(self):
        tool = WebSearchTool()
        schema = tool.get_schema()
        assert schema["function"]["name"] == "web_search"

    @pytest.mark.skipif(True, reason="Requires network; run manually")
    def test_real_search(self):
        tool = WebSearchTool()
        result = tool.execute(query="Python", max_results=3)
        assert result.success
        assert len(result.data) > 0


class TestDatabaseTool:
    @pytest.fixture(autouse=True)
    def setup_db(self, monkeypatch, tmp_path):
        # Force SQLite fallback for tests
        monkeypatch.setenv("FQ_DATA__MYSQL__ENABLED", "false")
        from futureQuant.core.config import reload_config
        reload_config()
        # Ensure a fresh sqlite file
        db_path = tmp_path / "test_tools.db"
        monkeypatch.setenv("FQ_DATA__DB_PATH", str(db_path))
        reload_config()

    def test_list_tables(self):
        tool = DatabaseTool()
        result = tool.execute(action="list_tables")
        assert result.success
        assert isinstance(result.data, list)

    def test_query_and_save(self):
        tool = DatabaseTool()
        # Drop and create table fresh
        tool.execute(action="execute", sql="DROP TABLE IF EXISTS test_items")
        create_result = tool.execute(
            action="execute",
            sql="CREATE TABLE test_items (id INTEGER PRIMARY KEY, name TEXT)",
        )
        assert create_result.success

        # Insert via save_dataframe
        result = tool.execute(
            action="save_dataframe",
            table="test_items",
            data=[{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}],
            if_exists="append",
        )
        assert result.success
        assert result.data["saved_rows"] == 2

        # Query
        query_result = tool.execute(action="query", sql="SELECT * FROM test_items")
        assert query_result.success
        assert len(query_result.data) == 2
