"""
Agent 基类测试

测试 AgentStatus 枚举、AgentResult 数据类和 BaseAgent 抽象类。
"""

from dataclasses import asdict
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from futureQuant.agent.base import AgentResult, AgentStatus, BaseAgent


# ============================================================================
# AgentStatus 枚举测试
# ============================================================================

class TestAgentStatus:
    """AgentStatus 枚举测试"""
    
    def test_status_values(self):
        """测试枚举值"""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.SUCCESS.value == "success"
        assert AgentStatus.FAILED.value == "failed"
    
    def test_status_count(self):
        """测试枚举数量"""
        assert len(AgentStatus) == 4
    
    def test_status_comparison(self):
        """测试枚举比较"""
        assert AgentStatus.SUCCESS == AgentStatus.SUCCESS
        assert AgentStatus.SUCCESS != AgentStatus.FAILED
        assert AgentStatus.SUCCESS != "success"  # 枚举与字符串不等


# ============================================================================
# AgentResult 数据类测试
# ============================================================================

class TestAgentResult:
    """AgentResult 数据类测试"""
    
    def test_result_creation_minimal(self):
        """测试最小化创建"""
        result = AgentResult(
            agent_name='test_agent',
            status=AgentStatus.SUCCESS
        )
        
        assert result.agent_name == 'test_agent'
        assert result.status == AgentStatus.SUCCESS
        assert result.data is None
        assert result.factors == []
        assert result.metrics == {}
        assert result.errors == []
        assert result.logs == []
        assert result.elapsed_seconds == 0.0
    
    def test_result_creation_full(self):
        """测试完整参数创建"""
        data = pd.DataFrame({'factor_1': [1, 2, 3]})
        factors = [Mock(), Mock()]
        metrics = {'ic': 0.05, 'icir': 2.5}
        errors = ['error1']
        logs = ['log1']
        
        result = AgentResult(
            agent_name='test_agent',
            status=AgentStatus.SUCCESS,
            data=data,
            factors=factors,
            metrics=metrics,
            errors=errors,
            logs=logs,
            elapsed_seconds=1.5
        )
        
        assert result.agent_name == 'test_agent'
        assert result.status == AgentStatus.SUCCESS
        assert result.data is data
        assert result.factors == factors
        assert result.metrics == metrics
        assert result.errors == errors
        assert result.logs == logs
        assert result.elapsed_seconds == 1.5
    
    def test_result_is_success(self):
        """测试 is_success 属性"""
        success_result = AgentResult(
            agent_name='test',
            status=AgentStatus.SUCCESS
        )
        failed_result = AgentResult(
            agent_name='test',
            status=AgentStatus.FAILED
        )
        
        assert success_result.is_success is True
        assert failed_result.is_success is False
    
    def test_result_n_factors(self):
        """测试 n_factors 属性"""
        result_empty = AgentResult(
            agent_name='test',
            status=AgentStatus.SUCCESS,
            factors=[]
        )
        result_with_factors = AgentResult(
            agent_name='test',
            status=AgentStatus.SUCCESS,
            factors=[Mock(), Mock(), Mock()]
        )
        
        assert result_empty.n_factors == 0
        assert result_with_factors.n_factors == 3
    
    def test_result_n_factors_none(self):
        """测试 factors 为 None 时的 n_factors"""
        result = AgentResult(
            agent_name='test',
            status=AgentStatus.SUCCESS,
            factors=None
        )
        
        assert result.n_factors == 0
    
    def test_result_repr(self):
        """测试字符串表示"""
        result = AgentResult(
            agent_name='test_agent',
            status=AgentStatus.SUCCESS,
            factors=[Mock()],
            elapsed_seconds=1.23
        )
        
        repr_str = repr(result)
        assert 'test_agent' in repr_str
        assert 'success' in repr_str
        assert 'n_factors=1' in repr_str
        assert '1.23s' in repr_str
    
    def test_result_post_init_defaults(self):
        """测试 __post_init__ 设置默认值"""
        result = AgentResult(
            agent_name='test',
            status=AgentStatus.SUCCESS
        )
        
        # 验证默认值
        assert result.factors == []
        assert result.metrics == {}
        assert result.errors == []
        assert result.logs == []


# ============================================================================
# BaseAgent 抽象类测试
# ============================================================================

class ConcreteAgent(BaseAgent):
    """用于测试的具体 Agent 实现"""
    
    def execute(self, context):
        """模拟执行逻辑"""
        if 'fail' in context:
            raise ValueError("Test error")
        
        if 'custom_status' in context:
            return AgentResult(
                agent_name=self.name,
                status=context['custom_status']
            )
        
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            metrics={'test_metric': 42}
        )


class TestBaseAgent:
    """BaseAgent 抽象类测试"""
    
    def test_agent_creation(self):
        """测试 Agent 创建"""
        agent = ConcreteAgent(name='test_agent', config={'param': 1})
        
        assert agent.name == 'test_agent'
        assert agent.config == {'param': 1}
        assert agent.status == AgentStatus.IDLE
        assert agent._history == []
    
    def test_agent_creation_no_config(self):
        """测试无配置创建"""
        agent = ConcreteAgent(name='test_agent')
        
        assert agent.name == 'test_agent'
        assert agent.config == {}
        assert agent.status == AgentStatus.IDLE
    
    def test_agent_status_property(self):
        """测试 status 属性"""
        agent = ConcreteAgent(name='test')
        
        assert agent.status == AgentStatus.IDLE
        agent._status = AgentStatus.RUNNING
        assert agent.status == AgentStatus.RUNNING
    
    def test_agent_run_success(self):
        """测试成功运行"""
        agent = ConcreteAgent(name='test_agent')
        context = {'data': 'test'}
        
        result = agent.run(context)
        
        assert result.is_success
        assert result.agent_name == 'test_agent'
        assert result.metrics == {'test_metric': 42}
        assert result.elapsed_seconds >= 0
        assert agent.status == AgentStatus.SUCCESS
    
    def test_agent_run_failure(self):
        """测试失败运行"""
        agent = ConcreteAgent(name='test_agent')
        context = {'fail': True}
        
        result = agent.run(context)
        
        assert not result.is_success
        assert result.agent_name == 'test_agent'
        assert len(result.errors) > 0
        assert 'ValueError' in result.errors[0]
        assert result.elapsed_seconds >= 0
        assert agent.status == AgentStatus.FAILED
    
    def test_agent_run_state_transition(self):
        """测试状态转换"""
        agent = ConcreteAgent(name='test')
        
        # 初始状态
        assert agent.status == AgentStatus.IDLE
        
        # 运行后
        agent.run({})
        assert agent.status == AgentStatus.SUCCESS
        
        # 再次运行
        agent.run({'fail': True})
        assert agent.status == AgentStatus.FAILED
    
    def test_agent_history(self):
        """测试历史记录"""
        agent = ConcreteAgent(name='test')
        
        # 运行多次
        agent.run({})
        agent.run({'fail': True})
        agent.run({})
        
        history = agent.get_history()
        
        assert len(history) == 3
        assert history[0].status == AgentStatus.SUCCESS
        assert history[1].status == AgentStatus.FAILED
        assert history[2].status == AgentStatus.SUCCESS
    
    def test_agent_get_last_result(self):
        """测试获取最近结果"""
        agent = ConcreteAgent(name='test')
        
        # 无历史时
        assert agent.get_last_result() is None
        
        # 有历史时
        agent.run({})
        last = agent.get_last_result()
        
        assert last is not None
        assert last.status == AgentStatus.SUCCESS
    
    def test_agent_reset(self):
        """测试重置"""
        agent = ConcreteAgent(name='test')
        
        # 运行几次
        agent.run({})
        agent.run({})
        
        # 重置
        agent.reset()
        
        assert agent.status == AgentStatus.IDLE
        assert len(agent._history) == 0
    
    def test_agent_custom_status(self):
        """测试自定义状态返回"""
        agent = ConcreteAgent(name='test')
        context = {'custom_status': AgentStatus.FAILED}
        
        result = agent.run(context)
        
        # 即使 execute 返回 FAILED，run 也应该保持该状态
        assert result.status == AgentStatus.FAILED
        assert agent.status == AgentStatus.FAILED
    
    def test_agent_repr(self):
        """测试字符串表示"""
        agent = ConcreteAgent(name='test_agent')
        
        repr_str = repr(agent)
        assert 'ConcreteAgent' in repr_str
        assert 'test_agent' in repr_str
        assert 'idle' in repr_str
        assert 'history_len=0' in repr_str


# ============================================================================
# BaseAgent 边界情况测试
# ============================================================================

class TestBaseAgentEdgeCases:
    """BaseAgent 边界情况测试"""
    
    def test_agent_with_none_context(self):
        """测试 None 上下文"""
        agent = ConcreteAgent(name='test')
        result = agent.run({})
        
        assert result.is_success
    
    def test_agent_with_empty_context(self):
        """测试空上下文"""
        agent = ConcreteAgent(name='test')
        result = agent.run({})
        
        assert result.is_success
    
    def test_agent_elapsed_seconds_precision(self):
        """测试耗时精度"""
        agent = ConcreteAgent(name='test')
        result = agent.run({})
        
        # 耗时应该是非负数且精确到小数点后两位
        assert result.elapsed_seconds >= 0
        assert isinstance(result.elapsed_seconds, float)
    
    def test_agent_multiple_runs_independence(self):
        """测试多次运行的独立性"""
        agent = ConcreteAgent(name='test')
        
        result1 = agent.run({'data': 'run1'})
        result2 = agent.run({'data': 'run2'})
        
        # 每次运行应该独立，不受之前运行影响
        assert result1.agent_name == 'test'
        assert result2.agent_name == 'test'
        assert len(agent.get_history()) == 2
    
    def test_agent_exception_traceback_logged(self):
        """测试异常堆栈被记录"""
        agent = ConcreteAgent(name='test')
        result = agent.run({'fail': True})
        
        assert len(result.errors) > 0
        assert len(result.logs) > 0
        # 日志中应该包含堆栈信息
        assert 'Traceback' in result.logs[0] or 'ValueError' in result.logs[0]
