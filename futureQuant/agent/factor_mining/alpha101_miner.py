"""
Alpha101 Miner - 专门负责 Alpha101 因子挖掘的 ReAct Agent

可以联网搜索 Alpha101 研报，并生成/验证 Alpha101 因子。
"""

from typing import Any, Dict, Optional

from ..react_base import ReActAgent
from ..tools import WebSearchTool, CodeExecutionTool, BlackboardReadTool, BlackboardWriteTool
from .alpha101_generator import Alpha101Generator
from .alpha101_pool import Alpha101Pool
from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.factor_mining.alpha101_miner")


class Alpha101Miner(ReActAgent):
    """
    Alpha101 因子挖掘 Agent

    context 支持:
    - target (str): 品种代码
    - alpha_names (List[str]): 指定要挖掘的 Alpha 名称列表（可选）
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        blackboard=None,
        llm_client=None,
    ):
        super().__init__(
            name="alpha101_miner",
            config=config,
            llm_client=llm_client,
        )
        self._generator = Alpha101Generator(llm_client=llm_client)
        self._pool = Alpha101Pool()
        self.register_tools(
            WebSearchTool(),
            CodeExecutionTool(),
            BlackboardReadTool(blackboard),
            BlackboardWriteTool(blackboard),
        )

    @property
    def system_prompt(self) -> str:
        return (
            "你是 futureQuant 的 Alpha101 因子挖掘专家。\n"
            "你的任务是：基于用户指定的期货品种，从 Alpha101 公式库中选择最有潜力的因子，"
            "生成对应的 pandas 代码，验证其有效性，并将结果写回黑板。\n\n"
            "执行规则：\n"
            "1. 如果用户未指定 alpha_names，优先选择经典因子如 Alpha001-Alpha010、Alpha012、Alpha028、Alpha030 等。\n"
            "2. 你可以使用 web_search 搜索 'WorldQuant Alpha101 {品种} 研报' 获取额外领域知识。\n"
            "3. 使用 code_execution 运行生成的 Alpha 代码。\n"
            "4. 使用 blackboard_write 将生成的因子代码和结果写入黑板，key 建议为 'alpha101_factors'。\n"
            "5. 完成后输出 FINAL_ANSWER: <总结>"
        )

    def execute(self, context: Dict[str, Any]) -> Any:
        target = context.get("target", "UNKNOWN")
        alpha_names = context.get("alpha_names", [])
        if not alpha_names:
            # 默认选取经典因子
            all_names = self._pool.list_factors()
            priority = [n for n in all_names if int(n.replace("Alpha", "")) <= 30]
            alpha_names = priority[:10]

        generated = []
        for name in alpha_names:
            logger.info(f"[Alpha101Miner] Generating {name} for {target}")
            info = self._generator.generate(name)
            if info:
                generated.append(info)

        # 将结果写回黑板
        self._tools.execute(
            "blackboard_write",
            key="alpha101_factors",
            value={
                "target": target,
                "generated": generated,
                "count": len([g for g in generated if g and g.get("validated")]),
            },
            agent_name=self.name,
        )

        success_count = len([g for g in generated if g and g.get("validated")])
        final = f"Alpha101 挖掘完成：{success_count}/{len(alpha_names)} 个因子验证通过。"

        from ..base import AgentResult, AgentStatus
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={"generated": generated},
            metrics={"success_count": success_count, "total": len(alpha_names)},
            logs=[final],
        )
