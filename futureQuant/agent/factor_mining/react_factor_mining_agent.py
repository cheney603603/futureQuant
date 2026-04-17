"""
React Factor Mining Agent - 基于 ReAct 的因子挖掘 Agent

功能：
1. 从黑板或数据库读取价格数据
2. 并行计算候选因子池 + Alpha101 因子
3. 计算 IC/ICIR，筛选 Top 因子
4. 将结果存入 factor_library（MySQL/SQLite）
5. 回写黑板
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import AgentResult, AgentStatus
from ..react_base import ReActAgent
from ..tools import (
    BlackboardReadTool,
    BlackboardWriteTool,
    CodeExecutionTool,
    DatabaseTool,
    WebSearchTool,
)
from .alpha101_miner import Alpha101Miner
from .factor_candidate_pool import FactorCandidatePool
from .factor_mysql_store import FactorMySQLStore
from .parallel_miner import ParallelFactorMiner
from ...factor.evaluator import FactorEvaluator
from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.factor_mining.react")


class ReactFactorMiningAgent(ReActAgent):
    """
    因子挖掘 Agent（ReAct 版）

    context 支持:
    - target (str): 品种代码，如 "RB"
    - price_data_key (str): blackboard 上价格数据的 key，默认 "price_data"
    - alpha101_enabled (bool): 是否启用 Alpha101，默认 True
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        blackboard=None,
        llm_client=None,
    ):
        super().__init__(
            name="factor_mining",
            config=config,
            llm_client=llm_client,
        )
        self._pool = FactorCandidatePool()
        self._miner = ParallelFactorMiner()
        self._evaluator = FactorEvaluator()
        self._store = FactorMySQLStore()
        self.register_tools(
            BlackboardReadTool(blackboard),
            BlackboardWriteTool(blackboard),
            CodeExecutionTool(),
            DatabaseTool(),
            WebSearchTool(),
        )

    @property
    def system_prompt(self) -> str:
        return (
            "你是 futureQuant 的因子挖掘 Agent。你的任务是对指定期货品种进行系统化因子挖掘。\n"
            "执行规则：\n"
            "1. 首先使用 blackboard_read 或 database 查询获取目标品种的价格数据。\n"
            "2. 使用 code_execution 计算候选因子池（技术指标、基本面因子、交叉因子）的 IC 值。\n"
            "3. 如果启用 alpha101_enabled，调用 alpha101_miner 生成并验证 Alpha101 因子。\n"
            "4. 使用 web_search 搜索 'WorldQuant Alpha101 {品种} 研报' 获取外部知识（可选）。\n"
            "5. 筛选 |IC| >= 0.02 且 ICIR >= 0.3 的因子，使用 database/factor_store 保存到 factor_library。\n"
            "6. 使用 blackboard_write 将 Top 因子列表写回黑板，key 为 'top_factors'。\n"
            "7. 最后输出 FINAL_ANSWER: <Top 因子数量、最佳因子名称及 IC 值摘要>"
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        start_time = time.time()
        target = context.get("target", "UNKNOWN")
        price_data_key = context.get("price_data_key", "price_data")
        alpha101_enabled = context.get("alpha101_enabled", True)

        # 1. 获取数据
        price_data = self._get_price_data(price_data_key, target, context)
        if price_data is None or price_data.empty:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[f"No price data found for {target}"],
            )

        returns = price_data["close"].pct_change().shift(-1)

        # 2. 计算候选池因子
        candidates = self._pool.get_all()
        factor_values = self._miner.mine(
            candidates=candidates,
            price_data=price_data,
        )
        factor_df = pd.DataFrame(factor_values)

        # 3. Alpha101 因子（可选）
        alpha_df = pd.DataFrame()
        if alpha101_enabled:
            alpha_miner = Alpha101Miner(config=self.config, llm_client=self._llm)
            alpha_result = alpha_miner.execute({"target": target})
            # 简化：从 alpha_result 提取生成代码并执行
            alpha_generated = alpha_result.data.get("generated", [])
            alpha_series = {}
            for item in alpha_generated:
                if item and item.get("validated") and item.get("code"):
                    try:
                        exec_res = self._tools.execute_tool_call(
                            {
                                "function": {
                                    "name": "code_execution",
                                    "arguments": '{"code": "' + item["code"].replace('"', '\\"') + '\\nresult = compute_alpha(price_data)", "context": {"price_data": price_data.to_dict("list")}}',
                                }
                            }
                        )
                        if exec_res.success:
                            alpha_series[item["name"]] = pd.Series(exec_res.data)
                    except Exception as exc:
                        logger.warning(f"Alpha execution failed: {exc}")
            if alpha_series:
                alpha_df = pd.DataFrame(alpha_series)

        # 4. 合并并评估 IC
        combined_df = pd.concat([factor_df, alpha_df], axis=1) if not alpha_df.empty else factor_df
        common_index = combined_df.index.intersection(returns.index)
        combined_aligned = combined_df.loc[common_index]
        returns_aligned = returns.loc[common_index]

        if len(common_index) < 30:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Insufficient samples for IC evaluation"],
            )

        ic_series = self._evaluator.calculate_ic(combined_aligned, returns_aligned, method="spearman")
        passed = ic_series[ic_series.abs() >= 0.02].sort_values(key=abs, ascending=False)

        top_factors: List[Dict[str, Any]] = []
        for factor_name in passed.head(20).index:
            s = combined_aligned[factor_name].dropna()
            r = returns_aligned.loc[s.index]
            if len(s) < 30:
                continue
            icir_dict = self._evaluator.calculate_icir(pd.Series(s.values, index=r.index))
            icir = icir_dict.get("annual_icir", 0)
            if icir < 0.3:
                continue

            top_factors.append({
                "name": factor_name,
                "ic": float(passed[factor_name]),
                "icir": float(icir),
                "source": "alpha101" if factor_name.startswith("Alpha") else "candidate_pool",
            })

            # 存入 factor_library
            self._store.save_factor({
                "factor_id": f"{target}_{factor_name}_{pd.Timestamp.now().strftime('%Y%m%d')}",
                "name": factor_name,
                "category": "alpha101" if factor_name.startswith("Alpha") else "technical",
                "variety": target,
                "frequency": "daily",
                "source": "alpha101" if factor_name.startswith("Alpha") else "candidate_pool",
                "logic_description": f"IC={passed[factor_name]:.4f}, ICIR={icir:.4f}",
                "ic": float(passed[factor_name]),
                "icir": float(icir),
            })

        # 5. 回写黑板
        self._tools.execute(
            "blackboard_write",
            key="top_factors",
            value={
                "target": target,
                "top_factors": top_factors,
                "n_candidates": len(combined_df.columns),
                "n_passed": len(top_factors),
            },
            agent_name=self.name,
        )

        elapsed = time.time() - start_time
        final_msg = (
            f"因子挖掘完成：候选 {len(combined_df.columns)} 个，"
            f"通过筛选 {len(top_factors)} 个。"
        )
        if top_factors:
            final_msg += f" 最佳因子：{top_factors[0]['name']} (IC={top_factors[0]['ic']:.4f})"

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=pd.DataFrame(top_factors),
            factors=top_factors,
            metrics={
                "n_candidates": len(combined_df.columns),
                "n_passed": len(top_factors),
                "top_factors": top_factors,
                "elapsed_seconds": elapsed,
            },
            logs=[final_msg],
            elapsed_seconds=elapsed,
        )

    def _get_price_data(self, key: str, target: str, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        # 优先从 blackboard 读取
        bb_res = self._tools.execute("blackboard_read", key=key)
        if bb_res.success and bb_res.data:
            try:
                return pd.DataFrame(bb_res.data)
            except Exception:
                pass

        # 其次从 database 查询
        db_res = self._tools.execute(
            "database",
            action="query",
            sql=f"SELECT * FROM daily_price WHERE symbol = '{target}' ORDER BY date",
        )
        if db_res.success and db_res.data:
            return pd.DataFrame(db_res.data)

        # 最后从 context 直接取
        return context.get("price_data")
