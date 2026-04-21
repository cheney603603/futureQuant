"""
因子进化挖掘 Agent

整合 GP 遗传规划引擎和自我反思模块的完整 Agent：

Pipeline:
  1. 加载数据（OHLCV + 收益率）
  2. 候选池因子挖掘（传统方法）
  3. GP 进化挖掘（可选，启用后扩展搜索空间）
  4. IC 筛选
  5. 自我反思评估
  6. 策略调整 + 迭代优化（可选）
  7. 综合评分 + 报告生成
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...core.logger import get_logger
from ...factor.evaluator import FactorEvaluator
from ..base import AgentResult, AgentStatus, BaseAgent
from .factor_candidate_pool import FactorCandidatePool
from .factor_report import FactorReport
from .gp_factor_engine import (
    EvolutionConfig,
    GPFactorEngine,
    Individual,
)
from .self_reflection import FactorMiningSelfReflection


class FactorEvolutionAgent(BaseAgent):
    """
    因子进化挖掘 Agent

    相比 FactorMiningAgent，新增：
    - GP 遗传规划因子进化
    - 自我反思迭代优化
    - 自动策略调整

    Attributes:
        use_gp: 是否启用 GP 进化
        use_reflection: 是否启用自我反思
        max_iterations: 最大反思迭代次数
        gp_config: GP 进化配置

    使用示例：
        >>> agent = FactorEvolutionAgent(config={
        ...     'use_gp': True,
        ...     'use_reflection': True,
        ...     'max_iterations': 2,
        ...     'gp_population_size': 50,
        ...     'gp_generations': 10,
        ... })
        >>> result = agent.run({
        ...     'target': 'RB',
        ...     'price_data': price_df,
        ...     'returns': returns_series,
        ... })
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        # GP 配置
        'use_gp': False,
        'gp_population_size': 50,
        'gp_generations': 10,
        'gp_max_depth': 4,
        'gp_mutation_prob': 0.25,
        'gp_crossover_prob': 0.70,
        'gp_elite_size': 3,
        # 自我反思配置
        'use_reflection': False,
        'max_iterations': 3,
        'min_improvement': 0.05,
        # 候选池配置
        'use_technical': True,
        'use_fundamental': True,
        'use_cross': True,
        # 评估配置
        'min_ic': 0.02,
        'min_icir': 0.3,
        'top_n': 20,
        'max_workers': 8,
        'report_dir': 'docs/reports',
        'var_set': ['close', 'open', 'high', 'low', 'volume', 'returns'],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name='factor_evolution', config=config)
        self._cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.candidate_pool = FactorCandidatePool()
        self.evaluator = FactorEvaluator()
        self.reflection = FactorMiningSelfReflection(
            max_iterations=self._cfg['max_iterations'],
            min_improvement=self._cfg['min_improvement'],
        )
        self.gp_engine: Optional[GPFactorEngine] = None
        self.logger = get_logger('agent.factor_mining.evolution')

    def _init_gp_engine(self) -> GPFactorEngine:
        """初始化 GP 引擎"""
        gp_cfg = EvolutionConfig(
            population_size=self._cfg['gp_population_size'],
            generations=self._cfg['gp_generations'],
            max_depth=self._cfg['gp_max_depth'],
            mutation_prob=self._cfg['gp_mutation_prob'],
            crossover_prob=self._cfg['gp_crossover_prob'],
            elite_size=self._cfg['gp_elite_size'],
            var_set=self._cfg['var_set'],
        )
        return GPFactorEngine(gp_cfg)

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行因子进化挖掘流程

        context 支持的 keys：
            - target (str): 标的代码
            - price_data (pd.DataFrame): 价格数据
            - returns (pd.Series): 收益率序列（可选）
            - basis_data (pd.DataFrame): 基本面数据（可选）
            - start_date (str): 开始日期
            - end_date (str): 结束日期
        """
        start_time = time.time()
        errors: List[str] = []
        logs: List[str] = []

        # =====================================================================
        # Step 1: 加载数据
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 1: 加载数据")
        self.logger.info("=" * 60)

        target = context.get('target', 'UNKNOWN')
        price_data = context.get('price_data')
        basis_data = context.get('basis_data')
        returns = context.get('returns')
        start_date = context.get('start_date', '2020-01-01')
        end_date = context.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        if price_data is None or price_data.empty:
            msg = "价格数据为空"
            self.logger.error(msg)
            return AgentResult(
                status=AgentStatus.FAILED,
                message=msg,
                data=None,
                errors=[msg],
            )

        # 计算收益率
        if returns is None:
            if 'close' in price_data.columns:
                returns = price_data['close'].pct_change().fillna(0)
            else:
                returns = price_data.iloc[:, 0].pct_change().fillna(0)

        self.logger.info(f"  标的: {target}, 日期: {start_date} ~ {end_date}")
        self.logger.info(f"  数据行数: {len(price_data)}")

        # =====================================================================
        # Step 2: 候选池因子挖掘（可选）
        # =====================================================================
        pool_factors: List[Dict[str, Any]] = []
        if self._cfg.get('use_technical') or self._cfg.get('use_fundamental') or self._cfg.get('use_cross'):
            self.logger.info("=" * 60)
            self.logger.info("Step 2: 候选池因子挖掘")
            self.logger.info("=" * 60)
            pool_factors, pool_log = self._mine_pool_factors(price_data, returns, basis_data)
            logs.extend(pool_log)
            self.logger.info(f"  候选池有效因子: {len([f for f in pool_factors if f.get('is_valid')])}")

        # =====================================================================
        # Step 3: GP 进化挖掘（可选）
        # =====================================================================
        gp_factors: List[Dict[str, Any]] = []
        if self._cfg.get('use_gp'):
            self.logger.info("=" * 60)
            self.logger.info("Step 3: GP 遗传规划因子进化")
            self.logger.info("=" * 60)
            gp_factors, gp_logs = self._mine_gp_factors(price_data, returns, pool_factors)
            logs.extend(gp_logs)
            self.logger.info(f"  GP 进化有效因子: {len([f for f in gp_factors if f.get('is_valid')])}")

        # =====================================================================
        # Step 4: 合并因子
        # =====================================================================
        all_factors = pool_factors + gp_factors

        # IC 筛选
        valid_factors = [f for f in all_factors if f.get('is_valid', False)]
        self.logger.info(f"  合并后有效因子: {len(valid_factors)}")

        # =====================================================================
        # Step 5: 自我反思（可选）
        # =====================================================================
        reflection_report = ""
        if self._cfg.get('use_reflection') and valid_factors:
            self.logger.info("=" * 60)
            self.logger.info("Step 5: 自我反思评估")
            self.logger.info("=" * 60)

            strategy = self._cfg.copy()
            should_continue, adjustment = self.reflection.reflect(
                valid_factors, price_data, returns, strategy, iteration=0
            )
            reflection_report = self.reflection.generate_report()
            logs.append(f"反思: {adjustment.reason}")

            if should_continue:
                logs.append("建议进行迭代优化（本次实现暂不执行迭代）")

        # =====================================================================
        # Step 6: 排序与 Top N
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 6: 综合排序")
        self.logger.info("=" * 60)

        top_n = self._cfg.get('top_n', 20)
        sorted_factors = sorted(
            all_factors,
            key=lambda x: x.get('overall_score', 0),
            reverse=True,
        )[:top_n]

        self.logger.info(f"  最终 Top {len(sorted_factors)} 因子:")
        for i, f in enumerate(sorted_factors[:5], 1):
            self.logger.info(
                f"    {i}. {f.get('name', 'unknown')} | "
                f"IC={f.get('ic_mean', 0):.4f} | "
                f"ICIR={f.get('icir', 0):.3f} | "
                f"score={f.get('overall_score', 0):.3f} | "
                f"source={f.get('source', 'unknown')}"
            )

        # =====================================================================
        # Step 7: 生成报告
        # =====================================================================
        elapsed = time.time() - start_time
        report = self._generate_report(
            target, f"{start_date} ~ {end_date}",
            len(all_factors), len(valid_factors),
            sorted_factors, elapsed,
        )

        self.logger.info("=" * 60)
        self.logger.info(f"因子进化挖掘完成，耗时 {elapsed:.1f}s")
        self.logger.info("=" * 60)

        result_data = {
            'target': target,
            'message': f"因子进化挖掘完成，有效因子 {len(valid_factors)}",
            'date_range': f"{start_date} ~ {end_date}",
            'total_factors': len(all_factors),
            'valid_factors': len(valid_factors),
            'top_factors': sorted_factors,
            'gp_history': self.gp_engine.history if self.gp_engine else [],
            'reflection_report': reflection_report,
            'elapsed_time': elapsed,
        }
        return AgentResult(
            agent_name='factor_evolution',
            status=AgentStatus.SUCCESS,
            data=result_data,
            errors=errors,
            logs=logs,
            elapsed_seconds=elapsed,
        )

    def _mine_pool_factors(
        self,
        price_data: pd.DataFrame,
        returns: pd.Series,
        basis_data: Optional[pd.DataFrame],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """候选池因子挖掘"""
        logs: List[str] = []
        factors: List[Dict[str, Any]] = []

        try:
            # 生成候选因子
            candidates = self.candidate_pool.get_all(
                use_technical=self._cfg.get('use_technical', True),
                use_fundamental=self._cfg.get('use_fundamental', True),
                use_cross=self._cfg.get('use_cross', True),
            )
            logs.append(f"候选因子总数: {len(candidates)}")

            # 计算因子值
            from .parallel_miner import ParallelFactorMiner
            miner = ParallelFactorMiner(max_workers=self._cfg.get('max_workers', 8))

            data_context = {
                'close': price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, 0],
                'open': price_data.get('open', price_data.iloc[:, 0]),
                'high': price_data.get('high', price_data.iloc[:, 0]),
                'low': price_data.get('low', price_data.iloc[:, 0]),
                'volume': price_data.get('volume', pd.Series(1, index=price_data.index)),
                'returns': returns,
            }

            factor_values = miner.compute_factors(candidates, data_context)

            # 评估
            for candidate in candidates:
                values = factor_values.get(candidate.name)
                if values is None or values.isna().all():
                    continue

                fitness = self.evaluator.evaluate(values, returns)
                if fitness and fitness.is_valid:
                    factors.append({
                        'name': candidate.name,
                        'category': candidate.category,
                        'description': candidate.description,
                        'expression': candidate.name,
                        'tree': None,
                        'ic_mean': fitness.ic_mean,
                        'ic_std': fitness.ic_std,
                        'icir': fitness.icir,
                        'ic_win_rate': fitness.ic_win_rate,
                        'turnover': fitness.turnover,
                        'overall_score': fitness.overall_score,
                        'is_valid': fitness.is_valid,
                        'source': 'candidate_pool',
                    })

        except Exception as e:
            logs.append(f"候选池挖掘异常: {str(e)}")
            self.logger.error(f"候选池挖掘异常: {e}")

        return factors, logs

    def _mine_gp_factors(
        self,
        price_data: pd.DataFrame,
        returns: pd.Series,
        pool_factors: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """GP 进化因子挖掘"""
        logs: List[str] = []
        factors: List[Dict[str, Any]] = []

        try:
            self.gp_engine = self._init_gp_engine()
            self.gp_engine.set_data(price_data, returns)

            history = self.gp_engine.evolve()
            logs.append(f"GP 进化完成，进化代数: {len(history)}")

            if history:
                best = history[-1]
                logs.append(f"  最终最佳分数: {best['best_score']:.4f}")
                logs.append(f"  最佳表达式: {best['best_expr']}")

            # 获取 Top GP 因子
            top_gp = self.gp_engine.get_top_factors(
                top_k=20,
                include_pool=[f['tree'] for f in pool_factors if f.get('tree') is not None],
            )
            factors.extend(top_gp)

        except Exception as e:
            logs.append(f"GP 进化异常: {str(e)}")
            self.logger.error(f"GP 进化异常: {e}")

        return factors, logs

    def _generate_report(
        self,
        target: str,
        date_range: str,
        total: int,
        valid: int,
        top_factors: List[Dict[str, Any]],
        elapsed: float,
    ) -> str:
        """生成 Markdown 报告"""
        lines: List[str] = []
        lines.append(f"# 因子进化挖掘报告")
        lines.append(f"")
        lines.append(f"**标的**: `{target}`  |  **时间范围**: `{date_range}`  |  **耗时**: `{elapsed:.1f}s`")
        lines.append(f"**GP 进化**: {'启用' if self._cfg.get('use_gp') else '未启用'}  |  **自我反思**: {'启用' if self._cfg.get('use_reflection') else '未启用'}")
        lines.append(f"")

        lines.append(f"## 执行摘要")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"| --- | --- |")
        lines.append(f"| 候选因子总数 | {total} |")
        lines.append(f"| 通过筛选因子 | {valid} |")
        lines.append(f"| 筛选通过率 | {valid / max(total, 1) * 100:.1f}% |")
        lines.append(f"")

        # GP 进化结果
        if self.gp_engine and self.gp_engine.history:
            lines.append(f"## GP 进化历史")
            lines.append(f"| 代数 | 最佳分数 | 平均分数 | 有效因子数 |")
            lines.append(f"| --- | --- | --- | --- |")
            for h in self.gp_engine.history:
                lines.append(f"| {h['generation']} | {h['best_score']:.4f} | {h['avg_score']:.4f} | {h['valid_count']} |")
            lines.append(f"")

        # Top 因子
        if top_factors:
            lines.append(f"## Top {len(top_factors)} 因子")
            lines.append(f"| 排名 | 因子名称 | IC | ICIR | 胜率 | 评分 | 来源 |")
            lines.append(f"| --- | --- | --- | --- | --- | --- | --- |")
            for i, f in enumerate(top_factors, 1):
                lines.append(
                    f"| {i} | `{f.get('name', 'N/A')[:30]}` | "
                    f"{f.get('ic_mean', 0):.4f} | {f.get('icir', 0):.3f} | "
                    f"{f.get('ic_win_rate', 0):.1%} | {f.get('overall_score', 0):.3f} | "
                    f"{f.get('source', 'N/A')} |"
                )
            lines.append(f"")

        lines.append(f"---")
        lines.append(f"*由 futureQuant 因子进化 Agent 自动生成 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        return '\n'.join(lines)
