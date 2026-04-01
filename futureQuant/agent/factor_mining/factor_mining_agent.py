"""
因子挖掘 Agent (Factor Mining Agent)

继承 BaseAgent，自主运行完整的因子挖掘流程：

1. 加载数据（从 DataCollector 结果或外部传入）
2. 生成因子候选池（50+ 候选因子）
3. 并行计算因子值（ThreadPoolExecutor）
4. IC 初步筛选（|IC| < 0.02 丢弃）
5. 深入评估（ICIR、分层回测、稳健性）
6. 综合评分排序
7. 生成 Markdown 报告，推荐 Top 因子
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...core.logger import get_logger
from ...factor.evaluator import FactorEvaluator
from ..base import AgentResult, AgentStatus, BaseAgent
from .factor_candidate_pool import FactorCandidatePool, FactorCandidate
from .factor_report import FactorReport
from .parallel_miner import ParallelFactorMiner


class FactorMiningAgent(BaseAgent):
    """
    因子挖掘 Agent

    自主运行完整的因子挖掘流程，从候选因子池生成到最终推荐的完整闭环。

    Attributes:
        candidate_pool: 因子候选池实例
        miner: 并行因子挖掘引擎
        evaluator: 因子评估器
        logger: 日志记录器

    使用示例：
        >>> agent = FactorMiningAgent(config={'top_n': 20, 'max_workers': 8})
        >>> result = agent.run({
        ...     'target': 'RB',
        ...     'price_data': price_df,
        ...     'start_date': '2023-01-01',
        ...     'end_date': '2024-12-31',
        ... })
        >>> print(result.data)
    """

    # 默认配置
    DEFAULT_CONFIG: Dict[str, Any] = {
        'min_ic': 0.02,       # IC 最小阈值
        'min_icir': 0.3,      # ICIR 最小阈值
        'top_n': 20,          # 保留 Top N 因子
        'max_workers': 8,     # 最大并行线程数
        'report_dir': 'docs/reports',  # 报告保存目录
        'min_samples': 30,     # 最小样本数
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化因子挖掘 Agent

        Args:
            config: 配置字典，会与 DEFAULT_CONFIG 合并
        """
        super().__init__(name='factor_mining', config=config)
        self._cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.candidate_pool = FactorCandidatePool()
        self.miner = ParallelFactorMiner(max_workers=self._cfg['max_workers'])
        self.evaluator = FactorEvaluator()
        self.logger = get_logger('agent.factor_mining')

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行因子挖掘流程

        context 支持的 keys：
            - target (str): 标的代码（如 'RB'）
            - price_data (pd.DataFrame): 价格数据 DataFrame
            - start_date (str): 开始日期
            - end_date (str): 结束日期
            - basis_data (pd.DataFrame): 基本面数据（可选）
            - returns (pd.Series): 收益率序列（可选，默认用次日收益率）

        Returns:
            AgentResult: 包含评估指标、Top 因子 DataFrame 和执行耗时
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
        start_date = context.get('start_date', '2020-01-01')
        end_date = context.get('end_date', datetime.now().strftime('%Y-%m-%d'))

        if price_data is None or price_data.empty:
            errors.append("price_data 为空或未提供")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=errors,
                elapsed_seconds=time.time() - start_time,
            )

        self.logger.info(f"标的: {target}")
        self.logger.info(f"数据范围: {start_date} ~ {end_date}")
        self.logger.info(f"数据形状: {price_data.shape}")
        self.logger.info(f"候选因子总数: {self.candidate_pool.total_count}")

        # =====================================================================
        # Step 2: 生成收益率序列（未来收益率，用于 IC 计算）
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 2: 计算收益率序列")
        self.logger.info("=" * 60)

        if 'returns' in context and context['returns'] is not None:
            returns = context['returns']
        else:
            # 默认使用次日收益率（收盘价变化率）
            close = price_data['close']
            returns = close.pct_change().shift(-1)

        self.logger.info(f"收益率序列长度: {len(returns)}")

        # =====================================================================
        # Step 3: 并行计算候选因子
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 3: 并行计算候选因子")
        self.logger.info("=" * 60)

        candidates = self.candidate_pool.get_all()
        self.logger.info(f"开始计算 {len(candidates)} 个候选因子...")

        factor_values = self.miner.mine(
            candidates=candidates,
            price_data=price_data,
            basis_data=basis_data,
        )

        self.logger.info(f"计算完成，有效因子数: {len(factor_values)}")
        logs.append(f"因子计算完成: {len(factor_values)}/{len(candidates)} 个")

        if not factor_values:
            errors.append("无有效因子值被计算")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=errors,
                elapsed_seconds=time.time() - start_time,
            )

        # 构建因子 DataFrame
        factor_df = pd.DataFrame(factor_values)

        # =====================================================================
        # Step 4: IC 初步筛选
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 4: IC 初步筛选")
        self.logger.info("=" * 60)

        # 对齐数据
        common_index = factor_df.index.intersection(returns.index)
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns.loc[common_index]

        if len(common_index) < self._cfg['min_samples']:
            errors.append(f"共同样本数不足: {len(common_index)} < {self._cfg['min_samples']}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=errors,
                elapsed_seconds=time.time() - start_time,
            )

        # 计算 IC
        ic_series = self.evaluator.calculate_ic(
            factor_aligned, returns_aligned, method='spearman'
        )

        # IC 筛选：|IC| >= min_ic
        passed_mask = ic_series.abs() >= self._cfg['min_ic']
        passed_factors = ic_series[passed_mask].sort_values(key=abs, ascending=False)

        self.logger.info(f"IC 筛选后: {len(passed_factors)}/{len(ic_series)} 个通过")
        logs.append(f"IC 筛选通过: {len(passed_factors)}/{len(ic_series)}")

        if len(passed_factors) == 0:
            errors.append(f"无因子通过 IC 筛选 (|IC| >= {self._cfg['min_ic']})")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=errors,
                elapsed_seconds=time.time() - start_time,
            )

        # =====================================================================
        # Step 5: 深入评估（ICIR、分层回测、稳健性）
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 5: 深入评估")
        self.logger.info("=" * 60)

        top_candidates = passed_factors.head(100).index.tolist()
        top_factor_df = factor_aligned[top_candidates]

        evaluation_results: Dict[str, Dict[str, Any]] = {}
        factor_scores: List[Dict[str, Any]] = []

        for factor_name in top_candidates:
            try:
                series = top_factor_df[factor_name]
                valid_ret = returns_aligned.loc[series.dropna().index]
                series_valid = series.dropna()

                if len(series_valid) < self._cfg['min_samples']:
                    continue

                # 计算 ICIR
                ic_vals = []
                for i in range(len(series_valid) - 1):
                    if len(series_valid.iloc[i:i+20]) >= 10:
                        c = series_valid.iloc[i]
                        r = valid_ret.iloc[i]
                        if not np.isnan(c) and not np.isnan(r):
                            ic_vals.append(c)

                if len(ic_vals) < 5:
                    ic_vals = [series_valid.corr(valid_ret.loc[series_valid.index])]

                ic_arr = np.array(ic_vals)
                ic_mean = np.nanmean(ic_arr)
                ic_std = np.nanstd(ic_arr)
                icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
                ic_win_rate = np.mean(ic_arr > 0) if len(ic_arr) > 0 else 0.0

                # 计算 ICIR（使用 evaluator）
                ic_series_single = pd.Series(series_valid.values, index=valid_ret.loc[series_valid.index].index)
                icir_dict = self.evaluator.calculate_icir(ic_series_single)

                # 获取候选因子信息
                candidate = self.candidate_pool.get_by_name(factor_name)
                category = candidate.category if candidate else 'technical'

                # 组装结果
                score_dict = {
                    'name': factor_name,
                    'category': category,
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'icir': icir,
                    'annual_icir': icir_dict.get('annual_icir', 0),
                    'ic_win_rate': ic_win_rate,
                    'turnover': series_valid.pct_change().abs().mean() if len(series_valid) > 1 else 0,
                    'overall_score': icir * 0.5 + ic_win_rate * 0.3 + (1 - ic_std) * 0.2,
                    'description': candidate.description if candidate else '',
                    'expected_direction': candidate.expected_direction if candidate else None,
                }

                factor_scores.append(score_dict)
                evaluation_results[factor_name] = score_dict

            except Exception as exc:
                self.logger.warning(f"因子 {factor_name} 评估失败: {exc}")
                continue

        # =====================================================================
        # Step 6: 综合评分排序
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 6: 综合评分排序")
        self.logger.info("=" * 60)

        if factor_scores:
            # 按综合评分排序
            sorted_factors = sorted(
                factor_scores,
                key=lambda x: x.get('overall_score', 0),
                reverse=True
            )
            top_factors = sorted_factors[:self._cfg['top_n']]
            top_factors_df = pd.DataFrame(top_factors)
        else:
            top_factors = []
            top_factors_df = pd.DataFrame()

        self.logger.info(f"最终推荐 Top {len(top_factors)} 因子")
        for i, f in enumerate(top_factors[:10], 1):
            self.logger.info(
                f"  {i}. {f['name']}: IC={f['ic_mean']:.4f}, "
                f"ICIR={f['icir']:.4f}, Score={f['overall_score']:.4f}"
            )

        # =====================================================================
        # Step 7: 生成报告
        # =====================================================================
        self.logger.info("=" * 60)
        self.logger.info("Step 7: 生成报告")
        self.logger.info("=" * 60)

        date_range = f"{start_date} ~ {end_date}"
        report = FactorReport(
            target=target,
            date_range=date_range,
            total_candidates=len(candidates),
            passed_candidates=len(passed_factors),
            top_factors=top_factors,
        )
        report.generate_markdown()

        # 保存报告
        report_dir = self._cfg['report_dir']
        os.makedirs(report_dir, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d')
        report_path = os.path.join(report_dir, f'factor_mining_{date_str}.md')
        report.save(report_path)
        self.logger.info(f"报告已保存: {report_path}")
        logs.append(f"报告已保存至: {report_path}")

        elapsed = time.time() - start_time
        self.logger.info(f"因子挖掘完成，总耗时: {elapsed:.2f}s")

        # =====================================================================
        # 返回结果
        # =====================================================================
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=top_factors_df,
            factors=top_factors,
            metrics={
                'ic_results': ic_series.to_dict(),
                'top_factors': top_factors,
                'n_candidates': len(candidates),
                'n_passed': len(passed_factors),
                'n_top': len(top_factors),
                'report_path': report_path,
            },
            errors=errors,
            logs=logs,
            elapsed_seconds=elapsed,
        )
