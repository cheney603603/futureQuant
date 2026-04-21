"""
因子研究流程编排器 (Factor Research Flow Orchestrator)

参考 quant_react_interview-main/engine/core/engine.py 设计，
但针对量化因子研究场景进行了定制。

核心功能：
- 协调完整因子研究流程：data → factor_mining → evaluation → fusion → backtest → report
- 支持流水线 YAML/JSON 配置加载
- 提供自然语言任务入口
- 整合增强型 ReAct 循环（EnhancedReActAgent）

使用示例:
    from futureQuant.agent.factor_research_flow import FactorResearchFlow

    # 方式1：直接构建流水线
    flow = FactorResearchFlow()
    flow.add_trigger(target="RB", start_date="2023-01-01", end_date="2024-12-31")
    flow.add_price_data(source="database")
    flow.add_technical_factors(indicators=["momentum", "volatility"])
    flow.add_ic_evaluation()
    flow.add_fusion(method="icir_weighted")
    result = flow.run()

    # 方式2：从 YAML 配置加载
    flow = FactorResearchFlow.from_yaml("pipeline_config.yaml")
    result = flow.run()

    # 方式3：自然语言任务
    flow = FactorResearchFlow()
    result = flow.run_nl_task("帮我研究 RB 的动量因子")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base import AgentResult, AgentStatus
from .react_base import EnhancedReActAgent
from ..engine.nodes.factor_catalog import get_catalog, get_details, FactorCatalog
from ..engine.nodes.pipeline_builder import FactorPipelineBuilder
from ..core.logger import get_logger

logger = get_logger("agent.factor_research_flow")


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class FlowStepResult:
    """流水线中单个步骤的结果。"""
    step_id: str
    kind: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class FlowResult:
    """
    因子研究流程执行结果。

    Attributes:
        success: 是否成功
        step_results: 各步骤的执行结果
        top_factors: 通过筛选的 Top 因子列表
        ic_results: IC 评估结果
        composite_factor: 合成后的因子
        backtest_result: 回测结果（如果有）
        report_path: 报告路径（如果有）
        elapsed_seconds: 总耗时
        summary: 摘要信息
    """
    success: bool
    step_results: Dict[str, FlowStepResult] = field(default_factory=dict)
    top_factors: List[Any] = field(default_factory=list)
    ic_results: Dict[str, float] = field(default_factory=dict)
    composite_factor: Optional[pd.Series] = None
    backtest_result: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "step_results": {
                k: {
                    "step_id": v.step_id,
                    "kind": v.kind,
                    "success": v.success,
                    "elapsed_seconds": v.elapsed_seconds,
                    "error": v.error,
                }
                for k, v in self.step_results.items()
            },
            "top_factors": self.top_factors,
            "ic_results": self.ic_results,
            "backtest_result": self.backtest_result,
            "report_path": self.report_path,
            "elapsed_seconds": self.elapsed_seconds,
            "summary": self.summary,
        }


# =============================================================================
# 默认执行器
# =============================================================================

class DefaultStepExecutors:
    """
    默认步骤执行器集合。

    提供了因子研究流程中各步骤类型的默认实现。
    用户也可以通过 register_executor() 注册自定义执行器。
    """

    def __init__(self, flow: "FactorResearchFlow"):
        self._flow = flow

    # -------------------------------------------------------------------------
    # Trigger
    # -------------------------------------------------------------------------

    async def execute_trigger(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 trigger.manual。"""
        return {
            "target": config.get("target", "UNKNOWN"),
            "start_date": config.get("start_date", "2020-01-01"),
            "end_date": config.get("end_date", "2024-12-31"),
            "frequency": config.get("frequency", "daily"),
            "universe": config.get("universe", []),
        }

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    async def execute_price_bars(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 data.price_bars。"""
        from ...data.manager import DataManager
        from ...data.processor.cleaner import DataCleaner

        target = config.get("target")
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        frequency = config.get("frequency", "daily")
        lookback_days = config.get("lookback_days", 60)
        source = config.get("source", "database")

        # 计算实际开始日期（向前追溯 lookback_days）
        if isinstance(start_date, str):
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            actual_start = (start_dt - timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
        else:
            actual_start = start_date

        logger.info(f"[Flow] Fetching price data: target={target}, range={actual_start}~{end_date}")

        try:
            dm = DataManager()
            if frequency == "daily":
                df = dm.get_daily_data(target, actual_start, end_date)
            else:
                # 分钟数据
                df = dm.get_minute_data(target, actual_start, end_date, frequency)

            if df is None or df.empty:
                # 尝试 fallback
                if source == "fallback":
                    df = self._fetch_from_akshare(target, actual_start, end_date, frequency)
                else:
                    return {
                        "error": f"No data found for {target}",
                        "hint": "Try source='fallback' to use akshare as fallback",
                    }

            # 清洗数据
            cleaner = DataCleaner()
            df = cleaner.clean(df)

            return {"data": df, "rows": len(df), "columns": list(df.columns)}

        except Exception as exc:
            logger.error(f"[Flow] Price data fetch failed: {exc}")
            return {"error": str(exc)}

    async def execute_fundamental(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 data.fundamental。"""
        from ...data.manager import DataManager

        target = config.get("target")
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        data_type = config.get("data_type", "basis")

        logger.info(f"[Flow] Fetching fundamental data: target={target}, type={data_type}")

        try:
            dm = DataManager()
            if data_type == "basis":
                df = dm.get_basis_data(target, start_date, end_date)
            elif data_type == "inventory":
                df = dm.get_inventory_data(target, start_date, end_date)
            elif data_type == "warehouse_receipt":
                df = dm.get_warehouse_receipt_data(target, start_date, end_date)
            else:
                return {"error": f"Unknown data_type: {data_type}"}

            if df is None or df.empty:
                return {"data": pd.DataFrame(), "message": "No fundamental data available"}

            return {"data": df, "rows": len(df), "data_type": data_type}

        except Exception as exc:
            logger.error(f"[Flow] Fundamental data fetch failed: {exc}")
            return {"error": str(exc)}

    # -------------------------------------------------------------------------
    # Factor
    # -------------------------------------------------------------------------

    async def execute_technical(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 factor.technical。"""
        from ...factor.technical.momentum import calculate_momentum
        from ...factor.technical.volatility import calculate_volatility
        from ...factor.technical.volume import calculate_volume_ratio

        data_ref = config.get("data")
        indicators = config.get("indicators", ["momentum", "volatility"])
        windows = config.get("windows", {})
        momentum_windows = windows.get("momentum", [5, 10, 20, 60])
        volatility_windows = windows.get("volatility", [10, 20, 60])
        volume_windows = windows.get("volume", [5, 10, 20])

        # 从 context 获取数据
        df = self._resolve_data(data_ref, context)
        if df is None or df.empty:
            return {"error": "No price data available"}

        factors: Dict[str, pd.Series] = {}

        # 动量因子
        if "momentum" in indicators:
            for w in momentum_windows:
                try:
                    mom = calculate_momentum(df, window=w)
                    factors[f"momentum_{w}"] = mom
                except Exception as exc:
                    logger.warning(f"Momentum {w} failed: {exc}")

        # 波动率因子
        if "volatility" in indicators:
            for w in volatility_windows:
                try:
                    vol = calculate_volatility(df, window=w)
                    factors[f"volatility_{w}"] = vol
                except Exception as exc:
                    logger.warning(f"Volatility {w} failed: {exc}")

        # 成交量因子
        if "volume" in indicators:
            for w in volume_windows:
                try:
                    vol_ratio = calculate_volume_ratio(df, window=w)
                    factors[f"volume_ratio_{w}"] = vol_ratio
                except Exception as exc:
                    logger.warning(f"Volume ratio {w} failed: {exc}")

        if not factors:
            return {"error": "No valid factors computed"}

        factor_df = pd.DataFrame(factors)
        logger.info(f"[Flow] Computed {len(factors)} technical factors")
        return {"factors": factor_df, "factor_names": list(factors.keys())}

    async def execute_alpha101(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 factor.alpha101。"""
        data_ref = config.get("data")
        top_n = config.get("top_n", 20)

        df = self._resolve_data(data_ref, context)
        if df is None or df.empty:
            return {"error": "No price data available"}

        try:
            from ...agent.factor_mining.alpha101_generator import Alpha101Generator
            generator = Alpha101Generator()
            alpha_factors = generator.generate_all(df)
            alpha_df = pd.DataFrame(alpha_factors)

            if len(alpha_df.columns) > top_n:
                # 简单按方差筛选（真实场景应该按 IC）
                variances = alpha_df.var().sort_values(ascending=False)
                alpha_df = alpha_df[variances.head(top_n).index]

            logger.info(f"[Flow] Computed {len(alpha_df.columns)} Alpha101 factors")
            return {"data": alpha_df, "factor_names": list(alpha_df.columns)}

        except Exception as exc:
            logger.warning(f"[Flow] Alpha101 execution failed: {exc}")
            return {"error": str(exc), "factor_names": []}

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    async def execute_ic(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 evaluation.ic。"""
        factors_ref = config.get("factors")
        returns_ref = config.get("returns")
        method = config.get("method", "spearman")
        ic_threshold = config.get("ic_threshold", 0.02)
        icir_threshold = config.get("icir_threshold", 0.3)

        # 从 context 获取因子
        factors_df = self._resolve_factor_data(factors_ref, context)
        if factors_df is None or factors_df.empty:
            return {"error": "No factors available"}

        # 计算收益率
        returns = self._resolve_returns(returns_ref, context, factors_df)
        if returns is None or returns.empty:
            return {"error": "No returns available"}

        # 对齐数据
        common_idx = factors_df.index.intersection(returns.index)
        if len(common_idx) < 30:
            return {"error": f"Insufficient samples: {len(common_idx)}"}

        factors_aligned = factors_df.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        # 计算 IC
        ic_series = {}
        icir_dict = {}

        for col in factors_aligned.columns:
            factor_vals = factors_aligned[col].dropna()
            ret_vals = returns_aligned.loc[factor_vals.index].dropna()
            common = factor_vals.index.intersection(ret_vals.index)
            if len(common) < 20:
                continue

            f = factor_vals.loc[common]
            r = ret_vals.loc[common]

            if method == "spearman":
                ic = f.corr(r, method="spearman")
            else:
                ic = f.corr(r, method="pearson")

            if not pd.isna(ic):
                ic_series[col] = ic

        # 筛选
        passed = {k: v for k, v in ic_series.items() if abs(v) >= ic_threshold}
        passed_sorted = dict(sorted(passed.items(), key=lambda x: abs(x[1]), reverse=True))

        logger.info(f"[Flow] IC evaluation: {len(passed)}/{len(ic_series)} passed (|IC| >= {ic_threshold})")

        return {
            "ic_series": passed_sorted,
            "all_ic_series": ic_series,
            "passed_factors": list(passed_sorted.keys()),
            "ic_threshold": ic_threshold,
            "method": method,
        }

    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------

    async def execute_fusion(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 fusion.icir_weighted 或 fusion.multifactor。"""
        method = config.get("method", "icir_weighted")
        factors_ref = config.get("factors")
        ic_series_ref = config.get("ic_series")
        corr_threshold = config.get("corr_threshold", 0.8)
        min_icir = config.get("min_icir", 0.3)
        normalize = config.get("normalize", "zscore")

        factors_df = self._resolve_factor_data(factors_ref, context)
        if factors_df is None or factors_df.empty:
            return {"error": "No factors available"}

        # 获取通过 IC 筛选的因子
        passed = context.get_output("_ic_passed") if hasattr(context, "get_output") else None
        if passed:
            available = [f for f in passed if f in factors_df.columns]
            if available:
                factors_df = factors_df[available]

        if factors_df.empty:
            return {"error": "No factors to fuse"}

        if method == "icir_weighted":
            # ICIR 加权合成
            ic_dict = ic_series_ref if isinstance(ic_series_ref, dict) else {}
            weights = {}
            total_icir = 0.0

            for col in factors_df.columns:
                ic = ic_dict.get(col, 0.01)
                weight = abs(ic)
                weights[col] = weight
                total_icir += weight

            # 归一化权重
            for col in weights:
                weights[col] /= total_icir

            # 计算合成因子
            composite = pd.Series(0.0, index=factors_df.index)
            for col, weight in weights.items():
                composite += factors_df[col].fillna(0) * weight

            # Z-score 标准化
            mean = composite.mean()
            std = composite.std()
            if std > 1e-8:
                composite = (composite - mean) / std

        elif method == "multifactor":
            # 多因子标准化合成
            z_scores = factors_df.apply(
                lambda col: (col - col.rolling(60, min_periods=20).mean())
                / col.rolling(60, min_periods=20).std()
            )
            composite = z_scores.mean(axis=1)

        else:
            return {"error": f"Unknown fusion method: {method}"}

        logger.info(f"[Flow] Fusion completed: {len(factors_df.columns)} factors combined")
        return {
            "composite_factor": composite,
            "weights": weights if method == "icir_weighted" else {},
            "method": method,
            "n_factors": len(factors_df.columns),
        }

    # -------------------------------------------------------------------------
    # Backtest
    # -------------------------------------------------------------------------

    async def execute_backtest(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 backtest.factor_signal。"""
        factor_ref = config.get("factor")
        price_ref = config.get("price_data")
        signal_threshold = config.get("signal_threshold", 1.0)
        cost_rate = config.get("cost_rate", 0.0003)

        # 获取因子
        if hasattr(factor_ref, "to_dict"):
            composite = pd.Series(factor_ref)
        elif isinstance(factor_ref, str):
            composite = self._resolve_composite(factor_ref, context)
        else:
            composite = factor_ref

        if composite is None or composite.empty:
            return {"error": "No composite factor available"}

        # 获取价格数据
        df = self._resolve_data(price_ref, context)
        if df is None or df.empty:
            return {"error": "No price data available"}

        close = df["close"]
        returns = close.pct_change().shift(-1)

        # 对齐
        common_idx = composite.index.intersection(returns.index)
        composite = composite.loc[common_idx]
        returns = returns.loc[common_idx]

        # 生成信号
        signals = pd.Series(0, index=composite.index)
        signals[composite > signal_threshold] = 1
        signals[composite < -signal_threshold] = -1

        # 计算策略收益
        strategy_returns = signals.shift(1).fillna(0) * returns
        # 扣除交易成本
        turnover = signals.diff().abs() / 2
        costs = turnover * cost_rate
        strategy_returns -= costs

        # 计算绩效
        total_return = (1 + strategy_returns.dropna()).prod() - 1
        annual_return = (1 + total_return) ** (252 / max(len(strategy_returns.dropna()), 1)) - 1
        volatility = strategy_returns.std() * (252 ** 0.5)
        sharpe = annual_return / volatility if volatility > 1e-8 else 0

        # 最大回撤
        cumulative = (1 + strategy_returns.dropna()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        result = {
            "returns": strategy_returns,
            "signals": signals,
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "turnover": float(turnover.mean()),
        }

        logger.info(
            f"[Flow] Backtest: Sharpe={sharpe:.2f}, Annual={annual_return:.2%}, MDD={max_drawdown:.2%}"
        )
        return result

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    async def execute_report(
        self, config: Dict[str, Any], context: Any
    ) -> Dict[str, Any]:
        """执行 output.report。"""
        top_factors_ref = config.get("top_factors")
        ic_results_ref = config.get("ic_results")
        report_dir = config.get("report_dir", "docs/reports")

        import os
        from datetime import datetime

        os.makedirs(report_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"factor_research_{date_str}.md")

        ic_results = ic_results_ref if isinstance(ic_results_ref, dict) else {}
        top_factors = top_factors_ref if isinstance(top_factors_ref, list) else []

        # 生成报告内容
        lines = [
            "# 因子研究报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## IC 评估结果",
            "",
            "| 因子 | IC |",
            "|------|-----|",
        ]
        for name, ic in list(ic_results.items())[:20]:
            direction = "✅ 正向" if ic > 0 else "❌ 反向"
            lines.append(f"| {name} | {ic:.4f} {direction} |")

        lines.extend(["", "## Top 因子", ""])
        for i, f in enumerate(top_factors[:10], 1):
            if isinstance(f, dict):
                lines.append(f"{i}. **{f.get('name', 'Unknown')}** (IC={f.get('ic', 'N/A')})")
            else:
                lines.append(f"{i}. {f}")

        report_content = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"[Flow] Report saved to {report_path}")
        return {"report_path": report_path, "summary": {"n_factors": len(ic_results)}}

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _resolve_data(self, data_ref: Any, context: Any) -> Optional[pd.DataFrame]:
        """从引用解析价格数据。"""
        if isinstance(data_ref, pd.DataFrame):
            return data_ref
        if hasattr(context, "get_output") and isinstance(data_ref, str):
            if data_ref.startswith("$"):
                step_id = data_ref.replace("$", "").split("[")[0]
                output = context.get_output(step_id)
                if isinstance(output, dict):
                    return output.get("data")
                return output
        return data_ref

    def _resolve_factor_data(self, data_ref: Any, context: Any) -> Optional[pd.DataFrame]:
        """从引用解析因子数据。"""
        if isinstance(data_ref, pd.DataFrame):
            return data_ref
        if hasattr(context, "get_output") and isinstance(data_ref, str):
            if data_ref.startswith("$"):
                step_id = data_ref.replace("$", "").split("[")[0]
                output = context.get_output(step_id)
                if isinstance(output, dict):
                    if "factors" in output:
                        return output["factors"]
                    if "data" in output:
                        return output["data"]
                return output
        return data_ref

    def _resolve_returns(
        self, returns_ref: Any, context: Any, fallback_data: pd.DataFrame
    ) -> Optional[pd.Series]:
        """从引用解析收益率序列。"""
        if isinstance(returns_ref, pd.Series):
            return returns_ref
        if hasattr(context, "get_output") and isinstance(returns_ref, str):
            output = context.get_output(returns_ref.replace("$", "").split("[")[0])
            if isinstance(output, dict):
                return output.get("returns")
            return output
        # 默认使用次日收益率
        close = fallback_data["close"]
        return close.pct_change().shift(-1)

    def _resolve_composite(
        self, composite_ref: Any, context: Any
    ) -> Optional[pd.Series]:
        """从引用解析合成因子。"""
        if isinstance(composite_ref, pd.Series):
            return composite_ref
        if hasattr(context, "get_output"):
            output = context.get_output(str(composite_ref).replace("$", "").split("[")[0])
            if isinstance(output, dict):
                return output.get("composite_factor")
            return output
        return composite_ref

    def _fetch_from_akshare(
        self, symbol: str, start_date: str, end_date: str, frequency: str
    ) -> Optional[pd.DataFrame]:
        """从 akshare 获取数据作为 fallback。"""
        try:
            import akshare as ak

            if frequency == "daily":
                df = ak.futures_zh_daily_sina(symbol=symbol)
                df["date"] = pd.to_datetime(df["date"])
                df = df.rename(columns={"volume": "vol"})
                df = df.set_index("date").sort_index()
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                return df
        except Exception as exc:
            logger.warning(f"Akshare fallback failed: {exc}")
        return None


# =============================================================================
# 因子研究流程编排器
# =============================================================================

class FactorResearchFlow:
    """
    因子研究流程编排器。

    提供声明式和配置式两种方式构建因子研究流程：
    - 声明式：add_* 方法逐步添加步骤
    - 配置式：从 YAML/JSON 加载流水线配置

    工作流程:
        trigger -> data -> factor -> evaluation -> fusion -> backtest -> report

    使用示例:
        flow = FactorResearchFlow()
        flow.add_trigger(target="RB", start_date="2023-01-01", end_date="2024-12-31")
        flow.add_price_data()
        flow.add_technical_factors(indicators=["momentum", "volatility"])
        flow.add_ic_evaluation()
        result = flow.run()
    """

    # 步骤类型到执行器的映射
    DEFAULT_EXECUTORS = {
        "trigger.manual": "execute_trigger",
        "data.price_bars": "execute_price_bars",
        "data.fundamental": "execute_fundamental",
        "factor.technical": "execute_technical",
        "factor.alpha101": "execute_alpha101",
        "factor.fundamental": "execute_fundamental",  # 复用
        "evaluation.ic": "execute_ic",
        "fusion.icir_weight": "execute_fusion",
        "fusion.multifactor": "execute_fusion",
        "backtest.factor_signal": "execute_backtest",
        "backtest.walk_forward": "execute_backtest",  # 简化版
        "output.report": "execute_report",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化因子研究流程。

        Args:
            config: 全局配置（会传递给各步骤）
        """
        self._config = config or {}
        self._builder = FactorPipelineBuilder()
        self._executor_registry: Dict[str, Any] = {}
        self._default_executors = DefaultStepExecutors(self)
        self._step_results: Dict[str, FlowStepResult] = {}
        self._catalog = FactorCatalog()

        # 注册默认执行器
        for kind, method_name in self.DEFAULT_EXECUTORS.items():
            method = getattr(self._default_executors, method_name, None)
            if method:
                self._builder.register_step(kind, _AsyncExecutorWrapper(method))
                self._executor_registry[kind] = method

    # -------------------------------------------------------------------------
    # 声明式 API（链式调用）
    # -------------------------------------------------------------------------

    def add_trigger(
        self,
        target: str = "RB",
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        frequency: str = "daily",
        universe: Optional[List[str]] = None,
        step_id: str = "trigger",
    ) -> "FactorResearchFlow":
        """添加触发步骤。"""
        config = {
            "target": target,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency,
        }
        if universe:
            config["universe"] = universe
        self._builder.add_step("trigger.manual", config=config, step_id=step_id)
        return self

    def add_price_data(
        self,
        source: str = "database",
        frequency: str = "daily",
        lookback_days: int = 60,
        data_step_id: str = "price_bars",
        depends_on: str = "trigger",
    ) -> "FactorResearchFlow":
        """添加价格数据步骤。"""
        config = {
            "target": f"${depends_on}['target']",
            "start_date": f"${depends_on}['start_date']",
            "end_date": f"${depends_on}['end_date']",
            "frequency": frequency,
            "lookback_days": lookback_days,
            "source": source,
        }
        self._builder.add_step("data.price_bars", config=config, step_id=data_step_id)
        self._builder.connect_steps(depends_on, data_step_id)
        return self

    def add_fundamental_data(
        self,
        data_type: str = "basis",
        fund_step_id: str = "fundamental",
        price_step_id: str = "price_bars",
        depends_on: str = "trigger",
    ) -> "FactorResearchFlow":
        """添加基本面数据步骤。"""
        config = {
            "target": f"${depends_on}['target']",
            "start_date": f"${depends_on}['start_date']",
            "end_date": f"${depends_on}['end_date']",
            "data_type": data_type,
        }
        self._builder.add_step("data.fundamental", config=config, step_id=fund_step_id)
        self._builder.connect_steps(depends_on, fund_step_id)
        return self

    def add_technical_factors(
        self,
        indicators: Optional[List[str]] = None,
        windows: Optional[Dict[str, List[int]]] = None,
        factor_step_id: str = "technical",
        depends_on: str = "price_bars",
    ) -> "FactorResearchFlow":
        """添加技术因子步骤。"""
        if indicators is None:
            indicators = ["momentum", "volatility"]
        if windows is None:
            windows = {
                "momentum": [5, 10, 20, 60],
                "volatility": [10, 20, 60],
                "volume": [5, 10, 20],
            }
        config = {
            "data": f"${depends_on}['data']",
            "indicators": indicators,
            "windows": windows,
        }
        self._builder.add_step("factor.technical", config=config, step_id=factor_step_id)
        self._builder.connect_steps(depends_on, factor_step_id)
        return self

    def add_alpha101_factors(
        self,
        top_n: int = 20,
        factor_step_id: str = "alpha101",
        depends_on: str = "price_bars",
    ) -> "FactorResearchFlow":
        """添加 Alpha101 因子步骤。"""
        config = {
            "data": f"${depends_on}['data']",
            "top_n": top_n,
        }
        self._builder.add_step("factor.alpha101", config=config, step_id=factor_step_id)
        self._builder.connect_steps(depends_on, factor_step_id)
        return self

    def add_ic_evaluation(
        self,
        ic_threshold: float = 0.02,
        icir_threshold: float = 0.3,
        method: str = "spearman",
        eval_step_id: str = "ic_eval",
        factors_step_id: str = "technical",
        returns_step_id: Optional[str] = None,
    ) -> "FactorResearchFlow":
        """添加 IC 评估步骤。"""
        config = {
            "factors": f"${factors_step_id}['factors']",
            "method": method,
            "ic_threshold": ic_threshold,
            "icir_threshold": icir_threshold,
        }
        if returns_step_id:
            config["returns"] = f"${returns_step_id}['returns']"
        self._builder.add_step("evaluation.ic", config=config, step_id=eval_step_id)
        self._builder.connect_steps(factors_step_id, eval_step_id)
        return self

    def add_fusion(
        self,
        method: str = "icir_weighted",
        fusion_step_id: str = "fusion",
        eval_step_id: str = "ic_eval",
        factors_step_id: str = "technical",
    ) -> "FactorResearchFlow":
        """添加因子融合步骤。"""
        config = {
            "factors": f"${factors_step_id}['factors']",
            "ic_series": f"${eval_step_id}['ic_series']",
            "method": method,
        }
        self._builder.add_step("fusion.icir_weight", config=config, step_id=fusion_step_id)
        self._builder.connect_steps(eval_step_id, fusion_step_id)
        self._builder.connect_steps(factors_step_id, fusion_step_id)
        return self

    def add_backtest(
        self,
        signal_threshold: float = 1.0,
        cost_rate: float = 0.0003,
        backtest_step_id: str = "backtest",
        fusion_step_id: str = "fusion",
        price_step_id: str = "price_bars",
    ) -> "FactorResearchFlow":
        """添加回测步骤。"""
        config = {
            "factor": f"${fusion_step_id}['composite_factor']",
            "price_data": f"${price_step_id}['data']",
            "signal_threshold": signal_threshold,
            "cost_rate": cost_rate,
        }
        self._builder.add_step("backtest.factor_signal", config=config, step_id=backtest_step_id)
        self._builder.connect_steps(fusion_step_id, backtest_step_id)
        self._builder.connect_steps(price_step_id, backtest_step_id)
        return self

    def add_report(
        self,
        report_dir: str = "docs/reports",
        report_step_id: str = "report",
        eval_step_id: str = "ic_eval",
    ) -> "FactorResearchFlow":
        """添加报告步骤。"""
        config = {
            "top_factors": f"${eval_step_id}['passed_factors']",
            "ic_results": f"${eval_step_id}['ic_series']",
            "report_dir": report_dir,
        }
        self._builder.add_step("output.report", config=config, step_id=report_step_id)
        self._builder.connect_steps(eval_step_id, report_step_id)
        return self

    # -------------------------------------------------------------------------
    # 执行
    # -------------------------------------------------------------------------

    def run(self) -> FlowResult:
        """
        执行完整流程。

        Returns:
            FlowResult
        """
        import asyncio

        logger.info(f"[Flow] Starting factor research flow: {len(self._builder.snapshot_step_ids())} steps")
        start_time = time.time()

        try:
            results = asyncio.run(self._builder.execute_pipeline())
        except Exception as exc:
            logger.error(f"[Flow] Pipeline execution failed: {exc}")
            return FlowResult(
                success=False,
                summary={"error": str(exc)},
                elapsed_seconds=time.time() - start_time,
            )

        # 收集结果
        ic_results = {}
        top_factors = []
        composite_factor = None
        backtest_result = None
        report_path = None
        all_success = True

        for step_id, result in results.items():
            self._step_results[step_id] = FlowStepResult(
                step_id=step_id,
                kind=self._builder.get_config(step_id).get("kind", "unknown") if self._builder.get_config(step_id) else "unknown",
                success=result.success,
                output=result.output,
                error=result.error,
            )
            if not result.success:
                all_success = False

            # 提取关键结果
            if result.success and result.output:
                if "ic_series" in result.output:
                    ic_results.update(result.output["ic_series"])
                if "passed_factors" in result.output:
                    top_factors = result.output["passed_factors"]
                if "composite_factor" in result.output:
                    composite_factor = result.output["composite_factor"]
                if "report_path" in result.output:
                    report_path = result.output["report_path"]
                # backtest 结果提取
                if all(k in result.output for k in ["sharpe_ratio", "annual_return"]):
                    backtest_result = result.output

        elapsed = time.time() - start_time
        logger.info(
            f"[Flow] Completed: success={all_success}, "
            f"top_factors={len(top_factors)}, "
            f"elapsed={elapsed:.2f}s"
        )

        return FlowResult(
            success=all_success,
            step_results=self._step_results,
            top_factors=top_factors,
            ic_results=ic_results,
            composite_factor=composite_factor,
            backtest_result=backtest_result,
            report_path=report_path,
            elapsed_seconds=elapsed,
            summary={
                "n_steps": len(results),
                "n_top_factors": len(top_factors),
                "best_ic": max(ic_results.values()) if ic_results else None,
            },
        )

    # -------------------------------------------------------------------------
    # 配置式 API
    # -------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "FactorResearchFlow":
        """
        从配置创建流程。

        Args:
            config: YAML/JSON 文件路径或配置字典

        Returns:
            FactorResearchFlow 实例
        """
        if isinstance(config, (str, Path)):
            if Path(config).suffix in (".yaml", ".yml", ".json"):
                with open(config, encoding="utf-8") as f:
                    config = json.load(f) if config.endswith(".json") else yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config}")

        flow = cls(config=config)
        steps = config.get("steps", [])

        for step_config in steps:
            kind = step_config["kind"]
            step_id = step_config.get("step_id", kind.replace(".", "_"))
            step_flow_config = step_config.get("config", {})
            depends_on = step_config.get("depends_on")

            flow._builder.add_step(kind, config=step_flow_config, step_id=step_id)
            if depends_on:
                if isinstance(depends_on, str):
                    flow._builder.connect_steps(depends_on, step_id)
                else:
                    for dep in depends_on:
                        flow._builder.connect_steps(dep, step_id)

        return flow

    # -------------------------------------------------------------------------
    # 自然语言入口
    # -------------------------------------------------------------------------

    def run_nl_task(self, task: str) -> FlowResult:
        """
        执行自然语言任务。

        使用 LLM 将自然语言任务转换为流水线配置，然后执行。

        Args:
            task: 自然语言任务描述

        Returns:
            FlowResult
        """
        logger.info(f"[Flow] NL task: {task}")

        # 简单实现：使用规则解析
        # 完整实现应该使用 EnhancedReActAgent
        task_lower = task.lower()

        # 提取目标品种
        target = "RB"
        for variety in ["RB", "HC", "I", "AL", "CU", "AU", "AG", "MA", "SM"]:
            if variety.lower() in task_lower or variety in task:
                target = variety
                break

        # 提取时间范围
        import re
        date_pattern = r"20\d{2}-\d{2}-\d{2}"
        dates = re.findall(date_pattern, task)
        start_date = dates[0] if len(dates) > 0 else "2023-01-01"
        end_date = dates[1] if len(dates) > 1 else "2024-12-31"

        # 构建流程
        self.add_trigger(target=target, start_date=start_date, end_date=end_date)
        self.add_price_data()
        self.add_technical_factors()

        # 检查是否需要基本面
        if "基差" in task or "库存" in task or "基本面" in task:
            self.add_fundamental_data()

        # 检查是否需要 Alpha101
        if "alpha" in task_lower or "Alpha101" in task:
            self.add_alpha101_factors()

        # 检查是否需要 IC 评估
        if "IC" in task or "评估" in task or "筛选" in task:
            self.add_ic_evaluation()

        # 检查是否需要融合
        if "融合" in task or "合成" in task or "组合" in task:
            self.add_fusion()

        # 检查是否需要回测
        if "回测" in task or "backtest" in task_lower:
            self.add_backtest()

        return self.run()

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------

    def register_executor(self, kind: str, executor: Any) -> "FactorResearchFlow":
        """注册自定义执行器。"""
        self._builder.register_step(kind, _AsyncExecutorWrapper(executor))
        self._executor_registry[kind] = executor
        return self

    def get_pipeline(self) -> Dict[str, Any]:
        """获取当前流水线定义。"""
        return self._builder.get_pipeline()

    def get_catalog(self) -> List[Dict[str, Any]]:
        """获取步骤目录。"""
        return get_catalog()

    def get_step_details(self, kind: str) -> Dict[str, Any]:
        """获取步骤详细信息。"""
        return get_details(kind)

    def __repr__(self) -> str:
        steps = self._builder.snapshot_step_ids()
        return f"FactorResearchFlow(steps={len(steps)}, step_ids={steps})"


# =============================================================================
# 辅助类
# =============================================================================

class _AsyncExecutorWrapper:
    """
    异步执行器包装器。

    将同步函数包装为异步函数，
    方便注册到 FactorPipelineBuilder。
    """

    def __init__(self, func):
        self._func = func

    async def execute(self, config: Dict[str, Any], context: Any) -> Any:
        """执行包装的函数。"""
        import asyncio
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(config, context)
        else:
            # 在线程池中执行同步函数
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self._func(config, context))
