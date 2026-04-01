"""
回测 Agent

功能：
- 接收量化 Agent 的信号（signal=1/-1/0）
- 叠加基本面信号增强（如果 context 中有 fundamental_signal）
- 使用向量化引擎执行回测
- 计算绩效指标（收益率、夏普比、最大回撤、胜率）
- 生成回测报告
- 收益归因分析

依赖：
- futureQuant.agent.base.BaseAgent
- futureQuant.agent.backtest_agent.backtest_result.BacktestResult
- futureQuant.agent.backtest_agent.attribution_analyzer.AttributionAnalyzer
- futureQuant.backtest.engine.BacktestEngine, BacktestMode
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import AgentResult, AgentStatus, BaseAgent
from .attribution_analyzer import AttributionAnalyzer
from .backtest_result import BacktestResult

# 延迟导入回测引擎（避免循环依赖）
_BACKTEST_ENGINE: Optional[Any] = None
_BACKTEST_MODE: Optional[Any] = None


def _get_backtest_engine():
    """延迟导入回测引擎"""
    global _BACKTEST_ENGINE, _BACKTEST_MODE
    if _BACKTEST_ENGINE is None:
        from futureQuant.backtest.engine import BacktestEngine, BacktestMode

        _BACKTEST_ENGINE = BacktestEngine
        _BACKTEST_MODE = BacktestMode
    return _BACKTEST_ENGINE, _BACKTEST_MODE


class BacktestAgent(BaseAgent):
    """
    回测 Agent

    执行信号回测，评估策略绩效，并生成归因分析报告。

    Attributes:
        name: Agent 名称
        config: 配置字典

    Example:
        >>> agent = BacktestAgent(config={"initial_capital": 1_000_000})
        >>> result = agent.run({
        ...     "target": "RB2105",
        ...     "signals": signal_df,
        ...     "price_data": price_df,
        ...     "fundamental_signal": 1.5,  # 可选
        ... })
        >>> print(result.metrics["sharpe_ratio"])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化回测 Agent

        Args:
            config: 配置字典，支持参数：
                - initial_capital (float): 初始资金，默认 1_000_000
                - commission (float): 手续费率，默认 0.0001
                - slippage (float): 滑点，默认 1
                - margin_rate (float): 保证金率，默认 0.1
                - sharpe_threshold (float): 夏普比达标阈值，默认 0.5
        """
        super().__init__(name="backtest", config=config)
        self.initial_capital: float = self.config.get("initial_capital", 1_000_000)
        self.commission: float = self.config.get("commission", 0.0001)
        self.slippage: float = self.config.get("slippage", 1.0)
        self.margin_rate: float = self.config.get("margin_rate", 0.1)
        self.sharpe_threshold: float = self.config.get("sharpe_threshold", 0.5)
        self.attribution = AttributionAnalyzer()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行回测

        Args:
            context: 执行上下文，必须包含：
                - signals (DataFrame): 信号 DataFrame，包含 signal 列
                - price_data (DataFrame): 价格数据
                - target (str): 标的代码（可选）
                - fundamental_signal (float, optional): 基本面信号分数

        Returns:
            AgentResult: 包含 metrics，回测绩效指标
        """
        signals: Optional[pd.DataFrame] = context.get("signals")
        price_data: Optional[pd.DataFrame] = context.get("price_data")
        target: str = context.get("target", "UNKNOWN")
        fundamental_signal: Optional[float] = context.get("fundamental_signal")

        self._logger.info(f"Running backtest for {target}")

        if signals is None or signals.empty:
            self._logger.warning("No signals provided for backtest")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                metrics={"target": target, "n_trades": 0, "total_return": 0.0},
            )

        if price_data is None or price_data.empty:
            self._logger.warning("No price data provided for backtest")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                metrics={"target": target, "n_trades": 0, "total_return": 0.0},
            )

        try:
            # Step 1: 叠加基本面信号增强
            enhanced_signals = self._enhance_signals(signals, fundamental_signal)

            # Step 2: 执行回测
            BacktestEngineCls, BacktestModeCls = _get_backtest_engine()

            engine = BacktestEngineCls(
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage,
                margin_rate=self.margin_rate,
            )

            # 准备回测数据（合并信号与价格）
            df = price_data.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            if "date" in enhanced_signals.columns:
                enhanced_signals["date"] = pd.to_datetime(enhanced_signals["date"])
                enhanced_signals = enhanced_signals.set_index("date")

            # 对齐信号
            join_cols = ["signal"] + (["confidence"] if "confidence" in enhanced_signals.columns else [])
            df = df.join(enhanced_signals[join_cols], how="inner")

            if df.empty or "signal" not in df.columns:
                self._logger.warning("No aligned data for backtest")
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.SUCCESS,
                    metrics={"target": target, "n_trades": 0, "total_return": 0.0},
                )

            # Step 3: 向量化回测（直接计算，无需策略对象）
            bt_result = self._run_vectorized_backtest(df, engine)

            # Step 4: 计算绩效指标
            perf_metrics = self._calculate_performance(bt_result)

            # Step 5: 归因分析（如果基本面信号存在）
            attribution_result: Optional[Dict[str, Any]] = None
            if fundamental_signal is not None:
                attribution_result = self.attribution.analyze(
                    price_data=df,
                    signals=enhanced_signals,
                    fundamental_signal=fundamental_signal,
                )

            # Step 6: 生成报告
            report_path = self._generate_report(
                target=target,
                perf_metrics=perf_metrics,
                attribution=attribution_result,
                signals=enhanced_signals,
                equity_curve=bt_result.get("equity_curve"),
            )

            # Step 7: 检查是否达标
            feedback = ""
            if perf_metrics["sharpe_ratio"] < self.sharpe_threshold:
                feedback = (
                    f"⚠️ 夏普比 {perf_metrics['sharpe_ratio']:.3f} "
                    f"< 阈值 {self.sharpe_threshold}，建议重新挖掘因子或调整策略"
                )
                self._logger.warning(feedback)

            self._logger.info(
                f"Backtest for {target}: "
                f"return={perf_metrics['total_return']*100:.2f}%, "
                f"sharpe={perf_metrics['sharpe_ratio']:.3f}, "
                f"max_dd={perf_metrics['max_drawdown']*100:.2f}%, "
                f"n_trades={perf_metrics['n_trades']}"
            )

            metrics: Dict[str, Any] = {
                "target": target,
                "total_return": perf_metrics["total_return"],
                "annual_return": perf_metrics["annual_return"],
                "sharpe_ratio": perf_metrics["sharpe_ratio"],
                "max_drawdown": perf_metrics["max_drawdown"],
                "win_rate": perf_metrics["win_rate"],
                "n_trades": perf_metrics["n_trades"],
                "equity_curve": bt_result.get("equity_curve"),
                "report_path": report_path,
                "attribution": attribution_result,
                "feedback": feedback,
                "below_threshold": perf_metrics["sharpe_ratio"] < self.sharpe_threshold,
            }

            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                metrics=metrics,
            )

        except Exception as exc:
            self._logger.error(f"Backtest failed for {target}: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc)],
            )

    def _enhance_signals(
        self,
        signals: pd.DataFrame,
        fundamental_signal: Optional[float],
    ) -> pd.DataFrame:
        """
        叠加基本面信号增强

        增强规则：
        - signal=1 且 fundamental_score > 1 → signal=1.5（强化多）
        - signal=-1 且 fundamental_score < -1 → signal=-1.5（强化空）
        - 信号矛盾时降低仓位（|signal| 乘以 0.5）

        Args:
            signals: 量化信号 DataFrame
            fundamental_signal: 基本面评分（-5~5）

        Returns:
            增强后的信号 DataFrame
        """
        df = signals.copy()

        if fundamental_signal is None:
            return df

        # 确保有 signal 列
        if "signal" not in df.columns:
            return df

        # 基本面方向
        fund_direction = 1 if fundamental_signal > 1 else (-1 if fundamental_signal < -1 else 0)

        enhanced = []
        for _, row in df.iterrows():
            sig = row["signal"]

            if sig > 0 and fund_direction > 0:
                # 多头 + 基本面利多 → 强化
                new_sig = 1.5
            elif sig < 0 and fund_direction < 0:
                # 空头 + 基本面利空 → 强化
                new_sig = -1.5
            elif sig > 0 and fund_direction < 0:
                # 多头 + 基本面利空 → 矛盾，降低仓位
                new_sig = sig * 0.5
            elif sig < 0 and fund_direction > 0:
                # 空头 + 基本面利多 → 矛盾，降低仓位
                new_sig = sig * 0.5
            else:
                new_sig = sig

            enhanced.append(new_sig)

        df = df.copy()
        df["signal"] = enhanced
        df["fundamental_signal"] = fundamental_signal

        return df

    def _run_vectorized_backtest(
        self,
        data: pd.DataFrame,
        engine: Any,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        简化实现：直接基于信号和价格计算收益。

        Args:
            data: 合并了信号的价格数据
            engine: BacktestEngine 实例

        Returns:
            回测结果字典
        """
        close_prices = data["close"]
        signals = data["signal"].fillna(0)

        # 计算收益率
        returns = close_prices.pct_change().fillna(0)

        # 持仓（信号延迟一期）
        positions = signals.shift(1).fillna(0)

        # 策略收益
        strategy_returns = positions * returns

        # 交易成本
        signal_changes = signals.diff().abs().fillna(0)
        transaction_costs = signal_changes * self.commission * 2
        net_returns = strategy_returns - transaction_costs

        # 净值曲线
        equity = self.initial_capital * (1 + net_returns).cumprod()

        return {
            "equity_curve": equity,
            "net_returns": net_returns,
            "positions": positions,
            "data": data,
        }

    def _calculate_performance(
        self,
        bt_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        计算绩效指标

        Args:
            bt_result: 回测结果

        Returns:
            绩效指标字典
        """
        equity = bt_result["equity_curve"]
        net_returns = bt_result["net_returns"]
        data = bt_result["data"]

        # 总收益率
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # 年化收益
        n_days = len(equity)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 30 else 0.0

        # 波动率
        volatility = net_returns.std() * np.sqrt(252)

        # 夏普比率（无风险利率 3% 年化）
        risk_free = 0.03
        if volatility > 1e-8:
            sharpe_ratio = (annual_return - risk_free) / volatility
        else:
            sharpe_ratio = 0.0

        # 最大回撤
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 持仓时间比
        positions = bt_result["positions"]
        position_time = (positions != 0).sum() / len(positions)

        # 交易次数
        signal_changes = positions.diff().abs().fillna(0)
        n_trades = int((signal_changes != 0).sum())

        # 胜率（只看有持仓的日子）
        trading_days = positions != 0
        if trading_days.sum() > 0:
            daily_pnl = net_returns[trading_days]
            win_rate = (daily_pnl > 0).sum() / trading_days.sum()
        else:
            win_rate = 0.0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "n_trades": n_trades,
            "position_time": float(position_time),
        }

    def _generate_report(
        self,
        target: str,
        perf_metrics: Dict[str, Any],
        attribution: Optional[Dict[str, Any]],
        signals: pd.DataFrame,
        equity_curve: Optional[pd.Series],
    ) -> str:
        """
        生成并保存回测报告

        Args:
            target: 标的代码
            perf_metrics: 绩效指标
            attribution: 归因分析结果
            signals: 信号 DataFrame
            equity_curve: 权益曲线

        Returns:
            报告保存路径
        """
        os.makedirs("D:/310Programm/futureQuant/docs/reports", exist_ok=True)
        report_path = os.path.join(
            "D:/310Programm/futureQuant/docs/reports",
            f"backtest_{target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        )

        lines: List[str] = []

        lines.append("# 📊 回测报告")
        lines.append("")
        lines.append(f"**标的**: `{target}`")
        lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**初始资金**: {self.initial_capital:,.2f}")
        lines.append("")

        # 收益指标
        lines.append("## 📈 收益指标")
        lines.append("")
        lines.append(f"- **总收益率**: `{perf_metrics['total_return']*100:+.2f}%`")
        lines.append(f"- **年化收益率**: `{perf_metrics['annual_return']*100:+.2f}%`")
        lines.append(f"- **夏普比率**: `{perf_metrics['sharpe_ratio']:.3f}`")
        lines.append(f"- **持仓时间比**: `{perf_metrics['position_time']:.1%}`")
        lines.append("")

        # 风险指标
        lines.append("## 📉 风险指标")
        lines.append("")
        lines.append(f"- **最大回撤**: `{perf_metrics['max_drawdown']*100:.2f}%`")
        lines.append(f"- **年化波动率**: `{perf_metrics['volatility']*100:.2f}%`")
        lines.append("")

        # 交易统计
        lines.append("## 📋 交易统计")
        lines.append("")
        lines.append(f"- **总交易次数**: `{perf_metrics['n_trades']}`")
        lines.append(f"- **胜率**: `{perf_metrics['win_rate']:.1%}`")
        lines.append("")

        # 归因分析
        if attribution:
            lines.append("## 🔍 收益归因")
            lines.append("")
            for key, value in attribution.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}**: `{value:.2%}`")
                else:
                    lines.append(f"- **{key}**: `{value}`")
            lines.append("")

        # 信号统计
        if not signals.empty and "signal" in signals.columns:
            lines.append("## 📊 信号统计")
            lines.append("")
            long_signals = (signals["signal"] > 0).sum()
            short_signals = (signals["signal"] < 0).sum()
            neutral_signals = (signals["signal"] == 0).sum()
            lines.append(f"- 做多信号: `{long_signals}` 次")
            lines.append(f"- 做空信号: `{short_signals}` 次")
            lines.append(f"- 空仓信号: `{neutral_signals}` 次")
            lines.append("")

        # 绩效评估
        sharpe_ok = perf_metrics["sharpe_ratio"] >= self.sharpe_threshold
        lines.append("## 🎯 绩效评估")
        lines.append("")
        if sharpe_ok:
            lines.append("✅ **夏普比率达标**（≥ {:.1f}）".format(self.sharpe_threshold))
        else:
            lines.append(
                f"⚠️ **夏普比率未达标**（{perf_metrics['sharpe_ratio']:.3f} < {self.sharpe_threshold}）"
            )
            lines.append("建议：重新挖掘有效因子或调整策略参数")
        lines.append("")

        # 免责
        lines.append("---")
        lines.append("*本报告由 futureQuant AI 自动生成，仅供参考，不构成投资建议。*")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path
