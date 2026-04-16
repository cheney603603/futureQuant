"""
价格行为分析 Agent

功能：
- 市场状态分类（趋势上升 / 趋势下降 / 震荡 / 通道）
- 形态识别（三角、矩形、楔形、旗形、双顶等）
- 突破成功率评估
- 入场推荐（方向、入场区间、止损、目标）
- 波动率分析（ATR、布林带）

依赖：
- futureQuant.agent.base.BaseAgent
- futureQuant.agent.price_behavior.market_state.MarketState, PatternResult
- futureQuant.agent.price_behavior.volatility_analyzer.VolatilityAnalyzer
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import AgentResult, AgentStatus, BaseAgent
from .market_state import MarketState, PatternResult
from .volatility_analyzer import VolatilityAnalyzer


class PriceBehaviorAgent(BaseAgent):
    """
    价格行为分析 Agent

    基于价格时序数据，分析市场状态、识别价格形态、评估突破概率，
    并给出入场推荐。

    Attributes:
        name: Agent 名称
        config: 配置字典

    Example:
        >>> agent = PriceBehaviorAgent()
        >>> result = agent.run({
        ...     "target": "RB2105",
        ...     "price_data": price_df,  # OHLCV DataFrame
        ... })
        >>> print(result.metrics["market_state"])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化价格行为 Agent

        Args:
            config: 配置字典，支持参数：
                - trend_window (int): 趋势判断窗口，默认 20
                - pattern_threshold (float): 形态识别阈值，默认 0.02
        """
        super().__init__(name="price_behavior", config=config)
        self.trend_window: int = self.config.get("trend_window", 20)
        self.pattern_threshold: float = self.config.get("pattern_threshold", 0.02)
        self.volatility_analyzer = VolatilityAnalyzer()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute price behavior analysis.

        Args:
            context: Execution context with price_data and target.

        Returns:
            AgentResult with market_state and pattern_result in metrics.
        """
        # Normalize price_data to DataFrame
        raw = context.get("price_data")
        if raw is None:
            price_data = None
        elif isinstance(raw, pd.Series):
            price_data = pd.DataFrame({"close": raw})
        elif isinstance(raw, pd.DataFrame):
            price_data = raw
        else:
            price_data = None
        target: str = context.get("target", "UNKNOWN")

        self._logger.info(f"Analyzing price behavior for {target}")

        if price_data is None or price_data.empty:
            self._logger.warning("No price data provided for price behavior analysis")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                metrics={"target": target, "market_state": "unknown", "pattern_type": "none"},
            )

        try:
            df = price_data.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # Step 1: 波动率分析
            vol_data = self.volatility_analyzer.analyze(df)
            atr = vol_data["atr"]
            atr_ratio = vol_data["atr_ratio"]  # 当前 ATR / 20日均值

            # Step 2: 市场状态分类
            market_state, state_confidence = self._classify_market_state(df)

            # Step 3: 形态识别
            pattern_result = self._identify_pattern(df, market_state, atr)

            # Step 4: 评估突破成功率
            breakout_prob = self._evaluate_breakout(
                pattern_result, vol_data, market_state
            )

            # Step 5: 生成入场推荐
            recommendation = self._generate_recommendation(
                df, pattern_result, breakout_prob, market_state
            )

            self._logger.info(
                f"Price behavior for {target}: "
                f"state={market_state.value}, "
                f"pattern={pattern_result.pattern_type}, "
                f"breakout_prob={breakout_prob:.2%}"
            )

            # Step 6: 生成报告
            report_path = self._generate_report(
                target=target,
                df=df,
                market_state=market_state,
                state_confidence=state_confidence,
                pattern_result=pattern_result,
                breakout_prob=breakout_prob,
                vol_data=vol_data,
                recommendation=recommendation,
            )

            metrics: Dict[str, Any] = {
                "target": target,
                "market_state": market_state.value,
                "state_confidence": state_confidence,
                "pattern_type": pattern_result.pattern_type,
                "breakout_probability": breakout_prob,
                "recommended_direction": recommendation["direction"],
                "entry_range": recommendation["entry_range"],
                "stop_loss": recommendation["stop_loss"],
                "target_price": recommendation["target"],
                "risk_ratio": recommendation["risk_ratio"],
                "atr": atr,
                "atr_ratio": atr_ratio,
                "report_path": report_path,
            }

            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data=df[["close"]].tail(60) if "close" in df.columns else pd.DataFrame(),
                metrics=metrics,
            )

        except Exception as exc:
            self._logger.error(f"Price behavior analysis failed: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc)],
            )

    def _classify_market_state(
        self, df: pd.DataFrame
    ) -> Tuple[MarketState, float]:
        """
        分类市场状态

        规则：
        - 最近 N 天不断创新高（HHV）且均线向上 → TREND_UP
        - 最近 N 天不断创新低（LLV）且均线向下 → TREND_DOWN
        - 价格在均线附近来回穿越 → RANGE
        - 其他 → CHANNEL

        Args:
            df: 价格数据

        Returns:
            (market_state, confidence)
        """
        close = df["close"] if "close" in df.columns else df.iloc[:, 0]
        n = min(self.trend_window, len(close) - 1)
        if n < 5:
            return MarketState.RANGE, 0.5

        recent = close.tail(n + 1)
        ma20 = close.rolling(20).mean().iloc[-1]

        # 均线方向
        ma_current = close.rolling(20).mean().iloc[-1]
        ma_past = close.rolling(20).mean().iloc[-5] if len(close) >= 5 else ma_current
        ma_slope = (ma_current - ma_past) / ma_past if ma_past != 0 else 0

        # 价格相对均线
        price_vs_ma = (close.iloc[-1] - ma20) / ma20 if ma20 != 0 else 0

        # 创新高/新低计数
        # recent is a Series (close.tail()), so we need to get high/low from df
        if isinstance(recent, pd.Series):
            # Use close values as proxy
            highs = recent.values
            lows = recent.values
        else:
            highs = recent['high'] if 'high' in recent.columns else recent['close'] if 'close' in recent.columns else recent
            lows = recent['low'] if 'low' in recent.columns else recent['close'] if 'close' in recent.columns else recent

        hh_count = sum(1 for i in range(1, len(recent)) if recent.iloc[i] > recent.iloc[:i].max())
        ll_count = sum(1 for i in range(1, len(recent)) if recent.iloc[i] < recent.iloc[:i].min())

        # 判断
        if hh_count >= n * 0.4 and ma_slope > 0.001:
            state = MarketState.TREND_UP
            confidence = min(0.5 + hh_count / n * 0.5, 0.95)
        elif ll_count >= n * 0.4 and ma_slope < -0.001:
            state = MarketState.TREND_DOWN
            confidence = min(0.5 + ll_count / n * 0.5, 0.95)
        elif abs(price_vs_ma) < 0.01:
            state = MarketState.RANGE
            # 震荡强度（穿越次数）
            cross_count = sum(
                1
                for i in range(1, len(close))
                if abs(close.iloc[i] - ma20) / ma20 < 0.005
                and abs(close.iloc[i - 1] - ma20) / ma20 >= 0.005
            )
            confidence = min(0.5 + cross_count / n * 0.3, 0.85)
        else:
            state = MarketState.CHANNEL
            confidence = 0.6

        return state, float(confidence)

    def _identify_pattern(
        self, df: pd.DataFrame, market_state: MarketState, atr: float
    ) -> PatternResult:
        """
        识别价格形态

        简化版：检测转折点（局部极值），判断形态类型。

        Args:
            df: 价格数据
            market_state: 市场状态
            atr: ATR 值

        Returns:
            PatternResult
        """
        close = df["close"] if "close" in df.columns else df.iloc[:, 0]
        recent = close.tail(60).values

        if len(recent) < 20:
            return PatternResult()

        # 检测转折点（简单方法：极值点）
        highs: List[int] = []
        lows: List[int] = []

        for i in range(2, len(recent) - 2):
            if (
                recent[i] > recent[i - 2]
                and recent[i] > recent[i - 1]
                and recent[i] > recent[i + 1]
                and recent[i] > recent[i + 2]
            ):
                highs.append(i)
            if (
                recent[i] < recent[i - 2]
                and recent[i] < recent[i - 1]
                and recent[i] < recent[i + 1]
                and recent[i] < recent[i + 2]
            ):
                lows.append(i)

        if len(highs) < 2 or len(lows) < 2:
            pattern_type = "none"
        else:
            # 简化判断：高低点趋势
            recent_high_prices = [recent[i] for i in highs[-3:]]
            recent_low_prices = [recent[i] for i in lows[-3:]]

            high_trend = np.polyfit(range(len(recent_high_prices)), recent_high_prices, 1)[0]
            low_trend = np.polyfit(range(len(recent_low_prices)), recent_low_prices, 1)[0]

            price_range = recent.max() - recent.min()
            threshold = price_range * 0.05  # 5% 阈值

            if high_trend > threshold and low_trend > threshold:
                pattern_type = "ascending_triangle"
            elif high_trend < -threshold and low_trend < -threshold:
                pattern_type = "descending_triangle"
            elif abs(high_trend) < threshold and abs(low_trend) < threshold:
                pattern_type = "rectangle"
            elif abs(high_trend) > threshold * 2 and abs(low_trend) < threshold:
                pattern_type = "flag"
            elif len(highs) >= 2 and abs(recent[highs[-1]] - recent[highs[-2]]) / atr < 2:
                pattern_type = "double_top"
            else:
                pattern_type = "none"

        # 基于市场状态的默认突破概率
        if market_state == MarketState.TREND_UP:
            breakout_prob = 0.65
            direction = "buy"
        elif market_state == MarketState.TREND_DOWN:
            breakout_prob = 0.60
            direction = "sell"
        elif market_state == MarketState.RANGE:
            breakout_prob = 0.50
            direction = "hold"
        else:
            breakout_prob = 0.55
            direction = "hold"

        return PatternResult(
            pattern_type=pattern_type,
            breakout_probability=breakout_prob,
            recommended_direction=direction,
            confidence=0.6,
        )

    def _evaluate_breakout(
        self,
        pattern: PatternResult,
        vol_data: Dict[str, float],
        market_state: MarketState,
    ) -> float:
        """
        评估突破成功率

        综合考虑：
        - ATR 收缩度（收敛越紧，突破概率越高）
        - 趋势强度加成
        - 形态加成

        Args:
            pattern: 形态结果
            vol_data: 波动率数据
            market_state: 市场状态

        Returns:
            突破概率（0~1）
        """
        base_prob = 0.5

        # ATR 收敛加成：ratio < 1 表示收缩
        atr_ratio = vol_data.get("atr_ratio", 1.0)
        if atr_ratio < 0.7:
            base_prob += 0.15
        elif atr_ratio < 0.85:
            base_prob += 0.08
        elif atr_ratio > 1.2:
            base_prob -= 0.1

        # 趋势加成
        if market_state == MarketState.TREND_UP:
            base_prob += 0.10
        elif market_state == MarketState.TREND_DOWN:
            base_prob += 0.05

        # 形态加成
        pattern_boost = {
            "ascending_triangle": 0.10,
            "descending_triangle": 0.10,
            "rectangle": 0.05,
            "flag": 0.08,
            "double_top": -0.05,
            "none": 0.0,
        }
        base_prob += pattern_boost.get(pattern.pattern_type, 0.0)

        return float(np.clip(base_prob, 0.1, 0.95))

    def _generate_recommendation(
        self,
        df: pd.DataFrame,
        pattern: PatternResult,
        breakout_prob: float,
        market_state: MarketState,
    ) -> Dict[str, Any]:
        """
        生成入场推荐

        Args:
            df: 价格数据
            pattern: 形态结果
            breakout_prob: 突破概率
            market_state: 市场状态

        Returns:
            推荐字典
        """
        close = df["close"].iloc[-1] if "close" in df.columns else 4000.0
        atr = self.volatility_analyzer.analyze(df)["atr"]

        # 方向
        direction_map = {
            "buy": "long",
            "sell": "short",
            "hold": "neutral",
        }
        direction = direction_map.get(pattern.recommended_direction, "neutral")

        # 入场区间（当前价格 ± 0.5 ATR）
        entry_min = close - atr * 0.5
        entry_max = close + atr * 0.5

        # 止损（反向 1.5 ATR）
        if pattern.recommended_direction == "buy":
            stop_loss = close - atr * 1.5
            target = close + atr * 3.0
        elif pattern.recommended_direction == "sell":
            stop_loss = close + atr * 1.5
            target = close - atr * 3.0
        else:
            stop_loss = close - atr
            target = close + atr

        # 风险收益比
        risk = abs(close - stop_loss)
        reward = abs(target - close)
        risk_ratio = reward / risk if risk > 0 else 0

        return {
            "direction": direction,
            "entry_range": (round(entry_min, 2), round(entry_max, 2)),
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "risk_ratio": round(risk_ratio, 2),
            "breakout_probability": breakout_prob,
        }

    def _generate_report(
        self,
        target: str,
        df: pd.DataFrame,
        market_state: MarketState,
        state_confidence: float,
        pattern_result: PatternResult,
        breakout_prob: float,
        vol_data: Dict[str, float],
        recommendation: Dict[str, Any],
    ) -> str:
        """生成并保存价格行为报告"""
        os.makedirs("D:/310Programm/futureQuant/docs/reports", exist_ok=True)
        report_path = os.path.join(
            "D:/310Programm/futureQuant/docs/reports",
            f"price_behavior_{target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        )

        close = df["close"].iloc[-1] if "close" in df.columns else 0.0

        lines: List[str] = []
        lines.append("# 📊 价格行为分析报告")
        lines.append("")
        lines.append(f"**标的**: `{target}`")
        lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**当前价格**: `{close:.2f}`")
        lines.append("")

        # 市场状态
        state_emoji = {
            MarketState.TREND_UP: "📈",
            MarketState.TREND_DOWN: "📉",
            MarketState.RANGE: "🔄",
            MarketState.CHANNEL: "📐",
        }
        lines.append("## 📊 市场状态")
        lines.append("")
        lines.append(
            f"{state_emoji.get(market_state, '❓')} "
            f"**{market_state.value.upper()}** "
            f"| 置信度: `{state_confidence:.0%}`"
        )
        lines.append("")

        state_desc = {
            MarketState.TREND_UP: "上升趋势：价格持续创阶段新高，均线向上，适合趋势跟踪策略",
            MarketState.TREND_DOWN: "下降趋势：价格持续创阶段新低，均线向下，适合做空或观望",
            MarketState.RANGE: "震荡格局：价格在均线附近来回穿越，适合高抛低吸区间操作",
            MarketState.CHANNEL: "通道运行：价格在两条平行趋势线内运行，关注通道突破",
        }
        lines.append(state_desc.get(market_state, ""))
        lines.append("")

        # 形态识别
        lines.append("## 🔺 形态识别")
        lines.append("")
        lines.append(f"- **形态类型**: `{pattern_result.pattern_type}`")
        lines.append(f"- **突破概率**: `{breakout_prob:.1%}`")
        lines.append(f"- **推荐方向**: `{recommendation['direction']}`")
        lines.append("")

        # 波动率
        lines.append("## 📉 波动率分析")
        lines.append("")
        lines.append(f"- **ATR(14)**: `{vol_data.get('atr', 0):.2f}`")
        lines.append(f"- **ATR均值比**: `{vol_data.get('atr_ratio', 0):.2f}`（<1 收缩，>1 扩张）")
        bb_upper = vol_data.get("bb_upper", 0)
        bb_lower = vol_data.get("bb_lower", 0)
        if bb_upper and bb_lower:
            lines.append(f"- **布林带**: 上轨 `{bb_upper:.2f}`, 下轨 `{bb_lower:.2f}`")
        lines.append("")

        # 入场推荐
        lines.append("## 🎯 入场推荐")
        lines.append("")
        lines.append(f"- **方向**: `{recommendation['direction']}`")
        lines.append(
            f"- **入场区间**: `{recommendation['entry_range'][0]:.2f}` ~ `{recommendation['entry_range'][1]:.2f}`"
        )
        lines.append(f"- **止损**: `{recommendation['stop_loss']:.2f}`")
        lines.append(f"- **目标**: `{recommendation['target']:.2f}`")
        lines.append(f"- **风险收益比**: `{recommendation['risk_ratio']:.2f}`")
        lines.append("")

        # 操作建议
        lines.append("## 💡 操作建议")
        lines.append("")
        if recommendation["direction"] == "long" and recommendation["risk_ratio"] >= 2.0:
            lines.append("✅ 趋势向上，风险收益比良好，可考虑做多")
        elif recommendation["direction"] == "short" and recommendation["risk_ratio"] >= 2.0:
            lines.append("✅ 趋势向下，风险收益比良好，可考虑做空")
        elif recommendation["direction"] == "neutral":
            lines.append("⚠️ 市场方向不明确，建议观望等待突破确认")
        else:
            lines.append("⚠️ 风险收益比不足，建议谨慎操作")
        lines.append("")

        lines.append("---")
        lines.append("*本报告由 futureQuant AI 自动生成，仅供参考，不构成投资建议。*")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path
