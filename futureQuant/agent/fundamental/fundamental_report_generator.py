"""
基本面报告生成器

生成 Markdown 格式的基本面分析报告，包含：
- 标的信息摘要
- 基差/库存/仓单表格
- 利多/利空事件列表
- 库存周期图示
- 评分结论
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .sentiment_result import SentimentResult


class FundamentalReportGenerator:
    """基本面 Markdown 报告生成器"""

    def generate(
        self,
        target: str,
        factors: pd.DataFrame,
        sentiment: SentimentResult,
    ) -> str:
        """
        生成完整的基本面分析报告

        Args:
            target: 标的代码
            factors: 基本面因子 DataFrame
            sentiment: 情绪分析结果

        Returns:
            Markdown 格式报告字符串
        """
        lines: list[str] = []

        # === 头部 ===
        lines.append("# 📊 基本面分析报告")
        lines.append("")
        lines.append(f"**标的**: `{target}`")
        lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**分析周期**: {factors.index[0].strftime('%Y-%m-%d')} ~ {factors.index[-1].strftime('%Y-%m-%d')}")
        lines.append("")

        # === 情绪评分摘要 ===
        lines.append("## 📈 情绪评分摘要")
        lines.append("")
        score = sentiment.sentiment_score
        emoji = "🟢" if score > 1 else ("🔴" if score < -1 else "🟡")
        direction = "利多" if score > 0.5 else ("利空" if score < -0.5 else "中性偏弱")
        lines.append(
            f"{emoji} **综合评分**: `{score:+.2f}` / ±5  "
            f"| 置信度: `{sentiment.confidence:.0%}`  "
            f"| 方向: **{direction}**"
        )
        lines.append("")
        lines.append(f"- 库存周期: `{sentiment.inventory_cycle}`")
        lines.append(f"- 供需格局: `{sentiment.supply_demand}`")
        lines.append(f"- 分析周期: `{sentiment.time_horizon}`")
        lines.append("")

        # === 关键因子表格 ===
        lines.append("## 📋 关键基本面因子")
        lines.append("")
        lines.append("| 因子 | 最新值 | 状态 | 说明 |")
        lines.append("|------|--------|------|------|")

        latest = factors.iloc[-1]
        prev = factors.iloc[-5] if len(factors) >= 5 else factors.iloc[-2]

        # 基差率
        basis = latest["basis_rate"]
        basis_prev = prev["basis_rate"]
        basis_chg = basis - basis_prev
        lines.append(
            f"| 基差率 | {basis:.2f}% | "
            f"{'📈' if basis > 0 else '📉'} | "
            f"{'正基差（现货升水）' if basis > 0 else '负基差（期货升水'}，"
            f"变化 {basis_chg:+.2f}% |"
        )

        # 库存水平
        inv = latest["inventory_level"]
        inv_prev = prev["inventory_level"]
        inv_chg = inv - inv_prev
        inv_status = "高" if inv > 70 else ("低" if inv < 30 else "中等")
        lines.append(
            f"| 库存水平 | {inv:.1f} | "
            f"{'🔺' if inv_chg > 2 else '🔻' if inv_chg < -2 else '➡️'} | "
            f"库存{inv_status}（{inv_chg:+.1f}），{inv_status}库存利多 |"
        )

        # 仓单
        wr = latest["warehouse_receipt"]
        wr_prev = prev["warehouse_receipt"]
        wr_chg = wr - wr_prev
        lines.append(
            f"| 仓单数量 | {wr:.1f} | "
            f"{'🔺' if wr_chg > 3 else '🔻' if wr_chg < -3 else '➡️'} | "
            f"仓单{'增加' if wr_chg > 3 else '减少' if wr_chg < -3 else '稳定'} |"
        )

        # 供需差
        gap = latest["demand_gap"]
        lines.append(
            f"| 供需差 | {gap:.2f} | "
            f"{'📉' if gap < 0 else '📈'} | "
            f"{'供不应求（利多）' if gap < 0 else '供过于求（利空'} |"
        )

        # 利润指数
        profit = latest["profit_index"]
        lines.append(
            f"| 利润指数 | {profit:.1f} | "
            f"{'📈' if profit > 10 else '📉' if profit < -10 else '➡️'} | "
            f"{'盈利改善' if profit > 10 else '盈利恶化' if profit < -10 else '盈亏平衡'} |"
        )

        # 开工率
        op = latest["operating_rate"]
        lines.append(
            f"| 开工率 | {op:.1f}% | "
            f"{'⚠️' if op > 85 else '📉' if op < 60 else '➡️'} | "
            f"{'产能偏紧' if op > 85 else '产能宽松' if op < 60 else '产能中性'} |"
        )

        # 进口利润
        imp = latest["import_profit"]
        lines.append(
            f"| 进口利润 | {imp:.2f} | "
            f"{'📈' if imp > 5 else '📉' if imp < -5 else '➡️'} | "
            f"{'进口盈利' if imp > 5 else '进口亏损' if imp < -5 else '进口盈亏平衡'} |"
        )

        lines.append("")

        # === 驱动因素 ===
        lines.append("## 🔍 驱动因素分析")
        lines.append("")

        bullish = [d for d in sentiment.drivers if d["direction"] == "利多"]
        bearish = [d for d in sentiment.drivers if d["direction"] == "利空"]
        neutral = [d for d in sentiment.drivers if d["direction"] == "中性"]

        if bullish:
            lines.append("### 🟢 利多因素")
            lines.append("")
            for d in bullish:
                lines.append(
                    f"- **{d['factor']}**: 得分 `{d['score']:+.2f}`，"
                    f"权重 {d['weight']:.0%}"
                )
            lines.append("")

        if bearish:
            lines.append("### 🔴 利空因素")
            lines.append("")
            for d in bearish:
                lines.append(
                    f"- **{d['factor']}**: 得分 `{d['score']:+.2f}`，"
                    f"权重 {d['weight']:.0%}"
                )
            lines.append("")

        if neutral:
            lines.append("### 🟡 中性因素")
            lines.append("")
            for d in neutral:
                lines.append(
                    f"- **{d['factor']}**: 得分 `{d['score']:+.2f}`，"
                    f"权重 {d['weight']:.0%}"
                )
            lines.append("")

        # === 库存周期 ===
        lines.append("## 📊 库存周期判断")
        lines.append("")
        cycle = sentiment.inventory_cycle
        cycle_desc: Dict[str, str] = {
            "主动补库": (
                "📈 **主动补库阶段**：企业积极采购，库存上升，利润改善。\n"
                "   特征：价格上涨 + 库存上升 + 需求向好。\n"
                "   信号含义：**中期利多**（需求驱动，库存重建中）"
            ),
            "被动补库": (
                "⚠️ **被动补库阶段**：需求下滑但生产未及时调整，库存被动累积。\n"
                "   特征：价格下跌 + 库存上升。\n"
                "   信号含义：**中期利空**（需求萎缩，库存压力积累）"
            ),
            "主动去库": (
                "📉 **主动去库阶段**：企业主动降价去化库存，利润压缩。\n"
                "   特征：价格下跌 + 库存下降。\n"
                "   信号含义：**短期利空，中期转利多**（库存去化后价格触底反弹概率大）"
            ),
            "被动去库": (
                "✅ **被动去库阶段**：需求回暖但生产未跟上，库存被动消耗。\n"
                "   特征：价格上涨 + 库存下降。\n"
                "   信号含义：**中期利多**（需求启动，库存去化支持价格上涨）"
            ),
        }
        lines.append(cycle_desc.get(cycle, f"当前处于 **{cycle}** 阶段"))
        lines.append("")

        # === 供需格局 ===
        sd_map = {"tight": "🔴 偏紧", "balanced": "🟡 平衡", "loose": "🟢 宽松"}
        sd_desc = {
            "tight": "当前供需格局偏紧，现货支撑较强，期货易涨难跌。",
            "balanced": "当前供需格局相对平衡，价格震荡为主。",
            "loose": "当前供需格局宽松，库存压力较大，价格上行阻力明显。",
        }
        lines.append("## ⚖️ 供需格局评估")
        lines.append("")
        lines.append(f"{sd_map.get(sentiment.supply_demand, '❓')} **{sentiment.supply_demand.upper()}**")
        lines.append("")
        lines.append(sd_desc.get(sentiment.supply_demand, ""))
        lines.append("")

        # === 时序摘要 ===
        lines.append("## 📅 因子时序变化（近10日）")
        lines.append("")
        recent = factors.tail(10).copy()
        lines.append("| 日期 | 基差率 | 库存 | 仓单 | 供需差 | 利润指数 |")
        lines.append("|------|--------|------|------|--------|----------|")
        for date, row in recent.iterrows():
            lines.append(
                f"| {date.strftime('%Y-%m-%d')} | "
                f"{row['basis_rate']:+.2f}% | "
                f"{row['inventory_level']:.1f} | "
                f"{row['warehouse_receipt']:.1f} | "
                f"{row['demand_gap']:+.2f} | "
                f"{row['profit_index']:+.1f} |"
            )
        lines.append("")

        # === 结论 ===
        lines.append("## 🎯 综合结论")
        lines.append("")
        sentiment_text = (
            "**情绪评级**: "
            + ("🟢 偏多" if score > 1 else "🔴 偏空" if score < -1 else "🟡 中性")
        )
        lines.append(sentiment_text)
        lines.append("")
        lines.append(f"- 建议关注周期: `{sentiment.time_horizon}`")
        lines.append(f"- 置信度: `{sentiment.confidence:.0%}`（因子间一致性评估）")
        lines.append("")

        # 操作建议
        if score > 2:
            lines.append("**操作建议**: 基本面偏多，建议逢低做多或持有多头思路")
        elif score > 0.5:
            lines.append("**操作建议**: 基本面略偏多，但力度不强，建议轻仓尝试")
        elif score < -2:
            lines.append("**操作建议**: 基本面偏空，建议逢高做空或持有空头思路")
        elif score < -0.5:
            lines.append("**操作建议**: 基本面略偏空，但驱动有限，建议轻仓偏空")
        else:
            lines.append("**操作建议**: 基本面中性，建议观望或对冲操作")
        lines.append("")

        # === 免责声明 ===
        lines.append("---")
        lines.append("*本报告由 futureQuant AI 自动生成，基于数学模拟的基本面因子分析，")
        lines.append("仅供参考，不构成投资建议。*")

        return "\n".join(lines)
