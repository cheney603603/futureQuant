"""
因子报告数据类 (Factor Report)

定义因子评估报告的数据结构，支持：
- Markdown 格式报告生成
- 报告持久化保存
- 多维度评估结果聚合
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FactorReport:
    """
    因子报告数据类

    包含因子挖掘与评估的完整结果，供后续使用或存档。

    Attributes:
        target: 标的代码（如 'RB'）
        date_range: 数据时间范围，格式 'YYYY-MM-DD ~ YYYY-MM-DD'
        total_candidates: 候选因子总数
        passed_candidates: 通过 IC 筛选的因子数
        top_factors: Top 因子列表，每项为 dict（含 name, ic, icir, score 等）
        ic_heatmap_data: IC 热力图数据（可选）
        quantile_returns: 分层回测收益数据（可选）
        correlation_matrix: 因子相关系数矩阵（可选）
        markdown_report: Markdown 格式报告正文
    """
    target: str
    date_range: str
    total_candidates: int
    passed_candidates: int
    top_factors: List[Dict[str, Any]] = field(default_factory=list)
    ic_heatmap_data: Optional[Dict[str, Any]] = None
    quantile_returns: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    markdown_report: str = ""

    def save(self, path: str) -> None:
        """
        将报告保存为 Markdown 文件

        Args:
            path: 保存路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.markdown_report)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            报告数据字典
        """
        result = {
            'target': self.target,
            'date_range': self.date_range,
            'total_candidates': self.total_candidates,
            'passed_candidates': self.passed_candidates,
            'top_factors': self.top_factors,
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        if self.ic_heatmap_data is not None:
            result['ic_heatmap_data'] = self.ic_heatmap_data

        if self.quantile_returns is not None:
            result['quantile_returns'] = self.quantile_returns

        return result

    def generate_markdown(self) -> str:
        """
        生成 Markdown 格式报告

        Returns:
            Markdown 报告字符串
        """
        lines: List[str] = []

        # 标题
        lines.append(f"# 因子挖掘报告")
        lines.append(f"")
        lines.append(f"**标的**: `{self.target}`  |  **时间范围**: `{self.date_range}`  |  **生成时间**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
        lines.append(f"")

        # 摘要
        lines.append(f"## 📊 执行摘要")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"| --- | --- |")
        lines.append(f"| 候选因子总数 | {self.total_candidates} |")
        lines.append(f"| 通过 IC 筛选 | {self.passed_candidates} |")
        lines.append(f"| 筛选通过率 | {self.passed_candidates / max(self.total_candidates, 1) * 100:.1f}% |")
        lines.append(f"")

        # Top 因子
        if self.top_factors:
            lines.append(f"## 🏆 Top {len(self.top_factors)} 因子")
            lines.append(f"")
            lines.append(f"| 排名 | 因子名称 | IC | ICIR | IC胜率 | 综合评分 | 类别 |")
            lines.append(f"| --- | --- | --- | --- | --- | --- | --- |")

            for i, factor in enumerate(self.top_factors, 1):
                name = factor.get('name', 'N/A')
                ic = factor.get('ic_mean', 0)
                icir = factor.get('icir', 0)
                win_rate = factor.get('ic_win_rate', 0)
                score = factor.get('overall_score', 0)
                category = factor.get('category', 'technical')
                lines.append(
                    f"| {i} | `{name}` | {ic:.4f} | {icir:.4f} | {win_rate:.1%} | {score:.3f} | {category} |"
                )
            lines.append(f"")

        # IC 统计详情
        if self.top_factors:
            lines.append(f"## 📈 IC 统计详情")
            lines.append(f"")
            for factor in self.top_factors:
                name = factor.get('name', 'N/A')
                ic_mean = factor.get('ic_mean', 0)
                ic_std = factor.get('ic_std', 0)
                icir = factor.get('icir', 0)
                annual_icir = factor.get('annual_icir', 0)
                win_rate = factor.get('ic_win_rate', 0)
                turnover = factor.get('turnover', 0)
                score = factor.get('overall_score', 0)

                lines.append(f"### `{name}`")
                lines.append(f"")
                lines.append(f"- **IC 均值**: `{ic_mean:.4f}`")
                lines.append(f"- **IC 标准差**: `{ic_std:.4f}`")
                lines.append(f"- **ICIR**: `{icir:.4f}`")
                lines.append(f"- **年化 ICIR**: `{annual_icir:.4f}`")
                lines.append(f"- **IC 胜率**: `{win_rate:.1%}`")
                lines.append(f"- **换手率**: `{turnover:.2%}`")
                lines.append(f"- **综合评分**: `{score:.3f}`")
                lines.append(f"")

        # 方法论
        lines.append(f"## 📝 方法论")
        lines.append(f"")
        lines.append(f"1. **候选因子池**: 技术因子 {sum(1 for f in self.top_factors if f.get('category') == 'technical')} 个, "
                     f"基本面因子 {sum(1 for f in self.top_factors if f.get('category') == 'fundamental')} 个, "
                     f"交叉因子 {sum(1 for f in self.top_factors if f.get('category') == 'cross')} 个")
        lines.append(f"2. **IC 筛选阈值**: |IC| ≥ 0.02")
        lines.append(f"3. **ICIR 筛选阈值**: ICIR ≥ 0.3")
        lines.append(f"4. **排序方法**: 综合评分（预测能力 30% + 稳定性 20% + 单调性 15% + 可交易性 15% + 稳健性 15% + 独立性 5%）")
        lines.append(f"")

        # 风险提示
        lines.append(f"## ⚠️ 风险提示")
        lines.append(f"")
        lines.append(f"- 历史表现不代表未来收益")
        lines.append(f"- 因子有效性可能随市场结构变化而衰减")
        lines.append(f"- 实际交易需考虑交易成本、滑点、流动性等因素")
        lines.append(f"")

        lines.append(f"---")
        lines.append(f"*由 futureQuant 因子挖掘 Agent 自动生成*")

        self.markdown_report = '\n'.join(lines)
        return self.markdown_report
