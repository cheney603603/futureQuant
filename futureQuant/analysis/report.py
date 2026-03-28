"""
绩效分析报告生成模块

生成专业级的量化策略绩效报告，包含：
- 收益指标（总收益、年化收益、滚动收益）
- 风险指标（波动率、最大回撤、VaR）
- 风险调整收益（夏普、索提诺、卡玛）
- 交易分析（胜率、盈亏比、持仓周期）
- 可视化图表（收益曲线、回撤、月度收益热力图）

输出格式：
- 文本报告（控制台/日志）
- HTML 报告（浏览器查看）
- JSON 数据（API 对接）
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np

from ..core.logger import get_logger

logger = get_logger('analysis.report')


@dataclass
class PerformanceMetrics:
    """绩效指标数据类"""
    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    cagr: float = 0.0  # 复合年均增长率
    
    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # 最大回撤持续天数
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    
    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_period: float = 0.0
    
    # 其他
    skewness: float = 0.0  # 收益偏度
    kurtosis: float = 0.0  # 收益峰度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'cagr': self.cagr,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_holding_period': self.avg_holding_period,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
        }


class PerformanceReport:
    """
    绩效报告生成器

    使用示例：
        >>> from futureQuant.backtest import BacktestEngine
        >>> engine = BacktestEngine()
        >>> result = engine.run(data, strategy)
        >>> 
        >>> report = PerformanceReport(result)
        >>> report.generate_text()  # 文本报告
        >>> report.save_html('report.html')  # HTML报告
        >>> report.save_json('report.json')  # JSON数据
    """

    def __init__(
        self,
        backtest_result: Optional[Dict] = None,
        equity_curve: Optional[pd.DataFrame] = None,
        trades: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        初始化报告生成器

        Args:
            backtest_result: BacktestEngine.run() 返回的字典
            equity_curve: 净值曲线 DataFrame（如果 backtest_result 中没有）
            trades: 交易记录 DataFrame
            benchmark_returns: 基准收益率序列（用于计算超额收益）
        """
        self.backtest_result = backtest_result or {}
        self.benchmark_returns = benchmark_returns
        
        # 提取净值曲线
        if 'equity_curve' in self.backtest_result:
            self.equity_curve = self.backtest_result['equity_curve']
        elif equity_curve is not None:
            self.equity_curve = equity_curve
        else:
            self.equity_curve = pd.DataFrame()
        
        # 提取交易记录
        if 'trades' in self.backtest_result:
            self.trades = self.backtest_result['trades']
        elif trades is not None:
            self.trades = trades
        else:
            self.trades = pd.DataFrame()
        
        # 计算指标
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """计算所有绩效指标"""
        metrics = PerformanceMetrics()
        
        if self.equity_curve.empty:
            return metrics
        
        # 确保 equity_curve 格式正确
        if isinstance(self.equity_curve, pd.DataFrame):
            if 'net_value' in self.equity_curve.columns:
                net_values = self.equity_curve['net_value']
            elif 'equity' in self.equity_curve.columns:
                net_values = self.equity_curve['equity']
            else:
                net_values = self.equity_curve.iloc[:, 0]
        else:
            net_values = self.equity_curve
        
        # 收益率序列
        returns = net_values.pct_change().dropna()
        
        if len(returns) == 0:
            return metrics
        
        # 收益指标
        initial_value = net_values.iloc[0]
        final_value = net_values.iloc[-1]
        metrics.total_return = (final_value - initial_value) / initial_value
        
        # 年化收益（假设252个交易日）
        n_days = len(returns)
        metrics.annual_return = (1 + metrics.total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        metrics.cagr = metrics.annual_return  # 期货通常用年化收益
        
        # 风险指标
        metrics.volatility = returns.std() * np.sqrt(252)
        
        # 最大回撤
        rolling_max = net_values.cummax()
        drawdown = (net_values - rolling_max) / rolling_max
        metrics.max_drawdown = drawdown.min()
        
        # 最大回撤持续时间
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_start = None
        for i, in_dd in enumerate(is_drawdown):
            if in_dd and current_start is None:
                current_start = i
            elif not in_dd and current_start is not None:
                drawdown_periods.append(i - current_start)
                current_start = None
        if current_start is not None:
            drawdown_periods.append(len(is_drawdown) - current_start)
        metrics.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # VaR
        metrics.var_95 = np.percentile(returns, 5)
        metrics.var_99 = np.percentile(returns, 1)
        
        # 风险调整收益
        risk_free_rate = 0.03
        excess_returns = returns - risk_free_rate / 252
        
        # 夏普比率
        if returns.std() > 0:
            metrics.sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics.sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        
        # 卡玛比率
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.annual_return / abs(metrics.max_drawdown)
        
        # 信息比率（如果有基准）
        if self.benchmark_returns is not None:
            aligned = pd.concat([returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned) > 0:
                active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                if active_returns.std() > 0:
                    metrics.information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
        
        # 交易统计
        if not self.trades.empty:
            metrics = self._calculate_trade_metrics(metrics)
        
        # 收益分布
        metrics.skewness = returns.skew()
        metrics.kurtosis = returns.kurtosis()
        
        return metrics
    
    def _calculate_trade_metrics(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """计算交易相关指标"""
        trades = self.trades
        
        if 'pnl' not in trades.columns:
            return metrics
        
        # 基本统计
        pnls = trades['pnl'].dropna()
        metrics.total_trades = len(pnls)
        
        winning = pnls[pnls > 0]
        losing = pnls[pnls <= 0]
        
        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(pnls) if len(pnls) > 0 else 0
        
        # 盈亏比
        total_wins = winning.sum() if len(winning) > 0 else 0
        total_losses = abs(losing.sum()) if len(losing) > 0 else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        metrics.avg_win = winning.mean() if len(winning) > 0 else 0
        metrics.avg_loss = losing.mean() if len(losing) > 0 else 0
        
        # 持仓周期
        if 'holding_period' in trades.columns:
            metrics.avg_holding_period = trades['holding_period'].mean()
        
        return metrics
    
    def generate_text(self, include_charts: bool = False) -> str:
        """
        生成文本报告

        Args:
            include_charts: 是否包含 ASCII 图表（可选）

        Returns:
            报告字符串
        """
        m = self.metrics
        
        lines = []
        lines.append("=" * 70)
        lines.append(" " * 20 + "期货量化策略绩效报告")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 收益指标
        lines.append("【收益指标】")
        lines.append(f"  总收益率:     {m.total_return * 100:>10.2f}%")
        lines.append(f"  年化收益率:   {m.annual_return * 100:>10.2f}%")
        lines.append(f"  CAGR:         {m.cagr * 100:>10.2f}%")
        lines.append("")
        
        # 风险指标
        lines.append("【风险指标】")
        lines.append(f"  年化波动率:   {m.volatility * 100:>10.2f}%")
        lines.append(f"  最大回撤:     {m.max_drawdown * 100:>10.2f}%")
        lines.append(f"  回撤持续期:   {m.max_drawdown_duration:>10} 天")
        lines.append(f"  VaR (95%):    {m.var_95 * 100:>10.2f}%")
        lines.append(f"  VaR (99%):    {m.var_99 * 100:>10.2f}%")
        lines.append("")
        
        # 风险调整收益
        lines.append("【风险调整收益】")
        lines.append(f"  夏普比率:     {m.sharpe_ratio:>10.3f}")
        lines.append(f"  索提诺比率:   {m.sortino_ratio:>10.3f}")
        lines.append(f"  卡玛比率:     {m.calmar_ratio:>10.3f}")
        if m.information_ratio != 0:
            lines.append(f"  信息比率:     {m.information_ratio:>10.3f}")
        lines.append("")
        
        # 交易统计
        if m.total_trades > 0:
            lines.append("【交易统计】")
            lines.append(f"  总交易次数:   {m.total_trades:>10}")
            lines.append(f"  盈利次数:     {m.winning_trades:>10}")
            lines.append(f"  亏损次数:     {m.losing_trades:>10}")
            lines.append(f"  胜率:         {m.win_rate * 100:>10.2f}%")
            lines.append(f"  盈亏比:       {m.profit_factor:>10.3f}")
            lines.append(f"  平均盈利:     {m.avg_win:>10.2f}")
            lines.append(f"  平均亏损:     {m.avg_loss:>10.2f}")
            lines.append(f"  平均持仓周期: {m.avg_holding_period:>10.1f} 天")
            lines.append("")
        
        # 收益分布
        lines.append("【收益分布】")
        lines.append(f"  偏度:         {m.skewness:>10.3f}")
        lines.append(f"  峰度:         {m.kurtosis:>10.3f}")
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def generate_html(self, title: str = "策略绩效报告") -> str:
        """
        生成 HTML 报告

        Args:
            title: 报告标题

        Returns:
            HTML 字符串
        """
        m = self.metrics
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .metric-value {{ color: #333; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        .neutral {{ color: #ff9800; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>收益指标</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">总收益率</div>
                <div class="metric-value {'positive' if m.total_return > 0 else 'negative'}">{m.total_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化收益率</div>
                <div class="metric-value {'positive' if m.annual_return > 0 else 'negative'}">{m.annual_return * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化波动率</div>
                <div class="metric-value neutral">{m.volatility * 100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value negative">{m.max_drawdown * 100:.2f}%</div>
            </div>
        </div>
        
        <h2>风险调整收益</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value {'positive' if m.sharpe_ratio > 1 else 'neutral'}">{m.sharpe_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">索提诺比率</div>
                <div class="metric-value">{m.sortino_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">卡玛比率</div>
                <div class="metric-value">{m.calmar_ratio:.3f}</div>
            </div>
        </div>
"""
        
        # 交易统计
        if m.total_trades > 0:
            html += f"""
        <h2>交易统计</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>总交易次数</td><td>{m.total_trades}</td></tr>
            <tr><td>盈利次数</td><td>{m.winning_trades}</td></tr>
            <tr><td>亏损次数</td><td>{m.losing_trades}</td></tr>
            <tr><td>胜率</td><td>{m.win_rate * 100:.2f}%</td></tr>
            <tr><td>盈亏比</td><td>{m.profit_factor:.3f}</td></tr>
            <tr><td>平均盈利</td><td>{m.avg_win:.2f}</td></tr>
            <tr><td>平均亏损</td><td>{m.avg_loss:.2f}</td></tr>
        </table>
"""
        
        html += """
        <div class="footer">
            <p>Generated by futureQuant Performance Report</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'metrics': self.metrics.to_dict(),
            'equity_curve_sample': self.equity_curve.head(10).to_dict() if not self.equity_curve.empty else {},
            'trades_sample': self.trades.head(10).to_dict() if not self.trades.empty else {},
            'generated_at': datetime.now().isoformat(),
        }
    
    def save_text(self, path: Union[str, Path]):
        """保存文本报告"""
        Path(path).write_text(self.generate_text(), encoding='utf-8')
        logger.info(f"Text report saved: {path}")
    
    def save_html(self, path: Union[str, Path], title: str = "策略绩效报告"):
        """保存 HTML 报告"""
        Path(path).write_text(self.generate_html(title), encoding='utf-8')
        logger.info(f"HTML report saved: {path}")
    
    def save_json(self, path: Union[str, Path]):
        """保存 JSON 数据"""
        data = self.to_dict()
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        logger.info(f"JSON report saved: {path}")
    
    def print_report(self):
        """打印报告到控制台"""
        print(self.generate_text())


class MultiStrategyReport:
    """
    多策略对比报告

    用于对比多个策略的绩效表现。
    """

    def __init__(self):
        self.reports: Dict[str, PerformanceReport] = {}
    
    def add_strategy(self, name: str, report: PerformanceReport):
        """添加策略报告"""
        self.reports[name] = report
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        data = []
        for name, report in self.reports.items():
            row = {'Strategy': name}
            row.update(report.metrics.to_dict())
            data.append(row)
        return pd.DataFrame(data)
    
    def generate_text(self) -> str:
        """生成对比报告"""
        df = self.generate_comparison_table()
        return df.to_string(index=False)
    
    def best_strategy(self, metric: str = 'sharpe_ratio') -> str:
        """返回最佳策略"""
        df = self.generate_comparison_table()
        if metric in df.columns:
            idx = df[metric].idxmax()
            return df.loc[idx, 'Strategy']
        return ""
