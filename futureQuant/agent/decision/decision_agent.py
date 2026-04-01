"""
DecisionAgent - 决策中枢

整合所有 Agent 输出，做出综合判断：
- 价格区间预测（乐观/基准/悲观）
- 交易策略推荐（趋势跟踪/均值回归/观望）
- 风险点识别与变量清单
- 最终决策报告

继承 BaseAgent，运行每日决策 Loop。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from ..base import AgentResult, AgentStatus, BaseAgent
from .decision_report import DecisionReport
from .dynamic_weight import DynamicWeightEngine, AgentWeights
from .scenario_analyzer import ScenarioAnalyzer
from ..price_behavior.market_state import MarketState

if TYPE_CHECKING:
    from ..fundamental import FundamentalAnalysisAgent
    from ..quant import QuantSignalAgent
    from ..backtest_agent import BacktestAgent
    from ..price_behavior import PriceBehaviorAgent

from ...core.logger import get_logger

logger = get_logger('agent.decision')


class DecisionAgent(BaseAgent):
    """
    决策中枢 Agent

    职责：
    1. 汇总各 Agent 输出（量化信号、基本面、行为、回测）
    2. 动态权重调整（根据市场状态）
    3. 价格区间预测
    4. 情景分析（乐观/基准/悲观）
    5. 风险识别
    6. 策略推荐
    7. 生成综合决策报告

    使用方式：
        >>> agent = DecisionAgent()
        >>> result = agent.execute({
        ...     'quant_signal': quant_agent_result,
        ...     'fundamental_result': fundamental_agent_result,
        ...     'price_behavior_result': price_behavior_result,
        ...     'backtest_result': backtest_agent_result,
        ...     'target': 'RB',
        ...     'current_price': 3800.0,
        ... })
    """

    DEFAULT_CONFIG = {
        'report_dir': 'docs/reports',
        'confidence_threshold': 0.55,
        'volatility_lookback': 20,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化决策 Agent

        Args:
            config: 自定义配置
        """
        name = "decision"
        super().__init__(name=name, config=config)
        self._cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self._weight_engine = DynamicWeightEngine()
        self._scenario_analyzer = ScenarioAnalyzer()
        self._logger = get_logger('agent.decision')

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行决策主逻辑

        Args:
            context: 执行上下文，包含：
                - quant_signal: QuantAgent 结果（AgentResult）
                - fundamental_result: FundamentalAgent 结果（AgentResult）
                - price_behavior_result: PriceBehaviorAgent 结果（AgentResult）
                - backtest_result: BacktestAgent 结果（AgentResult）
                - target: 标的代码（如 'RB'）
                - current_price: 当前价格（float）
                - volatility: 当前波动率（可选）

        Returns:
            AgentResult
        """
        start_time = datetime.now()

        target: str = context.get('target', 'UNKNOWN')
        current_price: float = context.get('current_price', 0.0)
        volatility: float = context.get('volatility', None)

        self._logger.info(f"Starting decision for {target} at price {current_price}")

        # 1. 汇总各 Agent 结果
        quant_sig = self._extract_quant_signal(context)
        fundamental = self._extract_fundamental(context)
        price_behavior = self._extract_price_behavior(context)
        backtest = self._extract_backtest(context)

        # 2. 确定市场状态
        regime = self._determine_regime(quant_sig, fundamental, volatility)
        self._logger.info(f"Market regime: {regime}")

        # 3. 动态权重调整
        weights = self._weight_engine.get_weights(regime)
        self._logger.info(f"Agent weights: {weights.to_dict()}")

        # 4. 综合信号计算
        composite_signal = self._compute_composite_signal(
            quant_sig, fundamental, price_behavior, weights
        )
        direction = self._signal_to_direction(composite_signal)

        # 5. 置信度计算
        confidence = self._compute_confidence(
            composite_signal, backtest, quant_sig, fundamental, price_behavior
        )

        # 6. 仓位建议
        position_size = self._compute_position_size(
            confidence, backtest, composite_signal
        )

        # 7. 价格预测
        price_target = self._compute_price_target(
            current_price, composite_signal, confidence, volatility
        )

        # 8. 止损设置
        stop_loss = self._compute_stop_loss(
            current_price, direction, volatility
        )

        # 9. 情景分析
        scenarios = self._scenario_analyzer.analyze(
            current_price=current_price,
            sentiment_score=fundamental.get('sentiment_score', 0.0),
            atr=volatility or 0.0,
            market_state=price_behavior.get('market_state', 'range'),
            fundamental_score=fundamental.get('sentiment_score', 0.0),
            quant_signal=1 if direction == 'long' else (-1 if direction == 'short' else 0),
        )

        # 10. 风险识别
        risk_points = self._identify_risk_points(
            fundamental, price_behavior, backtest, regime
        )

        # 11. 策略推荐
        strategy_type = self._recommend_strategy(
            regime, direction, composite_signal
        )

        # 12. 监控变量清单
        variables_to_watch = self._build_watchlist(
            fundamental, regime
        )

        # 13. 生成决策报告
        report = DecisionReport(
            target=target,
            date=datetime.now().strftime('%Y-%m-%d'),
            direction=direction,
            confidence=confidence,
            position_size=position_size,
            price_target=price_target,
            stop_loss=stop_loss,
            entry_range=(current_price * 0.995, current_price * 1.005),
            risk_points=risk_points,
            variables_to_watch=variables_to_watch,
            strategy_type=strategy_type,
            scenario_analysis=scenarios,
        )

        # 14. 保存报告
        self._save_report(report)

        elapsed = (datetime.now() - start_time).total_seconds()

        # 构建 AgentResult
        metrics = {
            'target': target,
            'direction': direction,
            'confidence': round(confidence, 4),
            'position_size': round(position_size, 4),
            'price_target': {
                'low': round(price_target[0], 2),
                'base': round(price_target[1], 2),
                'high': round(price_target[2], 2),
            },
            'stop_loss': round(stop_loss, 2),
            'strategy_type': strategy_type,
            'market_regime': regime,
            'weights': weights.to_dict(),
            'composite_signal': round(composite_signal, 4),
            'risk_points': risk_points,
            'scenarios': scenarios,
        }

        self._logger.info(
            f"Decision for {target}: {direction} | "
            f"confidence={confidence:.2f} | position={position_size:.0%} | "
            f"target={price_target[1]:.2f} | strategy={strategy_type}"
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            metrics=metrics,
            data=pd.DataFrame([{
                'target': target,
                'direction': direction,
                'confidence': confidence,
                'position_size': position_size,
                'price_low': price_target[0],
                'price_base': price_target[1],
                'price_high': price_target[2],
                'stop_loss': stop_loss,
                'strategy': strategy_type,
                'regime': regime,
            }]),
            elapsed_seconds=elapsed,
        )

    # ---- 提取各 Agent 结果 ----

    def _extract_quant_signal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取量化 Agent 信号"""
        try:
            result = context.get('quant_signal')
            if result is None:
                return {'signal': 0.0, 'confidence': 0.5, 'direction': 0}
            if isinstance(result, AgentResult):
                metrics = result.metrics or {}
                signal = metrics.get('signal', 0.0)
                confidence = metrics.get('confidence', 0.5)
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'direction': 1 if signal > 0 else (-1 if signal < 0 else 0),
                }
            if isinstance(result, dict):
                return result
            return {'signal': 0.0, 'confidence': 0.5, 'direction': 0}
        except Exception as exc:
            self._logger.warning(f"Failed to extract quant signal: {exc}")
            return {'signal': 0.0, 'confidence': 0.5, 'direction': 0}

    def _extract_fundamental(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取基本面 Agent 结果"""
        try:
            result = context.get('fundamental_result')
            if result is None:
                return {'sentiment_score': 0.0, 'confidence': 0.5, 'time_horizon': 'medium'}
            if isinstance(result, AgentResult):
                metrics = result.metrics or {}
                return {
                    'sentiment_score': metrics.get('sentiment_score', 0.0),
                    'confidence': metrics.get('confidence', 0.5),
                    'time_horizon': metrics.get('time_horizon', 'medium'),
                    'inventory_cycle': metrics.get('inventory_cycle', 'unknown'),
                    'supply_demand': metrics.get('supply_demand', 'balanced'),
                }
            if isinstance(result, dict):
                return result
            return {'sentiment_score': 0.0, 'confidence': 0.5}
        except Exception as exc:
            self._logger.warning(f"Failed to extract fundamental: {exc}")
            return {'sentiment_score': 0.0, 'confidence': 0.5}

    def _extract_price_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取价格行为 Agent 结果"""
        try:
            result = context.get('price_behavior_result')
            if result is None:
                return {'breakout_probability': 0.5, 'recommended_direction': 'hold', 'confidence': 0.5}
            if isinstance(result, AgentResult):
                metrics = result.metrics or {}
                return {
                    'breakout_probability': metrics.get('breakout_probability', 0.5),
                    'recommended_direction': metrics.get('recommended_direction', 'hold'),
                    'market_state': metrics.get('market_state', 'unknown'),
                    'confidence': metrics.get('confidence', 0.5),
                    'risk_ratio': metrics.get('risk_ratio', 1.0),
                }
            if isinstance(result, dict):
                return result
            return {'breakout_probability': 0.5, 'recommended_direction': 'hold', 'confidence': 0.5}
        except Exception as exc:
            self._logger.warning(f"Failed to extract price behavior: {exc}")
            return {'breakout_probability': 0.5, 'recommended_direction': 'hold', 'confidence': 0.5}

    def _extract_backtest(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取回测 Agent 结果"""
        try:
            result = context.get('backtest_result')
            if result is None:
                return {'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.5}
            if isinstance(result, AgentResult):
                metrics = result.metrics or {}
                return {
                    'sharpe': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'win_rate': metrics.get('win_rate', 0.5),
                    'total_return': metrics.get('total_return', 0.0),
                    'n_trades': metrics.get('n_trades', 0),
                }
            if isinstance(result, dict):
                return result
            return {'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.5}
        except Exception as exc:
            self._logger.warning(f"Failed to extract backtest: {exc}")
            return {'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.5}

    # ---- 市场状态判断 ----

    def _determine_regime(
        self,
        quant_sig: Dict[str, Any],
        fundamental: Dict[str, Any],
        volatility: Optional[float],
    ) -> str:
        """判断市场状态"""
        return self._weight_engine.determine_regime(
            adx=None,
            volatility=volatility,
            trend_direction=quant_sig.get('direction', 0),
        )

    # ---- 综合信号计算 ----

    def _compute_composite_signal(
        self,
        quant_sig: Dict[str, Any],
        fundamental: Dict[str, Any],
        price_behavior: Dict[str, Any],
        weights,
    ) -> float:
        """
        动态加权综合信号

        signal ∈ [-1, 1]，正值=多头，负值=空头
        """
        # 量化信号（归一化到 [-1, 1]）
        quant_raw = quant_sig.get('signal', 0.0)
        quant_conf = quant_sig.get('confidence', 0.5)
        quant_signal = self._normalize_signal(quant_raw) * quant_conf

        # 基本面信号（归一化到 [-1, 1]，sentiment ∈ [-5, 5]）
        sentiment = fundamental.get('sentiment_score', 0.0)
        fund_conf = fundamental.get('confidence', 0.5)
        fund_signal = (sentiment / 5.0) * fund_conf  # 归一化

        # 价格行为信号
        behavior_prob = price_behavior.get('breakout_probability', 0.5)
        behavior_dir = self._direction_to_sign(
            price_behavior.get('recommended_direction', 'hold')
        )
        behavior_conf = price_behavior.get('confidence', 0.5)
        behavior_signal = behavior_dir * (behavior_prob - 0.5) * 2 * behavior_conf

        # 加权综合
        composite = (
            quant_signal * weights.quant
            + fund_signal * weights.fundamental
            + behavior_signal * weights.price_behavior
        )

        return max(-1.0, min(1.0, composite))

    def _normalize_signal(self, raw: float) -> float:
        """归一化原始信号到 [-1, 1]"""
        if raw > 0.5:
            return 1.0
        elif raw < -0.5:
            return -1.0
        else:
            return raw * 2  # 缩放 [-0.5, 0.5] 到 [-1, 1]

    def _direction_to_sign(self, direction: str) -> int:
        """方向文字转符号"""
        mapping = {'buy': 1, 'long': 1, 'sell': -1, 'short': -1, 'hold': 0}
        return mapping.get(direction.lower(), 0)

    def _signal_to_direction(self, composite: float) -> str:
        """综合信号转方向文字"""
        threshold = 0.15
        if composite > threshold:
            return 'long'
        elif composite < -threshold:
            return 'short'
        else:
            return 'neutral'

    # ---- 置信度和仓位 ----

    def _compute_confidence(
        self,
        composite: float,
        backtest: Dict[str, Any],
        quant_sig: Dict[str, Any],
        fundamental: Dict[str, Any],
        price_behavior: Dict[str, Any],
    ) -> float:
        """
        计算综合置信度

        综合考虑：
        - 信号强度（|composite| 越大，置信度越高）
        - 回测绩效（Sharpe 越高，置信度越高）
        - 各 Agent 置信度一致性
        """
        # 信号强度贡献（|composite| ∈ [0, 1] → contribution ∈ [0, 0.4]）
        signal_contrib = abs(composite) * 0.4

        # 回测绩效贡献（Sharpe ∈ [-2, 3] → contribution ∈ [0, 0.25]）
        sharpe = backtest.get('sharpe', 0.0)
        sharpe_contrib = max(0.0, min(0.25, (sharpe + 0.5) / 14.0))

        # 一致性贡献（各 Agent 方向是否一致）
        directions = [
            quant_sig.get('direction', 0),
            1 if fundamental.get('sentiment_score', 0) > 0.5 else (-1 if fundamental.get('sentiment_score', 0) < -0.5 else 0),
            self._direction_to_sign(price_behavior.get('recommended_direction', 'hold')),
        ]
        # 计算方向一致性（所有方向相同 = 1.0，存在分歧 = 降低）
        signs = [d for d in directions if d != 0]
        if not signs:
            consistency = 0.5
        else:
            pos_count = sum(1 for d in signs if d > 0)
            neg_count = sum(1 for d in signs if d < 0)
            consistency = max(pos_count, neg_count) / len(signs)

        consistency_contrib = consistency * 0.2

        # Agent 置信度均值贡献
        confs = [
            quant_sig.get('confidence', 0.5),
            fundamental.get('confidence', 0.5),
            price_behavior.get('confidence', 0.5),
        ]
        avg_conf = sum(confs) / len(confs)
        conf_contrib = avg_conf * 0.15

        confidence = signal_contrib + sharpe_contrib + consistency_contrib + conf_contrib
        return max(0.1, min(0.95, confidence))

    def _compute_position_size(
        self,
        confidence: float,
        backtest: Dict[str, Any],
        composite: float,
    ) -> float:
        """
        建议仓位

        基础仓位 = 置信度
        回测调整：Sharpe > 1 → 可加仓；Sharpe < 0 → 降仓
        """
        base = confidence

        sharpe = backtest.get('sharpe', 0.0)
        if sharpe > 1.5:
            size = min(1.0, base * 1.2)
        elif sharpe > 1.0:
            size = min(1.0, base * 1.1)
        elif sharpe > 0.5:
            size = base
        elif sharpe > 0:
            size = base * 0.8
        else:
            size = base * 0.5

        # 信号强度调整
        size *= (0.5 + abs(composite))

        return max(0.0, min(1.0, size))

    # ---- 价格预测 ----

    def _compute_price_target(
        self,
        current_price: float,
        composite: float,
        confidence: float,
        volatility: Optional[float],
    ) -> tuple:
        """
        价格区间预测

        Returns:
            (pessimistic, base, optimistic)
        """
        if volatility is None:
            volatility = current_price * 0.02  # 默认 2% 波动率

        # 基准预测方向（composite ∈ [-1, 1]）
        # confidence ∈ [0.1, 0.95]，放大基础移动幅度
        base_move = abs(composite) * confidence * volatility * 5

        if composite > 0:
            # 多头情景
            base = current_price * (1 + base_move / current_price)
            optimistic = current_price * (1 + base_move / current_price * 1.5)
            pessimistic = current_price * (1 - volatility * 2 / current_price)
        elif composite < 0:
            # 空头情景
            base = current_price * (1 - base_move / current_price)
            pessimistic = current_price * (1 - base_move / current_price * 1.5)
            optimistic = current_price * (1 + volatility * 2 / current_price)
        else:
            # 中性情景（震荡）
            base = current_price
            optimistic = current_price * (1 + volatility / current_price)
            pessimistic = current_price * (1 - volatility / current_price)

        return (round(pessimistic, 2), round(base, 2), round(optimistic, 2))

    def _compute_stop_loss(
        self,
        current_price: float,
        direction: str,
        volatility: Optional[float],
    ) -> float:
        """计算止损位"""
        if volatility is None:
            volatility = current_price * 0.02

        if direction == 'long':
            # 多头止损：价格下跌 2 ATR
            return round(current_price - volatility * 2, 2)
        elif direction == 'short':
            # 空头止损：价格上浮 2 ATR
            return round(current_price + volatility * 2, 2)
        else:
            return 0.0

    # ---- 风险识别 ----

    def _identify_risk_points(
        self,
        fundamental: Dict[str, Any],
        price_behavior: Dict[str, Any],
        backtest: Dict[str, Any],
        regime: str,
    ) -> List[Dict[str, str]]:
        """识别主要风险点"""
        risks = []

        # 政策/宏观风险（基本面 Agent 提供）
        sentiment = fundamental.get('sentiment_score', 0.0)
        if abs(sentiment) > 3.0:
            risks.append({
                'risk': f'基本面极端信号（{sentiment:.1f}分），注意反转风险',
                'level': 'high',
            })

        # 模型风险
        sharpe = backtest.get('sharpe', 0.0)
        if sharpe < 0.3:
            risks.append({
                'risk': f'回测 Sharpe 过低（{sharpe:.2f}），模型可能存在过拟合',
                'level': 'medium',
            })

        # 最大回撤风险
        mdd = backtest.get('max_drawdown', 0.0)
        if mdd > 0.15:
            risks.append({
                'risk': f'历史最大回撤较大（{mdd:.1%}），需关注极端行情',
                'level': 'medium',
            })

        # 市场状态风险
        if regime == "high_volatility":
            risks.append({
                'risk': '当前高波动环境，止损需收紧',
                'level': 'medium',
            })

        # 流动性风险（根据交易次数推断）
        n_trades = backtest.get('n_trades', 0)
        if n_trades < 5:
            risks.append({
                'risk': '历史交易次数过少，策略样本不足',
                'level': 'low',
            })

        if not risks:
            risks.append({
                'risk': '当前未发现明显风险点',
                'level': 'low',
            })

        return risks

    # ---- 策略推荐 ----

    def _recommend_strategy(
        self,
        regime: str,
        direction: str,
        composite: float,
    ) -> str:
        """推荐策略类型"""
        if direction == 'neutral':
            return '观望'

        if regime == "trending":
            if composite > 0.3:
                return '趋势跟踪'
            else:
                return '均值回归'
        elif regime == "range":
            return '均值回归'
        elif regime == "high_volatility":
            return '短线交易 + 严格止损'
        else:
            return '趋势跟踪'

    # ---- 监控变量清单 ----

    def _build_watchlist(
        self,
        fundamental: Dict[str, Any],
        regime: str,
    ) -> List[str]:
        """构建需要监控的变量清单"""
        watchlist = [
            '每日库存数据（我的钢铁网/上海有色）',
            '期货交易所仓单日报',
            '主力合约持仓量变化',
            '美元指数走势',
        ]

        # 基本面特有监控项
        horizon = fundamental.get('time_horizon', 'medium')
        if horizon == 'short':
            watchlist.append('每日新闻事件')
            watchlist.append('持仓龙虎榜')
        elif horizon == 'long':
            watchlist.append('宏观经济数据（CPI/PPI/PMI）')
            watchlist.append('产业政策变化')

        inventory_cycle = fundamental.get('inventory_cycle', 'unknown')
        watchlist.append(f'库存周期：{inventory_cycle}')

        # 高波动环境额外监控
        if regime == "high_volatility":
            watchlist.append('隔夜外盘商品走势')
            watchlist.append('地缘政治事件')

        return watchlist

    # ---- 报告保存 ----

    def _save_report(self, report: DecisionReport) -> Path:
        """保存决策报告到 Markdown 文件"""
        report_dir = Path(self._cfg['report_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"decision_{report.target}_{date_str}.md"
        filepath = report_dir / filename

        # 生成 Markdown 报告
        md = self._render_markdown(report)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)

        self._logger.info(f"Decision report saved: {filepath}")
        return filepath

    def _render_markdown(self, report: DecisionReport) -> str:
        """渲染 Markdown 报告"""
        direction_emoji = {'long': '🟢', 'short': '🔴', 'neutral': '⚪'}
        emoji = direction_emoji.get(report.direction, '⚪')

        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}

        lines = [
            f"# 决策报告 - {report.target}",
            f"",
            f"**日期**: {report.date}",
            f"**生成时间**: {datetime.now().strftime('%H:%M:%S')}",
            f"",
            f"## 📊 综合判断",
            f"",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 方向 | {emoji} **{report.direction.upper()}** |",
            f"| 置信度 | {report.confidence:.1%} |",
            f"| 建议仓位 | {report.position_size:.0%} |",
            f"| 策略类型 | {report.strategy_type} |",
            f"",
            f"## 💰 价格预测",
            f"",
            f"| 情景 | 价格 | 说明 |",
            f"|------|------|------|",
        ]

        if report.scenario_analysis:
            for label, scenario in report.scenario_analysis.items():
                if isinstance(scenario, dict):
                    prob = scenario.get('probability', 0)
                    price = scenario.get('price_range', '-')
                    trigger = scenario.get('trigger_condition', '-')
                else:
                    prob = 0.0
                    price = str(scenario) if scenario else '-'
                    trigger = '-'
                lines.append(f"| {label} | {price} | 概率 {prob:.0%} | 触发条件: {trigger} |")
        else:
            pl, pb, ph = report.price_target
            lines.append(f"| 悲观 | {pl:.2f} | 可能下跌 |")
            lines.append(f"| 基准 | {pb:.2f} | 核心预测 |")
            lines.append(f"| 乐观 | {ph:.2f} | 可能上涨 |")

        lines += [
            f"",
            f"## 🎯 交易计划",
            f"",
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| 当前价格 | 参考实时行情 |",
            f"| 建议入场区间 | {report.entry_range[0]:.2f} ~ {report.entry_range[1]:.2f} |",
            f"| 止损位 | {report.stop_loss:.2f} |",
            f"",
            f"## ⚠️ 风险点",
            f"",
        ]

        for risk in report.risk_points:
            lvl = risk.get('level', 'low')
            lines.append(f"- {risk_emoji.get(lvl, '⚪')} **{risk['risk']}** (等级: {lvl})")

        lines += [
            f"",
            f"## 👀 需监控变量",
            f"",
        ]
        for var in report.variables_to_watch:
            lines.append(f"- {var}")

        lines += [
            f"",
            f"---",
            f"*本报告由 DecisionAgent 自动生成，仅供参考，不构成投资建议。*",
        ]

        return '\n'.join(lines)
