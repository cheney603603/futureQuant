"""
因子研究节点目录 (Factor Research Node Catalog)

为 LLM 提供因子研究流水线中每个步骤的完整元数据，
包含配置格式、输入输出结构、常见错误等信息。

参考 quant_react_interview-main/agent/catalog.py 的设计，
但针对量化因子研究场景进行了定制。

目录中的每个步骤代表因子研究流水线中的一个节点，
LLM 可以通过 get_catalog() 和 get_details() 查询元数据，
然后通过 PipelineBuilder 构建研究流水线。

使用示例:
    from futureQuant.engine.nodes import get_catalog, get_details

    # 列出所有可用步骤
    catalog = get_catalog()

    # 查看特定步骤的详细信息
    details = get_details("data.price_bars")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepDescriptor:
    """
    步骤描述符 - 包含构建因子研究流水线所需的所有元数据。

    Attributes:
        kind: 唯一标识符，如 "data.price_bars"
        purpose: 简要说明步骤的作用
        category: 分组标签 (trigger, data, factor, evaluation, fusion, backtest, output)
        required_fields: 必须在 config 中提供的字段
        optional_fields: 有默认值的字段，可省略
        field_descriptions: 每个字段的详细说明（不只是类型，还有语义）
        sample: 典型配置示例
        output_shape: 输出的数据结构描述，以及如何访问（如 $step_id['data']）
        notes: 使用注意事项和最佳实践
        common_mistakes: LLM 容易犯的错误，提前告知避免
    """

    kind: str
    purpose: str
    category: str
    required_fields: List[str]
    optional_fields: Dict[str, Any]
    field_descriptions: Dict[str, str]
    sample: Dict[str, Any]
    output_shape: str
    notes: List[str]
    common_mistakes: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """返回用于目录列表的简短摘要。"""
        return {
            "kind": self.kind,
            "category": self.category,
            "purpose": self.purpose,
            "required_fields": list(self.required_fields),
            "optional_fields": dict(self.optional_fields),
            "example_config": dict(self.sample),
        }

    def details(self) -> Dict[str, Any]:
        """返回步骤的完整详细信息。"""
        payload = self.summary()
        payload.update(
            {
                "field_descriptions": dict(self.field_descriptions),
                "output_shape": self.output_shape,
                "notes": list(self.notes),
            }
        )
        if self.common_mistakes:
            payload["common_mistakes"] = list(self.common_mistakes)
        return payload


# =============================================================================
# 步骤描述符定义
# =============================================================================

_DESCRIPTORS: List[StepDescriptor] = [
    # -------------------------------------------------------------------------
    # Trigger 节点 - 流水线起点
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="trigger.manual",
        purpose="定义研究目标，初始化流水线的输入参数（品种、时间范围等）。",
        category="trigger",
        required_fields=[],
        optional_fields={
            "target": "RB",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "frequency": "daily",
            "universe": ["RB", "HC", "I"],
        },
        field_descriptions={
            "target": "主要研究品种（期货合约代码，如 RB/HC/I/AL/CU 等）。",
            "start_date": "数据开始日期，格式 YYYY-MM-DD。",
            "end_date": "数据结束日期，格式 YYYY-MM-DD。",
            "frequency": "数据频率，支持 daily/5min/15min。",
            "universe": "品种列表，用于多品种因子研究。",
        },
        sample={
            "target": "RB",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "frequency": "daily",
            "universe": ["RB", "HC", "I"],
        },
        output_shape=(
            "透传配置值。使用 $step_id['target'] 访问品种，"
            "$step_id['universe'] 访问品种列表。"
        ),
        notes=[
            "通常作为流水线的第一个步骤。",
            "无执行逻辑，仅透传配置。",
            "trigger.manual 是触发器类型，其他触发器类型暂未实现。",
        ],
        common_mistakes=[
            "忘记指定日期范围，导致数据量过大或过小。",
            "universe 和 target 混淆：target 是主要品种，universe 是多品种分析时的完整列表。",
        ],
    ),

    # -------------------------------------------------------------------------
    # Data 节点 - 数据获取
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="data.price_bars",
        purpose="从数据库或 API 获取 OHLCV 价格数据（K线数据）。",
        category="data",
        required_fields=["target", "start_date", "end_date"],
        optional_fields={
            "frequency": "daily",
            "lookback_days": 60,
            "source": "database",
        },
        field_descriptions={
            "target": "品种代码，可以是单个品种字符串，也可以是品种列表。",
            "start_date": "数据开始日期，格式 YYYY-MM-DD。",
            "end_date": "数据结束日期，格式 YYYY-MM-DD。",
            "frequency": "K线频率：daily/5min/15min/1h。",
            "lookback_days": "向前追溯的天数（交易日），用于计算指标。",
            "source": "数据来源：database（默认，从 MySQL/SQLite 获取）或 fallback（从 akshare 获取）。",
        },
        sample={
            "target": "$trigger['target']",
            "start_date": "$trigger['start_date']",
            "end_date": "$trigger['end_date']",
            "frequency": "daily",
            "lookback_days": 60,
            "source": "database",
        },
        output_shape=(
            "返回 pandas DataFrame，包含列: date, open, high, low, close, volume。"
            "访问方式: $step_id['data'] 或 $step_id['df']。"
            "多品种时返回 dict: {品种: DataFrame}。"
        ),
        notes=[
            "数据质量直接影响因子质量，优先使用 database 来源。",
            "返回的 DataFrame 索引为 date（datetime 类型）。",
            "如果数据不足，会返回 error 信息，此时应检查日期范围和数据源。",
        ],
        common_mistakes=[
            "传入 target 时忘了加 $trigger['target']，直接写字符串。",
            "lookback_days 设置过小，导致指标计算（如均线）时数据不足。",
            "期望返回 dict 但实际返回 DataFrame，注意 output_shape 说明。",
        ],
    ),

    StepDescriptor(
        kind="data.fundamental",
        purpose="获取基本面数据（基差、库存、仓单、持仓量等）。",
        category="data",
        required_fields=["target", "start_date", "end_date"],
        optional_fields={
            "data_type": "basis",  # basis | inventory | warehouse_receipt | position
            "source": "database",
        },
        field_descriptions={
            "target": "品种代码。",
            "start_date": "数据开始日期。",
            "end_date": "数据结束日期。",
            "data_type": "基本面数据类型：basis(基差)/inventory(库存)/warehouse_receipt(仓单)/position(持仓结构)。",
            "source": "数据来源：database（默认）或 fallback。",
        },
        sample={
            "target": "$trigger['target']",
            "start_date": "$trigger['start_date']",
            "end_date": "$trigger['end_date']",
            "data_type": "basis",
            "source": "database",
        },
        output_shape=(
            "返回 dict，键为品种代码，值为 DataFrame。"
            "DataFrame 包含 date 列和其他基本面指标列（如 basis_spot/basis_future）。"
            "访问方式: $step_id['data'] 或 $step_id['df']。"
        ),
        notes=[
            "基本面数据通常频率低于价格数据（周频或月频），与日频数据对齐时会自动 forward-fill。",
            "basis（基差）是最常用的基本面因子来源。",
        ],
        common_mistakes=[
            "基本面数据的 date 频率与价格数据不同，直接合并会导致 NaN。",
            "data_type 拼写错误，写成 date_type。",
        ],
    ),

    # -------------------------------------------------------------------------
    # Factor 节点 - 因子计算
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="factor.technical",
        purpose="计算技术指标因子（动量、波动率、成交量、RSI 等）。",
        category="factor",
        required_fields=["data"],
        optional_fields={
            "indicators": ["momentum", "volatility", "volume"],
            "windows": {"momentum": [5, 10, 20], "volatility": [10, 20], "volume": [5, 10]},
        },
        field_descriptions={
            "data": "引用价格数据，必须使用 $price_bars_step['data'] 格式。",
            "indicators": "要计算的指标列表：momentum（动量）/ volatility（波动率）/ volume（成交量）/ rsi（RSI）/ macd（MACD）/ boll（布林带）。",
            "windows": "各指标的窗口参数，格式为 dict。momentum_windows: [5, 10, 20] 表示计算5/10/20日动量。",
        },
        sample={
            "data": "$price_bars['data']",
            "indicators": ["momentum", "volatility", "rsi"],
            "windows": {
                "momentum": [5, 10, 20, 60],
                "volatility": [10, 20, 60],
                "rsi": [6, 14, 21],
            },
        },
        output_shape=(
            "返回 dict，键为因子名称（如 'momentum_5', 'volatility_20', 'rsi_14'），"
            "值为 pandas Series。"
            "访问方式: $step_id['factors'] 或 $step_id['data']。"
        ),
        notes=[
            "技术因子是最常见的因子来源，通常能快速计算大量候选因子。",
            "indicators 和 windows 可以灵活组合，建议先跑宽窗口再精选。",
        ],
        common_mistakes=[
            "传入 $price_bars 而非 $price_bars['data']，忽略了 ['data'] 后缀。",
            "windows 格式错误，写成 windows: [5, 10, 20] 而非嵌套 dict。",
            "indicators 包含未实现的指标类型（如写成 'sma' 而非 'momentum'）。",
        ],
    ),

    StepDescriptor(
        kind="factor.fundamental",
        purpose="计算基本面因子（基差率、库存变化率、仓单比例等）。",
        category="factor",
        required_fields=["fundamental_data", "price_data"],
        optional_fields={
            "factor_types": ["basis_ratio", "inventory_change", "receipt_ratio"],
        },
        field_descriptions={
            "fundamental_data": "引用基本面数据 $fundamental_step['data']。",
            "price_data": "引用价格数据 $price_bars['data']，用于计算基差率等需要价格的比例因子。",
            "factor_types": "因子类型列表：basis_ratio（基差率）/ inventory_change（库存变化率）/ receipt_ratio（仓单比例）。",
        },
        sample={
            "fundamental_data": "$fundamental['data']",
            "price_data": "$price_bars['data']",
            "factor_types": ["basis_ratio", "inventory_change"],
        },
        output_shape=(
            "返回 dict，键为因子名称（如 'basis_ratio', 'inventory_change_3d'），"
            "值为 pandas Series。"
            "访问方式: $step_id['factors']。"
        ),
        notes=[
            "基本面因子通常与价格因子互补，组合使用效果更好。",
            "基差率 = (现货价 - 期货价) / 期货价，反映基差偏离程度。",
        ],
        common_mistakes=[
            "缺少 price_data 引用，导致基差率等需要价格的比例因子无法计算。",
            "fundamental_data 和 price_data 频率不同（基本面数据可能低频），直接运算会产生 NaN。",
        ],
    ),

    StepDescriptor(
        kind="factor.alpha101",
        purpose="计算 WorldQuant Alpha101 公式定义的因子。",
        category="factor",
        required_fields=["data"],
        optional_fields={
            "formulas": None,  # None 表示使用默认公式集
            "top_n": 20,
        },
        field_descriptions={
            "data": "引用价格数据 $price_bars['data']。",
            "formulas": "Alpha101 公式列表，为 None 时使用内置的常用公式子集。",
            "top_n": "从所有 Alpha101 因子中选取 IC 最高的 top_n 个。",
        },
        sample={
            "data": "$price_bars['data']",
            "formulas": None,
            "top_n": 20,
        },
        output_shape=(
            "返回 DataFrame，列为因子名（如 'Alpha001', 'Alpha010'），"
            "行为日期索引。"
            "访问方式: $step_id['data'] 或 $step_id['factors']。"
        ),
        notes=[
            "Alpha101 是业界经典因子库，包含 101 个公式。",
            "由于数据源限制（A股数据列名与原始公式不完全匹配），部分公式可能无法直接使用。",
        ],
        common_mistakes=[
            "Alpha101 因子在 A 股数据的适用性有限，部分公式结果可能不符合预期。",
            "期望返回 dict 但实际返回 DataFrame。",
        ],
    ),

    # -------------------------------------------------------------------------
    # Evaluation 节点 - 因子评估
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="evaluation.ic",
        purpose="计算因子的 IC（信息系数）和 ICIR（IC_IR 比率）。",
        category="evaluation",
        required_fields=["factors", "returns"],
        optional_fields={
            "method": "spearman",
            "ic_threshold": 0.02,
            "icir_threshold": 0.3,
        },
        field_descriptions={
            "factors": "引用因子数据，dict 格式: $technical['factors'] 或 DataFrame: $alpha101['data']。",
            "returns": "引用收益率序列，通常使用 $price_bars['data']['close'].pct_change().shift(-1)。",
            "method": "IC 计算方法：spearman（秩相关，默认）/ pearson（皮尔逊相关）。",
            "ic_threshold": "IC 绝对值的最低阈值，低于此值的因子会被丢弃。",
            "icir_threshold": "ICIR 的最低阈值，低于此值的因子会被丢弃。",
        },
        sample={
            "factors": "$technical['factors']",
            "returns": "$returns",
            "method": "spearman",
            "ic_threshold": 0.02,
            "icir_threshold": 0.3,
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - ic_series: dict {因子名: IC值}，已按 IC 绝对值排序\n"
            "  - icir_dict: dict {因子名: ICIR值}\n"
            "  - passed_factors: list，通过 IC 和 ICIR 筛选的因子名列表\n"
            "访问方式: $step_id['ic_series'], $step_id['passed_factors']。"
        ),
        notes=[
            "IC 是因子预测能力的核心指标，|IC| >= 0.02 通常被认为有显著预测能力。",
            "ICIR = IC均值 / IC标准差，反映因子预测能力的稳定性。",
            "method 推荐使用 spearman（秩相关），对非线性关系更鲁棒。",
        ],
        common_mistakes=[
            "factors 和 returns 的索引（日期）对不上，直接运算产生大量 NaN。",
            "使用 pearson 而非 spearman，导致对非线性关系的检测不足。",
            "ic_threshold 设得太高（如 0.05），导致大多数因子被丢弃。",
        ],
    ),

    StepDescriptor(
        kind="evaluation.robustness",
        purpose="评估因子稳健性（分层回测、换手率、衰减分析）。",
        category="evaluation",
        required_fields=["factors", "returns"],
        optional_fields={
            "n_groups": 5,
            "turnover_window": 20,
        },
        field_descriptions={
            "factors": "引用因子数据（通过 IC 筛选后的因子）。",
            "returns": "引用收益率序列。",
            "n_groups": "分层回测的组数，默认 5 组（按因子值排序分为 5 档）。",
            "turnover_window": "计算换手率的窗口天数。",
        },
        sample={
            "factors": "$ic_eval['passed_factors']",
            "returns": "$returns",
            "n_groups": 5,
            "turnover_window": 20,
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - group_returns: 各组收益率 dict\n"
            "  - long_short_return: 多空组合收益\n"
            "  - turnover: 各因子换手率 dict\n"
            "  - decay_analysis: 因子预测周期衰减 dict。"
        ),
        notes=[
            "稳健性评估是因子质量的重要保障，不能仅看 IC。",
            "分层回测中，第1组（因子值最高）和第5组（因子值最低）的收益差越大，因子越有效。",
            "高换手率意味着高交易成本，需要在收益和成本间权衡。",
        ],
        common_mistakes=[
            "直接传入原始因子而非通过 IC 筛选后的因子，导致评估量过大。",
            "n_groups 设得太大（如 10），每组样本不足。",
        ],
    ),

    # -------------------------------------------------------------------------
    # Fusion 节点 - 因子融合
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="fusion.icir_weight",
        purpose="基于 ICIR 加权合成多个有效因子。",
        category="fusion",
        required_fields=["factors", "ic_series"],
        optional_fields={
            "corr_threshold": 0.8,
            "min_icir": 0.3,
            "method": "icir_weighted",
        },
        field_descriptions={
            "factors": "引用因子数据（DataFrame，多列）。",
            "ic_series": "引用 IC 评估结果 $ic_eval['ic_series']。",
            "corr_threshold": "相关性阈值，超过此值的因子对会被合并或降权（避免冗余）。",
            "min_icir": "因子进入合成的最低 ICIR 要求。",
            "method": "合成方法：icir_weighted（ICIR 加权）/ equal（等权）/ corr_adjusted（相关性调整）。",
        },
        sample={
            "factors": "$technical['factors']",
            "ic_series": "$ic_eval['ic_series']",
            "corr_threshold": 0.8,
            "min_icir": 0.3,
            "method": "icir_weighted",
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - composite_factor: 合成后的因子 Series\n"
            "  - weights: 各因子权重 dict\n"
            "  - eliminated: 因相关性被剔除的因子列表。"
        ),
        notes=[
            "因子融合的关键是避免高度相关的因子重复计入。",
            "ICIR 加权是最直观的方法，但需要注意极端 ICIR 导致的权重集中。",
        ],
        common_mistakes=[
            "传入 dict 格式的因子而非 DataFrame 格式。",
            "忽略 corr_threshold，导致合成果汁中包含大量冗余信息。",
        ],
    ),

    StepDescriptor(
        kind="fusion.multifactor",
        purpose="多因子综合评分（Z-score 标准化 + 等权/IC 加权合成）。",
        category="fusion",
        required_fields=["factors"],
        optional_fields={
            "normalize": "zscore",
            "weights": None,
        },
        field_descriptions={
            "factors": "引用因子数据（DataFrame）。",
            "normalize": "标准化方法：zscore（Z-score 标准化）/ rank（百分位排名）。",
            "weights": "权重列表，与因子列一一对应。None 表示等权。",
        },
        sample={
            "factors": "$technical['factors']",
            "normalize": "zscore",
            "weights": None,
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - composite: 合成后的因子 Series\n"
            "  - factor_loadings: 各因子在合成中的权重（标准化后的）。"
        ),
        notes=[
            "多因子合成的核心是处理量纲差异（Z-score 或 Rank）。",
            "权重可以用 IC、ICIR 或机器学习确定。",
        ],
        common_mistakes=[
            "因子量纲差异大但不使用 normalize，直接相加导致量纲大的因子主导。",
        ],
    ),

    # -------------------------------------------------------------------------
    # Backtest 节点 - 回测验证
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="backtest.factor_signal",
        purpose="将合成因子转换为交易信号并进行回测。",
        category="backtest",
        required_fields=["factor", "price_data"],
        optional_fields={
            "signal_threshold": 1.0,
            "position_sizing": "equal",
            "cost_rate": 0.0003,
        },
        field_descriptions={
            "factor": "引用合成因子 $fusion['composite_factor'] 或单一因子。",
            "price_data": "引用价格数据 $price_bars['data']。",
            "signal_threshold": "开仓信号阈值（Z-score），超过此值才开仓。",
            "position_sizing": "仓位分配方式：equal（等权）/ ic_weighted（IC 加权）/ vol_adjusted（波动率调整）。",
            "cost_rate": "双边交易成本率（含滑点），默认 0.03%。",
        },
        sample={
            "factor": "$fusion['composite_factor']",
            "price_data": "$price_bars['data']",
            "signal_threshold": 1.0,
            "position_sizing": "equal",
            "cost_rate": 0.0003,
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - returns: 策略每日收益率 Series\n"
            "  - signals: 每日交易信号 Series\n"
            "  - positions: 每日持仓 Series\n"
            "  - equity_curve: 权益曲线。"
        ),
        notes=[
            "信号生成策略：因子值 > signal_threshold -> 做多，< -signal_threshold -> 做空，否则空仓。",
            "cost_rate 建议包含滑点，期货通常设 0.0003（万三）。",
        ],
        common_mistakes=[
            "signal_threshold 设得太高（如 2.0），导致交易信号稀少。",
            "忽略 cost_rate，导致回测收益虚高。",
            "仓位计算时没有考虑合约乘数（但 BacktestEngine 会自动处理）。",
        ],
    ),

    StepDescriptor(
        kind="backtest.walk_forward",
        purpose="Walk-Forward 分析（滚动窗口回测，检验因子稳健性）。",
        category="backtest",
        required_fields=["factor", "price_data"],
        optional_fields={
            "train_window": 250,
            "test_window": 60,
            "step": 20,
        },
        field_descriptions={
            "factor": "引用因子。",
            "price_data": "引用价格数据。",
            "train_window": "训练窗口（天数），用于计算因子权重/参数。",
            "test_window": "测试窗口（天数），在训练窗口外滚动。",
            "step": "滚动步长（天数）。",
        },
        sample={
            "factor": "$fusion['composite_factor']",
            "price_data": "$price_bars['data']",
            "train_window": 250,
            "test_window": 60,
            "step": 20,
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - window_results: list of dict，每窗口的结果\n"
            "  - mean_return: 平均收益\n"
            "  - win_rate: 盈利窗口占比。"
        ),
        notes=[
            "Walk-Forward 是因子实际应用中最重要的稳健性检验。",
            "如果 Walk-Forward 收益不稳定，即使 IC 高也不建议实盘。",
        ],
        common_mistakes=[
            "train_window 设得太小（如 60），因子参数估计不稳定。",
        ],
    ),

    # -------------------------------------------------------------------------
    # Output 节点 - 报告生成
    # -------------------------------------------------------------------------
    StepDescriptor(
        kind="output.report",
        purpose="生成因子研究报告（Markdown 格式）。",
        category="output",
        required_fields=["top_factors", "ic_results"],
        optional_fields={
            "report_dir": "docs/reports",
            "include_charts": False,
        },
        field_descriptions={
            "top_factors": "引用通过筛选的 Top 因子列表 $ic_eval['passed_factors']。",
            "ic_results": "引用 IC 评估结果 $ic_eval['ic_series']。",
            "report_dir": "报告保存目录。",
            "include_charts": "是否包含图表（目前不支持，留作扩展）。",
        },
        sample={
            "top_factors": "$ic_eval['passed_factors']",
            "ic_results": "$ic_eval['ic_series']",
            "report_dir": "docs/reports",
            "include_charts": False,
        },
        output_shape=(
            "返回 dict，包含:\n"
            "  - report_path: 报告文件路径\n"
            "  - summary: 摘要 dict。"
        ),
        notes=[
            "报告是因子研究的重要交付物，记录了研究过程和结论。",
            "报告路径格式: {report_dir}/factor_mining_{date}.md。",
        ],
        common_mistakes=[
            "传入整个 AgentResult 而非具体的因子列表和 IC 结果。",
        ],
    ),
]


# =============================================================================
# 公开 API
# =============================================================================

_catalog: Optional[Dict[str, StepDescriptor]] = None


def _get_catalog() -> Dict[str, StepDescriptor]:
    """获取以 kind 为键的目录字典（惰性初始化）。"""
    global _catalog
    if _catalog is None:
        _catalog = {d.kind: d for d in _DESCRIPTORS}
    return _catalog


def get_catalog() -> List[Dict[str, Any]]:
    """
    获取所有步骤的目录摘要列表。

    Returns:
        List of step summaries, each containing:
        - kind: 步骤类型
        - category: 分组
        - purpose: 简要说明
        - required_fields: 必填字段
        - optional_fields: 可选字段
        - example_config: 配置示例

    使用示例:
        catalog = get_catalog()
        for step in catalog:
            print(f"{step['kind']}: {step['purpose']}")
    """
    return [d.summary() for d in _DESCRIPTORS]


def get_details(kind: str) -> Dict[str, Any]:
    """
    获取特定步骤的完整详细信息。

    Args:
        kind: 步骤类型，如 "data.price_bars"

    Returns:
        包含完整元数据的 dict，包括：
        - summary() 的所有字段
        - field_descriptions: 字段详细说明
        - output_shape: 输出格式描述
        - notes: 使用注意事项
        - common_mistakes: 常见错误

    如果 kind 不存在，返回包含 error 信息的 dict。

    使用示例:
        details = get_details("factor.technical")
        print(f"Output shape: {details['output_shape']}")
        print(f"Common mistakes: {details.get('common_mistakes', [])}")
    """
    catalog = _get_catalog()
    descriptor = catalog.get(kind)
    if descriptor is None:
        valid_kinds = list(catalog.keys())
        return {
            "error": f"Unknown step kind: {kind}",
            "hint": f"Valid kinds: {', '.join(sorted(valid_kinds))}",
        }
    return descriptor.details()


def get_categories() -> List[str]:
    """获取所有步骤的类别列表（去重）。"""
    return sorted(set(d.category for d in _DESCRIPTORS))


def get_steps_by_category(category: str) -> List[Dict[str, Any]]:
    """获取指定类别的所有步骤摘要。"""
    return [d.summary() for d in _DESCRIPTORS if d.category == category]


class FactorCatalog:
    """
    因子目录类，提供面向对象的目录访问接口。

    Attributes:
        catalog: 所有步骤描述符的字典（kind -> StepDescriptor）

    使用示例:
        catalog = FactorCatalog()
        technical_steps = catalog.by_category("factor")
        details = catalog.details("data.price_bars")
    """

    def __init__(self):
        self._catalog = _get_catalog()

    def by_kind(self, kind: str) -> Optional[StepDescriptor]:
        """根据 kind 获取描述符。"""
        return self._catalog.get(kind)

    def by_category(self, category: str) -> List[StepDescriptor]:
        """获取指定类别的所有描述符。"""
        return [d for d in _DESCRIPTORS if d.category == category]

    def list_kinds(self) -> List[str]:
        """列出所有步骤类型。"""
        return sorted(self._catalog.keys())

    def validate_config(self, kind: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置是否符合步骤要求。

        Args:
            kind: 步骤类型
            config: 待验证的配置字典

        Returns:
            dict，包含:
            - valid: bool，是否有效
            - errors: list，错误信息列表
            - warnings: list，警告信息列表（如缺少可选字段）
        """
        descriptor = self._catalog.get(kind)
        if descriptor is None:
            return {
                "valid": False,
                "errors": [f"Unknown kind: {kind}"],
                "warnings": [],
            }

        errors = []
        warnings = []

        # 检查必填字段
        for field_name in descriptor.required_fields:
            if field_name not in config:
                errors.append(f"Missing required field: {field_name}")

        # 检查未知字段
        all_fields = set(descriptor.required_fields) | set(descriptor.optional_fields.keys())
        for key in config:
            if key not in all_fields:
                warnings.append(f"Unknown field: {key} (will be ignored)")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def suggest_sample(self, kind: str) -> Dict[str, Any]:
        """获取步骤的配置示例（用于 LLM 快速构建配置）。"""
        descriptor = self._catalog.get(kind)
        if descriptor is None:
            return {}
        return dict(descriptor.sample)
