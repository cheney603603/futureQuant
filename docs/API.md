# futureQuant API 参考文档

> 本文档涵盖 futureQuant 框架所有公开 API，按模块组织。

---

## 目录

1. [Core 模块](#1-core-模块)
2. [Data 模块](#2-data-模块)
3. [Factor 模块](#3-factor-模块)
4. [Strategy 模块](#4-strategy-模块)
5. [Backtest 模块](#5-backtest-模块)
6. [Model 模块](#6-model-模块)
7. [Analysis 模块](#7-analysis-模块)

---

## 1. Core 模块

> `from futureQuant.core import *`

### 1.1 Config — 全局配置

```python
from futureQuant.core import get_config, Config

config = get_config()        # 获取全局单例配置
config.data.cache_dir        # 数据缓存目录
config.data.db_path         # 数据库路径
config.backtest.commission   # 默认手续费率
```

### 1.2 Logger — 结构化日志

```python
from futureQuant.core import get_logger

logger = get_logger('module.name')  # 获取命名日志器
logger.info("message", extra={'key': 'value'})
logger.warning("...")
logger.error("...")
```

### 1.3 抽象基类

```python
from futureQuant.core.base import DataFetcher, Factor, Strategy, Model, BacktestEngine
```

所有自定义模块应继承这些基类以确保接口一致性。

---

## 2. Data 模块

> `from futureQuant.data import DataManager`

### 2.1 DataManager — 统一数据入口

```python
from futureQuant.data import DataManager

dm = DataManager(
    cache_dir="./data_cache",   # 缓存目录
    db_path="./data.db",         # 数据库路径
    auto_update=True             # 是否自动更新
)
```

#### 日线数据

```python
df = dm.get_daily_data(
    symbol="RB2501",             # 合约代码
    start_date="2023-01-01",    # 开始日期
    end_date="2024-12-31",      # 结束日期
    source="akshare",           # 数据源
    use_cache=True,             # 是否使用缓存
    auto_clean=True,            # 是否自动清洗
)
# 返回: DataFrame [date, open, high, low, close, volume, open_interest]
```

#### 连续合约

```python
continuous_df = dm.get_continuous_contract(
    variety="RB",                # 品种代码
    start_date="2020-01-01",
    end_date="2024-12-31",
    adjust_method="backward",    # 复权方式: backward | forward | none
    rollover_method="open_interest",  # 主力切换规则: open_interest | volume
)
```

#### 基本面数据

```python
basis_df = dm.get_basis_data(variety="RB")        # 基差数据
inventory_df = dm.get_inventory_data(variety="I") # 库存数据
receipts_df = dm.get_warehouse_receipts(variety="RB")  # 仓单数据
```

### 2.2 数据处理器

```python
from futureQuant.data.processor import DataCleaner, ContractManager, FuturesCalendar
```

```python
# 数据清洗
cleaner = DataCleaner()
df = cleaner.clean_ohlc(raw_df)   # 去除异常值（价格为0、high<low）

# 交易日历
calendar = FuturesCalendar()
calendar.is_trading_day("2024-03-15")   # True/False
calendar.get_next_trading_day("2024-03-15")  # 返回下一个交易日

# 合约管理器
cm = ContractManager()
cm.add_contract_data("RB2501", df)
continuous = cm.create_continuous_contract(
    variety="RB",
    adjust_method="backward",
    rollover_method="open_interest",
)
```

---

## 3. Factor 模块

> `from futureQuant.factor import *`

### 3.1 FactorEngine — 因子计算引擎

```python
from futureQuant.factor import FactorEngine

engine = FactorEngine()

# 注册因子
engine.register(MomentumFactor(window=20))
engine.register(RSIFactor(window=14))
engine.register(VolatilityFactor(window=20))

# 批量计算
factor_df = engine.compute_all(price_data)
# 返回: DataFrame, index=日期, columns=[momentum_20, rsi_14, volatility_20, ...]
```

### 3.2 FactorEvaluator — 因子评估

```python
from futureQuant.factor import FactorEvaluator

evaluator = FactorEvaluator()

# IC 分析
ic_series = evaluator.calculate_ic(factor_df, returns, method='spearman')
ic_stats = evaluator.calculate_icir(ic_series)
# ic_stats: {icir, annual_icir, ic_mean, ic_std, ic_win_rate, n_samples}

# IC 衰减分析
ic_decay = evaluator.calculate_ic_decay(factor_df, returns, max_lag=10)

# 分层回测
quantile_returns = evaluator.quantile_backtest(factor_df, returns, n_quantiles=5)
# 返回: DataFrame, columns=[Q1, Q2, Q3, Q4, Q5, long_short]

# 完整评估
results = evaluator.full_evaluation(factor_df, returns)

# 评估摘要
summary = evaluator.get_summary(results, factor_name='momentum')
```

### 3.3 技术因子

```python
from futureQuant.factor.technical import (
    # 动量因子
    MomentumFactor, RSIFactor, MACDFactor, MACDDIFFactor, RateOfChangeFactor,
    # 波动率因子
    ATRFactor, VolatilityFactor, BollingerBandWidthFactor,
    TrueRangeFactor, ParkisonVolatilityFactor,
    # 成交量因子
    OBVFactor, VolumeRatioFactor, VolumeMAFactor, VWAPFactor,
    MFI_Factor, VolumePriceTrendFactor,
)

# 动量因子
f = MomentumFactor(window=20)       # N日动量
f = RSIFactor(window=14)           # RSI
f = MACDFactor()                   # MACD (默认 12,26,9)
f = RateOfChangeFactor(window=12)  # 变动率

# 波动率因子
f = VolatilityFactor(window=20)     # 历史波动率（年化）
f = ATRFactor(window=14)           # 平均真实波幅
f = BollingerBandWidthFactor()     # 布林带宽度

# 成交量因子
f = OBVFactor()                    # 能量潮
f = VWAPFactor()                   # 成交量加权平均价
f = MFI_Factor(window=14)          # 资金流量指标
```

**通用使用方式：**
```python
factor_value = factor.compute(ohlcv_df)
# 返回: pd.Series, index 与输入数据的 date 对齐
```

### 3.4 基本面因子

```python
from futureQuant.factor import BasisFactor, BasisRateFactor, TermStructureFactor
from futureQuant.factor import InventoryChangeFactor, InventoryYoYFactor
from futureQuant.factor import WarehouseReceiptFactor, WarehousePressureFactor

f = BasisFactor()           # 基差 = 现货 - 期货
f = BasisRateFactor()       # 基差率 = 基差 / 期货价格
f = TermStructureFactor()   # 期限结构 = (远月 - 近月) / 近月
f = InventoryChangeFactor() # 库存变化率
f = WarehousePressureFactor()# 仓单压力 = 仓单量 / 历史均值
```

### 3.5 宏观因子

```python
from futureQuant.factor.macro import (
    DollarIndexFactor,
    InterestRateFactor,
    CommodityIndexFactor,
    InflationExpectationFactor,
)
```

---

## 4. Strategy 模块

> `from futureQuant.strategy import *`

### 4.1 BaseStrategy — 策略基类

```python
from futureQuant.strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # 返回 DataFrame: [date, signal, weight]
        # signal: 1(做多) | -1(做空) | 0(空仓)
        # weight: 0~1
        return signals_df
```

**风险管理参数（所有策略通用）：**

```python
strategy = MyStrategy(
    name="MyStrategy",
    symbols=["RB", "I"],
    stop_loss=0.02,        # 止损比例（价格百分比）
    take_profit=0.05,      # 止盈比例
    max_position=1.0,     # 最大仓位
    risk_per_trade=0.02,  # 单笔风险
    signal_threshold=0.0, # 信号阈值
)
```

**仓位计算：**

```python
position = strategy.calculate_position_size(
    capital=1_000_000,
    price=4000,
    volatility=0.02,   # 可选
    atr=100,          # 可选
)
```

### 4.2 预置策略

```python
from futureQuant.strategy import TrendFollowingStrategy, MeanReversionStrategy
from futureQuant.strategy import ArbitrageStrategy
```

**趋势跟踪策略：**

```python
strategy = TrendFollowingStrategy(
    name="RB_Trend",
    ma_period=20,              # 均线周期
    momentum_period=10,        # 动量周期
    use_atr_filter=False,      # 是否用ATR过滤信号
    atr_period=14,
    atr_threshold=1.0,
    stop_loss=0.02,
    take_profit=0.04,
)
```

**均值回归策略：**

```python
strategy = MeanReversionStrategy(
    name="RB_Reversion",
    lookback=20,              # 回看周期
    entry_threshold=2.0,      # 入场阈值（标准差倍数）
    exit_threshold=0.0,        # 出场阈值
    stop_loss=0.03,
)
```

**跨期套利策略：**

```python
strategy = ArbitrageStrategy(
    name="RB_Spread",
    near_contract="RB2501",   # 近月合约
    far_contract="RB2505",    # 远月合约
    entry_threshold=0.05,    # 价差偏离阈值
    z_score_window=20,       # Z-score 窗口
)
```

### 4.3 参数优化器

```python
from futureQuant.strategy import StrategyOptimizer

optimizer = StrategyOptimizer(
    strategy_class=TrendFollowingStrategy,
    data=price_data,
    metric="sharpe_ratio",   # 优化目标: sharpe_ratio | total_return | ...
)

# 网格搜索
result = optimizer.optimize_grid({
    'ma_period': [10, 20, 30],
    'momentum_period': [5, 10, 15],
    'stop_loss': [0.01, 0.02, 0.03],
})
print(result.best_params)
print(result.best_score)
print(result.optimization_history)
```

**随机搜索：**

```python
result = optimizer.optimize_random(
    param_bounds={
        'ma_period': (5, 60, 'int'),
        'momentum_period': (3, 30, 'int'),
        'stop_loss': (0.01, 0.1),
    },
    n_trials=100,
)
```

**贝叶斯优化（Optuna）：**

```python
result = optimizer.optimize_bayesian(
    param_bounds={
        'ma_period': (5, 60, 'int'),
        'momentum_period': (3, 30, 'int'),
    },
    n_trials=100,
)
```

**Walk-forward 滚动优化：**

```python
results = optimizer.walk_forward_optimization(
    train_size=252,           # 训练窗口（天数）
    test_size=63,            # 测试窗口（天数）
    step_size=63,            # 滚动步长
    method='random',         # 'grid' | 'random' | 'bayesian'
    param_bounds={...},
    n_trials=50,
)
# 返回: List[OptimizationResult]，每个窗口一个结果
```

---

## 5. Backtest 模块

> `from futureQuant.backtest import *`

### 5.1 BacktestEngine — 回测引擎

```python
from futureQuant.backtest import BacktestEngine, BacktestMode

engine = BacktestEngine(
    initial_capital=1_000_000,     # 初始资金
    commission=0.0001,              # 手续费率（万1，双边）
    slippage=1,                     # 滑点（跳数）
    margin_rate=0.1,                # 保证金率（10%）
    maintenance_margin_rate=0.08,   # 维持保证金率
    close_today_rate=0.0002,       # 平今手续费率（可选）
    contract_multipliers={'RB': 10}, # 合约乘数（可选）
    tick_sizes={'RB': 1},          # 最小变动价位（可选）
    max_concentration=0.3,         # 最大单品种集中度
    max_leverage=3.0,             # 最大杠杆
)
```

**向量化回测（快速）：**

```python
result = engine.run(
    data=continuous_df,
    strategy=TrendFollowingStrategy(...),
    mode=BacktestMode.VECTORIZED,
)
```

**事件驱动回测（精细）：**

```python
result = engine.run(
    data=continuous_df,
    strategy=TrendFollowingStrategy(...),
    mode=BacktestMode.EVENT_DRIVEN,
    use_margin_call=True,  # 是否启用强平
)
```

**结果：**

```python
result.keys()
# ['mode', 'initial_capital', 'final_equity', 'total_return',
#  'annual_return', 'volatility', 'sharpe_ratio', 'sortino_ratio',
#  'calmar_ratio', 'max_drawdown', 'total_trades', 'win_rate',
#  'profit_factor', 'equity_curve', 'trades', ...]

print(engine.generate_report())  # 文本报告
engine.plot_results()           # 绘图
```

### 5.2 TradeRecorder — 交易记录

```python
from futureQuant.backtest import TradeRecorder

recorder = TradeRecorder(initial_capital=1_000_000)
recorder.record_trade(trade_info)
recorder.record_daily_value(date, net_value, cash, margin)

# 绩效指标
metrics = recorder.get_performance_metrics()
trade_stats = recorder.get_trade_stats()
max_dd = recorder.calculate_max_drawdown()
sharpe = recorder.calculate_sharpe_ratio()
sortino = recorder.calculate_sortino_ratio()
calmar = recorder.calculate_calmar_ratio()
```

---

## 6. Model 模块

> `from futureQuant.model import *`

### 6.1 MLForecastPipeline — 端到端流水线

```python
from futureQuant.model import MLForecastPipeline

pipeline = MLForecastPipeline(
    config=PipelineConfig(
        model_type='xgboost',           # 'xgboost' | 'lightgbm'
        label_type='classification',     # 'classification' | 'regression'
        forward_period=5,               # 预测未来 N 日
        train_window=252,
        test_window=60,
        use_walk_forward=True,
    )
)

# 训练
pipeline.fit(price_data)

# Walk-forward 训练（防过拟合）
pipeline.fit_walk_forward(price_data)

# 预测
signals = pipeline.predict()
# signals: DataFrame [signal, confidence, proba]
# signal: 1(做多) | -1(做空) | 0(空仓)

# 特征重要性
importance = pipeline.get_feature_importance(top_n=20)
```

### 6.2 FeatureEngineer — 特征工程

```python
from futureQuant.model import FeatureEngineer, FeatureConfig

fe = FeatureEngineer()

fe.add_price_features()          # 价格特征（对数收益、滞后动量）
fe.add_technical_features()      # 技术指标（MA、RSI、MACD、布林带、ATR）
fe.add_volume_features()          # 成交量特征（OBV、OI变化）

# 构建特征矩阵
features, targets = fe.build(
    data=ohlcv_df,
    target_column='close',
    forward_periods=[1, 5],
    label_type='classification',  # 涨跌分类
)

# 数据泄漏检测
report = fe.detect_leakage(features, targets)
```

### 6.3 XGBoostModel — XGBoost 模型

```python
from futureQuant.model import XGBoostModel

model = XGBoostModel(
    objective='binary:logistic',  # 分类
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
pred = model.predict(X_test)              # 预测类别
proba = model.predict_proba(X_test)[:, 1]  # 预测概率

# 特征重要性
importance = model.get_feature_importance()
```

### 6.4 LightGBMModel — LightGBM 模型

```python
from futureQuant.model import LightGBMModel

model = LightGBMModel(
    objective='binary',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
)
model.fit(X_train, y_train)
```

### 6.5 LSTMModel — LSTM 时序模型

```python
from futureQuant.model import LSTMModel

model = LSTMModel(config=LSTMConfig(
    lookback=20,              # 回看窗口
    horizon=1,                # 预测步数
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    epochs=50,
))
model.fit(X, y, eval_set=(X_val, y_val))
pred = model.predict(X_test)              # 回归预测
direction = model.predict_direction(X_test)  # 涨跌分类

# 保存/加载
model.save('lstm_model.pt')
model.load('lstm_model.pt')
```

### 6.6 ARIMAModel — ARIMA 模型

```python
from futureQuant.model import ARIMAModel

model = ARIMAModel(config=ARIMAConfig(
    p=5, d=1, q=5,
    auto_order=True,        # 自动定阶
    max_p=5, max_q=5, max_d=2,
))
model.fit(price_series)
forecast = model.predict(steps=5)  # 预测5步

# 滚动预测
results_df = model.rolling_fit_predict(
    data=ohlcv_df,
    window=60,
    horizon=1,
)
```

---

## 7. Analysis 模块

> `from futureQuant.analysis import *`

### 7.1 PerformanceReport — 绩效报告

```python
from futureQuant.analysis import PerformanceReport

# 从回测结果生成报告
report = PerformanceReport(backtest_result)

# 文本报告
print(report.generate_text())

# HTML 报告
report.save_html('report.html', title='RB策略绩效报告')

# JSON 数据
report.save_json('report.json')

# 多策略对比
from futureQuant.analysis import MultiStrategyReport
multi = MultiStrategyReport()
multi.add_strategy('Trend', report1)
multi.add_strategy('MeanRev', report2)
print(multi.generate_comparison_table())
print(multi.best_strategy('sharpe_ratio'))
```

### 7.2 报告输出格式

**文本报告（控制台）：**
```
==================== 期货量化策略绩效报告 ====================
【收益指标】
  总收益率:           23.45%
  年化收益率:         12.67%
【风险指标】
  年化波动率:          8.23%
  最大回撤:           -6.78%
【风险调整收益】
  夏普比率:           1.243
  索提诺比率:          1.891
  卡玛比率:            1.869
【交易统计】
  总交易次数:              45
  胜率:                58.26%
  盈亏比:              1.452
```

---

## 附录 A：数据格式规范

### OHLCV 数据

| 列名 | 类型 | 说明 |
|------|------|------|
| date | str/datetime | 日期，格式 YYYY-MM-DD |
| symbol | str | 合约代码 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| volume | float | 成交量 |
| open_interest | float | 持仓量 |

### 信号格式

| 列名 | 类型 | 说明 |
|------|------|------|
| date | datetime | 日期 |
| signal | int | 1=做多, -1=做空, 0=空仓 |
| weight | float | 仓位权重 0~1 |
| confidence | float | 置信度 0~1（可选） |

---

## 附录 B：常见问题

**Q: 回测结果与实盘差异大？**
→ 检查滑点（`slippage`）、流动性（主力合约换月时成交能力）、手续费模型（平今 vs 平昨）。

**Q: 因子 IC 很高但策略不赚钱？**
→ 检查未来函数（特征是否用到了 `shift(-1)` 之后的数据），以及因子与交易成本的叠加效果。

**Q: 模型过拟合严重？**
→ 使用 `fit_walk_forward()` 进行滚动验证，增大 Walk-forward 训练窗口。

**Q: 内存不足？**
→ 启用数据缓存（`use_cache=True`），使用 DuckDB 处理分钟线级别数据。
