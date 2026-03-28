# futureQuant - 期货量化研究框架

> 模块化的期货量化研究与回测框架，支持因子挖掘、策略开发、双引擎回测、参数优化。

## 功能特性

| 模块 | 描述 |
|------|------|
| **数据层** | akshare 多合约获取、Web 爬虫（基差/库存/仓单）、主力连续合约合成、Parquet 本地缓存 |
| **因子库** | 技术因子（动量/波动率/成交量 16 种）、基本面因子（基差/库存/仓单）、宏观因子 |
| **策略层** | 趋势跟踪、均值回归、跨期套利三大模板，支持参数优化（网格/随机/贝叶斯/Walk-forward） |
| **回测层** | 向量化（快速研究）+ 事件驱动（精细验证）双引擎，含保证金/强平/手续费模型 |
| **模型层** | XGBoost / LightGBM 监督学习、特征工程Pipeline、LSTM/ARIMA 时序模型 |
| **分析层** | IC/ICIR 分析、分层回测、因子统计、换手率、绩效报告（文本/HTML/JSON） |

## 安装

### 前置要求

- Python 3.10+
- pip

### 安装步骤

```bash
# 克隆或进入项目目录
cd futureQuant

# 安装依赖（推荐使用虚拟环境）
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# 或
.venv\Scripts\activate           # Windows

# 安装
pip install -e .
```

### 依赖列表

| 类别 | 包 | 用途 |
|------|-----|------|
| 数据处理 | pandas>=2.0, numpy>=1.24 | 基础计算 |
| 数据获取 | akshare>=1.11, requests>=2.31 | 行情数据 |
| 配置管理 | pydantic>=2.0, pydantic-settings>=2.0, pyyaml | 配置 |
| 机器学习 | scikit-learn>=1.3, xgboost>=2.0, lightgbm>=4.0 | 模型 |
| 优化 | optuna>=3.3 | 参数寻优 |
| 可视化 | matplotlib>=3.7, seaborn>=0.12, plotly>=5.15 | 图表 |

## 快速开始

### 1. 获取数据

```python
from futureQuant.data import DataManager

dm = DataManager(cache_dir="./data_cache")

# 获取单一合约日线
df = dm.get_daily_data(
    symbol="RB2501",
    start_date="2023-01-01",
    end_date="2024-12-31",
)

# 获取主力连续合约（自动处理换月）
continuous_df = dm.get_continuous_contract(
    variety="RB",
    start_date="2020-01-01",
    end_date="2024-12-31",
    adjust_method="backward",       # 后复权
    rollover_method="open_interest", # 按持仓量切换主力
)

# 获取基本面数据
basis_df = dm.get_basis_data(variety="RB")
inventory_df = dm.get_inventory_data(variety="I")  # 铁矿石库存
```

### 2. 计算因子

```python
from futureQuant.factor import FactorEngine, MomentumFactor, RSIFactor, VolatilityFactor

engine = FactorEngine()

# 注册因子
engine.register(MomentumFactor(window=20))
engine.register(RSIFactor(window=14))
engine.register(VolatilityFactor(window=20))

# 批量计算
factor_df = engine.compute_all(price_data)

print(factor_df.columns.tolist())
# ['momentum_20', 'rsi_14', 'volatility_20']
```

### 3. 因子评估

```python
from futureQuant.factor import FactorEvaluator

evaluator = FactorEvaluator()

# 计算 IC
ic_series = evaluator.calculate_ic(factor_df, returns, method='spearman')

# 计算 ICIR
icir_stats = evaluator.calculate_icir(ic_series)
print(f"ICIR: {icir_stats['icir']:.3f}, 年化ICIR: {icir_stats['annual_icir']:.3f}")

# 分层回测
quantile_returns = evaluator.quantile_backtest(factor_df, returns, n_quantiles=5)

# 完整评估
results = evaluator.full_evaluation(factor_df, returns)
```

### 4. 策略回测

```python
from futureQuant.strategy import TrendFollowingStrategy
from futureQuant.backtest import BacktestEngine, BacktestMode

# 创建策略
strategy = TrendFollowingStrategy(
    name="RB_Trend",
    ma_period=20,
    momentum_period=10,
    stop_loss=0.02,
    risk_per_trade=0.02,
)

# 向量化回测（快速）
engine = BacktestEngine(
    initial_capital=1_000_000,
    commission=0.0001,
    slippage=1,
    margin_rate=0.1,
)

result = engine.run(
    data=continuous_df,
    strategy=strategy,
    mode=BacktestMode.VECTORIZED,
)

# 打印报告
print(engine.generate_report())
```

### 5. 参数优化

```python
from futureQuant.strategy.optimizer import StrategyOptimizer
from futureQuant.strategy import TrendFollowingStrategy

optimizer = StrategyOptimizer(
    strategy_class=TrendFollowingStrategy,
    data=price_data,
    metric="sharpe_ratio",  # 优化目标
)

# 网格搜索
best_params, results = optimizer.grid_search({
    'ma_period': range(10, 60, 5),
    'momentum_period': range(5, 30, 5),
    'stop_loss': [0.01, 0.02, 0.03, 0.05],
})

print(f"最优参数: {best_params}")
```

## 项目结构

```
futureQuant/
├── core/                 # 抽象基类、配置、日志、异常
│   ├── base.py          # DataFetcher/Factor/Strategy/Model/BacktestEngine 基类
│   ├── config.py        # 全局配置（pydantic-settings）
│   ├── logger.py        # 结构化日志
│   └── exceptions.py    # 自定义异常类
├── data/                 # 数据管理层
│   ├── manager.py       # 统一数据入口
│   ├── fetcher/
│   │   ├── akshare_fetcher.py   # akshare 适配器
│   │   └── crawler/              # 基差/库存/仓单爬虫
│   ├── processor/
│   │   ├── cleaner.py           # 数据清洗
│   │   ├── contract_manager.py  # 主力合约合成
│   │   └── calendar.py          # 交易日历
│   └── storage/
│       ├── db_manager.py        # Parquet 存储
│       └── updater.py           # 数据更新调度
├── factor/               # 因子库
│   ├── engine.py         # 因子计算引擎
│   ├── evaluator.py     # IC/分层回测评估
│   ├── technical/        # 技术因子
│   ├── fundamental/     # 基本面因子
│   └── macro/           # 宏观因子
├── strategy/             # 策略层
│   ├── base.py          # 策略基类
│   ├── trend_following.py
│   ├── mean_reversion.py
│   ├── arbitrage.py
│   └── optimizer.py      # 参数优化
├── backtest/            # 回测引擎
│   ├── engine.py         # 主引擎（向量化 + 事件驱动）
│   ├── broker.py         # 模拟交易所（手续费/保证金）
│   ├── portfolio.py      # 仓位管理
│   └── recorder.py       # 交易记录与绩效计算
├── model/               # 模型层（待完善）
│   ├── supervised/       # XGBoost / LightGBM
│   └── time_series/      # LSTM / ARIMA
├── config/
│   └── settings.yaml     # 全局配置
├── logs/                 # 日志输出
├── data_cache/           # Parquet 缓存
└── tests/               # 测试（待补充）
```

## 常见问题

### Q: akshare 数据获取失败？

akshare 数据源有时不稳定，建议：
1. 确认 akshare 版本为最新：`pip install akshare --upgrade`
2. 启用本地缓存，下次获取时优先从 `data_cache/` 读取
3. 检查网络连接，部分数据源需要代理

### Q: 如何处理主力合约换月跳空？

使用 `DataManager.get_continuous_contract()` 自动合成连续合约：

```python
# 后复权（推荐）：最新合约为基准，向前调整历史价格
df = dm.get_continuous_contract(variety="RB", adjust_method="backward")

# 不复权：保留跳空，适合价差套利
df = dm.get_continuous_contract(variety="RB", adjust_method="none")
```

### Q: 保证金和手续费如何设置？

在 `BacktestEngine` 初始化时指定：

```python
engine = BacktestEngine(
    commission=0.0001,          # 万1 双边
    slippage=1,                  # 1 个滑点（跳数）
    margin_rate=0.1,             # 10% 保证金
    maintenance_margin_rate=0.08, # 8% 维持保证金
)
```

各品种详细费率建议从[交易所官网](https://www.shfe.com.cn/,https://www.dce.com.cn/,https://www.czce.com.cn/)获取。

### Q: 回测结果和实盘差距大？

常见原因：
1. **滑点不足**：期货实盘滑点通常 1-3 个跳，考虑增加 `slippage`
2. **流动性不足**：主力合约换月时新主力成交量少，无法大资金成交
3. **手续费模型**：部分品种平今费率高于平昨，需配置 `close_today_rate`
4. **过拟合**：参数优化后用样本内最优参数在样本内测试，自然收益高

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v0.3.0 | 2026-03-24 | 核心功能完成（数据/因子/策略/回测） |
| v0.4.0 | 进行中 | 测试框架、文档完善 |
| v1.0.0 | 规划中 | 生产就绪 |

## 免责声明

本框架仅供研究学习使用，不构成任何投资建议。期货交易有风险，入市需谨慎。实盘交易前请充分回测并了解交易所规则。
