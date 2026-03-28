# 🎯 futureQuant 多智能体因子挖掘系统 - 快速入门指南

**版本**: v1.0  
**更新时间**: 2026-03-26

---

## 📋 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [基本用法](#基本用法)
4. [高级功能](#高级功能)
5. [示例代码](#示例代码)
6. [常见问题](#常见问题)

---

## 🔧 环境准备

### 系统要求

- **Python**: 3.10+
- **操作系统**: Windows/Linux/macOS
- **内存**: 建议 8GB+
- **磁盘**: 建议 10GB+

### 安装依赖

```bash
cd D:\310Programm\futureQuant

# 安装核心依赖
pip install numpy pandas scipy joblib
pip install sqlalchemy sqlalchemy-utils
pip install pyarrow fastparquet
pip install pytest pytest-cov pytest-xdist

# 或一次性安装所有依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 5 分钟快速入门

#### Step 1: 导入模块

```python
from futureQuant.agent import MultiAgentFactorMiner
from futureQuant.data import DataManager
```

#### Step 2: 准备数据

```python
# 初始化数据管理器
dm = DataManager(cache_dir="./data_cache")

# 获取期货数据
data = dm.get_continuous_contract(
    variety="RB",           # 螺纹钢
    start_date="2020-01-01",
    end_date="2024-12-31",
)
```

#### Step 3: 初始化挖掘器

```python
# 初始化因子挖掘器
miner = MultiAgentFactorMiner(
    symbols=['RB'],         # 交易品种
    start_date='2020-01-01',
    end_date='2024-12-31',
    data=data,              # 传入数据
    n_workers=4,            # 并行工作进程数
)
```

#### Step 4: 运行因子挖掘

```python
# 运行因子挖掘
result = miner.run()

# 查看结果
print(f"发现有效因子: {len(result.factors)}")
print(f"最佳因子: {result.best_factor}")
print(f"综合评分: {result.best_score:.3f}")
```

#### Step 5: 运行回测

```python
# 运行回测
backtest_result = miner.run_backtest(result.factors)

# 查看回测结果
print(f"夏普比率: {backtest_result['sharpe_ratio']:.3f}")
print(f"最大回撤: {backtest_result['max_drawdown']:.3f}")
print(f"年化收益: {backtest_result['annual_return']:.3f}")
```

---

## 📖 基本用法

### 1. 单因子挖掘

```python
from futureQuant.agent.miners import TechnicalMiningAgent

# 创建技术因子挖掘 Agent
tech_agent = TechnicalMiningAgent(
    symbols=['RB'],
    start_date='2020-01-01',
    end_date='2024-12-31',
)

# 运行挖掘
factors = tech_agent.run(data)

# 查看因子
for factor in factors[:5]:
    print(f"{factor.name}: IC={factor.ic:.4f}, ICIR={factor.icir:.3f}")
```

### 2. 多因子组合

```python
from futureQuant.agent.miners import FusionAgent

# 创建融合 Agent
fusion_agent = FusionAgent()

# 融合多个因子
combined_factor = fusion_agent.fuse(
    factors=[tech_factor, fundamental_factor, macro_factor],
    method='icir_weighted',  # ICIR 加权
)

print(f"融合因子评分: {combined_factor.score:.3f}")
```

### 3. 因子验证

```python
from futureQuant.agent.validators import LookAheadDetector
from futureQuant.agent.validators import TimeSeriesCrossValidator

# 未来函数检测
detector = LookAheadDetector()
has_lookahead = detector.check(factor)

if has_lookahead:
    print("⚠️ 因子存在未来函数风险！")
else:
    print("✅ 因子通过未来函数检测")

# 时序交叉验证
validator = TimeSeriesCrossValidator(
    method='walk_forward',
    train_size=252,
    test_size=63,
)

stability = validator.validate(factor, data)
print(f"因子稳定性评分: {stability:.3f}")
```

### 4. 因子评分

```python
from futureQuant.agent.validators import MultiDimensionalScorer

# 创建评分器
scorer = MultiDimensionalScorer()

# 计算综合评分
score = scorer.score(
    factor_values=factor_values,
    returns=returns,
    ic_mean=ic_mean,
    icir=icir,
    ic_win_rate=ic_win_rate,
)

print(f"预测能力评分: {score['predictability']:.3f}")
print(f"稳定性评分: {score['stability']:.3f}")
print(f"单调性评分: {score['monotonicity']:.3f}")
print(f"综合评分: {score['overall']:.3f}")
```

### 5. 策略回测

```python
from futureQuant.agent.backtest import StrategyGenerator
from futureQuant.agent.backtest import RiskController

# 生成策略
strategy_gen = StrategyGenerator()
strategy = strategy_gen.generate(factor)

# 配置风险控制
risk_ctrl = RiskController(
    stop_loss=0.05,      # 止损 5%
    take_profit=0.10,    # 止盈 10%
    max_position=0.30,   # 最大仓位 30%
    max_drawdown=0.15,   # 最大回撤 15%
)

# 运行回测
backtest_result = strategy.run_backtest(
    data=data,
    initial_capital=1000000,
    risk_controller=risk_ctrl,
)

# 查看结果
print(f"夏普比率: {backtest_result['sharpe_ratio']:.3f}")
print(f"索提诺比率: {backtest_result['sortino_ratio']:.3f}")
print(f"最大回撤: {backtest_result['max_drawdown']:.3f}")
```

### 6. 因子库管理

```python
from futureQuant.agent.repository import FactorRepository

# 初始化因子库
repo = FactorRepository('./factor_repo')

# 保存因子
factor_id = repo.save_factor(
    factor=factor,
    values=factor_values,
    performance={'ic_mean': 0.05, 'icir': 1.2},
)

# 查询因子
factor_data = repo.get_factor(factor_id)
print(f"因子名称: {factor_data['name']}")

# 列出所有因子
factors = repo.list_factors(category='technical')
print(f"技术因子数量: {len(factors)}")
```

---

## 🔬 高级功能

### 1. 性能优化

```python
from futureQuant.agent.optimization import ParallelCalculator
from futureQuant.agent.optimization import CacheManager

# 启用并行计算
calculator = ParallelCalculator(n_workers=8)
result = calculator.calculate(factors, data)

# 启用缓存
cache = CacheManager(cache_dir='./cache')
result = cache.get_or_compute('factor_key', compute_fn)
```

### 2. 增强评分（9 维度）

```python
from futureQuant.agent.validators import EnhancedMultiDimensionalScorer

# 创建增强评分器
scorer = EnhancedMultiDimensionalScorer()

# 计算 9 维评分
score = scorer.calculate_enhanced_score(
    factor_values=factor_values,
    returns=returns,
    ic_mean=ic_mean,
    icir=icir,
    ic_win_rate=ic_win_rate,
    volume=volume,              # 新增
    price=price,                 # 新增
    other_factors=other_factors, # 新增
)

print(f"可交易性评分: {score['tradability_score']:.3f}")
print(f"鲁棒性评分: {score['robustness_score']:.3f}")
print(f"独立性评分: {score['independence_score']:.3f}")
print(f"综合评分: {score['overall_score']:.3f}")
```

### 3. 数据质量管理

```python
from futureQuant.data import DataCleaner

# 初始化清洗器
cleaner = DataCleaner(sigma=3.0)

# 清洗数据
cleaned_data = cleaner.clean_data(
    data=raw_data,
    outlier_method='zscore',
    missing_method='ffill',
    smooth_method='rolling_mean',
    smooth_window=5,
)

# 查看清洗报告
report = cleaner.get_report()
print(f"去除极值数量: {report['outliers_removed']}")
```

### 4. 交易成本模型

```python
from futureQuant.agent.backtest import CostModel

# 配置成本模型
cost_model = CostModel(
    fixed_cost=5.0,              # 固定成本
    commission_rate=0.0001,      # 手续费率
    slippage_rate=0.0001,        # 滑点率
    impact_model='sqrt',         # 市场冲击模型
)

# 计算成本
cost = cost_model.calculate_total_cost(
    trade_value=100000,
    volume=1000000,
    price=5000,
    volatility=0.02,
)

print(f"总成本: {cost['total_cost']:.2f}")
print(f"成本比例: {cost['cost_ratio']:.4f}")
```

### 5. 因子相关性追踪

```python
from futureQuant.agent.repository import CorrelationTracker

# 初始化追踪器
tracker = CorrelationTracker(db_path='./factor_repo/metadata.db')

# 计算相关性矩阵
corr_matrix = tracker.calculate_correlation_matrix(factors)

# 生成相关性报告
report = tracker.generate_correlation_report(
    factors=factor_dict,
    threshold=0.8,
)

print(f"高相关性因子对: {len(report['high_correlation_pairs'])}")
```

---

## 💻 示例代码

### 示例 1: 完整因子挖掘流程

```python
from futureQuant.agent import MultiAgentFactorMiner
from futureQuant.data import DataManager

# 1. 准备数据
dm = DataManager(cache_dir="./data_cache")
data = dm.get_continuous_contract(
    variety="RB",
    start_date="2020-01-01",
    end_date="2024-12-31",
)

# 2. 初始化挖掘器
miner = MultiAgentFactorMiner(
    symbols=['RB'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    data=data,
    n_workers=4,
)

# 3. 运行挖掘
result = miner.run()

# 4. 查看结果
print(f"发现有效因子: {len(result.factors)}")
print(f"最佳因子: {result.best_factor}")
print(f"综合评分: {result.best_score:.3f}")

# 5. 运行回测
backtest_result = miner.run_backtest(result.factors)
print(f"夏普比率: {backtest_result['sharpe_ratio']:.3f}")
```

### 示例 2: 自定义因子挖掘

```python
from futureQuant.agent.miners import TechnicalMiningAgent
from futureQuant.agent.validators import LookAheadDetector

# 1. 创建挖掘 Agent
agent = TechnicalMiningAgent(
    symbols=['RB'],
    start_date='2020-01-01',
    end_date='2024-12-31',
)

# 2. 配置参数搜索空间
agent.config = {
    'momentum_windows': [5, 10, 20, 60],
    'volatility_windows': [10, 20],
    'rsi_windows': [6, 14, 21],
    'ic_threshold': 0.02,
}

# 3. 运行挖掘
factors = agent.run(data)

# 4. 验证因子
detector = LookAheadDetector()
valid_factors = [f for f in factors if not detector.check(f)]

print(f"有效因子: {len(valid_factors)}")
```

### 示例 3: 多因子组合

```python
from futureQuant.agent.miners import FusionAgent

# 1. 融合 Agent
fusion = FusionAgent()

# 2. 融合因子
combined = fusion.fuse(
    factors=[tech_factor, fundamental_factor, macro_factor],
    method='icir_weighted',
    correlation_threshold=0.8,
)

# 3. 查看结果
print(f"融合因子 IC: {combined.ic:.4f}")
print(f"融合因子 ICIR: {combined.icir:.3f}")
```

---

## ❓ 常见问题

### Q1: 如何处理缺失数据？

```python
from futureQuant.data import DataCleaner

cleaner = DataCleaner()
cleaned = cleaner.fill_missing_values(data, method='ffill')
```

### Q2: 如何调整评分权重？

```python
scorer = MultiDimensionalScorer()
scorer.weights = {
    'predictability': 0.30,
    'stability': 0.25,
    'monotonicity': 0.20,
    'turnover': 0.10,
    'risk': 0.15,
}
```

### Q3: 如何启用并行计算？

```python
miner = MultiAgentFactorMiner(
    symbols=['RB', 'HC', 'I'],
    data=data,
    n_workers=8,  # 启用 8 个并行进程
)
```

### Q4: 如何保存和加载因子？

```python
# 保存
repo = FactorRepository('./factor_repo')
factor_id = repo.save_factor(factor, values, performance)

# 加载
factor_data = repo.get_factor(factor_id)
```

### Q5: 如何生成回测报告？

```python
from futureQuant.agent.backtest import BacktestReportGenerator

generator = BacktestReportGenerator()

# 生成 HTML 报告
html_report = generator.generate_html(backtest_result)
with open('report.html', 'w') as f:
    f.write(html_report)

# 生成 JSON 报告
json_report = generator.generate_json(backtest_result)
```

---

## 📚 更多资源

- **完整文档**: `docs/user_guide/`
- **API 参考**: `docs/api/`
- **示例代码**: `examples/`
- **最佳实践**: `docs/best_practices/`

---

## 🚀 下一步

1. **查看示例代码**: 运行 `examples/` 下的示例
2. **阅读用户指南**: 查看 `docs/user_guide/`
3. **学习最佳实践**: 查看 `docs/best_practices/`
4. **查看 API 文档**: 查看 `docs/api/`

---

**祝您使用愉快！** 🎉

如有问题，请查看文档或联系开发团队。
