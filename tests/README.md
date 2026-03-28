# futureQuant 测试框架

本目录包含 futureQuant 项目的完整测试套件。

## 目录结构

```
tests/
├── conftest.py              # pytest 配置和共享 fixtures
├── fixtures/
│   ├── __init__.py
│   ├── sample_price_data.py # 价格数据生成器
│   └── sample_factor_data.py# 因子数据生成器
├── unit/
│   ├── __init__.py
│   ├── test_factor_engine.py      # FactorEngine 单元测试
│   ├── test_factor_evaluator.py   # FactorEvaluator 单元测试
│   ├── test_contract_manager.py   # ContractManager 单元测试
│   ├── test_data_cleaner.py       # DataCleaner 单元测试
│   ├── test_calendar.py            # FuturesCalendar 单元测试
│   └── test_strategy_base.py      # BaseStrategy 单元测试
├── integration/
│   ├── __init__.py
│   └── test_data_manager_flow.py  # DataManager 集成测试
└── README.md
```

## 运行测试

### 运行所有测试
```bash
cd D:\310Programm\futureQuant
pytest
```

### 运行单元测试
```bash
pytest tests/unit/
```

### 运行集成测试
```bash
pytest tests/integration/
```

### 运行特定文件
```bash
pytest tests/unit/test_factor_engine.py
```

### 运行特定测试类
```bash
pytest tests/unit/test_factor_engine.py::TestFactorEngineRegister
```

### 详细输出
```bash
pytest -v --tb=long
```

## 测试覆盖

### 单元测试

| 模块 | 测试内容 |
|------|----------|
| `FactorEngine` | 因子注册、批量计算、缓存管理 |
| `FactorEvaluator` | IC/ICIR 计算、分层回测、因子统计 |
| `ContractManager` | 主力合约识别、价格复权、合约解析 |
| `DataCleaner` | OHLC 清洗、异常值处理、缺失值填充 |
| `FuturesCalendar` | 交易日判断、节假日管理、夜盘信息 |
| `BaseStrategy` | 参数验证、仓位计算、信号生成 |

### 集成测试

| 模块 | 测试内容 |
|------|----------|
| `DataManager` | 初始化、缓存读取、数据源获取、爬虫降级 |

## Fixtures 说明

| Fixture | 用途 |
|---------|------|
| `sample_ohlcv` | 180天、5个品种的标准 OHLCV 数据 |
| `sample_continuous_contract` | 模拟主力合约切换的连续合约数据 |
| `sample_factor_panel` | MultiIndex (date, symbol) 横截面因子数据 |
| `mock_akshare` | Mock akshare 返回值 |

## 数据生成器

### `fixtures/sample_price_data.py`
- `generate_ohlcv()`: 生成真实感 OHLCV 数据
- `generate_returns()`: 生成收益率序列

### `fixtures/sample_factor_data.py`
- `generate_factor_panel()`: 生成横截面因子面板
- `generate_factor_and_returns()`: 生成因子-收益率配对数据

## 注意事项

1. **不修改源代码**：测试只验证行为，不修改任何源代码
2. **直接 import**：测试直接 import 真实模块
3. **使用 mock**：集成测试中使用 mock 隔离外部依赖
4. **pytest.importorskip**：处理可能不存在的模块
