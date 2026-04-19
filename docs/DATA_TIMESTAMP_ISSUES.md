# 数据时间戳问题诊断报告与解决方案

**日期**: 2026-04-19
**问题**: 数据时间戳不正确，可能使用错误年份的数据作为基准

---

## 一、发现的问题

### 问题1: 价格数据日期超出请求范围

**现象**: 
- 请求日期: `2026-03-20 ~ 2026-04-19`（最近30天）
- 实际返回: `2026-02-24 ~ 2026-04-17`

**原因分析**:
```
akshare.get_futures_daily() 可能没有严格按日期过滤
或 返回的缓存数据包含了更早的历史数据
```

**影响**: 
- ❌ 回测区间不准确
- ❌ 因子计算基于错误时间序列
- ❌ 策略信号时序错位

---

### 问题2: 价格数据异常波动

**现象**:
```
2026-02-24: +19.42% (不可能的单日涨幅)
2026-02-25: -16.47%
2026-02-26: -16.15%
```

**原因分析**:
1. 可能不同合约数据被混在一起
2. 价格单位不一致（有的可能是"元/吨"，有的是"元/手"）
3. 数据标准化处理不完整

---

### 问题3: 基本面数据品种代码错误

**现象**: FG 玻璃期货无法获取库存数据

**原因**: 
```
INVENTORY_SYMBOLS 映射表缺少 'FG': '玻璃' 的映射
akshare 接口需要 '玻璃' 不是 'FG'
```

---

## 二、解决方案

### 方案1: 数据时间戳验证（立即实施）

```python
def validate_data_freshness(df: pd.DataFrame, 
                             max_age_days: int = 5) -> bool:
    """
    验证数据新鲜度
    
    规则:
    1. 最新数据距今不超过 max_age_days 天
    2. 数据年份必须是当前年份或去年
    3. 数据不能来自未来
    """
    latest_date = pd.to_datetime(df['date']).max()
    now = datetime.now()
    
    # 检查1: 不能来自未来
    assert latest_date <= now, "数据包含未来日期！"
    
    # 检查2: 不能太旧
    days_old = (now - latest_date).days
    assert days_old <= max_age_days, f"数据过时: {days_old}天"
    
    # 检查3: 年份必须合理
    assert latest_date.year >= now.year - 1, "数据年份错误"
```

### 方案2: 数据日期严格过滤（立即实施）

```python
def fetch_with_strict_date_filter(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    严格按日期范围过滤数据
    """
    df = akshare_fetch(symbol, start_date, end_date)
    
    # 严格过滤
    df = df[
        (df['date'] >= pd.to_datetime(start_date)) & 
        (df['date'] <= pd.to_datetime(end_date))
    ]
    
    # 验证返回数量
    expected_days = (pd.to_datetime(end_date) - 
                     pd.to_datetime(start_date)).days
    
    if len(df) < expected_days * 0.5:  # 少于50%交易日
        logger.warning(f"数据量异常: 期望~{expected_days}, 实际{len(df)}")
    
    return df
```

### 方案3: 数据标准化增强（短期实施）

```python
def standardize_price_data(df: pd.DataFrame, 
                            symbol: str) -> pd.DataFrame:
    """
    增强数据标准化
    """
    # 1. 价格合理性检查
    close = df['close']
    
    # 对于期货，价格通常在 100~10000 元/吨
    assert (close > 50).all() and (close < 100000).all(), \
        "价格超出合理范围"
    
    # 2. 日收益率合理性检查（不超过10%）
    ret = close.pct_change().abs()
    if (ret > 0.10).any():
        logger.warning("发现异常价格波动，过滤中...")
        df = df[ret <= 0.10]
    
    # 3. 单品种过滤（避免多合约混在一起）
    if 'symbol' in df.columns:
        # 只取主力合约或指定合约
        df = df[df['symbol'].str.startswith(symbol[:2])]
    
    return df
```

### 方案4: 数据源健康检查（中期实施）

```python
class DataSourceHealthCheck:
    """数据源健康检查"""
    
    def check_akshare_health(self, symbol: str) -> dict:
        """
        检查 akshare 数据源是否正常
        """
        results = {
            'connection': False,
            'freshness': False,
            'completeness': False,
            'issues': []
        }
        
        # 1. 连接测试
        try:
            df = akshare.get_futures_daily(...)
            results['connection'] = True
        except Exception as e:
            results['issues'].append(f"连接失败: {e}")
        
        # 2. 新鲜度测试
        if results['connection']:
            latest = df['date'].max()
            if (datetime.now() - latest).days <= 3:
                results['freshness'] = True
            else:
                results['issues'].append(f"数据过时: {latest}")
        
        # 3. 完整性测试
        if results['connection']:
            # 检查必要列
            required_cols = ['date', 'open', 'high', 'low', 'close']
            missing = set(required_cols) - set(df.columns)
            if missing:
                results['issues'].append(f"缺少列: {missing}")
            else:
                results['completeness'] = True
        
        return results
```

---

## 三、实施计划

### Phase 1: 紧急修复（1-2天）

| 任务 | 优先级 | 工作内容 |
|------|--------|----------|
| P1.1 添加数据时间验证 | 🔴 高 | 在 fetch_daily 中添加日期范围验证 |
| P1.2 修复FG品种映射 | 🔴 高 | 添加 'FG': '玻璃' 映射 |
| P1.3 添加价格合理性检查 | 🔴 高 | 过滤异常价格数据 |
| P1.4 写数据验证单元测试 | 🟡 中 | 覆盖日期、价格、完整性检查 |

**产出**:
- `futureQuant/data/validator.py` - 数据验证模块
- 修复 `fundamental_fetcher.py` 品种映射
- 新增 `tests/unit/test_data_validator.py`

### Phase 2: 短期优化（3-7天）

| 任务 | 优先级 | 工作内容 |
|------|--------|----------|
| P2.1 数据缓存管理 | 🟡 中 | 实现缓存过期机制 |
| P2.2 多数据源降级 | 🟡 中 | akshare 失败时切换备用源 |
| P2.3 数据质量报告 | 🟡 中 | 自动生成数据质量报告 |
| P2.4 因子计算前校验 | 🟡 中 | 因子引擎添加输入校验 |

**产出**:
- `futureQuant/data/cache_manager.py` - 智能缓存
- `tests/integration/test_data_pipeline.py` - 集成测试

### Phase 3: 中期建设（2-4周）

| 任务 | 优先级 | 工作内容 |
|------|--------|----------|
| P3.1 数据源监控面板 | 🟢 低 | 监控各数据源健康状态 |
| P3.2 自动告警机制 | 🟢 低 | 数据异常时自动通知 |
| P3.3 数据版本控制 | 🟢 低 | 支持回溯到历史数据版本 |

**产出**:
- Web UI 数据监控页面
- Cron 定时检查任务

---

## 四、代码修改清单

### 文件1: `futureQuant/data/fetcher/akshare_fetcher.py`

```python
# 需要添加的方法
def _validate_date_range(df, start_date, end_date):
    """验证返回数据在请求范围内"""
    
def _check_data_quality(df):
    """检查数据质量"""
    
def _standardize_multi_contract(df, variety):
    """标准化多合约数据"""
```

### 文件2: `futureQuant/data/fetcher/fundamental_fetcher.py`

```python
# 需要修改的映射
INVENTORY_SYMBOLS = {
    # ... 原有 ...
    'FG': '玻璃',  # 添加这一行
}
```

### 文件3: 新增 `futureQuant/data/validator.py`

```python
class DataValidator:
    """数据验证器"""
    
    def validate_price_data(df):
        """验证价格数据"""
        
    def validate_fundamental_data(df):
        """验证基本面数据"""
        
    def validate_date_range(df, start, end):
        """验证日期范围"""
        
    def validate_freshness(df, max_days=5):
        """验证数据新鲜度"""
```

---

## 五、测试用例

```python
def test_date_range_validation():
    """测试日期范围验证"""
    # 场景1: 请求30天数据，应返回30天数据
    df = fetch_daily('RB', '2026-03-20', '2026-04-19')
    assert df['date'].min() >= '2026-03-20'
    assert df['date'].max() <= '2026-04-19'
    
def test_future_date_rejection():
    """测试拒绝未来日期"""
    with pytest.raises(ValidationError):
        fetch_daily('RB', '2026-04-19', '2026-06-01')  # 未来日期
        
def test_stale_data_warning():
    """测试数据过时警告"""
    df = fetch_daily('RB', '2026-01-01', '2026-01-31')
    # 应该产生警告
```

---

## 六、预期效果

修复后的数据流程:

```
请求数据 (2026-03-20 ~ 2026-04-19)
        ↓
akshare API 获取
        ↓
严格日期过滤 ← NEW: 只保留请求范围内的数据
        ↓
数据质量检查 ← NEW: 过滤异常价格/数据
        ↓
新鲜度验证  ← NEW: 确保数据不过时
        ↓
返回验证后的数据
```

---

## 七、风险与注意事项

1. **akshare API 限流**: 频繁请求可能触发限流
2. **数据源不稳定**: 部分接口可能返回空数据
3. **时区问题**: akshare 返回的时间可能是北京时间
4. **交易所节假日**: 交易日计算需要考虑节假日

---

**下一步行动**: 确认方案后，我将开始实施 Phase 1 的紧急修复。
