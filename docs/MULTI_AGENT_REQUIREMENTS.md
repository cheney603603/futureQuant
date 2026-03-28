# 多智能体因子挖掘与策略回测系统 - 需求分析文档

> **版本**: v1.0  
> **日期**: 2026-03-26  
> **状态**: 需求分析阶段

---

## 1. 背景与目标

### 1.1 背景

当前 futureQuant 框架已具备：
- ✅ 数据管理（`DataManager`）：支持日线、连续合约、基本面数据
- ✅ 因子库（`FactorEngine`）：16+ 技术因子 + 基本面因子 + 宏观因子
- ✅ 因子评估（`FactorEvaluator`）：IC、ICIR、分层回测
- ✅ 策略框架（`BaseStrategy`）：趋势跟踪、均值回归、套利策略
- ✅ 回测引擎（`BacktestEngine`）：向量化 + 事件驱动双引擎
- ✅ ML 模型（`MLForecastPipeline`）：XGBoost/LightGBM + Walk-forward
- ✅ 绩效报告（`PerformanceReport`）：文本/HTML/JSON 多格式输出

**现有痛点**：
1. **因子挖掘效率低**：依赖人工经验，缺乏自动化因子搜索和组合
2. **评估体系单一**：仅有 IC/分层回测，缺乏多维度综合评分
3. **过拟合风险高**：缺乏系统性的交叉验证和样本权重机制
4. **风控滞后**：回测阶段才发现风险，缺乏实时风控信号
5. **结果复现难**：因子和策略的演化过程缺乏可追溯记录

### 1.2 目标

构建一个 **多智能体协作的日频因子挖掘与策略自动回测系统**：

1. **自动化因子挖掘**：AI Agent 自动发现、评估、组合有效因子
2. **多维度评分体系**：IC、ICIR、换手率、稳定性、单调性等综合评分
3. **严格防泄漏**：自动检测未来函数，确保因子有效性
4. **时序交叉验证**：Purged K-Fold、Walk-forward 确保泛化能力
5. **智能样本权重**：近期样本权重更高，自动处理市场 regime 切换
6. **实时风控**：因子失效预警、仓位动态调整、止损机制

---

## 2. 功能需求

### 2.1 多智能体协作架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator Agent                           │
│  (任务调度、结果整合、最终决策)                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Factor Mining │   │  Validation   │   │  Risk Control │
│     Agent     │   │     Agent     │   │     Agent     │
│ (因子挖掘)    │   │ (验证评估)    │   │ (风控监控)    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  - 遗传规划   │   │  - Purged CV  │   │  - 因子衰减   │
│  - 表达式搜索 │   │  - Walk-Fwd   │   │  - 相关性监控 │
│  - 组合优化   │   │  - 样本权重   │   │  - 仓位限制   │
└───────────────┘   └───────────────┘   └───────────────┘
```

#### 2.1.1 Factor Mining Agent（因子挖掘智能体）

**职责**：
- 自动搜索新的因子表达式
- 组合现有因子生成合成因子
- 优化因子参数

**策略**：
1. **遗传规划（Genetic Programming）**：
   - 基础算子：`+`, `-`, `*`, `/`, `abs`, `log`, `rank`, `delta`, `delay`, `ts_mean`, `ts_std`
   - 基础数据：`close`, `open`, `high`, `low`, `volume`, `open_interest`, `vwap`
   - 进化策略：适应度 = IC × 稳定性 / 换手率

2. **表达式模板搜索**：
   - 动量类：`rank(return_N)`, `rank(delta(close, N))`
   - 反转类：`rank(-return_N)`, `rank(-delta(close, N))`
   - 波动类：`rank(-ts_std(return, N))`, `rank(ATR_N / close)`
   - 量价类：`rank(correlation(close, volume, N))`, `rank(OBV / volume)`

3. **因子组合**：
   - 正交化组合（去除共线性）
   - IC 加权组合
   - 机器学习组合（XGBoost/LightGBM 元学习）

**输出**：
- 新因子表达式
- 因子参数推荐
- 组合权重建议

#### 2.1.2 Validation Agent（验证评估智能体）

**职责**：
- 多维度因子评分
- 时序交叉验证
- 防泄漏检测
- 样本权重计算

**评分维度**：

| 维度 | 指标 | 权重 | 阈值 |
|------|------|------|------|
| 预测能力 | Rank IC | 30% | |IC| > 0.02 |
| 稳定性 | ICIR | 25% | ICIR > 0.5 |
| 单调性 | 分层收益单调性 | 15% | Spearman > 0.8 |
| 换手率 | 日均换手率 | 10% | Turnover < 50% |
| 覆盖率 | 有效数据占比 | 5% | Coverage > 80% |
| 正交性 | 与现有因子相关性 | 10% | |ρ| < 0.7 |
| 稳健性 | 不同市场环境下表现 | 5% | 通过率 > 60% |

**验证方法**：

1. **Purged K-Fold Cross Validation**：
   ```
   |----train----|purge|test|----train----|purge|test|...
   ```
   - Purge 区间：防止训练集信息泄漏到测试集
   - Embargo 区间：测试后等待期，防止标签泄漏

2. **Walk-Forward Validation**：
   ```
   |----train(N)----|test|----train(N+1)----|test|...
   ```
   - 滚动训练窗口
   - 样本外验证

3. **Combinatorial Purged CV**：
   - 同时测试多个窗口组合
   - 提高统计显著性

**样本权重**：
```python
# 时间衰减权重
time_weight = exp(-λ * (T - t) / T)

# 波动率调整权重
vol_weight = 1 / rolling_std(return, window)

# Regime 调整权重
regime_weight = regime_detector.get_weight(market_state)

# 综合权重
sample_weight = time_weight * vol_weight * regime_weight
```

#### 2.1.3 Risk Control Agent（风控监控智能体）

**职责**：
- 因子失效预警
- 实时相关性监控
- 仓位动态调整
- 止损/止盈信号

**监控指标**：

| 指标 | 预警阈值 | 强制阈值 | 动作 |
|------|----------|----------|------|
| 因子 IC 衰减 | IC < 0.01 | IC < 0 | 降低权重/剔除 |
| 因子相关性 | ρ > 0.6 | ρ > 0.8 | 正交化/剔除 |
| 回撤超限 | DD > 5% | DD > 10% | 减仓/止损 |
| 集中度超限 | Concentration > 30% | > 50% | 强制分散 |
| 杠杆超限 | Leverage > 2x | > 3x | 强制降仓 |

**风控动作**：
1. **预警**：发送通知，建议人工介入
2. **自动调整**：降低因子权重、减少仓位
3. **强制平仓**：触发止损线时自动平仓

---

### 2.2 因子评分体系

#### 2.2.1 单因子评分

```python
class FactorScore:
    """因子综合评分"""
    
    # 预测能力维度
    ic_mean: float          # Rank IC 均值
    ic_std: float           # Rank IC 标准差
    icir: float             # ICIR = IC_mean / IC_std
    ic_win_rate: float      # IC > 0 的比例
    
    # 分层表现维度
    quantile_monotonicity: float  # 分层收益单调性
    long_short_return: float      # 多空组合收益
    long_short_sharpe: float      # 多空组合夏普
    
    # 交易成本维度
    turnover: float         # 日均换手率
    correlation_cost: float # 换手相关成本
    
    # 稳定性维度
    ic_decay_rate: float    # IC 衰减速度
    regime_stability: float # 不同市场环境下的稳定性
    
    # 正交性维度
    max_correlation: float  # 与现有因子最大相关性
    
    # 综合得分
    composite_score: float  # 加权综合评分
```

#### 2.2.2 评分公式

```python
def calculate_composite_score(factor_score: FactorScore) -> float:
    """
    综合评分公式
    
    Score = 0.30 × IC_Score 
          + 0.25 × ICIR_Score 
          + 0.15 × Monotonicity_Score
          + 0.10 × Turnover_Score
          + 0.10 × Orthogonality_Score
          + 0.10 × Stability_Score
    """
    # IC Score: |IC| > 0.02 得 1 分，否则按比例
    ic_score = min(1.0, abs(factor_score.ic_mean) / 0.02)
    
    # ICIR Score: ICIR > 0.5 得 1 分
    icir_score = min(1.0, factor_score.icir / 0.5)
    
    # Monotonicity Score
    mono_score = max(0, factor_score.quantile_monotonicity)
    
    # Turnover Score: 换手率越低越好
    turnover_score = max(0, 1 - factor_score.turnover / 0.5)
    
    # Orthogonality Score: 相关性越低越好
    orth_score = max(0, 1 - abs(factor_score.max_correlation) / 0.7)
    
    # Stability Score
    stab_score = factor_score.regime_stability
    
    return (
        0.30 * ic_score +
        0.25 * icir_score +
        0.15 * mono_score +
        0.10 * turnover_score +
        0.10 * orth_score +
        0.10 * stab_score
    )
```

#### 2.2.3 因子等级

| 等级 | 综合得分 | 使用建议 |
|------|----------|----------|
| S 级 | ≥ 0.85 | 核心因子，权重 20-30% |
| A 级 | 0.70-0.85 | 重要因子，权重 10-20% |
| B 级 | 0.55-0.70 | 辅助因子，权重 5-10% |
| C 级 | 0.40-0.55 | 观察因子，权重 0-5% |
| D 级 | < 0.40 | 淘汰因子 |

---

### 2.3 防未来函数机制

#### 2.3.1 静态检测

```python
class FutureFunctionDetector:
    """未来函数检测器"""
    
    # 禁止的算子模式
    FORBIDDEN_PATTERNS = [
        r'shift\(\s*-[\d]+\s*\)',      # shift(-N)
        r'\.iloc\[[\d]+:\]',            # 未来索引
        r'\.loc\[.*:\]',                # 未来标签
        r'future_',                     # 未来变量命名
        r'lead\(',                      # lead 函数
        r'lookahead',                   # look-ahead
    ]
    
    def detect(self, factor_expression: str) -> List[str]:
        """检测因子表达式中的未来函数"""
        warnings = []
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, factor_expression):
                warnings.append(f"Potential future function: {pattern}")
        return warnings
```

#### 2.3.2 动态检测

```python
def detect_lookahead_bias(factor_values: pd.Series, 
                          returns: pd.Series,
                          max_lag: int = 5) -> Dict:
    """
    检测因子值与未来收益的相关性
    
    如果因子与 t+1 期以后收益的相关性异常高，
    可能存在未来函数。
    """
    correlations = {}
    for lag in range(1, max_lag + 1):
        future_ret = returns.shift(-lag)
        corr = factor_values.corr(future_ret, method='spearman')
        correlations[f'lag_{lag}'] = corr
    
    # 如果 lag_1 远高于其他，正常
    # 如果 lag_2+ 异常高，可能泄漏
    avg_later = np.mean([correlations[f'lag_{i}'] for i in range(2, max_lag + 1)])
    
    is_leaked = (correlations['lag_2'] > correlations['lag_1'] * 0.8) or \
                (abs(avg_later) > abs(correlations['lag_1']) * 0.5)
    
    return {
        'correlations': correlations,
        'is_leaked': is_leaked,
        'risk_level': 'HIGH' if is_leaked else 'LOW'
    }
```

#### 2.3.3 计算时保护

```python
class SafeFactorComputer:
    """安全因子计算器"""
    
    def compute(self, data: pd.DataFrame, expression: str) -> pd.Series:
        """
        计算因子值，确保无未来函数
        
        规则：
        1. 所有窗口函数只使用历史数据
        2. 禁止使用 shift(-N)
        3. 计算顺序严格按时间从前到后
        """
        # 解析表达式
        ast = self.parse_expression(expression)
        
        # 静态检测
        warnings = self.detector.detect(ast)
        if warnings:
            raise FutureFunctionError(warnings)
        
        # 安全计算
        result = self._safe_evaluate(ast, data)
        
        # 动态检测
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            leak_check = detect_lookahead_bias(result, returns)
            if leak_check['is_leaked']:
                logger.warning(f"Potential lookahead bias detected: {leak_check}")
        
        return result
```

---

### 2.4 时序交叉验证

#### 2.4.1 Purged K-Fold

```python
class PurgedKFold:
    """
    Purged K-Fold 交叉验证
    
    防止训练集信息泄漏到测试集。
    
    Example:
        >>> pkf = PurgedKFold(n_splits=5, purge_days=5, embargo_days=2)
        >>> for train_idx, test_idx in pkf.split(data):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 purge_days: int = 5,
                 embargo_days: int = 2):
        """
        Args:
            n_splits: 折数
            purge_days: 清除天数（训练集末尾）
            embargo_days: 禁运天数（测试集后）
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(self, data: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """生成分割索引"""
        n = len(data)
        fold_size = n // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            
            # 训练集：排除 purge 区间
            train_end = test_start - self.purge_days
            train_start = 0 if i == 0 else (i - 1) * fold_size + self.embargo_days
            
            if train_start < train_end:
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                yield train_idx, test_idx
```

#### 2.4.2 Walk-Forward 验证

```python
class WalkForwardValidator:
    """
    Walk-Forward 滚动验证
    
    模拟实盘部署场景。
    """
    
    def __init__(self,
                 train_window: int = 252,   # 1年训练
                 test_window: int = 63,     # 3个月测试
                 step_size: int = 63,       # 滚动步长
                 min_train_samples: int = 100):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples
    
    def split(self, data: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """生成训练/测试分割"""
        n = len(data)
        start = self.train_window
        
        while start + self.test_window <= n:
            train_data = data.iloc[start - self.train_window:start]
            test_data = data.iloc[start:start + self.test_window]
            
            if len(train_data) >= self.min_train_samples:
                yield train_data, test_data
            
            start += self.step_size
```

---

### 2.5 样本权重机制

#### 2.5.1 时间衰减权重

```python
def time_decay_weight(n_samples: int, 
                      decay_rate: float = 0.02) -> np.ndarray:
    """
    时间衰减权重
    
    近期样本权重更高，模拟市场状态演化。
    
    Args:
        n_samples: 样本数
        decay_rate: 衰减率（λ）
    
    Returns:
        权重数组，sum = n_samples
    """
    t = np.arange(n_samples)
    weights = np.exp(-decay_rate * (n_samples - 1 - t))
    weights = weights / weights.sum() * n_samples  # 归一化
    return weights
```

#### 2.5.2 波动率调整权重

```python
def volatility_weight(returns: pd.Series,
                      window: int = 20) -> pd.Series:
    """
    波动率调整权重
    
    高波动期降低权重，避免极端值主导训练。
    """
    rolling_vol = returns.rolling(window).std()
    weights = 1 / (rolling_vol + 1e-6)
    weights = weights / weights.mean()  # 均值为1
    return weights
```

#### 2.5.3 Regime 调整权重

```python
class RegimeWeightCalculator:
    """
    市场状态权重计算器
    
    不同市场状态下赋予不同权重：
    - 趋势市：趋势因子权重高
    - 震荡市：反转因子权重高
    - 极端市：降低权重或剔除
    """
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算 regime 权重"""
        regime = self.regime_detector.detect(data)
        
        weights = pd.Series(1.0, index=data.index)
        
        # 极端市场（暴涨暴跌）降低权重
        extreme_mask = regime['is_extreme']
        weights[extreme_mask] = 0.5
        
        # 根据 regime 类型调整
        weights[regime['regime'] == 'trending'] *= 1.2
        weights[regime['regime'] == 'ranging'] *= 0.8
        
        return weights
```

---

### 2.6 风险控制

#### 2.6.1 因子失效预警

```python
class FactorDecayMonitor:
    """因子衰减监控器"""
    
    def __init__(self, 
                 ic_threshold: float = 0.01,
                 decay_window: int = 20):
        self.ic_threshold = ic_threshold
        self.decay_window = decay_window
        self.ic_history = []
    
    def update(self, ic: float) -> Dict:
        """更新 IC 并检测衰减"""
        self.ic_history.append(ic)
        
        if len(self.ic_history) < self.decay_window:
            return {'status': 'insufficient_data'}
        
        recent_ic = np.mean(self.ic_history[-self.decay_window:])
        trend = np.polyfit(range(self.decay_window), 
                          self.ic_history[-self.decay_window:], 1)[0]
        
        # 预警条件
        is_decaying = trend < 0 and abs(trend) > 0.001
        is_below_threshold = recent_ic < self.ic_threshold
        
        status = 'normal'
        if is_below_threshold:
            status = 'critical'
        elif is_decaying:
            status = 'warning'
        
        return {
            'status': status,
            'recent_ic': recent_ic,
            'ic_trend': trend,
            'recommendation': self._get_recommendation(status)
        }
```

#### 2.6.2 仓位管理

```python
class PositionManager:
    """
    智能仓位管理
    
    根据因子强度、市场波动率、账户风险动态调整仓位。
    """
    
    def __init__(self,
                 max_position: float = 1.0,
                 max_leverage: float = 2.0,
                 risk_budget: float = 0.02,    # 单日最大风险敞口
                 vol_target: float = 0.15):     # 目标年化波动率
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.risk_budget = risk_budget
        self.vol_target = vol_target
    
    def calculate_position(self,
                          signal: float,
                          confidence: float,
                          volatility: float,
                          current_drawdown: float) -> float:
        """
        计算建议仓位
        
        Args:
            signal: 信号强度 (-1 to 1)
            confidence: 置信度 (0 to 1)
            volatility: 当前波动率
            current_drawdown: 当前回撤
        
        Returns:
            建议仓位比例 (-max_position to max_position)
        """
        # 基础仓位
        base_position = signal * confidence
        
        # 波动率调整
        vol_adjustment = self.vol_target / (volatility + 1e-6)
        vol_adjustment = min(2.0, max(0.5, vol_adjustment))
        
        # 回撤调整（回撤越大越保守）
        dd_adjustment = max(0.3, 1 - current_drawdown * 3)
        
        # 最终仓位
        position = base_position * vol_adjustment * dd_adjustment
        
        # 边界约束
        position = np.clip(position, -self.max_position, self.max_position)
        
        return position
```

---

## 3. 非功能需求

### 3.1 性能要求

| 指标 | 要求 | 说明 |
|------|------|------|
| 因子计算 | < 100ms/因子 | 单品种单日 |
| 因子评估 | < 5s/因子 | 全历史数据 |
| 回测速度 | > 1000 bar/s | 向量化模式 |
| 内存占用 | < 4GB | 单品种5年数据 |
| 并发支持 | 10 并发因子计算 | 多品种并行 |

### 3.2 可靠性要求

| 指标 | 要求 |
|------|------|
| 数据完整性 | > 99.9% |
| 计算准确性 | 与手工计算误差 < 0.01% |
| 异常恢复 | 支持断点续算 |
| 日志记录 | 全流程可追溯 |

### 3.3 可扩展性要求

- 支持自定义因子表达式
- 支持自定义评分维度
- 支持自定义风控规则
- 支持插件式智能体扩展

---

## 4. 接口需求

### 4.1 用户接口

```python
# 高层 API
from futureQuant.agent import FactorMiningSystem

system = FactorMiningSystem(config={
    'n_agents': 3,
    'validation_method': 'purged_kfold',
    'risk_control': True,
})

# 运行因子挖掘
results = system.run(
    data=price_data,
    target='5d_return',
    max_factors=50,
)

# 获取最优因子
best_factors = results.get_top_factors(n=10)

# 生成策略
strategy = results.generate_strategy()
```

### 4.2 输出报告

```
╔══════════════════════════════════════════════════════════════╗
║            多智能体因子挖掘报告 - 2026-03-26                  ║
╠══════════════════════════════════════════════════════════════╣
║ 【因子挖掘统计】                                              ║
║   搜索空间: 1,000+ 表达式                                     ║
║   有效因子: 23 个                                             ║
║   S级因子: 3 个 | A级: 7 个 | B级: 8 个 | C级: 5 个          ║
║                                                               ║
║ 【Top-5 因子】                                                ║
║   1. rank(delta(close, 5) / ts_std(close, 20)) [S级, 0.89]  ║
║   2. rank(-ts_corr(close, volume, 10)) [S级, 0.87]          ║
║   3. rank(delta(OBV, 5) / volume) [S级, 0.85]               ║
║   4. rank(ATR_14 / close * 100) [A级, 0.78]                 ║
║   5. rank(-delta(close, 20) / close) [A级, 0.76]            ║
║                                                               ║
║ 【验证结果】                                                  ║
║   Purged 5-Fold CV: 平均 IC = 0.032 ± 0.008                  ║
║   Walk-Forward: 样本外夏普 = 1.23                             ║
║   样本权重: 时间衰减 λ=0.02                                   ║
║                                                               ║
║ 【风控状态】                                                  ║
║   因子相关性矩阵: 最大 |ρ| = 0.42 (安全)                      ║
║   推荐仓位: 80% (基于当前信号强度)                            ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 5. 约束与假设

### 5.1 约束

1. **数据约束**：
   - 仅支持日线级别数据
   - 需要至少 1 年历史数据
   - 需要品种基准信息（合约乘数、保证金率）

2. **计算约束**：
   - 因子表达式长度 < 100 字符
   - 单次因子搜索时间 < 10 分钟
   - 并发因子数 < 20

3. **业务约束**：
   - 不支持高频交易
   - 不支持日内交易
   - 仅支持单品种或多品种独立策略

### 5.2 假设

1. 市场有效性假设：历史模式可能在未来重复
2. 流动性假设：主力合约流动性充足
3. 稳定性假设：因子在 1-3 个月内保持有效

---

## 6. 风险与挑战

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 因子过拟合 | 策略实盘失效 | Purged CV + Walk-forward |
| 市场 regime 切换 | 因子集体失效 | Regime 检测 + 动态权重 |
| 计算资源不足 | 挖掘效率低 | 分布式计算 + 增量更新 |
| 数据质量差 | 因子噪声大 | 数据清洗 + 异常检测 |
| 相关性崩溃 | 多因子失效 | 实时监控 + 分散化 |

---

## 7. 验收标准

### 7.1 功能验收

- [ ] 多智能体协作运行正常
- [ ] 因子自动挖掘产出 S 级因子
- [ ] 防未来函数检测有效
- [ ] Purged CV 验证通过
- [ ] 样本权重计算正确
- [ ] 风控预警及时触发

### 7.2 性能验收

- [ ] 因子计算耗时达标
- [ ] 回测速度达标
- [ ] 内存占用达标
- [ ] 并发处理正常

### 7.3 质量验收

- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
- [ ] 文档完整
- [ ] 代码审查通过

---

## 8. 术语表

| 术语 | 定义 |
|------|------|
| IC | Information Coefficient，因子值与未来收益的秩相关系数 |
| ICIR | IC 的信息比率，IC 均值 / IC 标准差 |
| Purged CV | 清除交叉验证，去除训练集末尾数据防止泄漏 |
| Embargo | 禁运期，测试后等待期防止标签泄漏 |
| Walk-Forward | 滚动前向验证，模拟实盘部署 |
| Future Function | 未来函数，使用未来数据计算当前值 |
| Regime | 市场状态，如趋势、震荡、极端 |
| Turnover | 换手率，因子值变化导致的持仓调整比例 |
