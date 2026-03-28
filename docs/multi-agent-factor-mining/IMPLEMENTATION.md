# 多智能体因子挖掘与策略自动回测系统 - 实现计划

## 1. 里程碑规划总览

```
v1.0 ✅ 基础框架 (2026-03-26) [当前]
   │
   ├── Phase 1: Agent 基础设施 (5 天)
   ├── Phase 2: 挖掘 Agent 开发 (7 天)
   ├── Phase 3: 验证 Agent 开发 (7 天)
   └── Phase 4: 回测与因子库 (5 天)
   │
   ▼
v1.1 🔄 完整功能 (预计 4 周)
   ├── Phase 5: 集成测试
   ├── Phase 6: 文档与示例
   └── Phase 7: 性能优化
   │
   ▼
v1.5 ⏳ 高级功能 (后续迭代)
   ├── 因子组合优化
   ├── 因子解释性分析
   └── 实时监控
   │
   ▼
v2.0 ⏳ 生产就绪
   ├── 分布式计算
   ├── 深度学习因子
   └── 高频因子
```

---

## Phase 1: Agent 基础设施

**目标**: 建立多智能体框架的基础架构

**时间**: 5 天

### 任务分解

| 任务 | 优先级 | 预估工时 | 依赖 | 产出 |
|------|--------|---------|------|------|
| T1.1 Agent 基类实现 | P0 | 2d | - | `agent/base.py` |
| T1.2 状态机与通信协议 | P0 | 1d | T1.1 | `agent/state_machine.py` |
| T1.3 日志与追踪系统 | P0 | 1d | T1.1 | 统一日志格式 |
| T1.4 配置管理模块 | P1 | 1d | - | `config/agent_settings.yaml` |

### 详细任务

#### T1.1 Agent 基类实现

```python
# agent/base.py 核心实现

class BaseAgent(ABC):
    """Agent 抽象基类"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.status = AgentStatus.IDLE
        self.logger = get_logger(f'agent.{name.lower()}')
    
    @abstractmethod
    def execute(self, context: Context) -> AgentResult:
        """执行任务"""
        pass
    
    def run(self, context: Context) -> AgentResult:
        """带状态管理的运行"""
        self.status = AgentStatus.RUNNING
        try:
            result = self.execute(context)
            self.status = result.status
            return result
        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(status=AgentStatus.FAILED, errors=[str(e)])
```

**验收标准:**
- [ ] BaseAgent 类完整实现
- [ ] AgentResult 数据类定义
- [ ] AgentStatus 枚举定义
- [ ] 单元测试覆盖 > 80%

#### T1.2 状态机与通信协议

```python
# agent/state_machine.py

class AgentState(Enum):
    IDLE = "idle"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentMessage:
    """Agent 间通信消息"""
    sender: str
    receiver: str
    message_type: MessageType  # TASK / RESULT / ERROR / HEARTBEAT
    payload: Dict
    timestamp: datetime

class StateMachine:
    """Agent 状态机"""
    
    def transition(self, agent: str, event: Event):
        """状态转换"""
        pass
```

#### T1.3 日志与追踪系统

统一日志格式，便于问题排查和性能分析：

```
[2026-03-26 10:00:00] [INFO] [TechnicalMiningAgent] Starting execution
[2026-03-26 10:00:01] [DEBUG] [TechnicalMiningAgent] Computed momentum_factor_20
[2026-03-26 10:00:02] [INFO] [TechnicalMiningAgent] IC=0.034, passed threshold
[2026-03-26 10:00:02] [INFO] [TechnicalMiningAgent] Completed: 5 factors discovered
```

#### T1.4 配置管理模块

配置文件结构：

```yaml
# config/agent_settings.yaml

agent:
  system:
    log_level: INFO
    log_dir: ./logs/agent
    n_workers: 4
  
  mining:
    technical:
      enabled: true
      ic_threshold: 0.02
      momentum_windows: [5, 10, 20, 60]
    
    fundamental:
      enabled: true
      data_lag:
        inventory: 3
        warehouse: 2
    
    macro:
      enabled: true
  
  validation:
    lookahead:
      enabled: true
      static_analysis: true
      dynamic_test: true
    
    cross_validation:
      method: walk_forward
      n_splits: 5
      train_size: 252
      test_size: 63
  
  repository:
    storage_dir: ./factor_repo
    metadata_db: ./factor_repo/metadata.db
```

---

## Phase 2: 挖掘 Agent 开发

**目标**: 实现技术因子、基本面因子、宏观因子挖掘 Agent

**时间**: 7 天

### 任务分解

| 任务 | 优先级 | 预估工时 | 依赖 | 产出 |
|------|--------|---------|------|------|
| T2.1 技术因子挖掘 Agent | P0 | 2d | Phase 1 | `agent/miners/technical_agent.py` |
| T2.2 基本面因子挖掘 Agent | P0 | 2d | Phase 1 | `agent/miners/fundamental_agent.py` |
| T2.3 宏观因子挖掘 Agent | P1 | 1.5d | Phase 1 | `agent/miners/macro_agent.py` |
| T2.4 因子融合 Agent | P0 | 1.5d | T2.1-T2.3 | `agent/miners/fusion_agent.py` |

### 详细任务

#### T2.1 技术因子挖掘 Agent

**实现要点:**

1. **动量因子模板:**
   ```python
   class MomentumTemplate:
       """动量因子计算模板"""
       def generate(window: int) -> Factor:
           """生成指定窗口的动量因子"""
           return MomentumFactor(window=window)
   ```

2. **因子搜索空间:**
   ```python
   SEARCH_SPACE = {
       'momentum': {
           'windows': [5, 10, 20, 60, 120, 250],
           'accelerations': [True, False],  # 是否计算加速度
       },
       'volatility': {
           'windows': [10, 20, 60],
           'types': ['std', 'atr', 'parkinson'],  # 不同波动率计算方式
       },
       'volume': {
           'windows': [5, 10, 20],
           'types': ['ratio', 'change', 'on_balance'],
       },
   }
   ```

3. **快速 IC 评估:**
   ```python
   def quick_ic_eval(factor: Factor, data: DataFrame) -> float:
       """快速评估因子 IC（用于筛选）"""
       # 使用采样数据快速计算
       sample = data.iloc[::5]  # 每 5 天采样
       factor_values = factor.compute(sample)
       returns = sample['close'].pct_change()
       return spearmanr(factor_values, returns)[0]
   ```

**验收标准:**
- [ ] 支持 20+ 技术因子模板
- [ ] IC 阈值过滤机制
- [ ] 因子去重逻辑
- [ ] 集成测试通过

#### T2.2 基本面因子挖掘 Agent

**实现要点:**

1. **数据延迟处理:**
   ```python
   class FundamentalAgent:
       def __init__(self):
           self.data_lag = {
               'inventory': 3,
               'warehouse': 2,
               'basis': 1,
           }
       
       def compute_factor(self, factor: Factor, data: DataFrame):
           """计算基本面因子时考虑数据延迟"""
           lag = self.data_lag.get(factor.category, 0)
           factor_values = factor.compute(data)
           return factor_values.shift(lag)  # 延迟生效
   ```

2. **因子模板:**
   ```python
   TEMPLATES = {
       'basis': [
           BasisRatio,           # 基差率
           BasisMomentum,        # 基差动量
           BasisBreakout,        # 基差突破
       ],
       'inventory': [
           InventoryChange,      # 库存变化率
           InventoryCycle,       # 库存周期
           InventoryPressure,    # 库存压力
       ],
       'warehouse': [
           WarrantRatio,         # 仓单比率
           WarrantCancel,        # 仓单注销率
       ],
   }
   ```

**验收标准:**
- [ ] 支持 15+ 基本面因子模板
- [ ] 数据延迟自动处理
- [ ] 基本面数据校验
- [ ] 与技术因子统一评分

#### T2.3 宏观因子挖掘 Agent

**实现要点:**

1. **宏观数据映射:**
   ```python
   class MacroAgent:
       """宏观因子挖掘"""
       
       def _map_to_daily(self, macro_data: DataFrame) -> DataFrame:
           """将低频宏观数据映射到日频"""
           # 使用前值填充或插值
           return macro_data.resample('D').ffill()
       
       def _compute_cross_market(self, data_dict: Dict) -> DataFrame:
           """计算跨市场联动因子"""
           pass
   ```

2. **因子模板:**
   ```python
   MACRO_INDICATORS = {
       'economic': ['gdp', 'cpi', 'ppi', 'pmi'],
       'financial': ['m2', 'interest_rate', 'exchange_rate'],
       'commodity': ['crb_index', 'nanhua_index'],
   }
   ```

#### T2.4 因子融合 Agent

**实现要点:**

1. **因子去相关:**
   ```python
   class FusionAgent:
       """因子融合 Agent"""
       
       def decorrelate(self, factors: List[Factor]) -> List[Factor]:
           """因子正交化处理"""
           # 计算因子间相关性矩阵
           corr_matrix = self._compute_correlation(factors)
           
           # 使用层次聚类分组
           clusters = self._hierarchical_clustering(corr_matrix)
           
           # 每组保留 IC 最高的因子
           return self._select_representatives(clusters)
       
       def combine(self, factors: List[Factor], method: str = 'icir_weighted'):
           """多因子加权合成"""
           if method == 'icir_weighted':
               return self._icir_weighted_combine(factors)
           elif method == 'max_sharpe':
               return self._max_sharpe_combine(factors)
   ```

2. **加权方法:**
   - ICIR 加权
   - 最大化夏普加权
   - 等权组合
   - 因子正交化后等权

---

## Phase 3: 验证 Agent 开发

**目标**: 实现防未来函数检测、时序交叉验证、样本权重、多维度评分

**时间**: 7 天

### 任务分解

| 任务 | 优先级 | 预估工时 | 依赖 | 产出 |
|------|--------|---------|------|------|
| T3.1 未来函数检测 Agent | P0 | 2d | Phase 1 | `agent/validators/lookahead_detector.py` |
| T3.2 时序交叉验证 Agent | P0 | 2d | Phase 1 | `agent/validators/cross_validator.py` |
| T3.3 样本权重 Agent | P1 | 1.5d | Phase 1 | `agent/validators/sample_weighter.py` |
| T3.4 多维度评分 Agent | P0 | 1.5d | T3.1-T3.3 | `agent/validators/scorer.py` |

### 详细任务

#### T3.1 未来函数检测 Agent

**实现要点:**

1. **静态检测规则库:**
   ```python
   DANGEROUS_PATTERNS = [
       # 负向 shift
       (r'shift\(-\d+\)', 'Negative shift detected'),
       (r'shift\(-[a-zA-Z_]+\)', 'Variable negative shift'),
       
       # 使用未来数据
       (r'\.rolling\(.*\).*\.shift\(-', 'Rolling with future shift'),
       
       # pct_change 使用不当
       (r'pct_change\(.*-1\)', 'Future pct_change'),
   ]
   ```

2. **动态测试框架:**
   ```python
   def dynamic_test(factor: Factor, data: DataFrame) -> TestResult:
       """
       动态测试：验证信号延迟后因子是否失效
       
       原理：如果因子使用了未来数据，
       延迟一期执行后 IC 应该显著下降。
       """
       # 原始 IC
       ic_original = compute_ic(factor, data)
       
       # 延迟执行后的 IC
       data_delayed = data.shift(-1)
       ic_delayed = compute_ic(factor, data_delayed)
       
       # 判断
       if ic_delayed < ic_original * 0.5:
           return TestResult(passed=False, reason="IC significantly dropped")
       return TestResult(passed=True)
   ```

3. **检测报告:**
   ```python
   class LookAheadReport:
       factor_name: str
       is_clean: bool
       issues: List[str]
       static_result: StaticAnalysis
       dynamic_result: DynamicTest
       recommendations: List[str]
   ```

#### T3.2 时序交叉验证 Agent

**实现要点:**

1. **三种验证模式:**

   ```python
   class TimeSeriesCrossValidator:
       """
       支持三种交叉验证模式：
       1. Walk-Forward: 滚动训练/测试窗口
       2. Expanding: 扩展训练窗口
       3. Purged K-Fold: 带清洗期的 K 折验证
       """
       
       def walk_forward_cv(self, factor, data, returns):
           """Walk-Forward 验证"""
           results = []
           for i in range(n_windows):
               train_data = data[train_start:train_end]
               test_data = data[test_start:test_end]
               
               # 训练
               train_ic = self._compute_ic(factor, train_data)
               
               # 测试
               test_ic = self._compute_ic(factor, test_data)
               
               results.append({'train_ic': train_ic, 'test_ic': test_ic})
           
           return results
   ```

2. **稳定性判断标准:**
   ```python
   def is_stable(cv_results: List[CVResult]) -> bool:
       """判断因子是否稳定"""
       
       # 条件1：测试集 IC 方向一致
       test_ics = [r.test_ic for r in cv_results]
       sign_consistency = (sum(ic > 0 for ic in test_ics) / len(test_ics))
       
       # 条件2：测试集 IC 均值 > 阈值
       mean_test_ic = np.mean(test_ics)
       
       # 条件3：训练-测试 IC 差异不过大（避免过拟合）
       train_ics = [r.train_ic for r in cv_results]
       ic_diff = abs(np.mean(train_ics) - mean_test_ic)
       
       return (
           sign_consistency > 0.7 and
           abs(mean_test_ic) > 0.02 and
           ic_diff < 0.05
       )
   ```

#### T3.3 样本权重 Agent

**实现要点:**

1. **权重计算方法:**

   ```python
   class SampleWeighter:
       """样本权重计算"""
       
       def compute_weights(
           self,
           data: DataFrame,
           returns: Series,
           method: str = 'volatility'
       ) -> Series:
           """
           计算样本权重
           
           Args:
               method: 'volatility' / 'liquidity' / 'regime' / 'custom'
           """
           if method == 'volatility':
               return self._volatility_weighting(data)
           elif method == 'liquidity':
               return self._liquidity_weighting(data)
           elif method == 'regime':
               return self._market_regime_weighting(data, returns)
           else:
               return pd.Series(1.0, index=data.index)
       
       def _volatility_weighting(self, data: DataFrame) -> Series:
           """基于波动率的权重：高波动期降权"""
           returns = data['close'].pct_change()
           volatility = returns.rolling(20).std()
           
           # 波动率越高，权重越低
           weights = 1 / volatility
           weights = weights / weights.sum()  # 归一化
           
           return weights.fillna(1.0)
   ```

2. **加权评估:**

   ```python
   def weighted_ic(
       factor: Series,
       returns: Series,
       weights: Series
   ) -> float:
       """加权 IC 计算"""
       # 使用加权相关系数
       return weighted_spearmanr(factor, returns, weights)[0]
   ```

#### T3.4 多维度评分 Agent

**实现要点:**

1. **评分维度与权重:**

   ```python
   SCORING_CONFIG = {
       'prediction': {
           'weight': 0.35,
           'metrics': ['ic_mean', 'icir', 'ic_win_rate'],
       },
       'stability': {
           'weight': 0.25,
           'metrics': ['monthly_ic_std', 'yearly_ic_std'],
       },
       'monotonicity': {
           'weight': 0.20,
           'metrics': ['spearman_rho', 'long_short_return'],
       },
       'turnover': {
           'weight': 0.10,
           'metrics': ['avg_turnover'],
       },
       'risk': {
           'weight': 0.10,
           'metrics': ['max_drawdown', 'downside_vol'],
       },
   }
   ```

2. **评分输出:**

   ```python
   class FactorScorecard:
       """因子评分卡"""
       
       def generate(self, factor: Factor, data: DataFrame) -> Scorecard:
           scores = {}
           
           for dimension, config in SCORING_CONFIG.items():
               score = self._compute_dimension_score(
                   factor, data, config
               )
               scores[dimension] = score
           
           overall = sum(
               s * SCORING_CONFIG[d]['weight']
               for d, s in scores.items()
           )
           
           return Scorecard(
               factor_name=factor.name,
               overall_score=overall,
               dimension_scores=scores,
               passed=overall >= 0.6,
           )
   ```

---

## Phase 4: 回测与因子库

**目标**: 实现策略自动生成、风险控制、回测报告和因子库管理

**时间**: 5 天

### 任务分解

| 任务 | 优先级 | 预估工时 | 依赖 | 产出 |
|------|--------|---------|------|------|
| T4.1 策略自动生成器 | P0 | 1.5d | Phase 2-3 | `agent/backtest/strategy_generator.py` |
| T4.2 风险控制器 | P0 | 1d | Phase 2-3 | `agent/backtest/risk_controller.py` |
| T4.3 回测报告生成器 | P0 | 1d | T4.1-T4.2 | `agent/backtest/report_generator.py` |
| T4.4 因子库管理器 | P1 | 1.5d | Phase 1-3 | `agent/repository/` |

### 详细任务

#### T4.1 策略自动生成器

**实现要点:**

```python
class StrategyGenerator:
    """策略自动生成"""
    
    def generate(
        self,
        factor: Factor,
        config: StrategyConfig
    ) -> Strategy:
        """
        将因子自动转化为策略
        """
        # 1. 确定信号生成规则
        # 因子值 > 上阈值 → 做多
        # 因子值 < 下阈值 → 做空
        # 其他 → 空仓
        
        # 2. 生成策略代码
        strategy_code = self._generate_code(factor, config)
        
        # 3. 返回策略实例
        return self._create_strategy(factor, strategy_code, config)
    
    def _generate_signal(
        self,
        factor_values: Series,
        upper_threshold: float,
        lower_threshold: float
    ) -> Series:
        """生成交易信号"""
        signal = pd.Series(0, index=factor_values.index)
        signal[factor_values > upper_threshold] = 1   # 做多
        signal[factor_values < lower_threshold] = -1  # 做空
        return signal
```

#### T4.2 风险控制器

**实现要点:**

```python
class RiskController:
    """风险控制器"""
    
    def __init__(self, config: RiskConfig):
        self.stop_loss = config.stop_loss
        self.take_profit = config.take_profit
        self.max_position = config.max_position
        self.max_drawdown = config.max_drawdown
    
    def apply_risk_rules(
        self,
        positions: Series,
        prices: DataFrame,
        equity: Series,
        context: RiskContext
    ) -> Series:
        """
        应用风险控制规则
        """
        adjusted_positions = positions.copy()
        
        # 1. 止损检查
        for sym in context.positions:
            pnl = context.positions[sym].unrealized_pnl
            if pnl < -self.stop_loss:
                adjusted_positions[sym] = 0  # 止损平仓
        
        # 2. 仓位限制
        adjusted_positions = self._apply_position_limit(adjusted_positions)
        
        # 3. 回撤控制
        if context.current_drawdown > self.max_drawdown:
            adjusted_positions *= 0.5  # 降低半仓
        
        return adjusted_positions
```

#### T4.3 回测报告生成器

**实现要点:**

```python
class BacktestReportGenerator:
    """回测报告生成"""
    
    def generate(
        self,
        backtest_result: BacktestResult,
        format: str = 'text'
    ) -> Report:
        """
        生成回测报告
        
        Args:
            backtest_result: 回测结果
            format: 'text' / 'html' / 'json'
        """
        report_data = {
            # 收益指标
            'total_return': backtest_result.total_return,
            'annual_return': backtest_result.annual_return,
            'excess_return': backtest_result.excess_return,
            
            # 风险指标
            'max_drawdown': backtest_result.max_drawdown,
            'volatility': backtest_result.volatility,
            'var_95': backtest_result.var_95,
            
            # 风险调整收益
            'sharpe_ratio': backtest_result.sharpe_ratio,
            'sortino_ratio': backtest_result.sortino_ratio,
            'calmar_ratio': backtest_result.calmar_ratio,
            
            # 交易统计
            'total_trades': backtest_result.total_trades,
            'win_rate': backtest_result.win_rate,
            'profit_factor': backtest_result.profit_factor,
            'avg_holding_days': backtest_result.avg_holding_days,
        }
        
        if format == 'text':
            return self._format_text(report_data)
        elif format == 'html':
            return self._format_html(report_data)
        else:
            return report_data
```

#### T4.4 因子库管理器

**实现要点:**

```python
class FactorRepository:
    """因子库管理器"""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.metadata_db = SQLiteDB(storage_dir / 'metadata.db')
        self.value_store = ParquetStore(storage_dir / 'values')
    
    def save_factor(
        self,
        factor: Factor,
        values: DataFrame,
        performance: PerformanceRecord
    ):
        """保存因子"""
        # 1. 保存元数据
        self.metadata_db.insert('factor_metadata', {
            'factor_id': factor.factor_id,
            'name': factor.name,
            'category': factor.category,
            'parameters': json.dumps(factor.params),
            'created_at': datetime.now(),
            'status': 'active',
        })
        
        # 2. 保存因子值
        self.value_store.save(factor.factor_id, values)
        
        # 3. 保存性能记录
        self.metadata_db.insert('factor_performance', {
            'factor_id': factor.factor_id,
            'ic_mean': performance.ic_mean,
            'icir': performance.icir,
            'overall_score': performance.overall_score,
        })
    
    def get_factor(
        self,
        factor_id: str,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[FactorRecord]:
        """获取因子"""
        pass
    
    def track_performance(
        self,
        factor_id: str,
        period: str = 'monthly'
    ) -> PerformanceTrend:
        """追踪因子表现"""
        # 滚动计算 IC 等指标
        # 检测因子衰减
        # 触发预警
        pass
```

---

## Phase 5: 集成测试

**目标**: 确保各模块正确协作

**时间**: 3 天

### 任务分解

| 任务 | 优先级 | 预估工时 | 依赖 | 产出 |
|------|--------|---------|------|------|
| T5.1 端到端测试 | P0 | 1d | Phase 1-4 | `tests/integration/test_e2e.py` |
| T5.2 Agent 协作测试 | P0 | 1d | Phase 1-4 | `tests/integration/test_agent_coordination.py` |
| T5.3 性能基准测试 | P1 | 1d | Phase 1-4 | `tests/performance/` |

### 验收标准

- [ ] 端到端流程测试通过（挖掘 → 验证 → 回测）
- [ ] Agent 间通信正常
- [ ] 并行计算正确
- [ ] 性能达标：单因子 < 5s，批量 100 因子 < 5min

---

## Phase 6: 文档与示例

**目标**: 完善用户文档和使用示例

**时间**: 2 天

### 任务分解

| 任务 | 优先级 | 预估工时 | 依赖 | 产出 |
|------|--------|---------|------|------|
| T6.1 API 文档 | P0 | 0.5d | Phase 1-4 | `docs/api.md` |
| T6.2 使用示例 | P0 | 1d | Phase 1-4 | `examples/agent/` |
| T6.3 用户指南 | P1 | 0.5d | T6.1 | `docs/user_guide.md` |

### 文档内容

1. **API 文档**: 所有公开类的接口说明
2. **使用示例**:
   - 基本用法（单品种因子挖掘）
   - 多品种批量挖掘
   - 自定义 Agent
   - 因子库使用
3. **用户指南**: 配置说明、常见问题

---

## Phase 7: 性能优化

**目标**: 提升系统性能

**时间**: 2 天

### 优化方向

1. **并行计算优化**
   - 使用 joblib 加速因子计算
   - 多进程数据加载

2. **存储优化**
   - Parquet 列存压缩
   - SQLite 索引优化

3. **算法优化**
   - 快速 IC 计算（采样）
   - 增量计算

---

## 风险控制

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| Agent 协作复杂 | 中 | 高 | 分阶段开发，每阶段完整测试 |
| 未来函数检测漏报 | 低 | 高 | 静态+动态双重检测，保守标记 |
| 性能不达标 | 中 | 中 | 预留优化时间，优先保证正确性 |
| 数据质量问题 | 中 | 中 | 数据校验机制，异常处理 |

---

## 开发建议

### 推荐的开发顺序

```
Phase 1 (Agent 基础)
    ↓
Phase 2 (挖掘 Agent) ←→ Phase 3 (验证 Agent)
    ↓
Phase 4 (回测与因子库)
    ↓
Phase 5-7 (测试/文档/优化)
```

### 关键技术难点

1. **多 Agent 协调**: 使用状态机模式，明确消息协议
2. **防未来函数**: 静态分析 + 动态测试双重保障
3. **因子评分**: 合理的多维度权重设置
4. **性能优化**: 并行计算与增量计算

### 测试策略

1. **单元测试**: 每个 Agent 独立测试
2. **集成测试**: Agent 间协作测试
3. **端到端测试**: 完整流程测试
4. **回归测试**: 防止引入新 Bug

---

## 工时汇总

| Phase | 任务数 | 总工时 | 说明 |
|-------|--------|--------|------|
| Phase 1 | 4 | 5d | Agent 基础设施 |
| Phase 2 | 4 | 7d | 挖掘 Agent |
| Phase 3 | 4 | 7d | 验证 Agent |
| Phase 4 | 4 | 5d | 回测与因子库 |
| Phase 5 | 3 | 3d | 集成测试 |
| Phase 6 | 3 | 2d | 文档与示例 |
| Phase 7 | 1 | 2d | 性能优化 |
| **总计** | **23** | **31d** | 约 6 周 |

> ⚠️ **说明**: 以上工时为保守估算，实际开发可能因团队熟悉度、需求变更等因素有所偏差。建议每周进行一次进度 review，及时调整计划。

---

**文档版本**: v1.0
**创建日期**: 2026-03-26
**最后更新**: 2026-03-26
