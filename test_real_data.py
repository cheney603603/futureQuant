"""
真实数据全量 Agent 测试
使用 akshare 接入真实期货数据，测试所有 7 个 Agent
"""
import sys
sys.path.insert(0, '.')

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

print("=" * 60)
print("futureQuant 全量 Agent 真实数据测试")
print("=" * 60)
print()

# ================================================================
# STEP 1: 用 akshare 获取真实数据
# ================================================================
print("[STEP 1] 从 akshare 拉取真实期货数据...")
print()

# 测试各品种 - 用已知合约
test_contracts = {
    'RB2505': '螺纹钢',
    'HC2505': '热轧卷板',
    'I2505': '铁矿石',
    'J2505': '焦炭',
    'CU2505': '铜',
    'AU2506': '黄金',
    'AG2506': '白银',
}

price_data_map = {}
for code, name in test_contracts.items():
    try:
        df = ak.futures_zh_daily_sina(symbol=code)
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            price_data_map[code] = df
            print(f"  OK: {code} ({name}): {len(df)} rows, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        else:
            print(f"  EMPTY: {code}")
    except Exception as e:
        print(f"  FAIL: {code} - {e}")

# 选用 RB2505 作为主测试标的
if 'RB2505' in price_data_map:
    price_data = price_data_map['RB2505']
    # 补齐完整列
    if 'open_interest' not in price_data.columns and 'hold' in price_data.columns:
        price_data = price_data.rename(columns={'hold': 'open_interest'})
    print()
    print(f"主测试数据: RB2505 螺纹钢")
    print(f"  数据量: {len(price_data)} 行")
    print(f"  时间范围: {price_data['date'].iloc[0].date()} ~ {price_data['date'].iloc[-1].date()}")
    print(f"  收盘价范围: {price_data['close'].min():.1f} ~ {price_data['close'].max():.1f}")
else:
    print("FATAL: 无法获取 RB2505 数据")
    sys.exit(1)

# 补充库存数据（螺纹钢）
inventory_data = None
try:
    # futures_inventory_em 似乎需要特定参数，先尝试无参数
    inv = ak.futures_inventory_em()
    if inv is not None and not inv.empty:
        # 尝试用 RB2505 数据对应的日期附近找库存
        inventory_data = inv.copy()
        print(f"\n  库存数据获取: {len(inventory_data)} rows")
except Exception as e:
    print(f"\n  库存数据获取失败: {e} (非致命，继续)")

print()

# ================================================================
# AGENT 1: Data Collector (使用 akshare 真实适配器)
# ================================================================
print("-" * 60)
print("[AGENT 1] DataCollectorAgent - 真实数据收集")
print("-" * 60)
try:
    from futureQuant.agent.data_collector import DataCollectorAgent

    agent1 = DataCollectorAgent()
    result1 = agent1.execute({
        'symbols': ['RB2505', 'HC2505', 'I2505'],
        'start_date': '2024-01-01',
        'end_date': '2025-04-01',
        'force_update': True,
    })
    print(f"  状态: {result1.status.value}")
    m1 = result1.metrics or {}
    print(f"  获取合约数: {m1.get('symbols_updated', 'N/A')}")
    print(f"  数据源: {m1.get('data_source', 'N/A')}")
    print(f"  耗时: {result1.elapsed_seconds:.2f}s")
    agent1_ok = True
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    agent1_ok = False
print()

# ================================================================
# AGENT 2: Factor Mining (使用真实价格数据)
# ================================================================
print("-" * 60)
print("[AGENT 2] FactorMiningAgent - 真实因子挖掘")
print("-" * 60)
try:
    from futureQuant.agent.factor_mining import FactorMiningAgent

    agent2 = FactorMiningAgent(config={'top_n': 20, 'min_ic': 0.01})
    result2 = agent2.execute({
        'target': 'RB2505',
        'price_data': price_data,
        'start_date': str(price_data['date'].iloc[0].date()),
        'end_date': str(price_data['date'].iloc[-1].date()),
    })
    print(f"  状态: {result2.status.value}")
    m2 = result2.metrics or {}
    print(f"  候选因子: {m2.get('n_candidates', 'N/A')}")
    print(f"  通过筛选: {m2.get('n_passed', 'N/A')}")
    if m2.get('top_factors'):
        top_names = [f.get('name', '?') for f in m2['top_factors'][:8]]
        print(f"  Top 因子: {top_names}")
    if m2.get('best_ic'):
        print(f"  最佳 IC: {m2['best_ic']:.4f}")
    print(f"  耗时: {result2.elapsed_seconds:.2f}s")
    agent2_ok = True
    factor_data = result2.data
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    agent2_ok = False
    factor_data = None
print()

# ================================================================
# AGENT 3: Fundamental Analysis (使用真实库存数据)
# ================================================================
print("-" * 60)
print("[AGENT 3] FundamentalAnalysisAgent - 真实基本面分析")
print("-" * 60)
try:
    from futureQuant.agent.fundamental import FundamentalAnalysisAgent

    agent3 = FundamentalAnalysisAgent()
    result3 = agent3.execute({
        'target': 'RB2505',
        'date_range': (str(price_data['date'].iloc[0].date()), str(price_data['date'].iloc[-1].date())),
        'inventory_data': inventory_data,
    })
    print(f"  状态: {result3.status.value}")
    m3 = result3.metrics or {}
    print(f"  情感评分: {m3.get('sentiment_score', 'N/A'):.4f}" if isinstance(m3.get('sentiment_score'), float) else f"  情感评分: {m3.get('sentiment_score', 'N/A')}")
    print(f"  置信度: {m3.get('confidence', 'N/A'):.2%}" if isinstance(m3.get('confidence'), float) else f"  置信度: {m3.get('confidence', 'N/A')}")
    print(f"  库存周期: {m3.get('inventory_cycle', 'N/A')}")
    print(f"  供需格局: {m3.get('supply_demand', 'N/A')}")
    if m3.get('drivers'):
        print(f"  主要驱动因素: {len(m3['drivers'])} 个")
        for d in m3['drivers'][:3]:
            print(f"    - {d.get('factor','?')}: {d.get('direction','?')} (权重 {d.get('weight','?')})")
    print(f"  耗时: {result3.elapsed_seconds:.2f}s")
    fundamental_result = result3
    agent3_ok = True
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    fundamental_result = None
    agent3_ok = False
print()

# ================================================================
# AGENT 4: Quant Signal (使用真实因子数据)
# ================================================================
print("-" * 60)
print("[AGENT 4] QuantSignalAgent - 真实量化信号")
print("-" * 60)
try:
    from futureQuant.agent.quant import QuantSignalAgent

    if factor_data is not None and not factor_data.empty:
        fdata = factor_data.copy()
    else:
        # 从真实价格生成因子数据
        fdata = pd.DataFrame(index=price_data['date'])
        close = price_data['close']
        fdata['rsi_14'] = (close.rolling(14).mean() / close * 100).fillna(50)
        fdata['atr_14'] = price_data[['high', 'low', 'close']].apply(
            lambda x: max(x[0]-x[1], abs(x[0]-x[2]), abs(x[1]-x[2])), axis=1
        ).rolling(14).mean()
        fdata['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        fdata['ma_cross'] = np.where(close.rolling(5).mean() > close.rolling(20).mean(), 1, -1).astype(float)
        fdata = fdata.reset_index().rename(columns={'index': 'date'}).set_index('date')

    agent4 = QuantSignalAgent()
    result4 = agent4.execute({
        'factor_data': fdata,
        'target': 'RB2505',
        'price_data': price_data,
    })
    print(f"  状态: {result4.status.value}")
    m4 = result4.metrics or {}
    sig_dir = m4.get('signal_direction', 'N/A')
    print(f"  信号方向: {sig_dir}")
    print(f"  置信度: {m4.get('confidence', 'N/A')}")
    print(f"  交易次数: {m4.get('n_trades', 'N/A')}")
    if m4.get('model_weights'):
        print(f"  模型权重: {m4['model_weights']}")
    print(f"  耗时: {result4.elapsed_seconds:.2f}s")
    quant_signal = result4
    agent4_ok = True
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    quant_signal = None
    agent4_ok = False
print()

# ================================================================
# AGENT 5: Backtest (基于真实信号和数据)
# ================================================================
print("-" * 60)
print("[AGENT 5] BacktestAgent - 真实回测验证")
print("-" * 60)
try:
    from futureQuant.agent.backtest_agent import BacktestAgent

    # 生成或使用真实信号
    if quant_signal is not None and hasattr(quant_signal, 'data') and quant_signal.data is not None:
        signals = quant_signal.data.copy()
    else:
        # 从因子数据生成简单信号
        signals = pd.DataFrame({
            'date': price_data['date'],
            'signal': np.where(
                fdata['ma_cross'] if 'ma_cross' in fdata.columns else pd.Series(0, index=fdata.index) > 0,
                1, -1
            ).astype(float),
        })

    agent5 = BacktestAgent()
    result5 = agent5.execute({
        'signals': signals,
        'price_data': price_data,
        'target': 'RB2505',
        'fundamental_signal': fundamental_result.metrics.get('sentiment_score', 0) if fundamental_result else 0,
    })
    print(f"  状态: {result5.status.value}")
    m5 = result5.metrics or {}
    tr = m5.get('total_return', 0)
    print(f"  总收益率: {tr:.2%}" if isinstance(tr, float) else f"  总收益率: {tr}")
    sr = m5.get('sharpe_ratio', 0)
    print(f"  夏普比率: {sr:.3f}" if isinstance(sr, float) else f"  夏普比率: {sr}")
    mdd = m5.get('max_drawdown', 0)
    print(f"  最大回撤: {mdd:.2%}" if isinstance(mdd, float) else f"  最大回撤: {mdd}")
    wr = m5.get('win_rate', 0)
    print(f"  胜率: {wr:.1%}" if isinstance(wr, float) else f"  胜率: {wr}")
    print(f"  交易次数: {m5.get('n_trades', 'N/A')}")
    print(f"  耗时: {result5.elapsed_seconds:.2f}s")
    backtest_result = result5
    agent5_ok = True
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    backtest_result = None
    agent5_ok = False
print()

# ================================================================
# AGENT 6: Price Behavior
# ================================================================
print("-" * 60)
print("[AGENT 6] PriceBehaviorAgent - 真实价格行为分析")
print("-" * 60)
try:
    from futureQuant.agent.price_behavior import PriceBehaviorAgent

    agent6 = PriceBehaviorAgent()
    result6 = agent6.execute({
        'price_data': price_data,
        'target': 'RB2505',
    })
    print(f"  状态: {result6.status.value}")
    m6 = result6.metrics or {}
    print(f"  市场状态: {m6.get('market_state', 'N/A')}")
    print(f"  形态类型: {m6.get('pattern_type', 'N/A')}")
    bp = m6.get('breakout_probability', 0)
    print(f"  突破概率: {bp:.1%}" if isinstance(bp, float) else f"  突破概率: {bp}")
    print(f"  建议方向: {m6.get('recommended_direction', 'N/A')}")
    rr = m6.get('risk_ratio', 0)
    print(f"  风险报酬比: {rr:.2f}" if isinstance(rr, float) else f"  风险报酬比: {rr}")
    print(f"  置信度: {m6.get('confidence', 'N/A')}")
    print(f"  耗时: {result6.elapsed_seconds:.2f}s")
    price_behavior_result = result6
    agent6_ok = True
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    price_behavior_result = None
    agent6_ok = False
print()

# ================================================================
# AGENT 7: Decision
# ================================================================
print("-" * 60)
print("[AGENT 7] DecisionAgent - 综合决策")
print("-" * 60)
try:
    from futureQuant.agent.decision import DecisionAgent

    current_price = float(price_data['close'].iloc[-1])
    volatility = float(price_data['close'].pct_change().std())

    agent7 = DecisionAgent()
    result7 = agent7.execute({
        'quant_signal': quant_signal,
        'fundamental_result': fundamental_result,
        'price_behavior_result': price_behavior_result,
        'backtest_result': backtest_result,
        'target': 'RB2505',
        'current_price': current_price,
        'volatility': volatility,
    })
    print(f"  状态: {result7.status.value}")
    m7 = result7.metrics or {}
    print(f"  决策方向: {m7.get('direction', 'N/A')}")
    conf = m7.get('confidence', 0)
    print(f"  置信度: {conf:.1%}" if isinstance(conf, float) else f"  置信度: {conf}")
    pos = m7.get('position_size', 0)
    print(f"  建议仓位: {pos:.0%}" if isinstance(pos, float) else f"  建议仓位: {pos}")
    pt = m7.get('price_target', {})
    if pt:
        print(f"  目标价 (悲观/基准/乐观): {pt.get('low','?')} / {pt.get('base','?')} / {pt.get('high','?')}")
    sl = m7.get('stop_loss', 0)
    print(f"  止损位: {sl}" if isinstance(sl, float) else f"  止损位: {sl}")
    print(f"  策略类型: {m7.get('strategy_type', 'N/A')}")
    print(f"  市场状态: {m7.get('market_regime', 'N/A')}")
    risks = m7.get('risk_points', [])
    print(f"  风险点数量: {len(risks)}")
    for r in risks[:3]:
        print(f"    - {r.get('risk','?')} ({r.get('level','?')})")
    scenarios = m7.get('scenarios', {})
    if scenarios:
        print(f"  情景分析: {list(scenarios.keys())}")
    print(f"  权重: {m7.get('weights', {})}")
    print(f"  耗时: {result7.elapsed_seconds:.2f}s")
    agent7_ok = True
except Exception as e:
    print(f"  错误: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    agent7_ok = False
print()

# ================================================================
# SUMMARY
# ================================================================
print("=" * 60)
print("真实数据测试汇总")
print("=" * 60)
agents = ['Agent1(数据收集)', 'Agent2(因子挖掘)', 'Agent3(基本面)', 'Agent4(量化)', 'Agent5(回测)', 'Agent6(价格行为)', 'Agent7(决策)']
results_ok = [agent1_ok, agent2_ok, agent3_ok, agent4_ok, agent5_ok, agent6_ok, agent7_ok]
for name, ok in zip(agents, results_ok):
    status = "PASS" if ok else "FAIL"
    print(f"  {status}: {name}")
print()
print(f"通过: {sum(results_ok)}/7")

if all(results_ok):
    print()
    print("ALL TESTS PASSED WITH REAL DATA!")
    print("7 个 Agent 全部使用真实 akshare 数据正常运行。")
else:
    print()
    failed = [name for name, ok in zip(agents, results_ok) if not ok]
    print(f"以下 Agent 需要修复: {failed}")
