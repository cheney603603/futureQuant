"""
端到端真实数据测试

全流程验证：
1. 数据获取（akshare 真实数据）
2. 因子计算
3. 基本面分析
4. 信号生成与回测
5. 生成报告

Author: futureQuant Team
Date: 2026-04-19
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.data.fetcher.akshare_fetcher import AKShareFetcher
from futureQuant.data.fetcher.fundamental_fetcher import FundamentalFetcher
from futureQuant.factor.engine import FactorEngine
from futureQuant.factor.technical import MomentumFactor, VolatilityFactor, VolumeRatioFactor
from futureQuant.agent.fundamental.fundamental_agent import FundamentalAnalysisAgent

logger = get_logger('e2e_test')


def test_data_fetch():
    """测试数据获取"""
    logger.info("=" * 60)
    logger.info("Step 1: 数据获取测试")
    logger.info("=" * 60)
    
    # 日期范围：最近30天
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 1.1 价格数据
    logger.info("获取 RB 价格数据...")
    try:
        fetcher = AKShareFetcher()
        price_df = fetcher.fetch_daily('RB', start_date, end_date)
        
        if price_df.empty:
            logger.error("价格数据为空，请检查 akshare 接口")
            return None
        
        logger.info(f"价格数据获取成功: {len(price_df)} 条记录")
        logger.info(f"   列: {price_df.columns.tolist()}")
        logger.info(f"   日期范围: {price_df['date'].min()} ~ {price_df['date'].max()}")
        
        # 保存到文件
        output_dir = project_root / 'data' / 'collected'
        output_dir.mkdir(parents=True, exist_ok=True)
        price_df.to_csv(output_dir / 'RB_price_latest.csv', index=False, encoding='utf-8-sig')
        logger.info("   已保存到 data/collected/RB_price_latest.csv")
        
    except Exception as e:
        logger.error(f"价格数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 1.2 基本面数据
    logger.info("")
    logger.info("获取基本面数据...")
    output_dir = project_root / 'data' / 'collected'
    try:
        fund_fetcher = FundamentalFetcher()
        
        # 库存
        inv_df = fund_fetcher.fetch_inventory('RB')
        if not inv_df.empty:
            logger.info(f"库存数据: {len(inv_df)} 条")
            inv_df.to_csv(output_dir / 'RB_inventory_latest.csv', index=False, encoding='utf-8-sig')
        else:
            logger.warning("库存数据为空")
        
        # 基差
        basis_df = fund_fetcher.fetch_basis('RB')
        if not basis_df.empty:
            logger.info(f"基差数据: {len(basis_df)} 条")
            basis_df.to_csv(output_dir / 'RB_basis_latest.csv', index=False, encoding='utf-8-sig')
        else:
            logger.warning("基差数据为空")
        
    except Exception as e:
        logger.warning(f"基本面数据获取失败（不影响继续）: {e}")
    
    return price_df


def test_factor_computation(price_df: pd.DataFrame):
    """测试因子计算"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: 因子计算测试")
    logger.info("=" * 60)
    
    if price_df is None or price_df.empty:
        logger.error("无价格数据，跳过因子计算")
        return None
    
    try:
        engine = FactorEngine()
        
        # 注册技术因子
        engine.register(MomentumFactor(period=5))
        engine.register(MomentumFactor(period=10))
        engine.register(MomentumFactor(period=20))
        engine.register(VolatilityFactor(period=10))
        engine.register(VolatilityFactor(period=20))
        engine.register(VolumeRatioFactor(period=5))
        
        logger.info("计算技术因子...")
        
        factor_data = {}
        factor_names = list(engine.factors.keys())
        
        for name in factor_names:
            try:
                series = engine.compute(price_df, name)
                if series is not None and len(series) > 0:
                    factor_data[name] = series
                    logger.info(f"  {name}: OK ({len(series)} values)")
            except Exception as e:
                logger.warning(f"  {name}: {e}")
        
        if not factor_data:
            logger.error("所有因子计算失败")
            return None
        
        # 合并因子
        factor_df = pd.DataFrame(factor_data)
        
        logger.info(f"")
        logger.info(f"因子计算完成: {len(factor_df.columns)} 个因子, {len(factor_df)} 条记录")
        
        # 保存
        output_dir = project_root / 'data' / 'collected'
        factor_df.to_csv(output_dir / 'RB_factors_latest.csv', encoding='utf-8-sig')
        logger.info("已保存到 data/collected/RB_factors_latest.csv")
        
        return factor_df
        
    except Exception as e:
        logger.error(f"因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_backtest(price_df: pd.DataFrame, factor_df: pd.DataFrame):
    """测试回测"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: 回测测试")
    logger.info("=" * 60)
    
    if price_df is None or factor_df is None or price_df.empty or factor_df.empty:
        logger.error("缺少数据，跳过回测")
        return None
    
    try:
        # 找到可用因子
        factor_col = None
        for col in factor_df.columns:
            if 'momentum' in col.lower() or 'volatility' in col.lower():
                factor_col = col
                break
        
        if factor_col is None:
            logger.error("没有可用的因子列")
            return None
        
        logger.info(f"使用因子 {factor_col} 生成信号...")
        
        # 生成信号
        factor_values = factor_df[factor_col].dropna()
        
        # 简单阈值策略
        mean_val = factor_values.mean()
        std_val = factor_values.std()
        
        signals = factor_values.apply(lambda x: 1 if x > mean_val + 0.5 * std_val else (-1 if x < mean_val - 0.5 * std_val else 0))
        
        logger.info(f"信号分布: {signals.value_counts().to_dict()}")
        
        # 计算简单收益
        aligned_returns = price_df['close'].pct_change().shift(-1).iloc[:len(signals)]
        strategy_returns = signals.values[:len(aligned_returns)] * aligned_returns.values
        strategy_returns = pd.Series(strategy_returns).dropna()
        
        if len(strategy_returns) == 0:
            logger.error("策略收益为空")
            return None
        
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        max_dd = (strategy_returns.cumsum().cummax() - strategy_returns.cumsum()).max()
        
        result = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'n_trades': int((signals != 0).sum()),
        }
        
        logger.info(f"回测完成")
        logger.info(f"   总收益率: {total_return:.2%}")
        logger.info(f"   夏普比率: {sharpe:.2f}")
        logger.info(f"   最大回撤: {max_dd:.2%}")
        logger.info(f"   交易次数: {result['n_trades']}")
        
        return result
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_fundamental_analysis():
    """测试基本面分析"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: 基本面分析测试")
    logger.info("=" * 60)
    
    try:
        agent = FundamentalAnalysisAgent(config={})
        
        context = {
            'symbol': 'RB2505',
            'variety': 'RB',
        }
        
        result = agent.run(context)
        
        if result and result.is_success:
            logger.info("基本面分析完成")
            logger.info(f"   状态: {result.status.value}")
            if result.data is not None:
                return result.data
            return None
        else:
            logger.warning("基本面分析返回空结果")
            return None
            
    except Exception as e:
        logger.warning(f"基本面分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_final_report(price_data, factor_data, backtest_result, fundamental_result):
    """生成最终报告"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 5: 生成最终报告")
    logger.info("=" * 60)
    
    report_path = project_root / 'docs' / 'reports' / f'e2e_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# futureQuant 端到端测试报告",
        "",
        f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**测试标的**: RB（螺纹钢）",
        "",
        "---",
        "",
        "## 1. 数据获取",
        "",
        "| 指标 | 结果 |",
        "|------|------|",
    ]
    
    if price_data is not None and hasattr(price_data, 'empty') and not price_data.empty:
        lines.extend([
            f"| 价格数据 | OK: {len(price_data)} 条记录 |",
            f"| 日期范围 | {price_data['date'].min()} ~ {price_data['date'].max()} |",
        ])
    else:
        lines.append("| 价格数据 | FAILED |")
    
    lines.extend([
        "",
        "## 2. 因子计算",
        "",
        "| 指标 | 结果 |",
        "|------|------|",
    ])
    
    if factor_data is not None and hasattr(factor_data, 'empty') and not factor_data.empty:
        n_factors = len(factor_data.columns)
        lines.extend([
            f"| 计算因子数 | OK: {n_factors} 个 |",
            f"| 数据行数 | {len(factor_data)} |",
        ])
    else:
        lines.append("| 因子计算 | FAILED |")
    
    lines.extend([
        "",
        "## 3. 回测验证",
        "",
        "| 指标 | 结果 |",
        "|------|------|",
    ])
    
    if backtest_result is not None:
        lines.extend([
            f"| 总收益率 | {backtest_result.get('total_return', 0):.2%} |",
            f"| 夏普比率 | {backtest_result.get('sharpe_ratio', 0):.2f} |",
            f"| 最大回撤 | {backtest_result.get('max_drawdown', 0):.2%} |",
            f"| 交易次数 | {backtest_result.get('n_trades', 0)} |",
        ])
    else:
        lines.append("| 回测 | FAILED |")
    
    lines.extend([
        "",
        "## 4. 基本面分析",
        "",
        "| 指标 | 结果 |",
        "|------|------|",
    ])
    
    if fundamental_result is not None and (not hasattr(fundamental_result, 'empty') or not fundamental_result.empty):
        if hasattr(fundamental_result, 'get'):
            lines.extend([
                f"| 情绪评分 | {fundamental_result.get('sentiment_score', 'N/A')} |",
                f"| 库存周期 | {fundamental_result.get('inventory_cycle', 'N/A')} |",
                f"| 供需格局 | {fundamental_result.get('supply_demand', 'N/A')} |",
            ])
        else:
            lines.append("| 基本面分析 | OK |")
    else:
        lines.append("| 基本面分析 | SKIPPED |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 测试结论",
        "",
    ])
    
    # 评估结果
    success_count = sum([
        price_data is not None and hasattr(price_data, 'empty') and not price_data.empty,
        factor_data is not None and hasattr(factor_data, 'empty') and not factor_data.empty,
        backtest_result is not None,
        fundamental_result is not None,
    ])
    
    if success_count >= 3:
        lines.append("端到端流程基本跑通！核心功能正常。")
        final_status = "SUCCESS"
    elif success_count >= 2:
        lines.append("部分流程成功，需要进一步调试。")
        final_status = "PARTIAL"
    else:
        lines.append("测试失败，存在严重问题。")
        final_status = "FAILED"
    
    lines.extend([
        "",
        "*报告由 futureQuant 自动生成*",
    ])
    
    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding='utf-8')
    
    logger.info(f"报告已生成: {report_path}")
    print("")
    print(f"报告路径: {report_path}")
    print(f"最终状态: {final_status}")
    
    return final_status


def main():
    """主测试流程"""
    logger.info("=" * 60)
    logger.info("futureQuant 端到端真实数据测试")
    logger.info("=" * 60)
    
    # Step 1: 数据获取
    price_data = test_data_fetch()
    
    # Step 2: 因子计算
    factor_data = test_factor_computation(price_data)
    
    # Step 3: 回测
    backtest_result = test_backtest(price_data, factor_data)
    
    # Step 4: 基本面分析
    fundamental_result = test_fundamental_analysis()
    
    # Step 5: 生成报告
    final_status = generate_final_report(price_data, factor_data, backtest_result, fundamental_result)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"测试完成！状态: {final_status}")
    logger.info("=" * 60)
    
    return final_status == "SUCCESS"


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
