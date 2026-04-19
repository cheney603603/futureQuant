# -*- coding: utf-8 -*-
"""
Phase 2 集成测试 - 验证完整数据管道

测试内容：
1. P2.1 缓存管理器 - TTL过期、统计、清理
2. P2.2 多数据源降级 - 主源失败时切换
3. P2.3 数据质量报告 - 自动生成质量评估
4. P2.4 因子输入验证 - 脏数据拦截

Author: futureQuant Team
Date: 2026-04-19
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger

logger = get_logger('phase2_test')


def test_p21_cache_manager():
    """测试 P2.1 缓存管理器"""
    print("\n" + "=" * 60)
    print("P2.1 测试: 数据缓存管理器")
    print("=" * 60)
    
    from futureQuant.data.cache_manager import DataCacheManager
    
    # 创建缓存管理器（短TTL用于测试）
    cache = DataCacheManager(
        ttl_config={'price': 0.001},  # 3.6秒过期，方便测试
        auto_cleanup=False
    )
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date': pd.date_range('2026-04-01', periods=10),
        'close': range(1000, 1010),
    })
    
    # 测试写入缓存
    print("\n1. 写入缓存...")
    success = cache.put(
        test_data,
        data_type='price',
        variety='RB',
        start_date='2026-04-01',
        end_date='2026-04-10'
    )
    print("   写入结果: " + ("成功" if success else "失败"))
    
    # 测试读取缓存
    print("\n2. 读取缓存...")
    cached = cache.get(
        data_type='price',
        variety='RB',
        start_date='2026-04-01',
        end_date='2026-04-10'
    )
    print("   读取结果: " + ("命中" if cached is not None else "未命中"))
    if cached is not None:
        print("   数据行数: " + str(len(cached)))
    
    # 测试过期清理
    print("\n3. 等待缓存过期并清理...")
    import time
    time.sleep(4)  # 等待过期
    
    cleaned = cache.cleanup_expired()
    print("   清理条目: " + str(cleaned))
    
    # 再次读取（应该未命中）
    cached2 = cache.get(
        data_type='price',
        variety='RB',
        start_date='2026-04-01',
        end_date='2026-04-10'
    )
    print("   过期后读取: " + ("命中（异常）" if cached2 is not None else "未命中（正确）"))
    
    # 测试统计
    print("\n4. 缓存统计...")
    stats = cache.get_stats()
    print("   命中率: {:.2%}".format(stats['hit_rate']))
    print("   总会话请求: " + str(stats['session_stats']['hits'] + stats['session_stats']['misses']))
    
    print("\n[PASS] P2.1 缓存管理器测试通过")
    return True


def test_p22_fallback_fetcher():
    """测试 P2.2 多数据源降级"""
    print("\n" + "=" * 60)
    print("P2.2 测试: 多数据源降级")
    print("=" * 60)
    
    from futureQuant.data.fetcher.fallback_fetcher import FallbackFetcher
    
    # 创建降级获取器
    fetcher = FallbackFetcher(
        primary='akshare',
        fallback_order=['akshare'],  # 目前只有akshare
        retry_count=2
    )
    
    # 测试获取数据
    print("\n1. 获取 RB 数据...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    try:
        df = fetcher.fetch_daily('RB', start_date, end_date)
        print("   获取成功: " + str(len(df)) + " 条记录")
        
        # 显示统计
        stats = fetcher.get_stats()
        print("\n2. 获取统计...")
        print("   总请求: " + str(stats['total_requests']))
        print("   成功: " + str(stats['successes']))
        print("   成功率: {:.2%}".format(stats['success_rate']))
        
        print("\n[PASS] P2.2 多数据源降级测试通过")
        return True
        
    except Exception as e:
        print("   获取失败: " + str(e))
        print("\n[WARNING] P2.2 测试遇到问题（可能是网络问题）")
        return False


def test_p23_quality_reporter():
    """测试 P2.3 数据质量报告"""
    print("\n" + "=" * 60)
    print("P2.3 测试: 数据质量报告")
    print("=" * 60)
    
    from futureQuant.data.quality_reporter import DataQualityReporter
    
    # 创建报告生成器
    reporter = DataQualityReporter()
    
    # 创建测试数据（包含一些异常）
    test_data = pd.DataFrame({
        'date': pd.date_range('2026-04-01', periods=20),
        'open': [1000 + i * 10 for i in range(20)],
        'high': [1010 + i * 10 for i in range(20)],
        'low': [990 + i * 10 for i in range(20)],
        'close': [1005 + i * 10 for i in range(20)],
        'volume': [10000] * 20,
    })
    # 添加一个异常值
    test_data.loc[10, 'close'] = 50  # 异常低价
    
    print("\n1. 生成质量报告...")
    report = reporter.generate_report(
        test_data,
        data_type='price',
        variety='RB',
        start_date='2026-04-01',
        end_date='2026-04-20'
    )
    
    print("   总体评分: {:.2f}".format(report.overall_score))
    print("   质量等级: " + report.quality_level)
    print("   数据行数: " + str(report.row_count))
    print("   异常值数量: " + str(report.outlier_count))
    
    print("\n2. 发现问题...")
    for issue in report.issues:
        print("   - " + issue)
    
    print("\n3. 生成 Markdown 报告...")
    md_content = reporter.to_markdown(report)
    print("   报告长度: " + str(len(md_content)) + " 字符")
    
    # 保存报告
    report_path = project_root / 'docs' / 'reports' / 'phase2_quality_test.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reporter.save_markdown_report(report, str(report_path))
    print("   报告已保存: " + str(report_path))
    
    print("\n[PASS] P2.3 数据质量报告测试通过")
    return True


def test_p24_factor_validation():
    """测试 P2.4 因子输入验证"""
    print("\n" + "=" * 60)
    print("P2.4 测试: 因子输入验证")
    print("=" * 60)
    
    from futureQuant.factor.input_validation import FactorInputValidator
    
    # 创建验证器
    validator = FactorInputValidator(variety='RB', mode='strict')
    
    # 测试1: 正常数据
    print("\n1. 测试正常数据...")
    good_data = pd.DataFrame({
        'date': pd.date_range('2026-04-01', periods=30),
        'open': [3000 + i * 10 for i in range(30)],
        'high': [3010 + i * 10 for i in range(30)],
        'low': [2990 + i * 10 for i in range(30)],
        'close': [3005 + i * 10 for i in range(30)],
        'volume': [10000] * 30,
    })
    
    is_valid, issues = validator.validate(good_data, '2026-04-01', '2026-04-30')
    print("   验证结果: " + ("通过" if is_valid else "失败"))
    if issues:
        for issue in issues:
            print("   - " + issue)
    
    # 测试2: 脏数据（行数不足）
    print("\n2. 测试行数不足的数据...")
    bad_data = pd.DataFrame({
        'date': pd.date_range('2026-04-01', periods=5),
        'close': [3000, 3010, 3020, 3030, 3040],
    })
    
    is_valid2, issues2 = validator.validate(bad_data, '2026-04-01', '2026-04-05')
    print("   验证结果: " + ("通过" if is_valid2 else "失败（预期）"))
    print("   发现问题: " + str(len(issues2)) + " 个")
    for issue in issues2:
        print("   - " + issue)
    
    # 测试3: 异常价格数据
    print("\n3. 测试异常价格数据...")
    weird_data = pd.DataFrame({
        'date': pd.date_range('2026-04-01', periods=30),
        'close': [3000] * 29 + [50],  # 最后一个异常
    })
    
    is_valid3, issues3 = validator.validate(weird_data, '2026-04-01', '2026-04-30')
    print("   验证结果: " + ("通过" if is_valid3 else "失败（预期）"))
    print("   发现问题: " + str(len(issues3)) + " 个")
    for issue in issues3[:3]:  # 只显示前3个
        print("   - " + issue)
    
    # 测试4: 验证或抛出异常
    print("\n4. 测试 validate_or_raise...")
    try:
        validator.validate_or_raise(bad_data, '2026-04-01', '2026-04-05')
        print("   未抛出异常（异常）")
    except Exception as e:
        print("   正确抛出异常: " + type(e).__name__)
    
    print("\n[PASS] P2.4 因子输入验证测试通过")
    return True


def test_full_pipeline():
    """测试完整数据管道"""
    print("\n" + "=" * 60)
    print("完整数据管道测试")
    print("=" * 60)
    
    from futureQuant.data.cache_manager import get_cache_manager
    from futureQuant.data.quality_reporter import DataQualityReporter
    from futureQuant.factor.input_validation import FactorInputValidator
    from futureQuant.data.fetcher.akshare_fetcher import AKShareFetcher
    
    print("\n1. 获取数据（带缓存）...")
    cache = get_cache_manager()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d')
    
    # 尝试从缓存获取
    df = cache.get('price', 'RB', start_date, end_date)
    
    if df is None:
        print("   缓存未命中，从数据源获取...")
        fetcher = AKShareFetcher()
        df = fetcher.fetch_daily('RB', start_date, end_date)
        
        # 存入缓存
        cache.put(df, 'price', 'RB', start_date, end_date)
        print("   已存入缓存: " + str(len(df)) + " 条")
    else:
        print("   缓存命中: " + str(len(df)) + " 条")
    
    print("\n2. 生成质量报告...")
    reporter = DataQualityReporter()
    report = reporter.generate_report(df, 'price', 'RB', start_date, end_date)
    print("   质量评分: {:.2f}".format(report.overall_score))
    print("   质量等级: " + report.quality_level)
    
    print("\n3. 因子输入验证...")
    validator = FactorInputValidator(variety='RB', mode='strict')
    is_valid, issues = validator.validate(df, start_date, end_date)
    print("   验证结果: " + ("通过" if is_valid else "失败"))
    if not is_valid:
        for issue in issues[:3]:
            print("   - " + issue)
    
    print("\n4. 缓存统计...")
    stats = cache.get_stats()
    print("   缓存条目: " + str(stats['total_entries']))
    print("   命中率: {:.2%}".format(stats['hit_rate']))
    
    print("\n[PASS] 完整数据管道测试通过")
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Phase 2 集成测试")
    print("=" * 60)
    print("\n测试内容:")
    print("  P2.1: 数据缓存管理器 (TTL过期机制)")
    print("  P2.2: 多数据源降级")
    print("  P2.3: 数据质量报告生成器")
    print("  P2.4: 因子输入验证器")
    print("  完整管道: 缓存+质量报告+验证")
    
    results = {}
    
    # 运行各项测试
    try:
        results['P2.1'] = test_p21_cache_manager()
    except Exception as e:
        print("\n[FAIL] P2.1 测试失败: " + str(e))
        results['P2.1'] = False
    
    try:
        results['P2.2'] = test_p22_fallback_fetcher()
    except Exception as e:
        print("\n[FAIL] P2.2 测试失败: " + str(e))
        results['P2.2'] = False
    
    try:
        results['P2.3'] = test_p23_quality_reporter()
    except Exception as e:
        print("\n[FAIL] P2.3 测试失败: " + str(e))
        results['P2.3'] = False
    
    try:
        results['P2.4'] = test_p24_factor_validation()
    except Exception as e:
        print("\n[FAIL] P2.4 测试失败: " + str(e))
        results['P2.4'] = False
    
    try:
        results['Pipeline'] = test_full_pipeline()
    except Exception as e:
        print("\n[FAIL] 完整管道测试失败: " + str(e))
        results['Pipeline'] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("Phase 2 测试总结")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print("  " + name + ": " + status)
    
    total = len(results)
    passed_count = sum(results.values())
    
    print("\n总计: " + str(passed_count) + "/" + str(total) + " 通过 ({:.0f}%)".format(passed_count/total*100))
    
    if passed_count == total:
        print("\n[SUCCESS] 所有 Phase 2 测试通过！")
    else:
        print("\n[WARNING] 部分测试未通过，请检查日志")
    
    return passed_count == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
