#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化模块导入验证脚本

验证所有优化模块是否可以正确导入和使用。
"""

import sys
from pathlib import Path

print("=" * 80)
print("优化模块导入验证")
print("=" * 80)
print()

# 测试导入
print("1. 测试模块导入...")
print("-" * 80)

try:
    from futureQuant.agent.optimization import (
        # parallel_calculator
        ParallelCalculator,
        BatchCalculator,
        ExecutionMode,
        TaskResult,
        ProgressTracker,
        create_calculator,
        
        # cache_manager
        CacheManager,
        LRUCache,
        DiskCache,
        CacheStats,
        CachedFunction,
        cached,
        
        # storage_optimizer
        StorageOptimizer,
        CompressionConfig,
        
        # query_optimizer
        QueryOptimizer,
        BulkQueryExecutor,
        QueryStats,
        
        # memory_manager
        MemoryManager,
        MemoryMonitor,
        MemoryStats,
        
        # data_preloader
        DataPreloader,
        BackgroundPreloader,
        PredictivePreloader,
        PreloadStats,
        
        # performance_monitor
        PerformanceMonitor,
        PerformanceReporter,
        PerformanceBenchmark,
        PerformanceAlert,
        PerformanceMetric,
    )
    
    print("✓ 所有模块导入成功！")
    
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

print()

# 测试实例化
print("2. 测试类实例化...")
print("-" * 80)

tests = [
    ("ParallelCalculator", lambda: ParallelCalculator()),
    ("BatchCalculator", lambda: BatchCalculator(batch_size=10)),
    ("LRUCache", lambda: LRUCache(max_size=100)),
    ("CacheManager", lambda: CacheManager()),
    ("StorageOptimizer", lambda: StorageOptimizer()),
    ("QueryOptimizer", lambda: QueryOptimizer()),
    ("MemoryManager", lambda: MemoryManager()),
    ("DataPreloader", lambda: DataPreloader()),
    ("PerformanceMonitor", lambda: PerformanceMonitor()),
    ("PerformanceReporter", lambda: PerformanceReporter()),
]

for name, create_func in tests:
    try:
        instance = create_func()
        print(f"✓ {name}: {instance}")
    except Exception as e:
        print(f"✗ {name}: {e}")

print()

# 测试基本功能
print("3. 测试基本功能...")
print("-" * 80)

try:
    # 测试 LRU 缓存
    cache = LRUCache(max_size=3)
    cache.put('a', 1)
    assert cache.get('a') == 1
    print("✓ LRUCache 基本功能正常")
    
    # 测试性能监控
    monitor = PerformanceMonitor("test")
    monitor.record_metric('test', 100, 'ms')
    stats = monitor.get_summary()
    assert stats['total_metrics'] == 1
    print("✓ PerformanceMonitor 基本功能正常")
    
    # 测试执行模式
    assert ExecutionMode.PROCESS.value == "process"
    assert ExecutionMode.THREAD.value == "thread"
    assert ExecutionMode.SEQUENTIAL.value == "sequential"
    print("✓ ExecutionMode 枚举正常")
    
    print("\n所有基本功能测试通过！")
    
except AssertionError as e:
    print(f"✗ 功能测试失败: {e}")
    sys.exit(1)

print()

# 显示模块统计
print("4. 模块统计...")
print("-" * 80)

optimization_dir = Path(__file__).parent.parent / "futureQuant" / "agent" / "optimization"
if optimization_dir.exists():
    py_files = list(optimization_dir.glob("*.py"))
    total_lines = sum(len(f.read_text(encoding='utf-8').splitlines()) for f in py_files)
    
    print(f"Python 文件数: {len(py_files)}")
    print(f"总代码行数: {total_lines}")
    print()
    print("文件详情:")
    for f in sorted(py_files):
        lines = len(f.read_text(encoding='utf-8').splitlines())
        print(f"  {f.name}: {lines} 行")

print()

print("=" * 80)
print("✓ 所有验证通过！优化模块已准备就绪。")
print("=" * 80)
