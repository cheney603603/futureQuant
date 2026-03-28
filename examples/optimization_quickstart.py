#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化模块快速入门示例

演示如何使用优化模块提升因子挖掘系统的性能。
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

# 导入优化模块
from futureQuant.agent.optimization import (
    ParallelCalculator,
    ExecutionMode,
    CacheManager,
    MemoryManager,
    PerformanceMonitor,
    cached,
)


def generate_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """生成示例数据"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 200, n_days),
        'high': np.random.uniform(100, 200, n_days),
        'low': np.random.uniform(100, 200, n_days),
        'close': np.random.uniform(100, 200, n_days),
        'volume': np.random.uniform(1e6, 1e8, n_days),
    })
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data.set_index('date')


print("=" * 80)
print("优化模块快速入门示例")
print("=" * 80)
print()

# 1. 性能监控示例
print("1. 性能监控")
print("-" * 80)

monitor = PerformanceMonitor("demo")

# 生成测试数据
data = generate_sample_data(1000)
print(f"生成数据: {len(data)} 行")
monitor.record_metric('data_rows', len(data), 'count')

# 2. 并行计算示例
print("\n2. 并行计算")
print("-" * 80)

def calculate_sma(data, window=20):
    """计算简单移动平均"""
    return data['close'].rolling(window=window).mean()

def calculate_ema(data, span=20):
    """计算指数移动平均"""
    return data['close'].ewm(span=span).mean()

def calculate_volatility(data, window=20):
    """计算波动率"""
    return data['close'].pct_change().rolling(window=window).std()

# 串行计算
print("\n串行计算:")
start_time = time.time()
sma = calculate_sma(data)
ema = calculate_ema(data)
vol = calculate_volatility(data)
serial_time = time.time() - start_time
print(f"  耗时: {serial_time:.4f}s")

# 并行计算
print("\n并行计算:")
calculator = ParallelCalculator(mode=ExecutionMode.THREAD, n_jobs=4)
factor_funcs = {
    'sma': lambda d: calculate_sma(d, window=20),
    'ema': lambda d: calculate_ema(d, span=20),
    'volatility': lambda d: calculate_volatility(d, window=20),
}

start_time = time.time()
results = calculator.calculate_factors(factor_funcs, data)
parallel_time = time.time() - start_time
print(f"  耗时: {parallel_time:.4f}s")
print(f"  加速比: {serial_time / parallel_time:.2f}x")

# 3. 缓存示例
print("\n3. 缓存管理")
print("-" * 80)

cache = CacheManager(memory_cache_size=100)

# 使用缓存装饰器
@cached(cache, key_prefix="factor")
def calculate_expensive_factor(data, param):
    """模拟复杂的因子计算"""
    time.sleep(0.1)  # 模拟耗时操作
    return data['close'].rolling(window=param).mean()

print("\n第一次调用（计算中...）:")
start_time = time.time()
result1 = calculate_expensive_factor(data, param=20)
time1 = time.time() - start_time
print(f"  耗时: {time1:.4f}s")

print("\n第二次调用（从缓存获取）:")
start_time = time.time()
result2 = calculate_expensive_factor(data, param=20)
time2 = time.time() - start_time
print(f"  耗时: {time2:.4f}s")
print(f"  加速比: {time1 / time2:.2f}x")

# 查看缓存统计
stats = cache.get_stats()
print(f"\n缓存统计:")
print(f"  内存缓存命中率: {stats['memory_cache']['hit_rate']:.1f}%")

# 4. 内存优化示例
print("\n4. 内存优化")
print("-" * 80)

memory_manager = MemoryManager()

# 生成大数据集
large_data = generate_sample_data(10000)
original_size = memory_manager.get_dataframe_size(large_data)
print(f"\n原始数据:")
print(f"  行数: {len(large_data)}")
print(f"  内存占用: {original_size / 1024 / 1024:.2f}MB")

# 优化内存
optimized_data, saved_memory = memory_manager.optimize_dataframe_memory(large_data)
print(f"\n优化后:")
print(f"  内存占用: {(original_size - saved_memory) / 1024 / 1024:.2f}MB")
print(f"  节省: {saved_memory / 1024 / 1024:.2f}MB ({saved_memory / original_size * 100:.1f}%)")

# 查看内存统计
mem_stats = memory_manager.get_stats()
print(f"\n内存统计:")
print(f"  进程内存: {mem_stats['process_memory_mb']:.1f}MB")
print(f"  系统内存使用率: {mem_stats['memory_percent']:.1f}%")

# 5. 性能报告
print("\n5. 性能报告")
print("-" * 80)

# 记录性能指标
monitor.record_metric('serial_time', serial_time, 'seconds')
monitor.record_metric('parallel_time', parallel_time, 'seconds')
monitor.record_metric('speedup', serial_time / parallel_time, 'x')
monitor.record_metric('cache_speedup', time1 / time2, 'x')

# 获取监控摘要
summary = monitor.get_summary()
print("\n监控摘要:")
print(f"  总指标数: {summary['total_metrics']}")
print(f"  监控时长: {summary['elapsed_seconds']:.2f}s")

print("\n指标详情:")
for name, stats in summary['metrics_by_name'].items():
    print(f"  {name}:")
    print(f"    平均值: {stats['avg']:.4f}")
    print(f"    最小值: {stats['min']:.4f}")
    print(f"    最大值: {stats['max']:.4f}")

# 生成报告
from futureQuant.agent.optimization import PerformanceReporter
reporter = PerformanceReporter(output_dir="./reports")
report_path = reporter.generate_report(monitor, "demo_report")
print(f"\n性能报告已保存: {report_path}")

print("\n" + "=" * 80)
print("示例完成！")
print("=" * 80)
