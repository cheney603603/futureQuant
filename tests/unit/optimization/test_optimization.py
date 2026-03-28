"""
优化模块单元测试

测试所有优化模块的基本功能。
"""

import unittest
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import numpy as np

from futureQuant.agent.optimization import (
    ParallelCalculator,
    ExecutionMode,
    BatchCalculator,
    CacheManager,
    LRUCache,
    DiskCache,
    StorageOptimizer,
    QueryOptimizer,
    MemoryManager,
    DataPreloader,
    BackgroundPreloader,
    PerformanceMonitor,
    PerformanceReporter,
)


class TestParallelCalculator(unittest.TestCase):
    """测试并行计算器"""
    
    def test_sequential_mode(self):
        """测试串行模式"""
        calculator = ParallelCalculator(mode=ExecutionMode.SEQUENTIAL)
        
        # 创建测试数据
        data = pd.DataFrame({
            'value': np.random.randn(100)
        })
        
        # 定义测试函数
        def test_func(d):
            return d['value'].mean()
        
        # 执行计算
        results = calculator.calculate_factors({'test': test_func}, data)
        
        # 验证结果
        self.assertIn('test', results)
        self.assertIsInstance(results['test'], float)
    
    def test_thread_mode(self):
        """测试多线程模式"""
        calculator = ParallelCalculator(
            mode=ExecutionMode.THREAD,
            n_jobs=2
        )
        
        data = pd.DataFrame({
            'value': np.random.randn(100)
        })
        
        def test_func(d):
            return d['value'].mean()
        
        results = calculator.calculate_factors({'test': test_func}, data)
        
        self.assertIn('test', results)


class TestCacheManager(unittest.TestCase):
    """测试缓存管理器"""
    
    def test_lru_cache(self):
        """测试 LRU 缓存"""
        cache = LRUCache(max_size=3)
        
        # 添加数据
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        
        # 验证数据
        self.assertEqual(cache.get('a'), 1)
        self.assertEqual(cache.get('b'), 2)
        self.assertEqual(cache.get('c'), 3)
        
        # 测试 LRU 淘汰
        cache.put('d', 4)
        self.assertIsNone(cache.get('a'))  # 应该被淘汰
        self.assertEqual(cache.get('d'), 4)
    
    def test_disk_cache(self):
        """测试磁盘缓存"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            cache = DiskCache(cache_dir=temp_dir)
            
            # 存储数据
            cache.put('test_key', {'value': 123})
            
            # 获取数据
            result = cache.get('test_key')
            self.assertEqual(result['value'], 123)
            
            # 测试不存在的键
            self.assertIsNone(cache.get('nonexistent'))
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_cache_manager(self):
        """测试缓存管理器"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = CacheManager(
                memory_cache_size=10,
                disk_cache_dir=temp_dir,
                use_disk_cache=True
            )
            
            # 存储数据
            manager.put('key1', 'value1')
            
            # 获取数据
            result = manager.get('key1')
            self.assertEqual(result, 'value1')
            
            # 获取统计信息
            stats = manager.get_stats()
            self.assertIn('memory_cache', stats)
            self.assertIn('disk_cache', stats)
            
        finally:
            shutil.rmtree(temp_dir)


class TestStorageOptimizer(unittest.TestCase):
    """测试存储优化器"""
    
    def test_save_and_load_dataframe(self):
        """测试保存和加载 DataFrame"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            storage = StorageOptimizer(storage_dir=temp_dir)
            
            # 创建测试数据
            data = pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            })
            
            # 保存
            storage.save_dataframe(data, 'test')
            
            # 加载
            loaded = storage.load_dataframe('test')
            
            # 验证
            self.assertEqual(len(loaded), 3)
            self.assertIn('a', loaded.columns)
            
        finally:
            shutil.rmtree(temp_dir)


class TestQueryOptimizer(unittest.TestCase):
    """测试查询优化器"""
    
    def test_execute_query(self):
        """测试执行查询"""
        optimizer = QueryOptimizer(db_path=':memory:')
        
        try:
            # 创建测试数据
            data = pd.DataFrame({
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            })
            
            # 加载到数据库
            optimizer.load_dataframe(data, 'test_table')
            
            # 执行查询
            result = optimizer.execute_query('SELECT * FROM test_table')
            
            # 验证
            self.assertEqual(len(result), 3)
            
        finally:
            optimizer.close()
    
    def test_query_cache(self):
        """测试查询缓存"""
        optimizer = QueryOptimizer(db_path=':memory:', enable_cache=True)
        
        try:
            data = pd.DataFrame({'a': [1, 2, 3]})
            optimizer.load_dataframe(data, 'test')
            
            # 第一次查询
            result1 = optimizer.execute_query('SELECT * FROM test')
            
            # 第二次查询（应该从缓存获取）
            result2 = optimizer.execute_query('SELECT * FROM test')
            
            # 验证缓存命中
            stats = optimizer.get_stats()
            self.assertEqual(stats['cache_hits'], 1)
            
        finally:
            optimizer.close()


class TestMemoryManager(unittest.TestCase):
    """测试内存管理器"""
    
    def test_optimize_dataframe(self):
        """测试 DataFrame 优化"""
        manager = MemoryManager()
        
        # 创建测试数据
        data = pd.DataFrame({
            'int_col': [1, 2, 3] * 1000,
            'float_col': [1.0, 2.0, 3.0] * 1000,
        })
        
        # 优化
        optimized, saved = manager.optimize_dataframe_memory(data)
        
        # 验证节省了内存
        self.assertGreater(saved, 0)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        manager = MemoryManager()
        
        stats = manager.get_stats()
        
        # 验证统计信息
        self.assertIn('process_memory_mb', stats)
        self.assertIn('memory_percent', stats)


class TestDataPreloader(unittest.TestCase):
    """测试数据预加载器"""
    
    def test_preload_data(self):
        """测试预加载数据"""
        preloader = DataPreloader(max_preload_size=10)
        
        # 定义加载函数
        def load_func():
            return pd.DataFrame({'a': [1, 2, 3]})
        
        # 预加载
        success = preloader.preload_data('test_key', load_func)
        
        # 验证
        self.assertTrue(success)
        self.assertTrue(preloader.is_preloaded('test_key'))
        
        # 获取数据
        data = preloader.get_preloaded_data('test_key')
        self.assertEqual(len(data), 3)


class TestPerformanceMonitor(unittest.TestCase):
    """测试性能监控器"""
    
    def test_record_metric(self):
        """测试记录指标"""
        monitor = PerformanceMonitor()
        
        # 记录指标
        monitor.record_metric('test_metric', 123.45, 'ms')
        
        # 验证
        df = monitor.get_metrics_dataframe()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['name'], 'test_metric')
    
    def test_measure_time(self):
        """测试测量时间"""
        monitor = PerformanceMonitor()
        
        def slow_func():
            import time
            time.sleep(0.1)
            return 'result'
        
        result, elapsed = monitor.measure_time(slow_func)
        
        # 验证
        self.assertEqual(result, 'result')
        self.assertGreater(elapsed, 0.1)


if __name__ == '__main__':
    unittest.main()
