"""
因子库管理单元测试

测试 FactorRepository、FactorVersionControl、PerformanceTracker、CorrelationTracker。
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
from datetime import datetime

from futureQuant.agent.repository import (
    FactorRepository,
    FactorVersionControl,
    PerformanceTracker,
    CorrelationTracker,
)


class TestFactorRepository:
    """测试因子存储"""
    
    @pytest.fixture
    def temp_repo(self):
        """创建临时因子库"""
        temp_dir = tempfile.mkdtemp()
        repo = FactorRepository(temp_dir)
        yield repo
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_factor(self):
        """创建测试因子"""
        class MockFactor:
            def __init__(self):
                self.name = 'test_factor'
                self.params = {'window': 10, 'threshold': 0.05}
                self.category = 'technical'
        
        return MockFactor()
    
    @pytest.fixture
    def test_values(self):
        """创建测试数据"""
        dates = pd.date_range('2026-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'value': np.random.randn(100).cumsum(),
        }, index=dates)
    
    def test_init(self, temp_repo):
        """测试初始化"""
        assert temp_repo.storage_dir.exists()
        assert temp_repo.db_path.exists()
        assert temp_repo.values_dir.exists()
    
    def test_save_factor(self, temp_repo, test_factor, test_values):
        """测试保存因子"""
        factor_id = temp_repo.save_factor(
            test_factor,
            test_values,
            performance={'ic_mean': 0.05},
        )
        
        assert factor_id is not None
        assert test_factor.name in factor_id
    
    def test_get_factor(self, temp_repo, test_factor, test_values):
        """测试获取因子"""
        factor_id = temp_repo.save_factor(test_factor, test_values)
        
        loaded = temp_repo.get_factor(factor_id)
        
        assert loaded is not None
        assert loaded['name'] == test_factor.name
        assert loaded['values'] is not None
        assert len(loaded['values']) == 100
    
    def test_get_factor_with_date_filter(self, temp_repo, test_factor, test_values):
        """测试带日期过滤的获取"""
        factor_id = temp_repo.save_factor(test_factor, test_values)
        
        loaded = temp_repo.get_factor(
            factor_id,
            start_date='2026-01-10',
            end_date='2026-01-20',
        )
        
        assert loaded['values'] is not None
        assert len(loaded['values']) == 11  # 10 日到 20 日
    
    def test_get_nonexistent_factor(self, temp_repo):
        """测试获取不存在的因子"""
        loaded = temp_repo.get_factor('nonexistent_factor')
        assert loaded is None
    
    def test_list_factors(self, temp_repo, test_factor, test_values):
        """测试列出因子"""
        # 保存多个因子
        test_factor.name = 'factor1'
        id1 = temp_repo.save_factor(test_factor, test_values)
        
        test_factor.name = 'factor2'
        test_factor.category = 'fundamental'
        id2 = temp_repo.save_factor(test_factor, test_values)
        
        # 列出所有因子
        all_factors = temp_repo.list_factors()
        assert len(all_factors) == 2
        
        # 按类别列出
        tech_factors = temp_repo.list_factors(category='technical')
        assert len(tech_factors) == 1
    
    def test_update_factor_status(self, temp_repo, test_factor, test_values):
        """测试更新因子状态"""
        factor_id = temp_repo.save_factor(test_factor, test_values)
        
        temp_repo.update_factor_status(factor_id, 'inactive')
        
        # 活跃因子列表不应包含该因子
        active_factors = temp_repo.list_factors(status='active')
        assert factor_id not in active_factors
        
        inactive_factors = temp_repo.list_factors(status='inactive')
        assert factor_id in inactive_factors
    
    def test_delete_factor(self, temp_repo, test_factor, test_values):
        """测试删除因子"""
        factor_id = temp_repo.save_factor(test_factor, test_values)
        
        temp_repo.delete_factor(factor_id)
        
        # 因子应该不存在
        loaded = temp_repo.get_factor(factor_id)
        assert loaded is None
    
    def test_save_with_performance(self, temp_repo, test_factor, test_values):
        """测试带性能指标保存"""
        performance = {
            'ic_mean': 0.05,
            'icir': 1.5,
            'ic_win_rate': 0.6,
            'monotonicity': 0.8,
            'turnover': 0.3,
            'max_drawdown': -0.1,
            'overall_score': 0.75,
        }
        
        factor_id = temp_repo.save_factor(
            test_factor,
            test_values,
            performance=performance,
        )
        
        assert factor_id is not None


class TestFactorVersionControl:
    """测试版本控制"""
    
    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / 'test.db'
        
        # 创建 factor_metadata 表（外键依赖）
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_metadata (
                factor_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            INSERT INTO factor_metadata (factor_id, name) VALUES ('test_factor', 'Test Factor')
        ''')
        conn.commit()
        conn.close()
        
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    def test_init(self, temp_db):
        """测试初始化"""
        vc = FactorVersionControl(temp_db)
        assert vc.db_path == temp_db
    
    def test_create_version(self, temp_db):
        """测试创建版本"""
        vc = FactorVersionControl(temp_db)
        
        version_id = vc.create_version(
            'test_factor',
            'v1.0',
            parameters={'window': 10},
            code='def calc(x): return x.mean()',
            change_reason='Initial version',
        )
        
        assert version_id == 'test_factor_v1.0'
    
    def test_get_version_history(self, temp_db):
        """测试获取版本历史"""
        vc = FactorVersionControl(temp_db)
        
        # 创建多个版本
        vc.create_version('test_factor', 'v1.0', {'window': 10}, 'code1', 'Initial')
        vc.create_version('test_factor', 'v1.1', {'window': 20}, 'code2', 'Update window')
        
        history = vc.get_version_history('test_factor')
        
        assert len(history) == 2
        assert history[0]['version_number'] == 'v1.1'  # 最新的在前
    
    def test_compare_versions(self, temp_db):
        """测试版本对比"""
        vc = FactorVersionControl(temp_db)
        
        vc.create_version('test_factor', 'v1.0', {'window': 10, 'threshold': 0.05}, 'code1', 'Initial')
        vc.create_version('test_factor', 'v1.1', {'window': 20, 'threshold': 0.05}, 'code2', 'Update window')
        
        diff = vc.compare_versions('test_factor_v1.0', 'test_factor_v1.1')
        
        assert 'parameter_changes' in diff
        assert 'window' in diff['parameter_changes']
        assert diff['parameter_changes']['window']['old'] == 10
        assert diff['parameter_changes']['window']['new'] == 20
    
    def test_rollback(self, temp_db):
        """测试回滚"""
        vc = FactorVersionControl(temp_db)
        
        vc.create_version('test_factor', 'v1.0', {'window': 10}, 'code1', 'Initial')
        vc.create_version('test_factor', 'v2.0', {'window': 100}, 'code2', 'Major update')
        
        # 回滚到 v1.0
        result = vc.rollback('test_factor', 'v1.0')
        
        assert result is True
        
        # 验证回滚后创建了新版本
        history = vc.get_version_history('test_factor')
        assert len(history) == 3  # v2.0, v1.0_rollback_xxx, v1.0


class TestPerformanceTracker:
    """测试性能追踪"""
    
    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / 'test.db'
        
        # 创建必要的表
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id TEXT NOT NULL,
                period TEXT,
                start_date DATE,
                end_date DATE,
                ic_mean REAL,
                icir REAL,
                ic_win_rate REAL,
                monotonicity REAL,
                turnover REAL,
                max_drawdown REAL,
                overall_score REAL,
                created_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    def test_init(self, temp_db):
        """测试初始化"""
        tracker = PerformanceTracker(temp_db)
        assert tracker.db_path == temp_db
    
    def test_track_monthly(self, temp_db):
        """测试月度性能记录"""
        tracker = PerformanceTracker(temp_db)
        
        tracker.track_monthly(
            factor_id='test_factor',
            period='2026-01',
            start_date='2026-01-01',
            end_date='2026-01-31',
            metrics={'ic_mean': 0.05, 'icir': 1.5, 'ic_win_rate': 0.6},
        )
        
        # 验证数据已写入
        import sqlite3
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM factor_performance WHERE factor_id = ?', ('test_factor',))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_detect_decay(self, temp_db):
        """测试衰减检测"""
        tracker = PerformanceTracker(temp_db)
        
        # 记录连续下降的 IC
        for i, ic in enumerate([0.08, 0.06, 0.04, 0.02]):
            tracker.track_monthly(
                'test_factor',
                f'2026-{i+1:02d}',
                f'2026-{i+1:02d}-01',
                f'2026-{i+1:02d}-28',
                {'ic_mean': ic, 'icir': 1.0, 'ic_win_rate': 0.5},
            )
        
        # 检测衰减
        is_decaying = tracker.detect_decay('test_factor', window=3)
        
        assert is_decaying is True
    
    def test_get_trend(self, temp_db):
        """测试获取趋势"""
        tracker = PerformanceTracker(temp_db)
        
        # 记录多个月的数据
        for i in range(6):
            tracker.track_monthly(
                'test_factor',
                f'2026-{i+1:02d}',
                f'2026-{i+1:02d}-01',
                f'2026-{i+1:02d}-28',
                {'ic_mean': 0.05 + i * 0.01, 'icir': 1.0, 'ic_win_rate': 0.5},
            )
        
        trend = tracker.get_trend('test_factor', months=6)
        
        assert not trend.empty
        assert len(trend) == 6
    
    def test_generate_warning_report(self, temp_db):
        """测试预警报告"""
        tracker = PerformanceTracker(temp_db)
        
        # 记录正常数据
        tracker.track_monthly(
            'test_factor',
            '2026-01',
            '2026-01-01',
            '2026-01-31',
            {'ic_mean': 0.05, 'icir': 1.5, 'ic_win_rate': 0.55},
        )
        
        report = tracker.generate_warning_report('test_factor')
        
        assert report['factor_id'] == 'test_factor'
        assert report['status'] in ['normal', 'warning', 'error']


class TestCorrelationTracker:
    """测试相关性追踪"""
    
    @pytest.fixture
    def tracker(self):
        """创建相关性追踪器"""
        return CorrelationTracker(window=20, correlation_threshold=0.7)
    
    @pytest.fixture
    def factor_dict(self):
        """创建测试因子数据"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2026-01-01', periods=n, freq='D')
        
        return {
            'factor_a': pd.Series(np.random.randn(n), index=dates),
            'factor_b': pd.Series(np.random.randn(n), index=dates),
            'factor_c': pd.Series(np.random.randn(n), index=dates),  # 高相关
        }
    
    def test_init(self, tracker):
        """测试初始化"""
        assert tracker.window == 20
        assert tracker.correlation_threshold == 0.7
    
    def test_calculate_matrix(self, tracker, factor_dict):
        """测试计算相关性矩阵"""
        matrix = tracker.calculate_matrix(factor_dict)
        
        assert matrix is not None
    
    def test_find_high_correlation_pairs(self, tracker):
        """测试查找高相关因子对"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2026-01-01', periods=n, freq='D')
        
        # 创建高相关的因子
        factor_dict = {
            'factor_a': pd.Series(np.random.randn(n), index=dates),
            'factor_b': pd.Series(np.random.randn(n), index=dates),
        }
        # 让 factor_c 与 factor_a 高相关
        factor_dict['factor_c'] = factor_dict['factor_a'] * 0.9 + pd.Series(np.random.randn(n) * 0.1, index=dates)
        
        pairs = tracker.find_high_correlation_pairs(factor_dict, threshold=0.7)
        
        # 应该找到高相关对
        assert len(pairs) >= 1
    
    def test_track_correlation_change(self, tracker):
        """测试追踪相关性变化"""
        np.random.seed(42)
        n = 150
        dates = pd.date_range('2026-01-01', periods=n, freq='D')
        
        factor_a = pd.Series(np.random.randn(n), index=dates)
        factor_b = pd.Series(np.random.randn(n), index=dates)
        
        result = tracker.track_correlation_change(factor_a, factor_b, windows=[20, 60])
        
        assert 'corr_20' in result
        assert 'trend' in result
    
    def test_generate_report(self, tracker, factor_dict):
        """测试生成报告"""
        report = tracker.generate_report(factor_dict)
        
        assert report.matrix is not None
        assert report.summary is not None
        assert 'mean_correlation' in report.summary
    
    def test_recommend_factor_removal(self, tracker):
        """测试建议移除因子"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2026-01-01', periods=n, freq='D')
        
        # 创建高相关的因子
        base = pd.Series(np.random.randn(n), index=dates)
        factor_dict = {
            'factor_a': base,
            'factor_b': pd.Series(np.random.randn(n), index=dates),
            'factor_c': base * 0.95 + pd.Series(np.random.randn(n) * 0.05, index=dates),
        }
        
        to_remove = tracker.recommend_factor_removal(factor_dict)
        
        # 应该建议移除一些因子
        assert isinstance(to_remove, list)


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def temp_repo(self):
        """创建临时因子库"""
        temp_dir = tempfile.mkdtemp()
        repo = FactorRepository(temp_dir)
        yield repo
        shutil.rmtree(temp_dir)
    
    def test_full_workflow(self, temp_repo):
        """测试完整工作流"""
        # 1. 创建因子
        class MockFactor:
            def __init__(self):
                self.name = 'momentum'
                self.params = {'window': 20}
                self.category = 'technical'
        
        factor = MockFactor()
        dates = pd.date_range('2026-01-01', periods=100, freq='D')
        values = pd.DataFrame({'value': np.random.randn(100).cumsum()}, index=dates)
        
        # 2. 保存因子
        factor_id = temp_repo.save_factor(
            factor,
            values,
            performance={'ic_mean': 0.06, 'icir': 1.8},
            version_id='v1.0',
        )
        
        # 3. 创建版本
        vc = FactorVersionControl(str(temp_repo.db_path))
        
        # 先创建外键记录
        import sqlite3
        conn = sqlite3.connect(str(temp_repo.db_path))
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO factor_metadata (factor_id, name) VALUES (?, ?)',
                      (factor_id, factor.name))
        conn.commit()
        conn.close()
        
        vc.create_version(factor_id, 'v1.0', factor.params, 'momentum_code', 'Initial')
        
        # 4. 验证
        loaded = temp_repo.get_factor(factor_id)
        assert loaded['name'] == 'momentum'
        
        history = vc.get_version_history(factor_id)
        assert len(history) == 1
        
        print('Full workflow test passed!')
