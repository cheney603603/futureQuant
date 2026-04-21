"""
自我反思模块单元测试
"""

import numpy as np
import pandas as pd
import pytest

from futureQuant.agent.factor_mining.self_reflection import (
    MiningResultEvaluator,
    StrategyAdjuster,
    FactorMiningSelfReflection,
    WeaknessReport,
    StrategyAdjustment,
)


def _make_price_data(n=200, seed=42) -> pd.DataFrame:
    """生成测试用价格数据"""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    close = 4000 + np.cumsum(np.random.randn(n) * 20)
    volume = np.random.randint(10000, 100000, n)
    df = pd.DataFrame({'date': dates, 'close': close, 'volume': volume})
    df.set_index('date', inplace=True)
    return df


class TestMiningResultEvaluator:
    """挖掘结果评估器测试"""

    def test_evaluate_empty_factors(self):
        evaluator = MiningResultEvaluator()
        weaknesses, summary = evaluator.evaluate([])
        assert len(weaknesses) == 1
        assert weaknesses[0].severity == 'critical'
        assert weaknesses[0].dimension == 'factor_count'
        assert summary == {}

    def test_evaluate_good_factors(self):
        evaluator = MiningResultEvaluator()
        factors = [
            {'name': 'factor_a', 'ic_mean': 0.05, 'icir': 0.8,
             'ic_win_rate': 0.6, 'turnover': 0.1,
             'overall_score': 0.7, 'is_valid': True, 'category': 'technical'},
            {'name': 'factor_b', 'ic_mean': 0.04, 'icir': 0.6,
             'ic_win_rate': 0.55, 'turnover': 0.15,
             'overall_score': 0.6, 'is_valid': True, 'category': 'fundamental'},
        ]
        data = _make_price_data(200)
        returns = data['close'].pct_change().fillna(0)
        weaknesses, summary = evaluator.evaluate(factors, data, returns)
        assert summary['total_factors'] == 2
        assert summary['valid_factors'] == 2
        assert len(weaknesses) <= 3  # 可能有 diversity warning

    def test_evaluate_weak_ic(self):
        evaluator = MiningResultEvaluator()
        factors = [
            {'name': 'weak_factor', 'ic_mean': 0.005, 'icir': 0.1,
             'ic_win_rate': 0.5, 'turnover': 0.5,
             'overall_score': 0.2, 'is_valid': False, 'category': 'technical'},
        ]
        weaknesses, summary = evaluator.evaluate(factors)
        assert any(w.dimension == 'ic_quality' for w in weaknesses)

    def test_evaluate_high_turnover(self):
        evaluator = MiningResultEvaluator()
        factors = [
            {'name': 'high_turn', 'ic_mean': 0.03, 'icir': 0.4,
             'ic_win_rate': 0.52, 'turnover': 0.8,
             'overall_score': 0.3, 'is_valid': True, 'category': 'technical'},
        ]
        weaknesses, summary = evaluator.evaluate(factors)
        assert any(w.dimension == 'turnover' for w in weaknesses)

    def test_summary_statistics(self):
        evaluator = MiningResultEvaluator()
        factors = [
            {'name': f'f{i}', 'ic_mean': 0.03 + i * 0.01,
             'icir': 0.4 + i * 0.1, 'ic_win_rate': 0.5 + i * 0.02,
             'turnover': 0.2 - i * 0.02, 'overall_score': 0.5 + i * 0.05,
             'is_valid': True}
            for i in range(5)
        ]
        _, summary = evaluator.evaluate(factors)
        assert summary['total_factors'] == 5
        assert summary['valid_factors'] == 5
        assert 0.03 < summary['avg_ic'] < 0.07


class TestStrategyAdjuster:
    """策略调整器测试"""

    def test_no_weaknesses_no_adjustment(self):
        adjuster = StrategyAdjuster()
        strategy = StrategyAdjuster.DEFAULT_STRATEGY.copy()
        adjustment = adjuster.adjust(strategy, [], iteration=0)
        # 无严重问题时策略不变（但迭代会微调）
        assert adjustment.adjusted_strategy is not None

    def test_critical_ic_triggers_gp(self):
        adjuster = StrategyAdjuster()
        strategy = StrategyAdjuster.DEFAULT_STRATEGY.copy()
        strategy['gp_evolution_enabled'] = False

        weaknesses = [
            WeaknessReport(
                dimension='ic_quality',
                severity='critical',
                description='IC 过低',
                suggestion='启用 GP 进化',
                current_value=0.005,
                threshold=0.02,
            )
        ]

        adjustment = adjuster.adjust(strategy, weaknesses, iteration=0)
        assert adjustment.adjusted_strategy['gp_evolution_enabled'] is True
        assert adjustment.reason != ''

    def test_diversity_warning_enables_fundamental(self):
        adjuster = StrategyAdjuster()
        strategy = StrategyAdjuster.DEFAULT_STRATEGY.copy()
        strategy['use_fundamental'] = False
        strategy['macro_enabled'] = False

        weaknesses = [
            WeaknessReport(
                dimension='diversity',
                severity='warning',
                description='类别不足',
                suggestion='引入基本面因子',
                current_value=1.0,
                threshold=2.0,
            )
        ]

        adjustment = adjuster.adjust(strategy, weaknesses, iteration=0)
        assert adjustment.adjusted_strategy['use_fundamental'] is True
        assert adjustment.adjusted_strategy['macro_enabled'] is True

    def test_iteration_increases_search(self):
        adjuster = StrategyAdjuster()
        strategy = StrategyAdjuster.DEFAULT_STRATEGY.copy()
        adjustment = adjuster.adjust(strategy, [], iteration=3)
        assert adjustment.adjusted_strategy['gp_generations'] > strategy.get('gp_generations', 10)


class TestFactorMiningSelfReflection:
    """因子挖掘自我反思器测试"""

    def test_init(self):
        reflection = FactorMiningSelfReflection(max_iterations=3)
        assert reflection.max_iterations == 3
        assert reflection.history == []

    def test_reflect_no_factors(self):
        reflection = FactorMiningSelfReflection(max_iterations=3)
        should_continue, adjustment = reflection.reflect(
            factors=[],
            price_data=None,
            returns=None,
            current_strategy={},
            iteration=0,
        )
        assert should_continue is False
        assert adjustment.adjusted_strategy is not None

    def test_reflect_max_iterations_reached(self):
        reflection = FactorMiningSelfReflection(max_iterations=2)
        factors = [{'overall_score': 0.5, 'is_valid': True, 'ic_mean': 0.03, 'icir': 0.4}]
        should_continue, _ = reflection.reflect(
            factors, None, None, {}, iteration=2,
        )
        assert should_continue is False

    def test_reflect_creates_history(self):
        reflection = FactorMiningSelfReflection(max_iterations=3)
        factors = [{'overall_score': 0.5, 'is_valid': True, 'ic_mean': 0.03, 'icir': 0.4}]
        reflection.reflect(factors, None, None, {}, iteration=0)
        assert len(reflection.history) == 1
        assert 'weaknesses' in reflection.history[0]
        assert 'summary' in reflection.history[0]

    def test_generate_report(self):
        reflection = FactorMiningSelfReflection(max_iterations=3)
        factors = [{'overall_score': 0.5, 'is_valid': True, 'ic_mean': 0.03, 'icir': 0.4}]
        reflection.reflect(factors, None, None, {}, iteration=0)
        report = reflection.generate_report()
        assert isinstance(report, str)
        assert '自我反思报告' in report
        assert '迭代 0' in report

    def test_reset(self):
        reflection = FactorMiningSelfReflection()
        factors = [{'overall_score': 0.5, 'is_valid': True}]
        reflection.reflect(factors, None, None, {}, iteration=0)
        reflection.reset()
        assert reflection.history == []


class TestWeaknessReport:
    """薄弱环节报告数据类测试"""

    def test_dataclass_fields(self):
        w = WeaknessReport(
            dimension='ic_quality',
            severity='critical',
            description='IC 过低',
            suggestion='启用 GP 进化',
            current_value=0.005,
            threshold=0.02,
        )
        assert w.dimension == 'ic_quality'
        assert w.severity == 'critical'
        assert w.current_value == 0.005
        assert w.threshold == 0.02


class TestIntegration:
    """集成测试"""

    def test_full_reflection_cycle(self):
        reflection = FactorMiningSelfReflection(max_iterations=2)
        data = _make_price_data(200)
        returns = data['close'].pct_change().fillna(0)

        factors = [
            {'name': 'weak_factor', 'ic_mean': 0.008, 'icir': 0.2,
             'ic_win_rate': 0.48, 'turnover': 0.7,
             'overall_score': 0.25, 'is_valid': False, 'category': 'technical'},
        ]

        should_continue, adjustment = reflection.reflect(
            factors, data, returns, {}, iteration=0
        )

        # 应该发现多个薄弱环节
        assert len(reflection.history[0]['weaknesses']) >= 2
        # 策略应该被调整
        assert adjustment.reason != ''

        # 第二次迭代
        if should_continue:
            better_factors = [
                {'name': 'better_factor', 'ic_mean': 0.04, 'icir': 0.6,
                 'ic_win_rate': 0.55, 'turnover': 0.2,
                 'overall_score': 0.6, 'is_valid': True, 'category': 'technical'},
            ]
            should_continue2, _ = reflection.reflect(
                better_factors, data, returns, adjustment.adjusted_strategy, iteration=1
            )
            # 不再有严重问题
            assert not any(w.severity == 'critical' for w in reflection.history[1]['weaknesses'])
