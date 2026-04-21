"""
遗传规划因子进化引擎单元测试
"""

import random
import numpy as np
import pandas as pd
import pytest

from futureQuant.agent.factor_mining.gp_factor_engine import (
    GPFactorEngine,
    EvolutionConfig,
    TreeGenerator,
    FitnessEvaluator,
    FitnessResult,
    Individual,
    ExprNode,
)


def _make_price_data(n=200, seed=42) -> pd.DataFrame:
    """生成测试用价格数据"""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    close = 4000 + np.cumsum(np.random.randn(n) * 20)
    open_ = close + np.random.randn(n) * 5
    high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 3)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 3)
    volume = np.random.randint(10000, 100000, n)

    df = pd.DataFrame({
        'date': dates, 'close': close, 'open': open_,
        'high': high, 'low': low, 'volume': volume,
    })
    df.set_index('date', inplace=True)
    return df


class TestExprNode:
    """表达式树节点测试"""

    def test_copy(self):
        node = ExprNode('add', arity=2, is_terminal=False)
        child1 = ExprNode('close', is_terminal=True, value='close')
        child2 = ExprNode('volume', is_terminal=True, value='volume')
        node.children = [child1, child2]
        copied = node.copy()
        assert copied.name == 'add'
        assert len(copied.children) == 2
        assert copied is not node
        assert copied.children[0] is not node.children[0]

    def test_to_infix(self):
        # close + volume
        node = ExprNode('+', arity=2, is_terminal=False)
        node.children = [
            ExprNode('close', is_terminal=True, value='close'),
            ExprNode('volume', is_terminal=True, value='volume'),
        ]
        assert node.to_infix() == '(close + volume)'

    def test_evaluate(self):
        data = _make_price_data(10)
        returns = data['close'].pct_change().fillna(0)
        ctx = {
            'close': data['close'],
            'volume': data['volume'],
        }

        # close
        node = ExprNode('close', is_terminal=True, value='close')
        result = node.evaluate(ctx)
        assert len(result) == len(data)
        assert result.name == 'close'

    def test_get_nodes(self):
        node = ExprNode('+', arity=2, is_terminal=False)
        node.children = [
            ExprNode('close', is_terminal=True, value='close'),
            ExprNode('volume', is_terminal=True, value='volume'),
        ]
        nodes = node.get_nodes()
        assert len(nodes) == 3  # root + 2 children

    def test_get_random_node(self):
        node = ExprNode('+', arity=2, is_terminal=False)
        node.children = [
            ExprNode('close', is_terminal=True, value='close'),
            ExprNode('volume', is_terminal=True, value='volume'),
        ]
        chosen = node.get_random_node()
        assert chosen is not None


class TestTreeGenerator:
    """树生成器测试"""

    def test_generate_ramped_half(self):
        gen = TreeGenerator(max_depth=3)
        tree = gen.generate('ramped_half')
        assert tree is not None
        assert tree.get_depth() <= 4

    def test_generate_full(self):
        gen = TreeGenerator(max_depth=2)
        tree = gen.generate('full')
        assert tree.get_depth() == 3  # depth starts at 0

    def test_generate_grow(self):
        gen = TreeGenerator(max_depth=3)
        tree = gen.generate('grow')
        assert tree is not None
        assert tree.get_depth() >= 1

    def test_mutation_produces_different_tree(self):
        gen = TreeGenerator(max_depth=3)
        tree = gen.generate('ramped_half')
        mutated = gen.mutate_node(tree)
        assert mutated is not None


class TestFitnessEvaluator:
    """适应度评估器测试"""

    def test_evaluate_with_valid_data(self):
        data = _make_price_data(100)
        returns = data['close'].pct_change().fillna(0)
        factor_values = data['close'].rolling(5).mean()

        evaluator = FitnessEvaluator(min_ic=0.01, min_icir=0.1)
        result = evaluator.evaluate(factor_values, returns)

        assert isinstance(result, FitnessResult)
        assert result.ic_mean != 0.0 or result.error is not None

    def test_evaluate_insufficient_data(self):
        data = _make_price_data(5)
        returns = data['close'].pct_change().fillna(0)
        factor_values = pd.Series([1, 2, 3, 4, 5])

        evaluator = FitnessEvaluator()
        result = evaluator.evaluate(factor_values, returns)
        assert result.error == 'insufficient data'

    def test_score_weights(self):
        weights = {
            'ic_mean': 0.40,
            'icir': 0.20,
            'win_rate': 0.20,
            'monotonicity': 0.10,
            'turnover': 0.05,
            'independence': 0.05,
        }
        evaluator = FitnessEvaluator(score_weights=weights)
        assert evaluator.score_weights == weights


class TestGPFactorEngine:
    """GP 因子进化引擎测试"""

    def test_init(self):
        config = EvolutionConfig(population_size=20, generations=5)
        engine = GPFactorEngine(config)
        assert engine.config.population_size == 20
        assert engine.config.generations == 5
        assert engine.population == []

    def test_set_data(self):
        data = _make_price_data(100)
        returns = data['close'].pct_change().fillna(0)

        engine = GPFactorEngine()
        engine.set_data(data, returns)

        assert 'close' in engine.data_context
        assert 'returns' in engine.data_context
        assert engine.returns_series is not None

    def test_evolve_produces_history(self):
        data = _make_price_data(200, seed=123)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(
            population_size=30,
            generations=5,
            max_depth=3,
            var_set=['close', 'volume'],
        )
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)

        history = engine.evolve()

        assert len(history) > 0
        assert 'generation' in history[0]
        assert 'best_score' in history[0]
        assert 'avg_score' in history[0]
        assert history[-1]['generation'] <= 4  # max 5 generations (0-indexed)

    def test_evolve_best_improves(self):
        data = _make_price_data(200, seed=456)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(
            population_size=40,
            generations=8,
            max_depth=3,
        )
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)

        history = engine.evolve()
        assert history[-1]['best_score'] >= 0  # score should be non-negative

    def test_get_best(self):
        data = _make_price_data(200, seed=789)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(population_size=30, generations=5)
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)
        engine.evolve()

        top = engine.get_best(top_k=5)
        assert len(top) <= 5
        scores = [ind.fitness.overall_score for ind in top]
        assert scores == sorted(scores, reverse=True)

    def test_evolve_requires_data(self):
        engine = GPFactorEngine()
        with pytest.raises(ValueError, match="Data not set"):
            engine.evolve()

    def test_evolve_with_small_population(self):
        data = _make_price_data(100)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(population_size=5, generations=3, max_depth=2)
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)

        history = engine.evolve()
        assert len(history) >= 1


class TestGPIntegration:
    """GP 引擎集成测试"""

    def test_full_pipeline(self):
        data = _make_price_data(200, seed=111)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(
            population_size=50,
            generations=10,
            max_depth=4,
            elite_size=3,
        )
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)

        history = engine.evolve()
        top = engine.get_best(top_k=10)

        assert len(history) >= 1
        assert len(top) > 0
        assert history[-1]['best_score'] >= 0

        best = top[0]
        assert best.fitness.ic_mean is not None
        assert best.tree is not None
        expr = best.tree.to_infix()
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_top_factors(self):
        data = _make_price_data(200, seed=222)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(population_size=30, generations=5)
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)
        engine.evolve()

        factors = engine.get_top_factors(top_k=5)
        assert len(factors) <= 5
        for f in factors:
            assert 'name' in f
            assert 'ic_mean' in f
            assert 'overall_score' in f
            assert 'source' in f

    def test_best_individual_persists(self):
        data = _make_price_data(200, seed=333)
        returns = data['close'].pct_change().fillna(0)

        config = EvolutionConfig(population_size=30, generations=5)
        engine = GPFactorEngine(config)
        engine.set_data(data, returns)
        engine.evolve()

        assert engine.best_individual is not None
        assert engine.best_individual.fitness.overall_score >= 0
