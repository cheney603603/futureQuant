"""
遗传规划因子进化引擎 (GP Factor Evolution Engine)

纯 numpy/scipy 实现，不依赖 gplearn。

功能：
1. 因子表达式树表示与解析
2. 遗传编程进化（交叉、变异、选择）
3. 适应度评估（IC/ICIR）
4. 与 DataFrame 数据集成
5. 进化历史记录

基于 Koza (1992) 遗传编程框架，针对量化因子优化。
"""

from __future__ import annotations

import random
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from ...core.logger import get_logger

logger = get_logger("agent.factor_mining.gp_engine")

# =============================================================================
# 因子表达式节点定义
# =============================================================================


class ExprNode:
    """
    表达式树节点

    支持两类节点：
    - func: 函数节点（如 add、mul、div），有 children
    - term: 终结符节点（如 close、volume），无 children
    """

    def __init__(
        self,
        name: str,
        func: Optional[Callable[..., float]] = None,
        arity: int = 0,
        is_terminal: bool = False,
        value: Any = None,
    ) -> None:
        self.name = name
        self.func = func
        self.arity = arity
        self.is_terminal = is_terminal
        self.value = value  # 仅终结符使用
        self.children: List[ExprNode] = []
        self.parent: Optional[ExprNode] = None
        self.depth: int = 0

    def __repr__(self) -> str:
        if self.is_terminal:
            return f"T({self.name})"
        return f"F({self.name})"

    def copy(self) -> ExprNode:
        """深拷贝节点及其子树"""
        new_node = ExprNode(
            name=self.name,
            func=self.func,
            arity=self.arity,
            is_terminal=self.is_terminal,
            value=copy.deepcopy(self.value) if self.value is not None else None,
        )
        for child in self.children:
            child_copy = child.copy()
            child_copy.parent = new_node
            new_node.children.append(child_copy)
        return new_node

    def evaluate(self, context: Dict[str, pd.Series]) -> pd.Series:
        """
        在 DataFrame context 上求值

        Args:
            context: 变量名 -> Series 的字典

        Returns:
            pd.Series 计算结果
        """
        if self.is_terminal:
            if self.value in context:
                return context[self.value].copy()
            # 常量
            name = self.name
            for var_name, series in context.items():
                return pd.Series(self.value, index=series.index, name=name)
            return pd.Series(self.value)
        else:
            # 函数节点
            child_results = [c.evaluate(context) for c in self.children]
            result = self.func(*child_results)
            return result

    def to_infix(self) -> str:
        """转换为中缀表达式字符串"""
        if self.is_terminal:
            return self.name
        if len(self.children) == 1:
            return f"{self.name}({self.children[0].to_infix()})"
        if len(self.children) == 2:
            left = self.children[0].to_infix()
            right = self.children[1].to_infix()
            return f"({left} {self.name} {right})"
        # 多参数
        args = ", ".join(c.to_infix() for c in self.children)
        return f"{self.name}({args})"

    def get_depth(self) -> int:
        """计算子树深度"""
        if not self.children:
            return 1
        return 1 + max(c.get_depth() for c in self.children)

    def get_nodes(self) -> List[ExprNode]:
        """返回所有节点的扁平列表（后序）"""
        nodes = []
        for child in self.children:
            nodes.extend(child.get_nodes())
        nodes.append(self)
        return nodes

    def get_random_node(self) -> ExprNode:
        """随机选择一个节点"""
        nodes = self.get_nodes()
        return random.choice(nodes)

    def replace(self, old_node: ExprNode, new_node: ExprNode) -> None:
        """替换子树"""
        if new_node.parent is not None:
            new_node = new_node.copy()
        new_node.parent = self
        for i, child in enumerate(self.children):
            if child is old_node:
                self.children[i] = new_node
                return
        raise ValueError("Node not found in children")

    def set_children(self, children: List[ExprNode]) -> None:
        """设置子节点"""
        self.children = []
        for child in children:
            child_copy = child.copy()
            child_copy.parent = self
            self.children.append(child_copy)


# =============================================================================
# 函数库（算子）
# =============================================================================

# 二元运算符
def _safe_add(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a + b
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_sub(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a - b
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_mul(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a * b
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < 1e-10, np.nan, a / b)
    return pd.Series(result, index=a.index)


# 一元运算符
def _safe_abs(x: pd.Series) -> pd.Series:
    return x.abs()


def _safe_sqrt(x: pd.Series) -> pd.Series:
    with np.errstate(invalid="ignore"):
        result = np.sqrt(np.maximum(x, 0))
    return pd.Series(result, index=x.index)


def _safe_log(x: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(np.maximum(x, 1e-10))
    return pd.Series(result, index=x.index)


def _safe_neg(x: pd.Series) -> pd.Series:
    return -x


# 聚合/滚动函数（生成终结符的工厂函数）
def _make_rolling(window: int, stat: str) -> Callable[[pd.Series], pd.Series]:
    def _fn(x: pd.Series) -> pd.Series:
        if stat == "mean":
            return x.rolling(window, min_periods=max(window // 2, 1)).mean()
        if stat == "std":
            return x.rolling(window, min_periods=max(window // 2, 1)).std()
        if stat == "sum":
            return x.rolling(window, min_periods=max(window // 2, 1)).sum()
        if stat == "max":
            return x.rolling(window, min_periods=max(window // 2, 1)).max()
        if stat == "min":
            return x.rolling(window, min_periods=max(window // 2, 1)).min()
        return x
    return _fn


# =============================================================================
# 因子表达式树生成器
# =============================================================================

class TreeGenerator:
    """表达式树随机生成器"""

    # 函数集（算子）
    FUNC_SET: List[Tuple[str, Callable, int]] = [
        ("+", _safe_add, 2),
        ("-", _safe_sub, 2),
        ("*", _safe_mul, 2),
        ("/", _safe_div, 2),
        ("abs", _safe_abs, 1),
        ("sqrt", _safe_sqrt, 1),
        ("log", _safe_log, 1),
        ("neg", _safe_neg, 1),
    ]

    # 终结符变量
    VAR_SET: List[str] = [
        "close", "open", "high", "low", "volume",
        "returns", "log_returns",
    ]

    # 常量
    CONST_VALUES: List[float] = [1, 2, 5, 10, 20, 0.5, 0.1, -1, -0.5]

    def __init__(
        self,
        max_depth: int = 4,
        var_set: Optional[List[str]] = None,
        custom_funcs: Optional[List[Tuple[str, Callable, int]]] = None,
    ) -> None:
        self.max_depth = max_depth
        self.var_set = var_set or self.VAR_SET
        self.func_set = custom_funcs or self.FUNC_SET

    def generate(self, method: str = "ramped_half") -> ExprNode:
        """
        生成随机表达式树

        Args:
            method: 'full'（满树）/ 'grow'（自由生长）/ 'ramped_half'（混合）

        Returns:
            ExprNode 根节点
        """
        if method == "full":
            return self._generate_full(0)
        elif method == "grow":
            return self._generate_grow(0)
        else:  # ramped_half
            depth = random.randint(1, self.max_depth)
            if random.random() < 0.5:
                return self._generate_full(depth)
            else:
                return self._generate_grow(depth)

    def _generate_full(self, depth: int) -> ExprNode:
        """满树生成：所有叶子节点深度相同"""
        if depth >= self.max_depth:
            return self._make_terminal()
        # 选择函数
        name, func, arity = random.choice(self.func_set)
        node = ExprNode(name=name, func=func, arity=arity, is_terminal=False)
        for _ in range(arity):
            child = self._generate_full(depth + 1)
            child.parent = node
            node.children.append(child)
        return node

    def _generate_grow(self, depth: int) -> ExprNode:
        """自由生长：叶子节点深度可变"""
        if depth >= self.max_depth:
            return self._make_terminal()
        # 函数和终结符随机
        if random.random() < 0.3:  # 30% 概率选终结符
            return self._make_terminal()
        name, func, arity = random.choice(self.func_set)
        node = ExprNode(name=name, func=func, arity=arity, is_terminal=False)
        for _ in range(arity):
            child = self._generate_grow(depth + 1)
            child.parent = node
            node.children.append(child)
        return node

    def _make_terminal(self) -> ExprNode:
        """创建终结符节点"""
        if random.random() < 0.7:
            # 变量
            name = random.choice(self.var_set)
            return ExprNode(name=name, is_terminal=True, value=name)
        else:
            # 常量（附加滚动统计）
            value = random.choice(self.CONST_VALUES)
            # 滚动窗口
            window = random.choice([3, 5, 10, 20])
            stat = random.choice(["mean", "std", "sum"])
            roll_fn = _make_rolling(window, stat)

            def const_func() -> pd.Series:
                return pd.Series(value, dtype=float)

            node = ExprNode(
                name=f"const_{value}_{window}d_{stat}",
                func=const_func,
                arity=0,
                is_terminal=True,
                value=value,
            )
            return node

    def mutate_node(self, node: ExprNode) -> ExprNode:
        """对节点进行变异"""
        if node.is_terminal:
            return self._make_terminal()
        else:
            return self._generate_grow(0)

    def crossover_node(self, parent1: ExprNode, parent2: ExprNode) -> Tuple[ExprNode, ExprNode]:
        """
        交叉两个子树

        随机选择两个父本的子树，交换它们。
        """
        p1_copy = parent1.copy()
        p2_copy = parent2.copy()

        # 随机选子树
        node1 = p1_copy.get_random_node()
        node2 = p2_copy.get_random_node()

        # 交换
        if node1.parent is None or node2.parent is None:
            return p1_copy, p2_copy

        node1.parent.replace(node1, node2.copy())
        node2.parent.replace(node2, node1.copy())

        return p1_copy, p2_copy


# =============================================================================
# 适应度评估
# =============================================================================


@dataclass
class FitnessResult:
    """适应度评估结果"""
    ic_mean: float = 0.0
    icir: float = 0.0
    ic_win_rate: float = 0.0
    turnover: float = 0.0
    overall_score: float = 0.0
    is_valid: bool = False
    error: Optional[str] = None


class FitnessEvaluator:
    """适应度评估器（基于 IC/ICIR）"""

    def __init__(
        self,
        min_ic: float = 0.02,
        min_icir: float = 0.3,
        score_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.min_ic = min_ic
        self.min_icir = min_icir
        self.score_weights = score_weights or {
            "ic_mean": 0.30,
            "icir": 0.20,
            "win_rate": 0.15,
            "monotonicity": 0.15,
            "turnover": 0.15,
            "independence": 0.05,
        }

    def evaluate(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> FitnessResult:
        """
        评估因子适应度

        Args:
            factor_values: 因子值序列
            returns: 收益率序列

        Returns:
            FitnessResult
        """
        try:
            # 对齐数据
            aligned = pd.concat([factor_values, returns], axis=1).dropna()
            if len(aligned) < 20:
                return FitnessResult(error="insufficient data")

            fv = aligned.iloc[:, 0].values
            rt = aligned.iloc[:, 1].values

            # IC（Spearman）
            ic, ic_p = stats.spearmanr(fv, rt)
            if np.isnan(ic):
                return FitnessResult(error="IC computation failed")

            # ICIR
            # 分期滚动 IC
            window = min(20, len(aligned) // 2)
            rolling_ic: List[float] = []
            for i in range(window, len(aligned)):
                sub = aligned.iloc[i - window:i]
                if len(sub) >= 10:
                    r, _ = stats.spearmanr(sub.iloc[:, 0].values, sub.iloc[:, 1].values)
                    if not np.isnan(r):
                        rolling_ic.append(r)

            if len(rolling_ic) < 3:
                icir = 0.0
            else:
                ic_std = np.std(rolling_ic) if np.std(rolling_ic) > 0 else 1e-10
                icir = np.mean(rolling_ic) / ic_std

            # IC 胜率
            ic_win_rate = sum(1 for x in rolling_ic if x * ic > 0) / max(len(rolling_ic), 1)

            # 换手率（因子值变化）
            turnover = factor_values.diff().abs().mean()

            # 综合评分
            abs_ic = abs(ic)
            ic_score = min(abs_ic / 0.05, 1.0)
            icir_score = min(icir / 2.0, 1.0)
            win_score = ic_win_rate
            turn_score = 1.0 / (1.0 + turnover * 10)

            overall = (
                self.score_weights["ic_mean"] * ic_score
                + self.score_weights["icir"] * icir_score
                + self.score_weights["win_rate"] * win_score
                + self.score_weights["turnover"] * turn_score
            )

            # 有效性判断
            is_valid = abs_ic >= self.min_ic and icir >= self.min_icir

            return FitnessResult(
                ic_mean=float(ic),
                icir=float(icir),
                ic_win_rate=float(ic_win_rate),
                turnover=float(turnover),
                overall_score=float(overall),
                is_valid=is_valid,
            )

        except Exception as e:
            return FitnessResult(error=str(e))


# =============================================================================
# 遗传规划主引擎
# =============================================================================


@dataclass
class EvolutionConfig:
    """进化配置"""
    population_size: int = 100
    generations: int = 20
    elite_size: int = 5
    crossover_prob: float = 0.70
    mutation_prob: float = 0.25
    max_depth: int = 4
    tournament_size: int = 3
    min_fitness_threshold: float = 1.0
    # 变量集
    var_set: Optional[List[str]] = None


@dataclass
class Individual:
    """进化个体"""
    tree: ExprNode
    fitness: FitnessResult
    age: int = 0  # 代数

    def __lt__(self, other: Individual) -> bool:
        return self.fitness.overall_score < other.fitness.overall_score


class GPFactorEngine:
    """
    遗传规划因子进化引擎

    使用遗传编程自动探索和进化优质因子表达式。

    使用示例：
        >>> config = EvolutionConfig(population_size=50, generations=10)
        >>> engine = GPFactorEngine(config)
        >>> engine.set_data(price_df, returns)
        >>> history = engine.evolve()
        >>> best = engine.get_best()
        >>> print(best.tree.to_infix())
    """

    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        self.config = config or EvolutionConfig()
        self.generator = TreeGenerator(
            max_depth=self.config.max_depth,
            var_set=self.config.var_set,
        )
        self.evaluator = FitnessEvaluator()
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []
        self.data_context: Dict[str, pd.Series] = {}
        self.returns_series: Optional[pd.Series] = None
        self.logger = logger

    def set_data(
        self,
        data: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        var_names: Optional[List[str]] = None,
    ) -> None:
        """
        设置输入数据

        Args:
            data: OHLCV 数据或包含因子的 DataFrame
            returns: 收益率序列（可选，自动计算）
            var_names: 使用的变量名列表（列名）
        """
        self.data_context = {}
        var_list = var_names or ["close", "open", "high", "low", "volume"]

        for col in var_list:
            if col in data.columns:
                self.data_context[col] = data[col].ffill().fillna(0)

        # 计算收益率
        if returns is not None:
            self.returns_series = returns.fillna(0)
        elif "close" in data.columns:
            self.returns_series = data["close"].pct_change().fillna(0)
        else:
            # 使用第一列
            first_col = data.columns[0]
            self.returns_series = data[first_col].pct_change().fillna(0)

        # 添加 returns 和 log_returns
        self.data_context["returns"] = self.returns_series.copy()
        self.data_context["log_returns"] = np.log(
            self.returns_series.clip(lower=1e-10) + 1
        ).fillna(0)

    def _init_population(self) -> None:
        """初始化种群"""
        self.population = []
        for _ in range(self.config.population_size):
            tree = self.generator.generate(method="ramped_half")
            fitness = self._evaluate(tree)
            self.population.append(Individual(tree=tree, fitness=fitness))

        self.population.sort(key=lambda x: x.fitness.overall_score, reverse=True)
        self._update_best()

    def _evaluate(self, tree: ExprNode) -> FitnessResult:
        """评估一个个体的适应度"""
        try:
            values = tree.evaluate(self.data_context)
            if self.returns_series is None:
                return FitnessResult(error="no returns data")
            return self.evaluator.evaluate(values, self.returns_series)
        except Exception as e:
            return FitnessResult(error=str(e))

    def _select_tournament(self) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(
            self.population, min(self.config.tournament_size, len(self.population))
        )
        return max(tournament, key=lambda x: x.fitness.overall_score)

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """交叉"""
        new_p1, new_p2 = self.generator.crossover_node(p1.tree, p2.tree)
        f1 = self._evaluate(new_p1)
        f2 = self._evaluate(new_p2)
        return Individual(new_p1, f1), Individual(new_p2, f2)

    def _mutate(self, ind: Individual) -> Individual:
        """变异"""
        new_tree = ind.tree.copy()
        node = new_tree.get_random_node()
        # 替换子树
        parent = node.parent
        if parent is not None:
            for i, child in enumerate(parent.children):
                if child is node:
                    parent.children[i] = self.generator.mutate_node(node)
                    parent.children[i].parent = parent
                    break
        else:
            # 根节点直接替换
            new_tree = self.generator.generate(method="ramped_half")
        fitness = self._evaluate(new_tree)
        return Individual(new_tree, fitness)

    def _update_best(self) -> None:
        """更新最优个体"""
        if not self.population:
            return
        current_best = max(self.population, key=lambda x: x.fitness.overall_score)
        if self.best_individual is None or current_best.fitness.overall_score > self.best_individual.fitness.overall_score:
            self.best_individual = copy.deepcopy(current_best)

    def evolve(self) -> List[Dict[str, Any]]:
        """
        执行进化流程

        Returns:
            进化历史记录
        """
        if not self.data_context:
            raise ValueError("Data not set. Call set_data() first.")
        if self.returns_series is None:
            raise ValueError("Returns not available.")

        self.logger.info(f"Starting GP evolution: pop={self.config.population_size}, "
                        f"gen={self.config.generations}")
        self.history = []

        # 初始化
        self._init_population()

        for gen in range(self.config.generations):
            gen_start = time.time()

            # 记录历史
            scores = [ind.fitness.overall_score for ind in self.population]
            valid_count = sum(1 for ind in self.population if ind.fitness.is_valid)
            best_score = max(scores) if scores else 0.0
            avg_score = np.mean(scores) if scores else 0.0

            gen_record = {
                "generation": gen,
                "best_score": float(best_score),
                "avg_score": float(avg_score),
                "valid_count": valid_count,
                "best_expr": self.population[0].tree.to_infix() if self.population else "",
                "best_ic": float(self.population[0].fitness.ic_mean) if self.population else 0.0,
                "best_icir": float(self.population[0].fitness.icir) if self.population else 0.0,
                "time": time.time() - gen_start,
            }
            self.history.append(gen_record)

            self.logger.info(
                f"  Gen {gen:2d} | best={best_score:.4f} | avg={avg_score:.4f} "
                f"| valid={valid_count} | {gen_record['time']:.1f}s"
            )

            # 早停
            if best_score >= self.config.min_fitness_threshold:
                self.logger.info(f"  Early stop: fitness threshold reached")
                break

            # 生成新一代
            new_population: List[Individual] = []

            # 精英保留
            elite = sorted(self.population, key=lambda x: x.fitness.overall_score, reverse=True)
            for ind in elite[: self.config.elite_size]:
                new_population.append(copy.deepcopy(ind))

            # 进化
            while len(new_population) < self.config.population_size:
                r = random.random()
                if r < self.config.crossover_prob:
                    # 交叉
                    p1 = self._select_tournament()
                    p2 = self._select_tournament()
                    c1, c2 = self._crossover(p1, p2)
                    new_population.extend([c1, c2])
                elif r < self.config.crossover_prob + self.config.mutation_prob:
                    # 变异
                    ind = self._select_tournament()
                    new_pop = self._mutate(ind)
                    new_population.append(new_pop)
                else:
                    # 复制
                    ind = self._select_tournament()
                    new_population.append(copy.deepcopy(ind))

            # 截断到目标大小
            self.population = new_population[: self.config.population_size]
            self._update_best()

        # 最终记录
        if self.best_individual:
            self.logger.info(f"Evolution complete. Best: {self.best_individual.tree.to_infix()}")
            self.logger.info(f"  IC={self.best_individual.fitness.ic_mean:.4f}, "
                            f"ICIR={self.best_individual.fitness.icir:.4f}, "
                            f"score={self.best_individual.fitness.overall_score:.4f}")

        return self.history

    def get_best(self, top_k: int = 10) -> List[Individual]:
        """获取最优个体列表"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness.overall_score, reverse=True)
        return sorted_pop[:top_k]

    def get_top_factors(
        self,
        top_k: int = 20,
        include_pool: Optional[List[ExprNode]] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取 Top 因子列表

        Args:
            top_k: 返回数量
            include_pool: 额外包含的因子节点（如从候选池来的）

        Returns:
            因子信息字典列表
        """
        factors: List[Dict[str, Any]] = []

        # GP 进化出的因子
        for ind in self.get_best(top_k):
            if ind.fitness.is_valid:
                factors.append({
                    "name": f"gp_{ind.tree.to_infix()[:40]}",
                    "expression": ind.tree.to_infix(),
                    "tree": ind.tree,
                    "ic_mean": ind.fitness.ic_mean,
                    "icir": ind.fitness.icir,
                    "ic_win_rate": ind.fitness.ic_win_rate,
                    "turnover": ind.fitness.turnover,
                    "overall_score": ind.fitness.overall_score,
                    "source": "gp_evolution",
                    "is_valid": ind.fitness.is_valid,
                })

        # 候选池因子
        if include_pool:
            for tree in include_pool[: top_k - len(factors)]:
                try:
                    values = tree.evaluate(self.data_context)
                    fitness = self.evaluator.evaluate(values, self.returns_series)
                    if fitness.is_valid:
                        factors.append({
                            "name": tree.name,
                            "expression": tree.to_infix(),
                            "tree": tree,
                            "ic_mean": fitness.ic_mean,
                            "icir": fitness.icir,
                            "ic_win_rate": fitness.ic_win_rate,
                            "turnover": fitness.turnover,
                            "overall_score": fitness.overall_score,
                            "source": "candidate_pool",
                            "is_valid": fitness.is_valid,
                        })
                except Exception:
                    pass

        # 按分数排序
        factors.sort(key=lambda x: x["overall_score"], reverse=True)
        return factors[:top_k]
