"""
策略参数优化模块

提供多种参数优化方法：
- 网格搜索优化 (Grid Search)
- 随机搜索优化 (Random Search)
- 贝叶斯优化 (Bayesian Optimization via Optuna)
- 滚动前向优化 (Walk-Forward Optimization)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from itertools import product
import random
from datetime import datetime

# 尝试导入optuna用于贝叶斯优化
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None

from ..core.logger import get_logger
from ..core.exceptions import StrategyError

logger = get_logger('strategy.optimizer')


@dataclass
class OptimizationResult:
    """
    优化结果数据类
    
    Attributes:
        best_params: 最优参数字典
        best_score: 最优分数
        all_results: 所有试验结果的DataFrame
        optimization_history: 优化历史记录
        metric: 优化使用的评价指标
        optimization_method: 优化方法名称
        total_trials: 总试验次数
        duration_seconds: 优化耗时（秒）
    """
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    optimization_history: List[Dict] = field(default_factory=list)
    metric: str = 'sharpe'
    optimization_method: str = ''
    total_trials: int = 0
    duration_seconds: float = 0.0
    
    def __repr__(self) -> str:
        return (f"OptimizationResult(method={self.optimization_method}, "
                f"metric={self.metric}, best_score={self.best_score:.4f}, "
                f"total_trials={self.total_trials})")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'metric': self.metric,
            'optimization_method': self.optimization_method,
            'total_trials': self.total_trials,
            'duration_seconds': self.duration_seconds,
        }


class StrategyOptimizer:
    """
    策略参数优化器
    
    提供多种参数优化方法，支持网格搜索、随机搜索、贝叶斯优化和滚动前向优化。
    
    Attributes:
        strategy_class: 策略类（如 DualMAStrategy）
        data: 回测数据 (DataFrame)
        metric: 优化目标指标，可选 'sharpe'(夏普比率), 'return'(收益率), 'calmar'(卡玛比率)
        
    Example:
        >>> from futureQuant.strategy import DualMAStrategy
        >>> optimizer = StrategyOptimizer(DualMAStrategy, data, metric='sharpe')
        >>> result = optimizer.optimize_grid({
        ...     'fast_period': [5, 10, 15],
        ...     'slow_period': [20, 30, 40]
        ... })
        >>> print(result.best_params)
    """
    
    def __init__(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        metric: str = 'sharpe'
    ):
        """
        初始化策略优化器
        
        Args:
            strategy_class: 策略类，必须继承自 BaseStrategy
            data: 回测数据，包含OHLCV等列
            metric: 优化目标指标，'sharpe'/'return'/'calmar'
            
        Raises:
            StrategyError: 当metric不支持时
        """
        self.strategy_class = strategy_class
        self.data = data.copy()
        
        valid_metrics = ['sharpe', 'return', 'calmar', 'max_drawdown', 'win_rate']
        if metric not in valid_metrics:
            raise StrategyError(f"Unsupported metric: {metric}. "
                              f"Valid options: {valid_metrics}")
        self.metric = metric
        
        self._results: List[Dict] = []
        self._best_score = -np.inf
        self._best_params: Optional[Dict] = None
        
        logger.info(f"StrategyOptimizer initialized: strategy={strategy_class.__name__}, "
                   f"metric={metric}, data_shape={data.shape}")
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """
        评估一组参数的表现
        
        创建策略实例，生成信号，并计算指定的评价指标。
        
        Args:
            params: 参数字典
            
        Returns:
            评价指标分数（越高越好）
            
        Raises:
            StrategyError: 当策略执行出错时
        """
        try:
            # 创建策略实例
            strategy = self.strategy_class(**params)
            
            # 生成信号
            signals = strategy.generate_signals(self.data)
            
            if signals.empty or 'signal' not in signals.columns:
                logger.warning(f"Empty signals generated with params: {params}")
                return -np.inf
            
            # 计算收益（简化版本，假设信号直接对应仓位）
            returns = self._calculate_returns(signals)
            
            if len(returns) == 0 or returns.std() == 0:
                return -np.inf
            
            # 根据metric计算分数
            score = self._calculate_metric(returns)
            
            # 记录结果
            trial_result = {
                'params': params.copy(),
                'score': score,
                'metric': self.metric,
            }
            self._results.append(trial_result)
            
            # 更新最优结果
            if score > self._best_score:
                self._best_score = score
                self._best_params = params.copy()
                logger.debug(f"New best score: {score:.4f} with params: {params}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}")
            return -np.inf
    
    def _calculate_returns(self, signals: pd.DataFrame) -> pd.Series:
        """
        根据信号计算收益率序列
        
        Args:
            signals: 信号DataFrame，包含'signal'列
            
        Returns:
            收益率序列
        """
        # 获取价格数据
        if 'close' not in self.data.columns:
            raise StrategyError("Data must contain 'close' column")
        
        # 计算价格收益率
        price_returns = self.data['close'].pct_change()
        
        # 信号作为仓位，计算策略收益
        # 使用信号的shift(1)来避免未来函数
        if 'signal' in signals.columns:
            positions = signals['signal'].shift(1).fillna(0)
        else:
            positions = pd.Series(0, index=signals.index)
        
        # 策略收益 = 仓位 * 价格收益
        strategy_returns = positions * price_returns
        
        return strategy_returns.dropna()
    
    def _calculate_metric(self, returns: pd.Series) -> float:
        """
        计算评价指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            评价指标值
        """
        if len(returns) == 0:
            return -np.inf
        
        if self.metric == 'sharpe':
            # 夏普比率 = 年化收益 / 年化波动
            if returns.std() == 0:
                return -np.inf
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            return sharpe
        
        elif self.metric == 'return':
            # 总收益率
            total_return = (1 + returns).prod() - 1
            return total_return
        
        elif self.metric == 'calmar':
            # 卡玛比率 = 年化收益 / 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            if max_drawdown == 0:
                return -np.inf
            
            annual_return = returns.mean() * 252
            calmar = annual_return / abs(max_drawdown)
            return calmar
        
        elif self.metric == 'max_drawdown':
            # 最大回撤（越小越好，所以取负）
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            return -abs(max_drawdown)  # 取负，因为优化器是最大化
        
        elif self.metric == 'win_rate':
            # 胜率
            if len(returns) == 0:
                return 0
            win_rate = (returns > 0).sum() / len(returns)
            return win_rate
        
        return -np.inf
    
    def optimize_grid(self, param_grid: Dict[str, List]) -> OptimizationResult:
        """
        网格搜索优化
        
        遍历参数网格中的所有组合，找到最优参数。
        
        Args:
            param_grid: 参数网格字典，格式：{param_name: [value1, value2, ...]}
            
        Returns:
            OptimizationResult: 优化结果
            
        Example:
            >>> result = optimizer.optimize_grid({
            ...     'fast_period': [5, 10, 15],
            ...     'slow_period': [20, 30, 40]
            ... })
        """
        start_time = datetime.now()
        logger.info(f"Starting grid search optimization with param_grid: {param_grid}")
        
        self._reset_state()
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        logger.info(f"Total combinations to evaluate: {total_combinations}")
        
        # 遍历所有组合
        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))
            
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Grid search progress: {i + 1}/{total_combinations}")
            
            self._evaluate_params(params)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 构建结果
        result = self._build_result('grid_search', duration)
        logger.info(f"Grid search completed. Best score: {result.best_score:.4f}, "
                   f"Best params: {result.best_params}")
        
        return result
    
    def optimize_random(
        self,
        param_distributions: Dict[str, Tuple],
        n_iter: int = 100
    ) -> OptimizationResult:
        """
        随机搜索优化
        
        从参数分布中随机采样进行优化。
        
        Args:
            param_distributions: 参数分布字典，格式：
                - 对于整数参数: (low, high) 或 (low, high, 'int')
                - 对于浮点数参数: (low, high, 'float')
                - 对于离散参数: [value1, value2, ...]
            n_iter: 迭代次数，默认100
            
        Returns:
            OptimizationResult: 优化结果
            
        Example:
            >>> result = optimizer.optimize_random({
            ...     'fast_period': (5, 20, 'int'),
            ...     'slow_period': (20, 60, 'int'),
            ...     'ma_type': ['sma', 'ema']
            ... }, n_iter=50)
        """
        start_time = datetime.now()
        logger.info(f"Starting random search optimization with n_iter={n_iter}")
        
        self._reset_state()
        
        for i in range(n_iter):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Random search progress: {i + 1}/{n_iter}")
            
            # 随机采样参数
            params = self._sample_params(param_distributions)
            self._evaluate_params(params)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = self._build_result('random_search', duration)
        logger.info(f"Random search completed. Best score: {result.best_score:.4f}, "
                   f"Best params: {result.best_params}")
        
        return result
    
    def _sample_params(self, param_distributions: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        从参数分布中随机采样
        
        Args:
            param_distributions: 参数分布字典
            
        Returns:
            采样得到的参数字典
        """
        params = {}
        
        for param_name, distribution in param_distributions.items():
            if isinstance(distribution, (list, tuple)):
                if len(distribution) == 3 and distribution[2] in ['int', 'float']:
                    # 连续分布 (low, high, type)
                    low, high, param_type = distribution
                    if param_type == 'int':
                        params[param_name] = random.randint(int(low), int(high))
                    else:
                        params[param_name] = random.uniform(low, high)
                elif len(distribution) == 2 and isinstance(distribution[0], (int, float)):
                    # 默认整数 (low, high)
                    low, high = distribution
                    params[param_name] = random.randint(int(low), int(high))
                else:
                    # 离散分布
                    params[param_name] = random.choice(distribution)
            else:
                # 单个值
                params[param_name] = distribution
        
        return params
    
    def optimize_bayesian(
        self,
        param_bounds: Dict[str, Tuple],
        n_trials: int = 100
    ) -> OptimizationResult:
        """
        贝叶斯优化（使用Optuna）
        
        使用TPE（Tree-structured Parzen Estimator）算法进行高效的全局优化。
        
        Args:
            param_bounds: 参数边界字典，格式：
                - 整数参数: (low, high, 'int')
                - 浮点数参数: (low, high, 'float')
                - 离散参数: [value1, value2, ...]
            n_trials: 试验次数，默认100
            
        Returns:
            OptimizationResult: 优化结果
            
        Raises:
            StrategyError: 当optuna未安装时
            
        Example:
            >>> result = optimizer.optimize_bayesian({
            ...     'fast_period': (5, 20, 'int'),
            ...     'slow_period': (20, 60, 'int'),
            ...     'ma_type': ['sma', 'ema']
            ... }, n_trials=100)
        """
        if not OPTUNA_AVAILABLE:
            raise StrategyError(
                "Optuna is required for Bayesian optimization. "
                "Install it with: pip install optuna"
            )
        
        start_time = datetime.now()
        logger.info(f"Starting Bayesian optimization with n_trials={n_trials}")
        
        self._reset_state()
        
        # 创建Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # 定义目标函数
        def objective(trial):
            params = {}
            for param_name, bounds in param_bounds.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 3:
                    if bounds[2] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, int(bounds[0]), int(bounds[1])
                        )
                    elif bounds[2] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, bounds[0], bounds[1]
                        )
                elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    # 默认整数
                    params[param_name] = trial.suggest_int(
                        param_name, int(bounds[0]), int(bounds[1])
                    )
                elif isinstance(bounds, list):
                    # 离散选择
                    params[param_name] = trial.suggest_categorical(param_name, bounds)
                else:
                    params[param_name] = bounds
            
            return self._evaluate_params(params)
        
        # 执行优化
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 更新最优结果
        self._best_score = study.best_value
        self._best_params = study.best_params
        
        # 构建结果
        result = self._build_result('bayesian', duration)
        result.total_trials = len(study.trials)
        
        logger.info(f"Bayesian optimization completed. Best score: {result.best_score:.4f}, "
                   f"Best params: {result.best_params}")
        
        return result
    
    def walk_forward_optimization(
        self,
        train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
        param_bounds: Optional[Dict[str, Tuple]] = None,
        param_grid: Optional[Dict[str, List]] = None,
        n_trials: int = 50,
        method: str = 'random'
    ) -> List[OptimizationResult]:
        """
        滚动前向优化 (Walk-Forward Optimization)
        
        将数据分割为多个训练/测试窗口，在每个窗口上优化参数并在下一个窗口测试，
        以避免过拟合。
        
        Args:
            train_size: 训练集大小（数据点数量）
            test_size: 测试集大小（数据点数量）
            step_size: 滚动步长，默认为test_size
            param_bounds: 参数边界（用于random/bayesian方法）
            param_grid: 参数网格（用于grid方法）
            n_trials: 每个窗口的优化次数
            method: 优化方法，'grid'/'random'/'bayesian'
            
        Returns:
            List[OptimizationResult]: 每个窗口的优化结果列表
            
        Example:
            >>> results = optimizer.walk_forward_optimization(
            ...     train_size=252,
            ...     test_size=63,
            ...     param_bounds={'fast_period': (5, 20, 'int'), 'slow_period': (20, 60, 'int')},
            ...     method='random'
            ... )
        """
        if step_size is None:
            step_size = test_size
        
        logger.info(f"Starting walk-forward optimization: "
                   f"train_size={train_size}, test_size={test_size}, step_size={step_size}")
        
        results = []
        data_length = len(self.data)
        
        # 计算窗口数量
        n_windows = (data_length - train_size - test_size) // step_size + 1
        
        if n_windows <= 0:
            raise StrategyError(
                f"Invalid window parameters: data_length={data_length}, "
                f"train_size={train_size}, test_size={test_size}"
            )
        
        logger.info(f"Number of walk-forward windows: {n_windows}")
        
        for i in range(n_windows):
            start_idx = i * step_size
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > data_length:
                break
            
            logger.info(f"Walk-forward window {i + 1}/{n_windows}: "
                       f"train=[{start_idx}:{train_end}], test=[{train_end}:{test_end}]")
            
            # 分割数据
            train_data = self.data.iloc[start_idx:train_end]
            test_data = self.data.iloc[train_end:test_end]
            
            # 在训练集上优化
            train_optimizer = StrategyOptimizer(
                self.strategy_class, train_data, self.metric
            )
            
            if method == 'grid' and param_grid is not None:
                window_result = train_optimizer.optimize_grid(param_grid)
            elif method == 'bayesian' and param_bounds is not None:
                window_result = train_optimizer.optimize_bayesian(param_bounds, n_trials)
            else:
                # 默认使用随机搜索
                if param_bounds is None:
                    raise StrategyError("param_bounds is required for random/bayesian method")
                window_result = train_optimizer.optimize_random(param_bounds, n_trials)
            
            # 在测试集上验证
            test_optimizer = StrategyOptimizer(
                self.strategy_class, test_data, self.metric
            )
            test_score = test_optimizer._evaluate_params(window_result.best_params)
            
            # 记录测试集表现
            window_result.optimization_history.append({
                'window': i + 1,
                'train_score': window_result.best_score,
                'test_score': test_score,
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
            })
            
            results.append(window_result)
            
            logger.info(f"Window {i + 1} completed: train_score={window_result.best_score:.4f}, "
                       f"test_score={test_score:.4f}")
        
        logger.info(f"Walk-forward optimization completed. Total windows: {len(results)}")
        
        return results
    
    def _reset_state(self):
        """重置优化状态"""
        self._results = []
        self._best_score = -np.inf
        self._best_params = None
    
    def _build_result(self, method: str, duration: float) -> OptimizationResult:
        """
        构建优化结果
        
        Args:
            method: 优化方法名称
            duration: 优化耗时
            
        Returns:
            OptimizationResult
        """
        # 构建all_results DataFrame
        if self._results:
            all_results_df = pd.DataFrame([
                {**{'score': r['score']}, **r['params']}
                for r in self._results
            ])
        else:
            all_results_df = pd.DataFrame()
        
        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score,
            all_results=all_results_df,
            optimization_history=self._results.copy(),
            metric=self.metric,
            optimization_method=method,
            total_trials=len(self._results),
            duration_seconds=duration
        )
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """获取当前最优参数"""
        return self._best_params
    
    def get_best_score(self) -> float:
        """获取当前最优分数"""
        return self._best_score
