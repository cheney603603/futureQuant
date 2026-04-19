# -*- coding: utf-8 -*-
"""
Optuna 参数优化器 - 策略参数自动搜索

B 方向实现：
- 贝叶斯优化搜索最优参数
- 支持多目标优化（收益 + 夏普 + 回撤）
- 可视化优化过程
- 与 Strategy 基类集成

Author: futureQuant Team
Date: 2026-04-19
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.core.exceptions import OptimizationError

logger = get_logger('strategy.optimizer')


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_value: float
    all_trials: List[Dict]
    optimization_history: pd.DataFrame
    study_name: str
    n_trials: int
    duration_seconds: float


class OptunaOptimizer:
    """
    Optuna 参数优化器
    
    使用贝叶斯优化自动搜索策略最优参数组合。
    
    Example:
        >>> optimizer = OptunaOptimizer()
        >>> result = optimizer.optimize(
        ...     strategy_class=MyStrategy,
        ...     data=df,
        ...     param_space={
        ...         'lookback': {'type': 'int', 'low': 5, 'high': 50},
        ...         'threshold': {'type': 'float', 'low': 0.01, 'high': 0.1},
        ...     },
        ...     n_trials=100
        ... )
    """
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = 'maximize',
        sampler: Optional[Any] = None
    ):
        """
        初始化优化器
        
        Args:
            study_name: 研究名称，用于持久化
            storage: 存储路径，如 'sqlite:///optuna.db'
            direction: 'maximize' 或 'minimize'
            sampler: 自定义采样器，默认 TPE
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        self.study_name = study_name or f"optimization_{datetime.now():%Y%m%d_%H%M%S}"
        self.storage = storage
        self.direction = direction
        self.sampler = sampler or TPESampler(seed=42)
        self.study: Optional[optuna.Study] = None
        
    def _create_param_from_space(self, trial, name: str, space: Dict) -> Any:
        """根据参数空间定义创建参数"""
        param_type = space.get('type', 'float')
        
        if param_type == 'int':
            return trial.suggest_int(
                name,
                space['low'],
                space['high'],
                step=space.get('step', 1)
            )
        elif param_type == 'float':
            return trial.suggest_float(
                name,
                space['low'],
                space['high'],
                step=space.get('step', None),
                log=space.get('log', False)
            )
        elif param_type == 'categorical':
            return trial.suggest_categorical(name, space['choices'])
        else:
            raise ValueError(f"Unknown param type: {param_type}")
    
    def optimize(
        self,
        objective_func: Callable[[Dict], float],
        param_space: Dict[str, Dict],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True,
        n_jobs: int = 1
    ) -> OptimizationResult:
        """
        执行参数优化
        
        Args:
            objective_func: 目标函数，接收参数字典返回评分
            param_space: 参数空间定义
            n_trials: 优化次数
            timeout: 超时时间（秒）
            show_progress: 是否显示进度条
            n_jobs: 并行作业数
            
        Returns:
            OptimizationResult 优化结果
        """
        start_time = datetime.now()
        
        # 创建或加载 study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            load_if_exists=True
        )
        
        # 定义目标函数包装器
        def objective(trial):
            params = {
                name: self._create_param_from_space(trial, name, space)
                for name, space in param_space.items()
            }
            
            try:
                value = objective_func(params)
                # 处理 NaN/Inf
                if not np.isfinite(value):
                    return float('-inf') if self.direction == 'maximize' else float('inf')
                return value
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {e}")
                return float('-inf') if self.direction == 'maximize' else float('inf')
        
        # 执行优化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            n_jobs=n_jobs
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 收集结果
        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials_data.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'datetime': trial.datetime_complete
                })
        
        history_df = pd.DataFrame([
            {'trial': t.number, 'value': t.value}
            for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_value=self.study.best_value,
            all_trials=trials_data,
            optimization_history=history_df,
            study_name=self.study_name,
            n_trials=len(trials_data),
            duration_seconds=duration
        )
        
        logger.info(f"Optimization completed: best_value={result.best_value:.4f}, "
                   f"best_params={result.best_params}")
        
        return result
    
    def optimize_strategy(
        self,
        strategy_class,
        data: pd.DataFrame,
        param_space: Dict[str, Dict],
        metric: str = 'sharpe_ratio',
        n_trials: int = 100,
        **kwargs
    ) -> OptimizationResult:
        """
        优化策略参数
        
        Args:
            strategy_class: 策略类（继承自 Strategy）
            data: 回测数据
            param_space: 参数空间
            metric: 优化目标指标（sharpe_ratio/return/max_drawdown/etc）
            n_trials: 优化次数
            **kwargs: 传递给 optimize()
            
        Returns:
            OptimizationResult
        """
        from ...backtest.engine import BacktestEngine
        
        def objective(params):
            # 创建策略实例
            strategy = strategy_class(**params)
            
            # 运行回测
            engine = BacktestEngine(initial_capital=100000)
            result = engine.run(data, strategy)
            
            # 返回目标指标
            if metric == 'sharpe_ratio':
                return result.get('sharpe_ratio', 0)
            elif metric == 'return':
                return result.get('total_return', 0)
            elif metric == 'max_drawdown':
                # 回撤是负数，需要反转
                return -result.get('max_drawdown', 0)
            else:
                return result.get(metric, 0)
        
        return self.optimize(objective, param_space, n_trials, **kwargs)
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """绘制优化历史"""
        if self.study is None:
            raise OptimizationError("No study available. Run optimize() first.")
        
        fig = plot_optimization_history(self.study)
        
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Optimization history plot saved to {save_path}")
        
        return fig
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """绘制参数重要性"""
        if self.study is None:
            raise OptimizationError("No study available. Run optimize() first.")
        
        fig = plot_param_importances(self.study)
        
        if save_path:
            fig.write_image(save_path)
            logger.info(f"Parameter importance plot saved to {save_path}")
        
        return fig
    
    def save_results(self, path: str):
        """保存优化结果到文件"""
        if self.study is None:
            raise OptimizationError("No study available. Run optimize() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'study_name': self.study_name,
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'direction': self.direction
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimization results saved to {path}")


# 便捷函数
def quick_optimize(
    objective_func: Callable[[Dict], float],
    param_space: Dict[str, Dict],
    n_trials: int = 50,
    **kwargs
) -> OptimizationResult:
    """
    快速优化函数
    
    Example:
        >>> result = quick_optimize(
        ...     lambda p: evaluate(p),
        ...     {'x': {'type': 'float', 'low': -10, 'high': 10}},
        ...     n_trials=50
        ... )
        >>> print(result.best_params)
    """
    optimizer = OptunaOptimizer()
    return optimizer.optimize(objective_func, param_space, n_trials, **kwargs)