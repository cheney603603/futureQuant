"""
XGBoost 监督学习模型

用于期货收益率分类（涨跌）或回归（收益率预测）。
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ...core.base import Model
from ...core.logger import get_logger

logger = get_logger('model.xgboost')

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed, XGBoostModel will be disabled")


@dataclass
class XGBoostConfig:
    """XGBoost 配置"""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    objective: str = 'binary:logistic'  # 'binary:logistic' / 'reg:squarederror'
    eval_metric: str = 'auc'           # 'auc' / 'rmse'
    random_state: int = 42
    n_jobs: int = -1


class XGBoostModel(Model if _XGB_AVAILABLE else object):
    """
    XGBoost 模型封装

    支持分类（涨跌预测）和回归（收益率预测）两种模式。

    使用示例：
        model = XGBoostModel(
            objective='binary:logistic',
            n_estimators=200,
            max_depth=5,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        proba = model.predict_proba(X_test)   # 分类
        pred = model.predict(X_test)          # 回归
    """

    def __init__(self, name: Optional[str] = None, **params):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required: pip install xgboost")
        super().__init__(name=name or 'XGBoostModel', **params)

        self.config = XGBoostConfig(**params)
        self._xgb_model = None
        self._feature_names: Optional[List[str]] = None
        self._eval_history: List[Dict] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = 50,
        verbose: bool = True,
        **kwargs
    ) -> 'XGBoostModel':
        """
        训练模型

        Args:
            X: 特征 DataFrame
            y: 目标变量（分类用 0/1，回归用收益率）
            eval_set: (X_val, y_val) 验证集，用于早停
            early_stopping_rounds: 早停轮数
            verbose: 是否打印训练日志

        Returns:
            self
        """
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'min_child_weight': self.config.min_child_weight,
            'gamma': self.config.gamma,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'objective': self.config.objective,
            'eval_metric': self.config.eval_metric,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'verbosity': 0 if not verbose else 1,
        }

        self._xgb_model = xgb.XGBClassifier(**params) if 'binary' in params['objective'] else xgb.XGBRegressor(**params)
        self._feature_names = list(X.columns)

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._prepare_X(X_val)
            y_val = self._prepare_y(y_val)
            self._xgb_model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
            self._eval_history = self._xgb_model.evals_result()
        else:
            self._xgb_model.fit(X, y)
            self._eval_history = []

        self._is_trained = True
        logger.info(f"XGBoostModel trained: {len(X)} samples, objective={params['objective']}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        X = self._prepare_X(X)
        return self._xgb_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（仅分类模式）"""
        X = self._prepare_X(X)
        return self._xgb_model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        if self._xgb_model is None:
            return pd.Series()
        imp = self._xgb_model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    def get_eval_history(self) -> Dict:
        """获取验证曲线历史"""
        return self._eval_history

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # 保持列顺序一致
        if self._feature_names:
            for col in self._feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self._feature_names]
        else:
            self._feature_names = list(X.columns)
        return X

    def _prepare_y(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.values
        return np.array(y)

    def __repr__(self) -> str:
        return f"XGBoostModel(objective={self.config.objective}, n_estimators={self.config.n_estimators})"
