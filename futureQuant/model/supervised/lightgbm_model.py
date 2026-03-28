"""
LightGBM 监督学习模型

用于期货收益率分类（涨跌）或回归（收益率预测）。
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ...core.base import Model
from ...core.logger import get_logger

logger = get_logger('model.lightgbm')

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    logger.warning("lightgbm not installed, LightGBMModel will be disabled")


@dataclass
class LightGBMConfig:
    """LightGBM 配置"""
    n_estimators: int = 100
    max_depth: int = -1              # -1 means no limit
    learning_rate: float = 0.1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    objective: str = 'binary'        # 'binary' / 'regression'
    metric: str = 'auc'              # 'auc' / 'rmse'
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


class LightGBMModel(Model if _LGB_AVAILABLE else object):
    """
    LightGBM 模型封装

    支持分类（涨跌预测）和回归（收益率预测）两种模式。
    支持自动识别分类特征。

    使用示例：
        model = LightGBMModel(
            objective='binary',
            n_estimators=200,
            num_leaves=31,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), categorical_feature=['category_col'])
        proba = model.predict_proba(X_test)   # 分类
        pred = model.predict(X_test)          # 回归
    """

    def __init__(self, name: Optional[str] = None, **params):
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is required: pip install lightgbm")
        super().__init__(name=name or 'LightGBMModel', **params)

        self.config = LightGBMConfig(**params)
        self._lgb_model = None
        self._feature_names: Optional[List[str]] = None
        self._categorical_feature: Optional[List[str]] = None
        self._eval_history: Dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: Optional[int] = 50,
        verbose: bool = True,
        categorical_feature: Optional[List[str]] = None,
        **kwargs
    ) -> 'LightGBMModel':
        """
        训练模型

        Args:
            X: 特征 DataFrame
            y: 目标变量（分类用 0/1，回归用收益率）
            eval_set: (X_val, y_val) 验证集，用于早停
            early_stopping_rounds: 早停轮数
            verbose: 是否打印训练日志
            categorical_feature: 分类特征列名列表

        Returns:
            self
        """
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        # 自动识别分类特征
        if categorical_feature is None:
            categorical_feature = self._auto_detect_categorical(X)
        self._categorical_feature = categorical_feature

        # 转换分类特征为 category 类型
        X = X.copy()
        if categorical_feature:
            for col in categorical_feature:
                if col in X.columns:
                    X[col] = X[col].astype('category')

        params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'min_child_samples': self.config.min_child_samples,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'objective': self.config.objective,
            'metric': self.config.metric,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'verbose': 1 if verbose else -1,
        }

        self._lgb_model = lgb.LGBMClassifier(**params) if 'binary' in params['objective'] else lgb.LGBMRegressor(**params)
        self._feature_names = list(X.columns)

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._prepare_X(X_val)
            y_val = self._prepare_y(y_val)

            # 验证集也转换分类特征
            X_val = X_val.copy()
            if categorical_feature:
                for col in categorical_feature:
                    if col in X_val.columns:
                        X_val[col] = X_val[col].astype('category')

            self._lgb_model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=verbose)],
            )
            self._eval_history = self._lgb_model.evals_result_
        else:
            self._lgb_model.fit(X, y)
            self._eval_history = {}

        self._is_trained = True
        logger.info(f"LightGBMModel trained: {len(X)} samples, objective={params['objective']}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        X = self._prepare_X(X)
        # 转换分类特征
        if self._categorical_feature:
            X = X.copy()
            for col in self._categorical_feature:
                if col in X.columns:
                    X[col] = X[col].astype('category')
        return self._lgb_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（仅分类模式）"""
        X = self._prepare_X(X)
        # 转换分类特征
        if self._categorical_feature:
            X = X.copy()
            for col in self._categorical_feature:
                if col in X.columns:
                    X[col] = X[col].astype('category')
        return self._lgb_model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        if self._lgb_model is None:
            return pd.Series()
        imp = self._lgb_model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    def get_eval_history(self) -> Dict:
        """获取验证曲线历史"""
        return self._eval_history

    def _auto_detect_categorical(self, X: pd.DataFrame) -> List[str]:
        """自动检测分类特征"""
        categorical = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical.append(col)
            elif X[col].dtype in ['int8', 'int16', 'int32', 'int64']:
                # 对于整数类型，如果唯一值较少，可能是分类特征
                n_unique = X[col].nunique()
                if n_unique <= 10 and n_unique < len(X) * 0.05:
                    categorical.append(col)
        return categorical

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
        return f"LightGBMModel(objective={self.config.objective}, n_estimators={self.config.n_estimators})"
