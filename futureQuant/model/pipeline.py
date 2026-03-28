"""
ML 预测 Pipeline

将特征工程、模型训练、预测整合为端到端流水线。
支持 Walk-forward 滚动验证（防止过拟合）。
"""

from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .feature_engineering import FeatureEngineer, FeatureConfig
from .supervised import XGBoostModel, LightGBMModel

from ..core.logger import get_logger

logger = get_logger('model.pipeline')


@dataclass
class PipelineConfig:
    """Pipeline 配置"""
    model_type: Literal['xgboost', 'lightgbm'] = 'xgboost'
    label_type: Literal['classification', 'regression'] = 'classification'
    forward_period: int = 1              # 预测未来 N 日
    train_window: int = 252             # 训练窗口（天数）
    test_window: int = 60                # 测试窗口（天数）
    min_train_samples: int = 100          # 最少训练样本数
    use_walk_forward: bool = True         # 是否使用 Walk-forward
    early_stopping_rounds: int = 50


class MLForecastPipeline:
    """
    机器学习预测流水线

    整合特征工程 → 训练 → 预测全流程，支持 Walk-forward 滚动验证。

    使用示例：
        >>> pipeline = MLForecastPipeline()
        >>> pipeline.fit(price_data, forward_period=5)
        >>> signals = pipeline.predict(price_data)  # 返回交易信号
        >>> report = pipeline.generate_report()
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        self.config = config or PipelineConfig()
        self.fe = FeatureEngineer(feature_config or FeatureConfig())
        self._model: Optional[Any] = None
        self._feature_names: Optional[List[str]] = None
        self._train_history: List[Dict] = []
        self._latest_features: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame) -> 'MLForecastPipeline':
        """
        训练模型（单次训练，非 Walk-forward）

        Args:
            data: 包含 OHLCV 的 DataFrame

        Returns:
            self
        """
        # 特征工程
        self.fe.add_price_features()
        self.fe.add_technical_features()
        self.fe.add_volume_features()
        
        label_type = 'classification' if self.config.label_type == 'classification' else 'regression'
        features, targets = self.fe.build(
            data,
            target_column='close',
            forward_periods=[self.config.forward_period],
            label_type=label_type
        )
        
        target_col = f'label_{self.config.forward_period}d'
        y = targets[target_col]
        
        # 分割训练/验证集（最后 20% 为验证集）
        split = int(len(features) * 0.8)
        X_train = features.iloc[:split]
        y_train = y.iloc[:split]
        X_val = features.iloc[split:]
        y_val = y.iloc[split:]
        
        # 创建模型
        if self.config.model_type == 'xgboost':
            self._model = XGBoostModel(
                objective='binary:logistic' if label_type == 'classification' else 'reg:squarederror',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
            )
        else:
            self._model = LightGBMModel(
                objective='binary' if label_type == 'classification' else 'regression',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
            )
        
        self._model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        self._feature_names = list(features.columns)
        self._latest_features = features
        
        # 训练历史
        self._train_history.append({
            'n_train': len(X_train),
            'n_val': len(X_val),
            'model': self.config.model_type,
        })
        
        logger.info(f"Pipeline trained: {len(X_train)} train, {len(X_val)} val samples")
        return self

    def fit_walk_forward(self, data: pd.DataFrame) -> 'MLForecastPipeline':
        """
        Walk-forward 滚动训练

        每个测试窗口结束后，用新的数据更新模型，模拟实盘部署。
        
        Args:
            data: 包含 OHLCV 的 DataFrame

        Returns:
            self
        """
        # 特征工程
        self.fe.add_price_features()
        self.fe.add_technical_features()
        self.fe.add_volume_features()
        
        label_type = 'classification' if self.config.label_type == 'classification' else 'regression'
        features, targets = self.fe.build(
            data,
            target_column='close',
            forward_periods=[self.config.forward_period],
            label_type=label_type
        )
        
        target_col = f'label_{self.config.forward_period}d'
        y = targets[target_col]
        
        n = len(features)
        walk_forward_results = []
        
        i = self.config.train_window
        while i + self.config.test_window <= n:
            train_end = i
            test_end = min(i + self.config.test_window, n)
            
            X_train = features.iloc[train_end - self.config.train_window:train_end]
            y_train = y.iloc[train_end - self.config.train_window:train_end]
            X_test = features.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            if len(X_train) < self.config.min_train_samples:
                i += self.config.test_window
                continue
            
            # 训练模型
            if self.config.model_type == 'xgboost':
                model = XGBoostModel(
                    objective='binary:logistic' if label_type == 'classification' else 'reg:squarederror',
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                )
            else:
                model = LightGBMModel(
                    objective='binary' if label_type == 'classification' else 'regression',
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                )
            
            model.fit(X_train, y_train, verbose=False)
            pred = model.predict(X_test)
            
            if label_type == 'classification':
                acc = (pred == y_test.values).mean()
                logger.info(f"Walk-forward window {train_end}-{test_end}: accuracy={acc:.3f}")
                walk_forward_results.append({'window': (train_end, test_end), 'accuracy': acc})
            else:
                mse = ((pred - y_test.values) ** 2).mean()
                mae = np.abs(pred - y_test.values).mean()
                logger.info(f"Walk-forward window {train_end}-{test_end}: MSE={mse:.4f}, MAE={mae:.4f}")
                walk_forward_results.append({'window': (train_end, test_end), 'mse': mse, 'mae': mae})
            
            # 更新全局模型（用最新窗口的数据）
            self._model = model
            i += self.config.test_window
        
        self._train_history = walk_forward_results
        self._feature_names = list(features.columns)
        logger.info(f"Walk-forward completed: {len(walk_forward_results)} windows")
        return self

    def predict(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        预测（生成交易信号）

        Args:
            data: 新的数据，如果为 None 则使用训练时的最新特征

        Returns:
            DataFrame，index=日期，columns=['signal', 'confidence']
            signal: 1（做多）, -1（做空）, 0（空仓）
            confidence: 信号强度（概率）
        """
        if self._model is None:
            raise ValueError("Model not trained. Call fit() or fit_walk_forward() first.")
        
        if data is not None:
            # 用新数据重新计算特征
            features, _ = self.fe.build(
                data,
                target_column='close',
                forward_periods=[self.config.forward_period],
                label_type='classification'
            )
        else:
            features = self._latest_features
        
        if features is None or features.empty:
            return pd.DataFrame(columns=['signal', 'confidence'])
        
        # 预测
        proba = self._model.predict_proba(features)[:, 1]  # P(y=1)
        pred = (proba > 0.5).astype(int)
        
        # 生成信号
        signal = pred * 2 - 1  # 0→-1, 1→1
        confidence = np.abs(proba - 0.5) * 2  # 0~1，越高越自信
        
        result = pd.DataFrame({
            'signal': signal,
            'confidence': confidence,
            'proba': proba,
        }, index=features.index)
        
        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """获取最重要的特征"""
        if self._model is None:
            return pd.Series()
        return self._model.get_feature_importance().head(top_n)

    def generate_report(self) -> Dict:
        """生成训练报告"""
        return {
            'config': {
                'model_type': self.config.model_type,
                'label_type': self.config.label_type,
                'forward_period': self.config.forward_period,
                'train_window': self.config.train_window,
                'test_window': self.config.test_window,
            },
            'train_history': self._train_history,
            'n_features': len(self._feature_names) if self._feature_names else 0,
            'feature_names': self._feature_names[:10] if self._feature_names else [],
        }
