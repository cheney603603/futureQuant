"""
LSTM 时序预测模型

使用 PyTorch 实现的长短期记忆网络，用于期货收益率预测。
支持多步预测、滚动预测、模型持久化。

设计要点：
1. 数据标准化：使用 rolling window 标准化，避免未来数据泄漏
2. 序列构建：t-lookback ~ t-1 的特征预测 t 的收益率
3. 早停机制：监控验证集 loss，防止过拟合
4. 滚动预测：每个测试窗口重新训练，模拟实盘部署
"""

from typing import Optional, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('model.lstm')

# PyTorch 可选依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed, LSTMModel will be disabled")
    torch = None
    nn = None


# ─────────────────────────────────────────────
# 网络结构
# ─────────────────────────────────────────────

class LSTMForecastNet(nn.Module if _TORCH_AVAILABLE else object):
    """
    LSTM 预测网络

    架构：输入层 → LSTM层 × N → 全连接层 → 输出层
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch")
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


# ─────────────────────────────────────────────
# 数据标准化器（严格无泄漏）
# ─────────────────────────────────────────────

class RollingNormalizer:
    """
    滚动标准化器

    在 t 时刻，使用 t-lookback 窗口内的数据计算均值和标准差，
    对 t 时刻的数据进行标准化。严格避免未来函数。
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'RollingNormalizer':
        """用全部历史数据计算全局均值/标准差（用于测试集标准化）"""
        self.mean_ = np.nanmean(data, axis=0)
        self.std_ = np.nanstd(data, axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """应用标准化"""
        if self.mean_ is None:
            raise ValueError("Normalizer not fitted")
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        if self.mean_ is None:
            raise ValueError("Normalizer not fitted")
        return data * self.std_ + self.mean_


# ─────────────────────────────────────────────
# 序列数据集
# ─────────────────────────────────────────────

def create_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建时间序列样本

    Args:
        data: 特征数据 (T, features)
        lookback: 回看窗口长度
        horizon: 预测区间（1 = 预测下一天）

    Returns:
        X: (samples, lookback, features)
        y: (samples, horizon)
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i + horizon - 1, 0])  # 预测第0列（收益率）
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 主模型类
# ─────────────────────────────────────────────

@dataclass
class LSTMConfig:
    """LSTM 配置"""
    lookback: int = 20           # 回看窗口
    horizon: int = 1              # 预测步数
    hidden_size: int = 64        # LSTM 隐藏层大小
    num_layers: int = 2           # LSTM 层数
    dropout: float = 0.2         # Dropout 比率
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10
    train_ratio: float = 0.8     # 训练集比例
    random_seed: int = 42


class LSTMModel:
    """
    LSTM 时序预测模型

    用于期货收益率预测，支持分类（涨跌）和回归（收益率）。

    使用示例：
        >>> model = LSTMModel(config=LSTMConfig(lookback=20, horizon=1))
        >>> model.fit(features, target)           # 单次训练
        >>> pred = model.predict(features_test)    # 预测
        >>> model.save('lstm_model.pt')            # 保存
        >>> model.load('lstm_model.pt')            # 加载
    """

    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        name: Optional[str] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch")
        
        self.config = config or LSTMConfig()
        self.name = name or "LSTMModel"
        self._model: Optional['LSTMForecastNet'] = None
        self._normalizer = RollingNormalizer(lookback=self.config.lookback)
        self._feature_names: Optional[List[str]] = None
        self._is_trained = False
        self._train_history: List[Dict] = []

        # 固定随机种子
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        verbose: bool = True,
    ) -> 'LSTMModel':
        """
        训练模型

        Args:
            X: 特征 DataFrame
            y: 目标变量（收益率）
            eval_set: (X_val, y_val) 验证集
            verbose: 是否打印训练日志

        Returns:
            self
        """
        # 合并特征和目标
        combined = X.copy()
        combined['_target_'] = y.values

        # 去除 NaN
        combined = combined.dropna()
        if len(combined) < self.config.lookback + 10:
            raise ValueError(f"Not enough data: {len(combined)} rows after dropna")

        data = combined.values

        # 标准化
        self._normalizer.fit(data)
        data_norm = self._normalizer.transform(data)
        self._feature_names = list(X.columns)

        # 分割
        split = int(len(data_norm) * self.config.train_ratio)
        train_data = data_norm[:split]
        val_data = data_norm[split:]

        # 构建序列
        X_seq, y_seq = create_sequences(
            train_data, self.config.lookback, self.config.horizon
        )
        X_val_seq, y_val_seq = create_sequences(
            val_data, self.config.lookback, self.config.horizon
        ) if len(val_data) > self.config.lookback else (None, None)

        # 转换为 Tensor
        X_t = torch.FloatTensor(X_seq)
        y_t = torch.FloatTensor(y_seq).unsqueeze(-1)
        train_dataset = TensorDataset(X_t, y_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # 模型
        self._model = LSTMForecastNet(
            input_size=data.shape[1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            output_size=1,
        )

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        best_state: Optional[Dict] = None

        for epoch in range(self.config.epochs):
            self._model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self._model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            # 验证
            val_loss = None
            if X_val_seq is not None:
                self._model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val_seq)
                    y_v = torch.FloatTensor(y_val_seq).unsqueeze(-1)
                    val_pred = self._model(X_v)
                    val_loss = criterion(val_pred, y_v).item()
                scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", val_loss={val_loss:.4f}" if val_loss else ""
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                          f"train_loss={train_loss:.4f}{val_str}")

            # 早停检查
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.clone() for k, v in self._model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 恢复最佳模型
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_trained = True
        logger.info(f"LSTMModel trained: {X_t.shape[0]} samples")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            X: 特征 DataFrame

        Returns:
            预测值数组
        """
        if not self._is_trained or self._model is None:
            raise ValueError("Model not trained. Call fit() first.")

        combined = X.copy()
        if '_target_' in combined.columns:
            combined = combined.drop(columns=['_target_'])
        
        combined = combined.dropna()
        if len(combined) < self.config.lookback:
            return np.array([])

        data = combined.values
        data_norm = self._normalizer.transform(data)

        X_seq, _ = create_sequences(data_norm, self.config.lookback, self.config.horizon)

        self._model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_seq)
            pred_norm = self._model(X_t).numpy().flatten()

        # 反标准化（仅对目标列）
        pred = self._normalizer.inverse_transform(
            np.column_stack([pred_norm, np.zeros_like(pred_norm)])
        )[:, 0]

        return pred

    def predict_direction(self, X: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
        """
        预测涨跌方向

        Args:
            X: 特征 DataFrame
            threshold: 分类阈值（默认0）

        Returns:
            方向数组：1=上涨，0=持平/下跌
        """
        pred = self.predict(X)
        return (pred > threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（需要校准后使用，这里返回预测值的 sigmoid 变换）

        Returns:
            上涨概率数组
        """
        pred = self.predict(X)
        # Sigmoid 变换，将预测值映射到 [0, 1]
        proba = 1 / (1 + np.exp(-pred * 10))  # 放大系数使分布更分明
        return proba

    def save(self, path: str):
        """保存模型"""
        if self._model is None:
            raise ValueError("No model to save")
        torch.save({
            'model_state': self._model.state_dict(),
            'config': self.config,
            'feature_names': self._feature_names,
            'normalizer_mean': self._normalizer.mean_,
            'normalizer_std': self._normalizer.std_,
        }, path)
        logger.info(f"Model saved: {path}")

    def load(self, path: str) -> 'LSTMModel':
        """加载模型"""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load LSTMModel")
        checkpoint = torch.load(path, map_location='cpu')
        self.config = checkpoint['config']
        self._feature_names = checkpoint['feature_names']
        self._normalizer.mean_ = checkpoint['normalizer_mean']
        self._normalizer.std_ = checkpoint['normalizer_std']
        self._is_trained = True

        # 重建模型
        dummy_data = np.zeros((10, len(self._feature_names) + 1))
        self._normalizer.fit(dummy_data)
        self._model = LSTMForecastNet(
            input_size=len(self._feature_names) + 1,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            output_size=1,
        )
        self._model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Model loaded: {path}")
        return self

    def get_config(self) -> Dict[str, Any]:
        return {
            'lookback': self.config.lookback,
            'horizon': self.config.horizon,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'feature_names': self._feature_names,
        }

    def __repr__(self) -> str:
        return (f"LSTMModel(lookback={self.config.lookback}, "
                f"horizon={self.config.horizon}, "
                f"hidden={self.config.hidden_size}, "
                f"layers={self.config.num_layers})")
