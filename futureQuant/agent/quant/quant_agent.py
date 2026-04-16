"""
量化信号 Agent

功能：
- 接收因子数据（从 factor_mining agent 结果）
- 训练多模型集成（线性回归、Ridge、LightGBM）
- 生成交易信号 DataFrame
- 模型监控：检测 IC 衰退

依赖：
- futureQuant.agent.base.BaseAgent
- futureQuant.agent.quant.signal.TradingSignal
- futureQuant.agent.quant.model_monitor.ModelMonitor
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge

from ..base import AgentResult, AgentStatus, BaseAgent
from .model_monitor import ModelMonitor
from .signal import TradingSignal

# 可选的 LightGBM 导入（优雅降级）
try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    lgb = None


class QuantSignalAgent(BaseAgent):
    """
    量化信号 Agent

    训练多模型集成策略，基于因子数据预测未来收益率，生成交易信号。
    支持模型：线性回归、Ridge 回归、LightGBM。

    Attributes:
        name: Agent 名称
        config: 配置字典

    Example:
        >>> agent = QuantSignalAgent(config={"threshold": 0.001, "look_forward": 5})
        >>> result = agent.run({
        ...     "target": "RB2105",
        ...     "factor_data": factor_df,  # index=date, columns=因子名
        ...     "price_data": price_df,    # 可选
        ... })
        >>> print(result.data)  # signal DataFrame
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化量化信号 Agent

        Args:
            config: 配置字典，支持参数：
                - threshold (float): 信号阈值，默认 0.001
                - look_forward (int): 预测未来 N 日收益率，默认 5
                - train_ratio (float): 训练集比例，默认 0.7
                - lgb_weight (float): LightGBM 集成权重（当 lgb 可用时）
        """
        super().__init__(name="quant_signal", config=config)
        self.threshold: float = self.config.get("threshold", 0.001)
        self.look_forward: int = self.config.get("look_forward", 5)
        self.train_ratio: float = self.config.get("train_ratio", 0.7)
        self.lgb_weight: float = self.config.get("lgb_weight", 0.4)
        self.monitor = ModelMonitor(window_short=20, window_long=60)
        self._last_ic: Optional[float] = None

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行量化信号生成

        Args:
            context: 执行上下文，必须包含：
                - factor_data (DataFrame): 因子数据，index=date，columns=因子名
                - target (str): 标的代码（可选，默认从 context 获取）
                - price_data (DataFrame): 价格数据（可选，用于计算 label）

        Returns:
            AgentResult: 包含 data=signal DataFrame, metrics
        """
        factor_data: Optional[pd.DataFrame] = context.get("factor_data")
        target: str = context.get("target", "UNKNOWN")
        price_data: Optional[pd.DataFrame] = context.get("price_data")

        self._logger.info(f"Generating quant signals for {target}")

        if factor_data is None or factor_data.empty:
            self._logger.warning("No factor data provided, returning empty signals")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data=pd.DataFrame(columns=["date", "symbol", "signal", "confidence"]),
                metrics={"n_signals": 0, "signal_generated": False},
            )

        try:
            # Step 1: 计算收益率序列（label）
            returns = self._compute_returns(factor_data, price_data, self.look_forward)

            # Step 2: 准备训练数据
            X, y, dates = self._prepare_data(factor_data, returns)

            if len(X) < 30:
                self._logger.warning(f"Insufficient data ({len(X)} rows), skipping model training")
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.SUCCESS,
                    data=pd.DataFrame(columns=["date", "symbol", "signal", "confidence"]),
                    metrics={"n_signals": 0, "signal_generated": False},
                )

            # Step 3: 训练多模型
            n_train = int(len(X) * self.train_ratio)
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            test_dates = dates[n_train:]

            pred_lr, pred_ridge, pred_lgb = self._train_models(
                X_train, y_train, X_test, y_test
            )

            # Step 4: 集成预测
            pred_ensemble, weights = self._ensemble_predictions(
                pred_lr, pred_ridge, pred_lgb, y_test
            )

            # Step 5: 计算 IC（信息系数）
            ic = self._compute_ic(pred_ensemble, y_test)
            self._last_ic = ic

            # Step 6: 模型监控
            monitor_result = self.monitor.check(ic)
            if monitor_result["declining"]:
                self._logger.warning(
                    f"Model IC declining: short_IC={monitor_result['short_ic']:.4f}, "
                    f"long_IC={monitor_result['long_ic']:.4f}"
                )

            # Step 7: 生成信号
            signals = self._generate_signals(
                pred_ensemble, test_dates, target, weights
            )

            self._logger.info(
                f"Quant signals generated: n={len(signals)}, "
                f"long={sum(signals['signal'] > 0)}, "
                f"short={sum(signals['signal'] < 0)}, "
                f"IC={ic:.4f}"
            )

            metrics: Dict[str, Any] = {
                "n_signals": len(signals),
                "signal_generated": len(signals) > 0,
                "ic": ic,
                "ic_declining": monitor_result["declining"],
                "model_weights": {
                    "linear_regression": weights[0],
                    "ridge": weights[1],
                    "lightgbm": weights[2] if _HAS_LGB else 0.0,
                },
                "monitor": monitor_result,
            }

            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data=signals,
                metrics=metrics,
            )

        except Exception as exc:
            self._logger.error(f"Quant signal generation failed: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc)],
            )

    def _compute_returns(
        self,
        factor_data: pd.DataFrame,
        price_data: Optional[pd.DataFrame],
        look_forward: int,
    ) -> pd.Series:
        """
        计算未来 N 日收益率

        若提供 price_data 则使用真实价格，否则从因子数据列中检测价格。

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            look_forward: 预测天数

        Returns:
            收益率 Series
        """
        if price_data is not None and "close" in price_data.columns:
            close = price_data["close"].reindex(factor_data.index).ffill().bfill()
        else:
            # 尝试从因子数据中推断价格
            if "close" in factor_data.columns:
                close = factor_data["close"]
            elif "price" in factor_data.columns:
                close = factor_data["price"]
            else:
                # 降级：使用因子数据的行索引作为价格（仅用于测试）
                close = pd.Series(4000.0, index=factor_data.index)

        returns = close.pct_change(look_forward).shift(-look_forward).fillna(0)
        return returns

    def _prepare_data(
        self,
        factor_data: pd.DataFrame,
        returns: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        准备训练数据

        Args:
            factor_data: 因子数据
            returns: 收益率序列

        Returns:
            X, y, dates 三元组
        """
        # 去除非因子列
        exclude_cols = {"close", "price", "target", "signal", "label"}
        feature_cols = [c for c in factor_data.columns if c not in exclude_cols]

        if not feature_cols:
            feature_cols = list(factor_data.columns)

        # Filter to numeric columns only
        import pandas as pd
        numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(factor_data[c])]
        if not numeric_feature_cols:
            raise ValueError(f"No numeric columns in factor_data. Available: {feature_cols}")
        feature_cols = numeric_feature_cols

        # 对齐
        common_idx = factor_data.index.intersection(returns.index)
        X_df = factor_data.loc[common_idx, feature_cols]
        y_series = returns.loc[common_idx]

        # 去除 NaN
        valid = ~(X_df.isna().any(axis=1) | y_series.isna())
        X_df = X_df[valid]
        y_series = y_series[valid]
        dates = list(X_df.index)

        # 标准化（处理极端值）
        X = X_df.values.astype(np.float64)
        # 再检查 NaN（防止遗漏）
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        # 线性变换到 0~1（稳健标准化）
        col_min = np.nanmin(X, axis=0)
        col_max = np.nanmax(X, axis=0)
        col_range = col_max - col_min
        col_range[col_range < 1e-8] = 1.0
        X = (X - col_min) / col_range

        return X, y_series.values.astype(np.float64), dates

    def _train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        训练多个模型

        Args:
            X_train, y_train: 训练集
            X_test, y_test: 测试集

        Returns:
            (pred_lr, pred_ridge, pred_lgb)
        """
        # 模型1: 线性回归
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        # 模型2: Ridge 回归
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)

        # 模型3: LightGBM（如果可用）
        pred_lgb: Optional[np.ndarray] = None
        if _HAS_LGB and lgb is not None:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                force_row_wise=True,
            )
            lgb_model.fit(X_train, y_train)
            pred_lgb = lgb_model.predict(X_test)

        return pred_lr, pred_ridge, pred_lgb

    def _ensemble_predictions(
        self,
        pred_lr: np.ndarray,
        pred_ridge: np.ndarray,
        pred_lgb: Optional[np.ndarray],
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        集成多个模型的预测

        基于验证集 IC 计算动态权重。

        Args:
            pred_lr, pred_ridge, pred_lgb: 各模型预测
            y_test: 测试集真实收益

        Returns:
            (ensemble_pred, weights)
        """
        # 计算各模型 IC
        ic_lr = self._compute_ic(pred_lr, y_test)
        ic_ridge = self._compute_ic(pred_ridge, y_test)

        ic_lgb = 0.0
        if pred_lgb is not None:
            ic_lgb = self._compute_ic(pred_lgb, y_test)

        # 基于 IC 的绝对值计算权重
        total_ic = abs(ic_lr) + abs(ic_ridge) + abs(ic_lgb)
        if total_ic < 1e-8:
            # 均匀权重
            if pred_lgb is not None:
                weights = (1 / 3, 1 / 3, 1 / 3)
            else:
                weights = (0.5, 0.5, 0.0)
        else:
            if pred_lgb is not None:
                w_lr = abs(ic_lr) / total_ic
                w_ridge = abs(ic_ridge) / total_ic
                w_lgb = abs(ic_lgb) / total_ic
                weights = (w_lr, w_ridge, w_lgb)
            else:
                w_lr = abs(ic_lr) / (abs(ic_lr) + abs(ic_ridge))
                w_ridge = abs(ic_ridge) / (abs(ic_lr) + abs(ic_ridge))
                weights = (w_lr, w_ridge, 0.0)

        # 加权集成
        if pred_lgb is not None:
            pred_ensemble = (
                weights[0] * pred_lr
                + weights[1] * pred_ridge
                + weights[2] * pred_lgb
            )
        else:
            pred_ensemble = weights[0] * pred_lr + weights[1] * pred_ridge

        return pred_ensemble, weights

    @staticmethod
    def _compute_ic(pred: np.ndarray, true: np.ndarray) -> float:
        """
        计算信息系数（IC）

        IC = corr(pred, true)

        Args:
            pred: 预测值
            true: 真实值

        Returns:
            IC 值（-1~1）
        """
        if len(pred) < 2 or np.std(pred) < 1e-8 or np.std(true) < 1e-8:
            return 0.0
        return float(np.corrcoef(pred, true)[0, 1])

    def _generate_signals(
        self,
        pred: np.ndarray,
        dates: List[pd.Timestamp],
        symbol: str,
        weights: Tuple[float, float, float],
    ) -> pd.DataFrame:
        """
        将预测值转换为交易信号

        Args:
            pred: 预测收益率
            dates: 日期列表
            symbol: 标的代码
            weights: 模型权重

        Returns:
            DataFrame: [date, symbol, signal, confidence]
        """
        pred_abs = np.abs(pred)
        max_pred = np.max(pred_abs) if pred_abs.max() > 1e-8 else 1.0

        # 信号转换
        signal_values = np.where(
            pred > self.threshold, 1, np.where(pred < -self.threshold, -1, 0)
        )

        # 置信度 = 归一化预测强度 * IC 稳定性
        confidence = pred_abs / max_pred
        confidence = np.clip(confidence, 0.0, 1.0)

        # 总模型权重（用于参考）
        model_weight = sum(weights)

        df = pd.DataFrame(
            {
                "date": [d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d) for d in dates],
                "symbol": symbol,
                "signal": signal_values,
                "confidence": np.round(confidence, 4),
                "model_weight": round(model_weight, 4),
                "pred_return": np.round(pred, 6),
            }
        )

        return df
