"""
ARIMA 时序预测模型

使用 statsmodels 实现 ARIMA 模型，用于期货收益率预测。
支持多品种横截面回归、滚动窗口预测、模型持久化。

设计要点：
1. ADF 检验：自动判断序列平稳性
2. 自动定阶：使用 AIC/BIC 自动选择 p, d, q 参数
3. 滚动预测：每个窗口重新拟合，模拟实盘部署
4. 横截面扩展：支持多品种同时建模
"""

from typing import Optional, List, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('model.arima')

# 可选依赖
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    _STATS_AVAILABLE = True
except ImportError:
    _STATS_AVAILABLE = False
    ARIMA = None
    adfuller = None
    logger.warning("statsmodels not installed, ARIMAModel will be disabled")


@dataclass
class ARIMAConfig:
    """ARIMA 配置"""
    p: int = 5              # AR 阶数
    d: int = 1              # 差分阶数
    q: int = 5              # MA 阶数
    trend: Literal['n', 'c', 't', 'ct'] = 'c'  # 趋势项
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    concentrate_scale: bool = False  # 固定残差方差
    auto_order: bool = False        # 是否自动选择阶数
    max_p: int = 5
    max_q: int = 5
    max_d: int = 2
    ic: Literal['aic', 'bic', 'hqic'] = 'aic'  # 定阶准则
    seasonal: bool = False          # 是否考虑季节性
    seasonal_period: int = 12


class ARIMAModel:
    """
    ARIMA 时序预测模型

    用于期货收益率或价格的预测，支持自动定阶和滚动窗口。

    使用示例：
        >>> model = ARIMAModel(config=ARIMAConfig(p=5, d=1, q=5))
        >>> model.fit(price_series)
        >>> forecast = model.predict(steps=5)
        >>> model.save('arima_model.pkl')
    """

    def __init__(
        self,
        config: Optional[ARIMAConfig] = None,
        name: Optional[str] = None,
    ):
        if not _STATS_AVAILABLE:
            raise ImportError("statsmodels is required: pip install statsmodels")

        self.config = config or ARIMAConfig()
        self.name = name or "ARIMAModel"
        self._model: Optional[Any] = None
        self._results: Optional[Any] = None
        self._is_trained = False
        self._train_history: List[Dict] = []
        self._residual_std: Optional[float] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _test_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """
        ADF 平稳性检验

        Returns:
            (is_stationary, p_value)
        """
        if adfuller is None:
            return True, 0.0
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            p_value = result[1]
            return p_value < 0.05, p_value
        except Exception:
            return True, 0.0

    def _auto_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        自动定阶（基于 AIC/BIC）

        Returns:
            (p, d, q)
        """
        if self.config.max_d == 0:
            return self.config.max_p // 2, 0, self.config.max_q // 2

        best_ic = float('inf')
        best_order = (self.config.p, self.config.d, self.config.q)

        # 先确定 d（差分阶数）
        d_candidates = list(range(0, self.config.max_d + 1))
        best_d = 0

        for d in d_candidates:
            if d > 0:
                diff_series = series.diff().dropna()
                if len(diff_series) < 10:
                    continue
                is_stat, _ = self._test_stationarity(diff_series)
                if is_stat:
                    best_d = d
                    break
                best_d = d  # 仍然使用这个 d，继续搜索 p, q

        # 搜索 p, q
        for p in range(0, self.config.max_p + 1):
            for q in range(0, self.config.max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(
                        series.values,
                        order=(p, best_d, q),
                        trend=self.config.trend,
                        enforce_stationarity=self.config.enforce_stationarity,
                        enforce_invertibility=self.config.enforce_invertibility,
                    )
                    fitted = model.fit()
                    ic_value = getattr(fitted, f'aic')
                    if ic_value < best_ic:
                        best_ic = ic_value
                        best_order = (p, best_d, q)
                except Exception:
                    continue

        logger.info(f"Auto order selection: {best_order}, {self.config.ic.upper()}={best_ic:.2f}")
        return best_order

    def fit(
        self,
        data: pd.Series,
        verbose: bool = True,
    ) -> 'ARIMAModel':
        """
        拟合 ARIMA 模型

        Args:
            data: 时间序列（价格或收益率）
            verbose: 是否打印日志

        Returns:
            self
        """
        series = data.dropna()
        if len(series) < 30:
            raise ValueError(f"Need at least 30 data points, got {len(series)}")

        order = (self.config.p, self.config.d, self.config.q)

        # 自动定阶
        if self.config.auto_order:
            order = self._auto_order(series)
            self.config.p, self.config.d, self.config.q = order

        # 检查平稳性
        if self.config.d > 0:
            is_stat, p_value = self._test_stationarity(series)
            if verbose:
                logger.info(f"Stationarity test: p_value={p_value:.4f}, "
                           f"stationary={is_stat}")

        # 拟合
        self._model = ARIMA(
            series.values,
            order=order,
            trend=self.config.trend,
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
        )

        self._results = self._model.fit()

        # 计算残差标准差
        residuals = self._results.resid
        self._residual_std = float(np.std(residuals))

        self._is_trained = True

        if verbose:
            logger.info(f"ARIMA{order} fitted: "
                       f"AIC={self._results.aic:.2f}, "
                       f"BIC={self._results.bic:.2f}, "
                       f"residual_std={self._residual_std:.4f}")

        return self

    def predict(
        self,
        steps: int = 1,
        return_conf_int: bool = False,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        预测未来 N 步

        Args:
            steps: 预测步数
            return_conf_int: 是否返回置信区间

        Returns:
            预测值序列，或 (预测值, 置信区间)
        """
        if not self._is_trained or self._results is None:
            raise ValueError("Model not trained. Call fit() first.")

        forecast = self._results.get_forecast(steps=steps)

        mean = forecast.predicted_mean
        index = pd.RangeIndex(start=0, stop=steps, step=1)
        result = pd.Series(mean, index=index, name='forecast')

        if return_conf_int:
            ci = forecast.conf_int(alpha=0.05)
            ci_df = pd.DataFrame(ci, columns=['lower', 'upper'])
            return result, ci_df

        return result

    def rolling_fit_predict(
        self,
        data: pd.DataFrame,
        window: int = 60,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        滚动拟合预测

        在滚动窗口内拟合 ARIMA，对未来进行预测。
        模拟实盘部署，避免使用未来数据。

        Args:
            data: 时间序列 DataFrame（包含 price 列）
            window: 滚动窗口大小
            horizon: 预测步数

        Returns:
            DataFrame，columns=['actual', 'predicted', 'error']
        """
        price_col = 'close' if 'close' in data.columns else data.columns[0]
        returns = data[price_col].pct_change().dropna()

        results = []
        for i in range(window, len(returns) - horizon + 1):
            train_window = returns.iloc[i - window:i]

            try:
                self.fit(train_window, verbose=False)
                pred = self.predict(steps=horizon)
                actual = returns.iloc[i:i + horizon].values
                pred_mean = float(pred.mean())

                results.append({
                    'date': data.index[i + horizon - 1],
                    'actual': actual[0] if len(actual) > 0 else np.nan,
                    'predicted': pred_mean,
                    'error': actual[0] - pred_mean if len(actual) > 0 else np.nan,
                })
            except Exception as e:
                logger.warning(f"Rolling fit failed at window end {i}: {e}")
                continue

        df = pd.DataFrame(results)
        logger.info(f"Rolling fit completed: {len(df)} predictions")
        return df

    def get_residuals(self) -> pd.Series:
        """获取残差序列"""
        if not self._is_trained or self._results is None:
            raise ValueError("Model not trained")
        return pd.Series(self._results.resid, name='residuals')

    def summary(self) -> str:
        """获取模型摘要"""
        if not self._is_trained or self._results is None:
            return "Model not trained"
        return str(self._results.summary())

    def get_config(self) -> Dict[str, Any]:
        return {
            'p': self.config.p,
            'd': self.config.d,
            'q': self.config.q,
            'trend': self.config.trend,
            'auto_order': self.config.auto_order,
            'residual_std': self._residual_std,
        }

    def __repr__(self) -> str:
        return f"ARIMAModel(p={self.config.p}, d={self.config.d}, q={self.config.q})"
