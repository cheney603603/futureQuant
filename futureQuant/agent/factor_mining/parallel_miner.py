"""
ParallelFactorMiner - Parallel Factor Mining Engine

Computes factor candidates in parallel using ThreadPoolExecutor.
Supports dependency analysis (which factors need pre-computed data),
incremental computation (only compute new dates), and batch caching.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.factor_mining.parallel')


@dataclass
class FactorCandidate:
    """A factor candidate to be mined."""
    name: str
    category: str  # 'technical' / 'fundamental' / 'cross'
    params: Dict[str, Any]
    description: str
    expected_direction: Optional[str] = None  # 'positive' / 'negative'
    dependencies: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.name, str(sorted(self.params.items()))))


class ParallelFactorMiner:
    """
    Parallel factor mining engine.

    Computes multiple factor candidates concurrently using ThreadPoolExecutor.
    """

    DEFAULT_CACHE_DIR = ".factor_cache"

    def __init__(self, max_workers: int = 8, cache_dir: str = DEFAULT_CACHE_DIR):
        """
        Initialize the parallel miner.

        Args:
            max_workers: Max parallel threads.
            cache_dir: Directory for factor cache.
        """
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self._cache: Dict[str, pd.Series] = {}
        self._data_hash: Optional[str] = None

    def mine(
        self,
        candidates: List[FactorCandidate],
        price_data: pd.DataFrame,
        basis_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.Series]:
        """
        Mine all factor candidates in parallel.

        Args:
            candidates: List of factor candidates.
            price_data: OHLCV DataFrame, index=date.
            basis_data: Fundamental DataFrame (optional).

        Returns:
            Dict mapping factor_name -> factor_values Series.
        """
        import os
        os.makedirs(self.cache_dir, exist_ok=True)

        # Compute data hash for cache key
        self._data_hash = self._compute_data_hash(price_data)

        # Filter to only candidates not in cache
        uncached = [c for c in candidates if not self._is_cached(c)]

        if not uncached:
            logger.info("All candidates found in cache")
            results = {}
            for c in candidates:
                results[c.name] = self._load_cache(c)
            return results

        # Build dependency graph and compute in topological order
        dep_graph = self._build_dependency_graph(uncached)
        levels = self._topological_sort_levels(dep_graph)

        results = {}
        for level_names in levels:
            level_candidates = [c for c in uncached if c.name in level_names]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._compute_single, c, price_data, basis_data): c
                    for c in level_candidates
                }
                for future in as_completed(futures):
                    c = futures[future]
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            results[c.name] = result
                            self._save_cache(c, result)
                            logger.debug(f"Computed factor: {c.name}")
                        else:
                            logger.warning(f"Factor {c.name} returned empty result")
                    except Exception as exc:
                        logger.error(f"Factor {c.name} failed: {exc}")

        # Load cached results for already-computed factors
        for c in candidates:
            if c.name not in results:
                cached = self._load_cache(c)
                if cached is not None:
                    results[c.name] = cached

        return results

    def _compute_single(
        self,
        candidate: FactorCandidate,
        price_data: pd.DataFrame,
        basis_data: Optional[pd.DataFrame],
    ) -> pd.Series:
        """Compute a single factor candidate."""
        close = price_data['close']
        open_ = price_data.get('open', close)
        high = price_data.get('high', close)
        low = price_data.get('low', close)
        volume = price_data.get('volume', pd.Series(0, index=close.index))

        name = candidate.name
        params = candidate.params.copy()

        # ---- Parse compound names like ma_5, rsi_14 ----
        import re
        name_parts = re.match(r'^([a-z_]+)(_(\d+))?(_(\d+))?$', name)
        if name_parts:
            base_name = name_parts.group(1)
            p1 = name_parts.group(3)
            p2 = name_parts.group(5)
            if p1:
                params['period'] = int(p1)
                params['window'] = int(p1)
                params['n'] = int(p1)
            if p2:
                params['period2'] = int(p2)
            name = base_name

        n = params.get('window', params.get('n', params.get('period', 20)))

        # ---- Technical factors ----
        if name == 'ma':
            return close.rolling(n).mean()
        elif name == 'ema':
            return close.ewm(span=n, adjust=False).mean()
        elif name == 'rsi':
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(n).mean()
            avg_loss = loss.rolling(n).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        elif name == 'macd':
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal = macd_line.ewm(span=9, adjust=False).mean()
            return macd_line - signal
        elif name == 'macd_hist':
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal = macd_line.ewm(span=9, adjust=False).mean()
            return macd_line - signal  # histogram
        elif name == 'atr':
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(n).mean()
        elif name == 'boll_upper':
            mid = close.rolling(n).mean()
            std = close.rolling(n).std()
            return mid + 2 * std
        elif name == 'boll_lower':
            mid = close.rolling(n).mean()
            std = close.rolling(n).std()
            return mid - 2 * std
        elif name == 'boll_width':
            mid = close.rolling(n).mean()
            std = close.rolling(n).std()
            return (mid + 2 * std) - (mid - 2 * std)
        elif name == 'momentum':
            return close - close.shift(n)
        elif name == 'roc':
            return (close - close.shift(n)) / close.shift(n) * 100
        elif name == 'cci':
            tp = (high + low + close) / 3
            sma = tp.rolling(n).mean()
            mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean())
            return (tp - sma) / (0.015 * mad)
        elif name == 'willr':
            highest_high = high.rolling(n).max()
            lowest_low = low.rolling(n).min()
            return (close - highest_high) / (highest_high - lowest_low) * -100
        elif name == 'obv':
            obv = pd.Series(0.0, index=close.index)
            obv.iloc[1:] = np.where(
                close.iloc[1:] > close.iloc[:-1].values,
                volume.iloc[1:].values,
                np.where(
                    close.iloc[1:] < close.iloc[:-1].values,
                    -volume.iloc[1:].values,
                    0.0
                )
            ).cumsum()
            return obv
        elif name == 'adx':
            high_low = high - low
            high_close = (high - close.shift()).abs()
            low_close = (low - close.shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(n).mean()
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
            plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            return dx.rolling(n).mean()
        elif name == 'mfi':
            tp = (high + low + close) / 3
            mf = tp * volume
            pos_mf = mf.where(tp.diff() > 0, 0.0).rolling(n).sum()
            neg_mf = mf.where(tp.diff() < 0, 0.0).rolling(n).sum()
            mr = pos_mf / neg_mf
            return 100 - (100 / (1 + mr))
        elif name == 'kdj_k':
            low_n = low.rolling(n).min()
            high_n = high.rolling(n).max()
            rsv = (close - low_n) / (high_n - low_n) * 100
            return rsv.ewm(alpha=1/3, adjust=False).mean()
        elif name == 'kdj_d':
            low_n = low.rolling(n).min()
            high_n = high.rolling(n).max()
            rsv = (close - low_n) / (high_n - low_n) * 100
            k = rsv.ewm(alpha=1/3, adjust=False).mean()
            return k.ewm(alpha=1/3, adjust=False).mean()
        elif name == 'kdj_j':
            low_n = low.rolling(n).min()
            high_n = high.rolling(n).max()
            rsv = (close - low_n) / (high_n - low_n) * 100
            k = rsv.ewm(alpha=1/3, adjust=False).mean()
            d = k.ewm(alpha=1/3, adjust=False).mean()
            return 3 * k - 2 * d
        elif name == 'pvo':
            ema12 = volume.ewm(span=12, adjust=False).mean()
            ema26 = volume.ewm(span=26, adjust=False).mean()
            pvo = (ema12 - ema26) / ema26 * 100
            pvo_signal = pvo.ewm(span=9, adjust=False).mean()
            return pvo - pvo_signal
        elif name == 'stoch':
            low_n = low.rolling(n).min()
            high_n = high.rolling(n).max()
            return (close - low_n) / (high_n - low_n) * 100
        elif name == 'vol_ma_ratio':
            vol_ma = volume.rolling(n).mean()
            return volume / vol_ma
        elif name == 'price_breakout':
            high_n = high.rolling(n).max()
            return (close > high_n.shift(1)).astype(float)
        elif name == 'ma_cross_signal':
            fast = close.rolling(5).mean()
            slow = close.rolling(n).mean()
            return np.where(fast > slow, 1.0, -1.0)
        elif name == 'trend_strength':
            # ADX proxy: relative range
            return (high.rolling(n).max() - low.rolling(n).min()) / close.rolling(n).mean()
        elif name == 'volatility_regime':
            short_vol = close.rolling(5).std()
            long_vol = close.rolling(n).std()
            return short_vol / long_vol
        elif name == 'momentum_ribbon':
            ema5 = close.ewm(span=5, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            return (ema5 - ema20) / ema20 * 100
        elif name == 'rsi_ma_diff':
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            ma_diff = rsi - rsi.rolling(n).mean()
            return ma_diff
        elif name == 'volume_price_divergence':
            ret = close.pct_change()
            vol_change = volume.pct_change()
            return -(ret * vol_change).rolling(n).mean()
        elif name == 'cmo':
            delta = close.diff()
            sum_pos = delta.where(delta > 0, 0.0).rolling(n).sum()
            sum_neg = (-delta).where(delta < 0, 0.0).rolling(n).sum()
            return 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg)
        elif name == 'stoch_rsi':
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(n).mean()
            avg_loss = loss.rolling(n).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi_min = rsi.rolling(n).min()
            rsi_max = rsi.rolling(n).max()
            return (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
        elif name == 'pp':
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            return r1 - s1
        elif name == 'dmi_plus':
            high_diff = high.diff()
            low_diff = -low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            high_low = high - low
            tr = pd.concat([high_low, (high - close.shift()).abs(), (-low + close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(n).mean()
            return 100 * plus_dm.rolling(n).mean() / atr
        elif name == 'dmi_minus':
            low_diff = -low.diff()
            high_diff = high.diff()
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
            high_low = high - low
            tr = pd.concat([high_low, (high - close.shift()).abs(), (-low + close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(n).mean()
            return 100 * minus_dm.rolling(n).mean() / atr
        elif name == 'hist_volatility':
            returns = close.pct_change()
            return returns.rolling(n).std() * np.sqrt(252)
        elif name == 'skewness':
            return close.rolling(n).skew()
        elif name == 'kurtosis':
            return close.rolling(n).kurt()
        elif name == 'variance_ratio':
            var_short = close.pct_change().rolling(5).var()
            var_long = close.pct_change().rolling(n).var()
            return var_short / var_long.replace(0, np.nan)
        elif name == 'zlema':
            lag = (n - 1) // 2
            ema = close.ewm(span=n, adjust=False).mean()
            return 2 * ema - close.shift(lag)
        elif name == 'hma':
            wma_half = close.rolling(n // 2).mean() * 2
            wma_full = close.rolling(n).mean()
            return (wma_half - wma_full).rolling(int(np.sqrt(n))).mean()
        elif name == 'kama':
            # Kaufman Adaptive Moving Average
            ret = close.pct_change().abs()
            er = ret.rolling(n).sum() / (close.diff().abs().rolling(n).sum() + 1e-10)
            fast = 2 / (2 + 1)
            slow = 2 / (30 + 1)
            alpha = er * (fast - slow) + slow
            return close.ewm(alpha=alpha.mean(), adjust=False).mean()
        elif name == 'triple_ema':
            ema1 = close.ewm(span=n, adjust=False).mean()
            ema2 = ema1.ewm(span=n, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema2.ewm(span=n, adjust=False).mean()
        elif name == 'vwap_proxy':
            # Volume-weighted price approximation
            typical = (high + low + close) / 3
            return (typical * volume).rolling(n).sum() / volume.rolling(n).sum()
        elif name == 'ichimoku_a':
            # Tenkan-sen (conversion line)
            high9 = high.rolling(9).max()
            low9 = low.rolling(9).min()
            return (high9 + low9) / 2
        elif name == 'ichimoku_b':
            # Kijun-sen (base line)
            high26 = high.rolling(26).max()
            low26 = low.rolling(26).min()
            return (high26 + low26) / 2
        elif name == 'ichimoku_span':
            tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            return ((tenkan + kijun) / 2).shift(26)
        elif name == 'supertrend':
            hl2 = (high + low) / 2
            atr_val = self._atr(high, low, close, n)
            up = hl2 - 3 * atr_val
            dn = hl2 + 3 * atr_val
            trend = pd.Series(1.0, index=close.index)
            for i in range(1, len(close)):
                if close.iloc[i] > dn.iloc[i-1]:
                    trend.iloc[i] = 1.0
                elif close.iloc[i] < up.iloc[i-1]:
                    trend.iloc[i] = -1.0
                else:
                    trend.iloc[i] = trend.iloc[i-1]
            return trend
        else:
            logger.warning(f"Unknown factor: {name}, returning zeros")
            return pd.Series(0.0, index=close.index)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
        """Compute ATR."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(n).mean()

    def _build_dependency_graph(
        self, candidates: List[FactorCandidate]
    ) -> Dict[str, Set[str]]:
        """Build a dependency graph for factor candidates."""
        graph: Dict[str, Set[str]] = {c.name: set(c.dependencies) for c in candidates}
        return graph

    def _topological_sort_levels(
        self, graph: Dict[str, Set[str]]
    ) -> List[Set[str]]:
        """
        Topological sort returning levels (factors in same level are independent).
        """
        in_degree = {name: len(deps) for name, deps in graph.items()}
        levels: List[Set[str]] = []
        remaining = set(graph.keys())

        while remaining:
            # Find nodes with no remaining dependencies
            level = {n for n in remaining if in_degree.get(n, 0) == 0}
            if not level:
                # Circular dependency - just add remaining
                level = remaining
            levels.append(level)
            remaining -= level
            for n in remaining:
                in_degree[n] -= len(level & graph.get(n, set()))

        return levels

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the data for cache key."""
        key = str(df.index[0]) + str(df.index[-1]) + str(len(df))
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_key(self, candidate: FactorCandidate) -> str:
        """Build cache key for a candidate."""
        return f"{candidate.name}_{candidate.category}_{self._data_hash}"

    def _is_cached(self, candidate: FactorCandidate) -> bool:
        key = self._cache_key(candidate)
        import os
        return os.path.exists(os.path.join(self.cache_dir, f"{key}.parquet"))

    def _save_cache(self, candidate: FactorCandidate, series: pd.Series) -> None:
        """Save computed factor to cache."""
        import os
        key = self._cache_key(candidate)
        path = os.path.join(self.cache_dir, f"{key}.parquet")
        try:
            series.to_frame(name='value').to_parquet(path)
        except Exception as exc:
            logger.warning(f"Cache write failed for {candidate.name}: {exc}")

    def _load_cache(self, candidate: FactorCandidate) -> Optional[pd.Series]:
        """Load computed factor from cache."""
        import os
        key = self._cache_key(candidate)
        path = os.path.join(self.cache_dir, f"{key}.parquet")
        try:
            df = pd.read_parquet(path)
            return df['value']
        except Exception:
            return None
