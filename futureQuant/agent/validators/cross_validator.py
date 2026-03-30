"""
时序交叉验证器 (Time Series Cross Validator)

支持三种时序验证模式：
- Walk-Forward: 滚动窗口验证
- Expanding Window: 扩展窗口验证
- Purged K-Fold: 带清洗期的 K 折验证
"""

from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger
from ...core.base import Factor

logger = get_logger('agent.cross_validator')


def _to_dataframe(
    factor_values: Any,
    returns: Any = None,
    n_splits: Optional[int] = None,
) -> pd.DataFrame:
    """
    将 (factor_values, returns) 两个 Series 合并为 DataFrame，
    供 TimeSeriesCrossValidator.split() 使用。

    若 factor_values 本身已是 DataFrame，则直接返回。
    """
    if isinstance(factor_values, pd.DataFrame):
        return factor_values

    if returns is None:
        # 仅有因子值，无法合并
        return pd.DataFrame({'factor': factor_values})

    df = pd.DataFrame({
        'factor': factor_values,
        'return': returns,
    })
    return df


class TimeSeriesCrossValidator:
    """
    时序交叉验证器

    防止数据泄露，正确评估因子在时序数据上的泛化能力。
    """

    METHODS = ('walk_forward', 'expanding', 'purged_kfold')

    def __init__(
        self,
        method: str = 'walk_forward',
        train_size: int = 252,
        test_size: int = 63,
        n_splits: int = 5,
        purge_size: int = 5,
        min_train_size: int = 126,
    ) -> None:
        """
        Args:
            method: 验证方法，'walk_forward' / 'expanding' / 'purged_kfold'
            train_size: 训练集大小（交易日数）
            test_size: 测试集大小（交易日数）
            n_splits: K 折数量（purged_kfold 模式）
            purge_size: 清洗期大小（防止训练/测试集重叠）
            min_train_size: 最小训练集大小
        """
        if method not in self.METHODS:
            raise ValueError(f'method must be one of {self.METHODS}')
        self.method = method
        self.train_size = train_size
        self.test_size = test_size
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.min_train_size = min_train_size

    def split(
        self,
        data: pd.DataFrame,
    ) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """
        生成训练/测试集索引对

        Args:
            data: 时序数据

        Yields:
            (train_index, test_index) 元组
        """
        if self.method == 'walk_forward':
            yield from self._walk_forward_split(data)
        elif self.method == 'expanding':
            yield from self._expanding_split(data)
        else:
            yield from self._purged_kfold_split(data)

    def _walk_forward_split(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """滚动窗口分割"""
        n = len(data)
        start = 0
        while start + self.train_size + self.purge_size + self.test_size <= n:
            train_end = start + self.train_size
            test_start = train_end + self.purge_size
            test_end = test_start + self.test_size

            train_idx = data.index[start:train_end]
            test_idx = data.index[test_start:test_end]
            yield train_idx, test_idx

            start += self.test_size

    def _expanding_split(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """扩展窗口分割"""
        n = len(data)
        start = self.min_train_size
        while start + self.purge_size + self.test_size <= n:
            train_idx = data.index[:start]
            test_start = start + self.purge_size
            test_end = min(test_start + self.test_size, n)
            test_idx = data.index[test_start:test_end]
            yield train_idx, test_idx
            start += self.test_size

    def _purged_kfold_split(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """带清洗期的 K 折分割"""
        n = len(data)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n

            # 清洗期
            purge_start = max(0, test_start - self.purge_size)
            purge_end = min(n, test_end + self.purge_size)

            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:purge_end] = False

            train_idx = data.index[train_mask]
            test_idx = data.index[test_start:test_end]

            if len(train_idx) >= self.min_train_size:
                yield train_idx, test_idx

    def validate(
        self,
        factor: Any,
        data: pd.DataFrame,
        factor_col: str = 'factor',
        return_col: str = 'return',
    ) -> float:
        """
        验证因子稳定性

        Args:
            factor: 因子对象（需有 compute 方法）或因子值 Series
            data: 包含因子值和收益率的 DataFrame
            factor_col: 因子列名
            return_col: 收益率列名

        Returns:
            稳定性评分 [0, 1]
        """
        ic_list: List[float] = []

        for train_idx, test_idx in self.split(data):
            test_data = data.loc[test_idx]
            if factor_col not in test_data.columns or return_col not in test_data.columns:
                continue
            valid = test_data[[factor_col, return_col]].dropna()
            if len(valid) < 10:
                continue
            corr, _ = stats.spearmanr(valid[factor_col], valid[return_col])
            if not np.isnan(corr):
                ic_list.append(corr)

        if not ic_list:
            return 0.0

        ic_arr = np.array(ic_list)
        # 稳定性 = IC 均值 / IC 标准差（ICIR），归一化到 [0,1]
        if ic_arr.std() < 1e-8:
            return 1.0
        icir = abs(ic_arr.mean()) / ic_arr.std()
        return float(min(icir / 3.0, 1.0))

    def get_cv_stats(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """
        计算交叉验证统计指标

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            统计指标字典
        """
        data = pd.DataFrame({'factor': factor_values, 'return': returns})
        ic_list: List[float] = []

        for _, test_idx in self.split(data):
            test_data = data.loc[test_idx].dropna()
            if len(test_data) < 10:
                continue
            corr, _ = stats.spearmanr(test_data['factor'], test_data['return'])
            if not np.isnan(corr):
                ic_list.append(corr)

        if not ic_list:
            return {'ic_mean': 0.0, 'ic_std': 0.0, 'icir': 0.0, 'win_rate': 0.0, 'n_folds': 0}

        ic_arr = np.array(ic_list)
        return {
            'ic_mean': float(ic_arr.mean()),
            'ic_std': float(ic_arr.std()),
            'icir': float(ic_arr.mean() / ic_arr.std()) if ic_arr.std() > 1e-8 else 0.0,
            'win_rate': float((ic_arr > 0).mean()),
            'n_folds': len(ic_list),
        }


# ---------------------------------------------------------------------------
# 语义化别名类（供外部按验证方式直接导入使用）
# ---------------------------------------------------------------------------

class CrossValidator(TimeSeriesCrossValidator):
    """
    时序交叉验证器通用别名。

    等同于 TimeSeriesCrossValidator，提供统一入口，默认使用 walk_forward 方法。
    额外支持 config 字典形式的参数传入。
    """

    def __init__(
        self,
        method: str = 'walk_forward',
        train_size: int = 252,
        test_size: int = 63,
        n_splits: int = 5,
        purge_size: int = 5,
        min_train_size: int = 126,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        # config 字典中的值覆盖具名参数
        if config:
            method = config.get('method', method)
            train_size = config.get('train_size', train_size)
            test_size = config.get('test_size', test_size)
            n_splits = config.get('n_splits', n_splits)
            purge_size = config.get('purge_size', purge_size)
            min_train_size = config.get('min_train_size', min_train_size)

        super().__init__(
            method=method,
            train_size=train_size,
            test_size=test_size,
            n_splits=n_splits,
            purge_size=purge_size,
            min_train_size=min_train_size,
        )
        # 保存 config 供外部访问
        self.config: Dict[str, Any] = config or {
            'method': self.method,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'n_splits': self.n_splits,
            'purge_size': self.purge_size,
            'min_train_size': self.min_train_size,
        }


class WalkForwardValidator(TimeSeriesCrossValidator):
    """
    Walk-Forward 滚动窗口验证器（语义化子类）。

    自动锁定 method='walk_forward'，无需手动指定。
    额外支持 config 字典形式的参数传入，以及 split(factor_values, returns) 签名。
    """

    def __init__(
        self,
        train_size: int = 252,
        test_size: int = 63,
        purge_size: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config:
            train_size = config.get('train_size', train_size)
            test_size = config.get('test_size', test_size)
            purge_size = config.get('purge_size', purge_size)
            n_splits = config.get('n_splits', 5)
        else:
            n_splits = 5

        super().__init__(
            method='walk_forward',
            train_size=train_size,
            test_size=test_size,
            purge_size=purge_size,
            n_splits=n_splits,
        )
        self.config: Dict[str, Any] = config or {
            'train_size': self.train_size,
            'test_size': self.test_size,
            'purge_size': self.purge_size,
            'n_splits': self.n_splits,
        }

    def split(self, factor_values: Any, returns: Any = None) -> Generator[Tuple[Any, Any], None, None]:  # type: ignore[override]
        """
        Walk-Forward 分割，支持 (Series, Series) 或 (DataFrame,) 签名。

        当 config 指定 n_splits 时，按比例划分确保精确生成 n_splits 折；
        否则使用固定 train_size/test_size 滑动窗口。
        """
        data = _to_dataframe(factor_values, returns, self.n_splits)
        n = len(data)
        n_splits = self.config.get('n_splits', 5)

        # 若有明确 n_splits，动态按比例划分（确保精确折数）
        if n_splits and n_splits > 1:
            train_ratio = self.config.get('train_ratio', 0.6)
            fold_size = n // (n_splits + 1) or 1
            train_size = max(1, int(fold_size * (n_splits * train_ratio / (1 - train_ratio + n_splits * train_ratio))))

            for i in range(n_splits):
                test_start = (i + 1) * fold_size
                test_end = test_start + fold_size
                if test_end > n:
                    break
                train_end = test_start
                train_idx = data.index[:train_end]
                test_idx = data.index[test_start:test_end]
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx
        else:
            yield from super().split(data)

    def validate_stability(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> Dict[str, float]:
        """
        验证因子稳定性，返回 IC 统计信息。

        Returns:
            {'ic_mean', 'ic_std', 'icir', 'win_rate', 'n_folds', 'is_stable'}
        """
        stats = self.get_cv_stats(factor_values, returns)
        stats['is_stable'] = (
            abs(stats.get('icir', 0)) > 0.5 and stats.get('n_folds', 0) >= 3
        )
        return stats


class ExpandingWindowValidator(TimeSeriesCrossValidator):
    """
    Expanding Window 扩展窗口验证器（语义化子类）。

    自动锁定 method='expanding'，无需手动指定。
    """

    def __init__(
        self,
        test_size: int = 63,
        purge_size: int = 5,
        min_train_size: int = 126,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config:
            test_size = config.get('test_size', test_size)
            purge_size = config.get('purge_size', purge_size)
            min_train_size = config.get('min_train_size', min_train_size)
            n_splits = config.get('n_splits', 5)
        else:
            n_splits = 5

        super().__init__(
            method='expanding',
            test_size=test_size,
            purge_size=purge_size,
            min_train_size=min_train_size,
            n_splits=n_splits,
        )
        self.config: Dict[str, Any] = config or {
            'test_size': self.test_size,
            'purge_size': self.purge_size,
            'min_train_size': self.min_train_size,
            'n_splits': self.n_splits,
        }

    def split(self, factor_values: Any, returns: Any = None) -> Generator[Tuple[Any, Any], None, None]:  # type: ignore[override]
        """
        Expanding Window 分割，支持 (Series, Series) 或 (DataFrame,) 签名。

        当 config 指定 n_splits 时，精确生成 n_splits 折，每折训练集递增。
        """
        data = _to_dataframe(factor_values, returns, self.n_splits)
        n = len(data)
        n_splits = self.config.get('n_splits', 5)
        min_train = self.config.get('min_train_size', self.min_train_size)

        if n_splits and n_splits > 1:
            # 均匀划分 test 区间
            test_size = max(1, (n - min_train) // n_splits)
            for i in range(n_splits):
                test_start = min_train + i * test_size
                test_end = test_start + test_size
                if test_end > n:
                    break
                train_idx = data.index[:test_start]
                test_idx = data.index[test_start:test_end]
                if len(train_idx) >= min_train and len(test_idx) > 0:
                    yield train_idx, test_idx
        else:
            yield from super().split(data)

    def validate_stability(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> Dict[str, float]:
        """验证因子稳定性，返回 IC 统计信息。"""
        stats = self.get_cv_stats(factor_values, returns)
        stats['is_stable'] = abs(stats.get('icir', 0)) > 0.5
        return stats


class PurgedKFoldValidator(TimeSeriesCrossValidator):
    """
    Purged K-Fold 带清洗期的 K 折验证器（语义化子类）。

    自动锁定 method='purged_kfold'，无需手动指定。
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_size: int = 5,
        min_train_size: int = 126,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config:
            n_splits = config.get('n_splits', n_splits)
            purge_size = config.get('purge_gap', config.get('purge_size', purge_size))
            min_train_size = config.get('min_train_size', min_train_size)

        super().__init__(
            method='purged_kfold',
            n_splits=n_splits,
            purge_size=purge_size,
            min_train_size=min_train_size,
        )
        self.config: Dict[str, Any] = config or {
            'n_splits': self.n_splits,
            'purge_size': self.purge_size,
            'min_train_size': self.min_train_size,
        }
        # purge_gap 别名
        if 'purge_gap' not in self.config:
            self.config['purge_gap'] = self.purge_size

    def split(self, factor_values: Any, returns: Any = None) -> Generator[Tuple[Any, Any], None, None]:  # type: ignore[override]
        """
        Purged K-Fold 分割，支持 (Series, Series) 或 (DataFrame,) 签名。
        """
        data = _to_dataframe(factor_values, returns, None)
        yield from super().split(data)

    def validate_stability(
        self, factor_values: pd.Series, returns: pd.Series
    ) -> Dict[str, float]:
        """验证因子稳定性，返回 IC 统计信息。"""
        stats = self.get_cv_stats(factor_values, returns)
        stats['is_stable'] = abs(stats.get('icir', 0)) > 0.5
        return stats


__all__ = [
    'TimeSeriesCrossValidator',
    'CrossValidator',
    'WalkForwardValidator',
    'ExpandingWindowValidator',
    'PurgedKFoldValidator',
]

