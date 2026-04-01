"""
因子候选池 (Factor Candidate Pool)

定义 50+ 种候选因子，涵盖三大类别：
- 技术因子（Technical）：30+ 种，基于 OHLCV 数据的各种技术指标
- 基本面因子（Fundamental）：8 种，基于期货基本面数据
- 交叉因子（Cross）：6 种，多指标组合或复合信号

每个候选因子用 dataclass 表示，包含名称、类别、参数、描述、预期方向和依赖项。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass
class FactorCandidate:
    """
    因子候选描述数据类

    Attributes:
        name: 因子唯一名称（英文，snake_case）
        category: 因子类别，'technical' | 'fundamental' | 'cross'
        params: 因子计算参数字典
        description: 因子中文描述
        expected_direction: 因子与收益率的预期关系，'positive' | 'negative' | None
        dependencies: 依赖的其他因子名列表（仅交叉因子需要）
    """
    name: str
    category: str  # 'technical' / 'fundamental' / 'cross'
    params: Dict[str, Any]
    description: str
    expected_direction: Optional[str] = None  # 'positive' / 'negative'
    dependencies: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 候选因子列表
# ---------------------------------------------------------------------------

def build_candidate_pool() -> List[FactorCandidate]:
    """
    构建完整候选因子池

    Returns:
        包含所有候选因子的列表
    """

    candidates: List[FactorCandidate] = []

    # ================================================================
    # 一、技术因子候选（30+ 种）
    # ================================================================

    # --- 简单移动平均线 ---
    for period in [5, 10, 20, 60, 120, 250]:
        candidates.append(FactorCandidate(
            name=f'ma_{period}',
            category='technical',
            params={'period': period, 'ma_type': 'sma'},
            description=f'{period}日简单移动平均',
            expected_direction=None,
        ))

    # --- 指数移动平均线 ---
    for period in [5, 10, 20, 60]:
        candidates.append(FactorCandidate(
            name=f'ema_{period}',
            category='technical',
            params={'period': period, 'ma_type': 'ema'},
            description=f'{period}日指数移动平均',
            expected_direction=None,
        ))

    # --- 相对强弱指标 (RSI) ---
    for period in [6, 14, 28]:
        candidates.append(FactorCandidate(
            name=f'rsi_{period}',
            category='technical',
            params={'period': period, 'indicator': 'rsi'},
            description=f'{period}日相对强弱指标',
            expected_direction=None,
        ))

    # --- MACD (12, 26, 9) ---
    candidates.append(FactorCandidate(
        name='macd',
        category='technical',
        params={'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'output': 'histogram'},
        description='MACD 柱状图 (12,26,9)',
        expected_direction=None,
    ))

    # --- MACD DIF 线 ---
    candidates.append(FactorCandidate(
        name='macd_diff',
        category='technical',
        params={'fast_period': 12, 'slow_period': 26, 'output': 'diff'},
        description='MACD DIF 快线 (12,26)',
        expected_direction=None,
    ))

    # --- KDJ 指标 (9,3,3) ---
    candidates.append(FactorCandidate(
        name='kdj',
        category='technical',
        params={'n': 9, 'm1': 3, 'm2': 3, 'output': 'j'},
        description='KDJ 随机指标 J 值 (9,3,3)',
        expected_direction=None,
    ))

    # --- KDJ K 值 ---
    candidates.append(FactorCandidate(
        name='kdj_k',
        category='technical',
        params={'n': 9, 'm1': 3, 'm2': 3, 'output': 'k'},
        description='KDJ 随机指标 K 值 (9,3,3)',
        expected_direction=None,
    ))

    # --- 布林带 (20, 2) ---
    candidates.append(FactorCandidate(
        name='boll_width',
        category='technical',
        params={'period': 20, 'num_std': 2.0, 'output': 'width'},
        description='布林带宽度 (20,2)',
        expected_direction=None,
    ))

    # --- 布林带位置 ---
    candidates.append(FactorCandidate(
        name='boll_position',
        category='technical',
        params={'period': 20, 'num_std': 2.0, 'output': 'position'},
        description='布林带价格位置 (20,2)',
        expected_direction=None,
    ))

    # --- ATR (14) ---
    candidates.append(FactorCandidate(
        name='atr_14',
        category='technical',
        params={'period': 14, 'indicator': 'atr'},
        description='14日平均真实波幅',
        expected_direction=None,
    ))

    # --- OBV 能量潮 ---
    candidates.append(FactorCandidate(
        name='obv',
        category='technical',
        params={'indicator': 'obv'},
        description='能量潮指标',
        expected_direction=None,
    ))

    # --- ADX (14) ---
    candidates.append(FactorCandidate(
        name='adx_14',
        category='technical',
        params={'period': 14, 'indicator': 'adx'},
        description='14日平均趋向指数',
        expected_direction='positive',
    ))

    # --- DMI+ ---
    candidates.append(FactorCandidate(
        name='dmi_plus',
        category='technical',
        params={'period': 14, 'indicator': 'dmi_plus'},
        description='DMI 正向指标',
        expected_direction=None,
    ))

    # --- DMI- ---
    candidates.append(FactorCandidate(
        name='dmi_minus',
        category='technical',
        params={'period': 14, 'indicator': 'dmi_minus'},
        description='DMI 负向指标',
        expected_direction=None,
    ))

    # --- CCI (14) ---
    candidates.append(FactorCandidate(
        name='cci_14',
        category='technical',
        params={'period': 14, 'indicator': 'cci'},
        description='14日顺势指标',
        expected_direction=None,
    ))

    # --- 威廉指标 (14) ---
    candidates.append(FactorCandidate(
        name='willr_14',
        category='technical',
        params={'period': 14, 'indicator': 'willr'},
        description='14日威廉指标',
        expected_direction=None,
    ))

    # --- ROC (12) ---
    candidates.append(FactorCandidate(
        name='roc_12',
        category='technical',
        params={'period': 12, 'indicator': 'roc'},
        description='12日变动率指标',
        expected_direction=None,
    ))

    # --- 动量指标 (10) ---
    candidates.append(FactorCandidate(
        name='mom_10',
        category='technical',
        params={'period': 10, 'indicator': 'momentum'},
        description='10日动量指标',
        expected_direction=None,
    ))

    # --- 动量指标 (20) ---
    candidates.append(FactorCandidate(
        name='mom_20',
        category='technical',
        params={'period': 20, 'indicator': 'momentum'},
        description='20日动量指标',
        expected_direction=None,
    ))

    # --- MFI (14) ---
    candidates.append(FactorCandidate(
        name='mfi_14',
        category='technical',
        params={'period': 14, 'indicator': 'mfi'},
        description='14日资金流量指标',
        expected_direction=None,
    ))

    # --- CMO (14) ---
    candidates.append(FactorCandidate(
        name='cmo_14',
        category='technical',
        params={'period': 14, 'indicator': 'cmo'},
        description='14日钱德动量 oscillator',
        expected_direction=None,
    ))

    # --- PPO ---
    candidates.append(FactorCandidate(
        name='ppo',
        category='technical',
        params={'fast_period': 12, 'slow_period': 26, 'indicator': 'ppo'},
        description='价格百分比振荡器',
        expected_direction=None,
    ))

    # --- 历史波动率 (14) ---
    candidates.append(FactorCandidate(
        name='hist_vol_14',
        category='technical',
        params={'period': 14, 'indicator': 'historical_volatility'},
        description='14日历史波动率（对数收益率标准差年化）',
        expected_direction='negative',
    ))

    # --- 成交量均线比 ---
    candidates.append(FactorCandidate(
        name='vol_ma_ratio',
        category='technical',
        params={'short_period': 5, 'long_period': 20, 'indicator': 'volume_ma_ratio'},
        description='成交量短期/长期均线比',
        expected_direction=None,
    ))

    # --- 价格突破 20 日高点 ---
    candidates.append(FactorCandidate(
        name='price_breakout_20',
        category='technical',
        params={'period': 20, 'indicator': 'price_breakout'},
        description='价格突破 20 日高点（1=突破，0=未突破）',
        expected_direction='positive',
    ))

    # --- 均线发散度 ---
    candidates.append(FactorCandidate(
        name='ma_divergence',
        category='technical',
        params={'short_period': 5, 'long_period': 60, 'indicator': 'ma_divergence'},
        description='短期均线与长期均线之差标准化',
        expected_direction=None,
    ))

    # --- 量价相关性 ---
    candidates.append(FactorCandidate(
        name='volume_price_corr',
        category='technical',
        params={'period': 20, 'indicator': 'volume_price_correlation'},
        description='20日量价相关系数',
        expected_direction=None,
    ))

    # --- 日收益率偏度 ---
    candidates.append(FactorCandidate(
        name='return_skew_20',
        category='technical',
        params={'period': 20, 'indicator': 'return_skewness'},
        description='20日收益率偏度',
        expected_direction=None,
    ))

    # --- 日收益率峰度 ---
    candidates.append(FactorCandidate(
        name='return_kurt_20',
        category='technical',
        params={'period': 20, 'indicator': 'return_kurtosis'},
        description='20日收益率峰度',
        expected_direction=None,
    ))

    # --- 上影线比例 ---
    candidates.append(FactorCandidate(
        name='upper_shadow_ratio',
        category='technical',
        params={'indicator': 'upper_shadow_ratio'},
        description='K线上影线/实体比例',
        expected_direction=None,
    ))

    # --- 下影线比例 ---
    candidates.append(FactorCandidate(
        name='lower_shadow_ratio',
        category='technical',
        params={'indicator': 'lower_shadow_ratio'},
        description='K线下影线/实体比例',
        expected_direction=None,
    ))

    # ================================================================
    # 二、基本面因子候选（8 种）
    # ================================================================

    # --- 基差率 ---
    candidates.append(FactorCandidate(
        name='basis_rate',
        category='fundamental',
        params={'indicator': 'basis_rate', 'spot_col': 'spot_price', 'close_col': 'close'},
        description='基差率：(现货-期货)/期货 × 100%',
        expected_direction=None,
    ))

    # --- 基差变化 ---
    candidates.append(FactorCandidate(
        name='basis_change',
        category='fundamental',
        params={'indicator': 'basis_change', 'period': 1},
        description='基差变化率（环比）',
        expected_direction=None,
    ))

    # --- 库存变化率 ---
    candidates.append(FactorCandidate(
        name='inventory_change',
        category='fundamental',
        params={'indicator': 'inventory_change', 'inventory_col': 'inventory', 'method': 'rate'},
        description='库存环比变化率',
        expected_direction='negative',
    ))

    # --- 库存同比 ---
    candidates.append(FactorCandidate(
        name='inventory_yoy',
        category='fundamental',
        params={'indicator': 'inventory_yoy', 'inventory_col': 'inventory', 'periods_per_year': 52},
        description='库存同比变化率',
        expected_direction='negative',
    ))

    # --- 仓单压力 ---
    candidates.append(FactorCandidate(
        name='warehouse_pressure',
        category='fundamental',
        params={'indicator': 'warehouse_pressure', 'receipt_col': 'receipt', 'window': 252, 'method': 'zscore'},
        description='仓单压力（Z-score）',
        expected_direction='negative',
    ))

    # --- 仓单变化 ---
    candidates.append(FactorCandidate(
        name='receipt_change',
        category='fundamental',
        params={'indicator': 'receipt_change', 'receipt_col': 'receipt', 'period': 1, 'method': 'rate'},
        description='仓单环比变化率',
        expected_direction='negative',
    ))

    # --- 期限结构 ---
    candidates.append(FactorCandidate(
        name='term_structure',
        category='fundamental',
        params={'indicator': 'term_structure', 'near_col': 'near_price', 'far_col': 'far_price', 'method': 'ratio'},
        description='期限结构：(近月-远月)/远月 × 100%',
        expected_direction=None,
    ))

    # --- 期限结构曲率 ---
    candidates.append(FactorCandidate(
        name='term_curvature',
        category='fundamental',
        params={'indicator': 'term_curvature'},
        description='期限结构曲率（二阶导数近似）',
        expected_direction=None,
    ))

    # ================================================================
    # 三、交叉因子候选（6 种）
    # ================================================================

    # --- 均线金叉/死叉信号 ---
    candidates.append(FactorCandidate(
        name='ma_cross_signal',
        category='cross',
        params={'short_period': 5, 'long_period': 20, 'indicator': 'ma_cross'},
        description='均线金叉/死叉信号（1=金叉，-1=死叉，0=无信号）',
        expected_direction=None,
        dependencies=['ma_5', 'ma_20'],
    ))

    # --- 量价背离 ---
    candidates.append(FactorCandidate(
        name='vol_price_divergence',
        category='cross',
        params={'price_period': 20, 'volume_period': 20, 'indicator': 'vol_price_divergence'},
        description='量价背离指数（标准化差值）',
        expected_direction=None,
        dependencies=['volume_price_corr'],
    ))

    # --- 趋势强度 (ADX 相关) ---
    candidates.append(FactorCandidate(
        name='trend_strength',
        category='cross',
        params={'period': 14, 'indicator': 'trend_strength'},
        description='趋势强度（ADX 标准化）',
        expected_direction='positive',
        dependencies=['adx_14'],
    ))

    # --- 波动率区间 ---
    candidates.append(FactorCandidate(
        name='volatility_regime',
        category='cross',
        params={'period': 20, 'indicator': 'volatility_regime'},
        description='波动率区间分类（低/中/高 = 0/1/2）',
        expected_direction=None,
        dependencies=['hist_vol_14'],
    ))

    # --- 动量丝带 ---
    candidates.append(FactorCandidate(
        name='momentum_ribbon',
        category='cross',
        params={'periods': [5, 10, 20], 'indicator': 'momentum_ribbon'},
        description='动量丝带（多周期动量标准化差值）',
        expected_direction=None,
        dependencies=['mom_10', 'mom_20'],
    ))

    # --- RSI 与均线差 ---
    candidates.append(FactorCandidate(
        name='rsi_ma_diff',
        category='cross',
        params={'rsi_period': 14, 'ma_period': 20, 'indicator': 'rsi_ma_diff'},
        description='RSI 与收盘价均线的标准化差值',
        expected_direction=None,
        dependencies=['rsi_14'],
    ))

    return candidates


# ---------------------------------------------------------------------------
# 候选池封装类
# ---------------------------------------------------------------------------

class FactorCandidatePool:
    """
    因子候选池封装类

    提供候选因子的查询、筛选和分组功能。

    使用示例：
        >>> pool = FactorCandidatePool()
        >>> tech_candidates = pool.get_by_category('technical')
        >>> print(f"共 {pool.total_count} 个候选因子")
    """

    def __init__(self) -> None:
        """初始化候选池，加载全部候选因子"""
        self._candidates: List[FactorCandidate] = build_candidate_pool()

    @property
    def total_count(self) -> int:
        """返回候选因子总数"""
        return len(self._candidates)

    def get_all(self) -> List[FactorCandidate]:
        """
        获取所有候选因子

        Returns:
            候选因子列表
        """
        return list(self._candidates)

    def get_by_category(self, category: str) -> List[FactorCandidate]:
        """
        按类别筛选候选因子

        Args:
            category: 'technical' | 'fundamental' | 'cross'

        Returns:
            符合条件的候选因子列表
        """
        return [c for c in self._candidates if c.category == category]

    def get_by_name(self, name: str) -> Optional[FactorCandidate]:
        """
        按名称获取单个候选因子

        Args:
            name: 因子名称

        Returns:
            候选因子或 None
        """
        for c in self._candidates:
            if c.name == name:
                return c
        return None

    def get_by_names(self, names: List[str]) -> List[FactorCandidate]:
        """
        按名称列表批量获取候选因子

        Args:
            names: 因子名称列表

        Returns:
            候选因子列表（按输入顺序）
        """
        pool = {c.name: c for c in self._candidates}
        return [pool[n] for n in names if n in pool]

    def get_technical_count(self) -> int:
        """返回技术因子数量"""
        return len(self.get_by_category('technical'))

    def get_fundamental_count(self) -> int:
        """返回基本面因子数量"""
        return len(self.get_by_category('fundamental'))

    def get_cross_count(self) -> int:
        """返回交叉因子数量"""
        return len(self.get_by_category('cross'))

    def get_summary(self) -> Dict[str, int]:
        """
        获取候选池统计摘要

        Returns:
            包含各类型数量的字典
        """
        return {
            'total': self.total_count,
            'technical': self.get_technical_count(),
            'fundamental': self.get_fundamental_count(),
            'cross': self.get_cross_count(),
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"FactorCandidatePool("
            f"total={summary['total']}, "
            f"technical={summary['technical']}, "
            f"fundamental={summary['fundamental']}, "
            f"cross={summary['cross']})"
        )
