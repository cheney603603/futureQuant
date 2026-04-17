"""
WorldQuant Alpha101 公式库（核心子集 + 扩展接口）

参考：
- WorldQuant Alpha101 Research Paper
- 社区公开实现

每个因子以数学表达式字符串形式存储，供 LLM 翻译为可执行 pandas 代码。
"""

from typing import Dict, List, Optional


ALPHA101_FORMULAS: Dict[str, str] = {
    # 经典动量/反转
    "Alpha001": "(rank(ts_argmax(power(((returns < 0) ? ts_std(returns, 20) : close), 2), 5)) - 0.5)",
    "Alpha002": "(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))",
    "Alpha003": "(-1 * correlation(rank(open), rank(volume), 10))",
    "Alpha004": "(-1 * ts_rank(rank(low), 9))",
    "Alpha005": "(rank((open - ts_sum(vwap, 10) / 10)) * (-1 * abs(rank((close - vwap)))))",
    "Alpha006": "(-1 * correlation(open, volume, 10))",

    # 波动率/成交量
    "Alpha010": "rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))",
    "Alpha012": "(sign(volume) * (-1 * delta(close, 1)))",
    "Alpha013": "(-1 * rank(covariance(rank(close), rank(volume), 5)))",
    "Alpha014": "((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))",
    "Alpha015": "(-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3))",

    # 价格位置/形态
    "Alpha020": "((-1 * rank(open - ts_delay(high, 1))) * rank(open - ts_delay(close, 1)))",
    "Alpha021": "((ts_sum(close, 8) / 8 + ts_std(close, 8)) < (ts_sum(close, 2) / 2)) ? (-1 * 1) : ((ts_sum(close, 2) / 2) < (ts_sum(close, 8) / 8 - ts_std(close, 8)) ? 1 : ((volume / adv20 < 1) ? (-1 * 1) : 1)))",
    "Alpha023": "((ts_sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)",
    "Alpha024": "((((delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100)) <= 0.05) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3))))",
    "Alpha026": "(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))",

    # 均值回归/趋势
    "Alpha028": "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))",
    "Alpha029": "(ts_min(product(rank(rank(scale(log(ts_sum(ts_min(correlation(rank(vwap), rank(volume), 3), 3), 1)))), 1), 5) + ts_rank(ts_delay((-1 * returns), 6), 5))",
    "Alpha030": "(((1.0 - rank(((sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2))) + sign(ts_delay(close, 2) - ts_delay(close, 3)))))) * ts_sum(volume, 5)) / ts_sum(volume, 20))",
    "Alpha031": "((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))",
    "Alpha033": "rank((-1 * ((1 - (open / close)) ** 1)))",
    "Alpha034": "rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(delta(close, 1)))))",
    "Alpha035": "((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))",
    "Alpha037": "(rank(correlation(ts_delay((open - close), 1), close, 200)) + rank((open - close)))",
    "Alpha038": "((-1 * rank(ts_rank(close, 10))) * rank((close / open)))",
    "Alpha039": "((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(ts_sum(returns, 250))))",
    "Alpha040": "((-1 * rank(ts_std(high, 10))) * correlation(high, volume, 10))",
    "Alpha043": "(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))",
    "Alpha044": "(-1 * correlation(high, rank(volume), 5))",
    "Alpha045": "(-1 * ((rank((ts_sum(ts_delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))))",
    "Alpha046": "((0.25 < (((ts_delay(close, 5) - ts_delay(close, 4)) / 4) - ((ts_delay(close, 4) - ts_delay(close, 3)) / 3))) ? 1 : ((((ts_delay(close, 4) - ts_delay(close, 3)) / 3) < ((ts_delay(close, 3) - ts_delay(close, 2)) / 2)) ? (-1 * 1) : (((ts_delay(close, 3) - ts_delay(close, 2)) / 2) < 0.25 ? 1 : ((-1 * 1) * sign((close - ts_delay(close, 1)))))))",
    "Alpha049": "(((((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - ts_delay(close, 1))))",
    "Alpha050": "(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))",
    "Alpha051": "(((((ts_delay(close, 20) - ts_delay(close, 10)) / 10) - ((ts_delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - ts_delay(close, 1))))",
    "Alpha052": "((((ts_delay(close, 5) - ts_delay(close, 4)) / 4) - ((ts_delay(close, 4) - ts_delay(close, 3)) / 3)) + ((ts_delay(close, 3) - ts_delay(close, 2)) / 2)) - ((ts_delay(close, 2) - ts_delay(close, 1)) / 1))",
    "Alpha053": "(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))",
    "Alpha054": "((-1 * ((low - close) * (open ** 5))) / ((low - high) * (close ** 5)))",
    "Alpha055": "(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))",
    "Alpha060": "(-1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10)))))",
}


# 辅助函数映射（供 LLM 参考）
ALPHA_HELPER_FUNCTIONS: Dict[str, str] = {
    "ts_delay(x, d)": "x.shift(d)",
    "ts_delta(x, d)": "x.diff(d)",
    "ts_corr(x, y, d)": "x.rolling(d).corr(y)",
    "ts_cov(x, y, d)": "x.rolling(d).cov(y)",
    "ts_std(x, d)": "x.rolling(d).std()",
    "ts_sum(x, d)": "x.rolling(d).sum()",
    "ts_mean(x, d)": "x.rolling(d).mean()",
    "ts_max(x, d)": "x.rolling(d).max()",
    "ts_min(x, d)": "x.rolling(d).min()",
    "ts_rank(x, d)": "x.rolling(d).apply(lambda s: s.rank().iloc[-1])",
    "ts_argmax(x, d)": "x.rolling(d).apply(np.argmax)",
    "ts_argmin(x, d)": "x.rolling(d).apply(np.argmin)",
    "rank(x)": "x.rank()",
    "scale(x)": "(x - x.mean()) / x.std()",
    "sign(x)": "np.sign(x)",
    "abs(x)": "np.abs(x)",
    "log(x)": "np.log(x)",
    "power(x, a)": "np.power(x, a)",
    "sqrt(x)": "np.sqrt(x)",
    "decay_linear(x, d)": "x.ewm(span=d, adjust=False).mean()",
    "adv20": "volume.rolling(20).mean()",
    "vwap": "(close * volume).cumsum() / volume.cumsum()",
    "returns": "close.pct_change()",
    "correlation(x, y, d)": "x.rolling(d).corr(y)",
    "covariance(x, y, d)": "x.rolling(d).cov(y)",
    "condition ? a : b": "np.where(condition, a, b)",
}


class Alpha101Pool:
    """Alpha101 公式池管理"""

    def __init__(self):
        self._formulas = dict(ALPHA101_FORMULAS)

    def list_factors(self) -> List[str]:
        return list(self._formulas.keys())

    def get_formula(self, name: str) -> Optional[str]:
        return self._formulas.get(name)

    def get_helper_doc(self) -> str:
        lines = ["Alpha101 辅助函数映射（供代码生成参考）："]
        for k, v in ALPHA_HELPER_FUNCTIONS.items():
            lines.append(f"  {k} -> {v}")
        return "\n".join(lines)

    def get_all(self) -> Dict[str, str]:
        return dict(self._formulas)

    def add_formula(self, name: str, formula: str):
        self._formulas[name] = formula
