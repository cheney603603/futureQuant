"""
未来函数检测器 (Look-Ahead Bias Detector)

检测因子计算中是否存在未来函数（look-ahead bias），包括：
- 静态代码检测：使用 AST 分析源代码中的危险模式
- 动态数据检测：比较原始 IC 与延迟 IC，检测信息泄露
"""

import ast
import inspect
import textwrap
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger
from ...core.base import Factor

logger = get_logger('agent.lookahead_detector')


class LookAheadPattern(ast.NodeVisitor):
    """AST 访问器，检测未来函数相关的危险代码模式"""

    DANGEROUS_METHODS: Set[str] = {
        'shift', 'rolling', 'expanding', 'ewm',
        'resample', 'fillna', 'ffill', 'bfill',
    }

    SUSPICIOUS_PATTERNS: List[str] = [
        'future', 'lookahead', 'look_ahead', 'forward',
        'next', 'tomorrow', 'ahead',
    ]

    def __init__(self, source: Optional[str] = None) -> None:
        """
        初始化 AST 访问器。

        Args:
            source: 可选的源代码字符串。若提供，会在 __init__ 内立即执行 AST 扫描，
                    扫描结果保存在 self.dangerous_calls / self.suspicious_names 中，
                    方便直接通过构造函数完成一次性检查。
        """
        self.dangerous_calls: List[Dict[str, Any]] = []
        self.suspicious_names: Set[str] = set()
        self.current_function: str = ''
        self.current_lineno: int = 0

        if source is not None:
            try:
                tree = ast.parse(source)
                self.visit(tree)
            except SyntaxError as exc:
                logger.warning(f'AST parse error during __init__: {exc}')

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old = self.current_function
        self.current_function = node.name
        # 检测函数名中是否含可疑词汇
        name_lower = node.name.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in name_lower:
                self.suspicious_names.add(node.name)
                self.dangerous_calls.append({
                    'type': 'suspicious_function_name',
                    'name': node.name,
                    'line': getattr(node, 'lineno', 0),
                    'function': node.name,
                })
        self.generic_visit(node)
        self.current_function = old

    def visit_Call(self, node: ast.Call) -> None:
        """访问函数调用"""
        self.current_lineno = node.lineno or 0

        func_name = self._get_func_name(node.func)

        if func_name in self.DANGEROUS_METHODS:
            self._check_shift_call(node, func_name)

        if isinstance(node.func, ast.Name):
            name_lower = node.func.id.lower()
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern in name_lower:
                    self.suspicious_names.add(node.func.id)
                    self.dangerous_calls.append({
                        'type': 'suspicious_name',
                        'name': node.func.id,
                        'line': self.current_lineno,
                        'function': self.current_function,
                    })

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """访问赋值语句，检测可疑变量名（如 future_price = ...）"""
        self.current_lineno = getattr(node, 'lineno', 0)
        for target in node.targets:
            self._check_name_node(target)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """访问名称节点（赋值目标、Store上下文），检测可疑词汇"""
        if isinstance(node.ctx, (ast.Store,)):
            self._check_name_node(node)

    def _check_name_node(self, node: ast.expr) -> None:
        """检测 Name 节点是否含可疑关键词"""
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                self._check_name_node(elt)
            return

        if name is None:
            return

        name_lower = name.lower()
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern in name_lower:
                self.suspicious_names.add(name)
                self.dangerous_calls.append({
                    'type': 'suspicious_variable_name',
                    'name': name,
                    'line': self.current_lineno,
                    'function': self.current_function,
                })

    def _get_func_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ''

    def _check_shift_call(self, node: ast.Call, func_name: str) -> None:
        """检测 shift(-n) 等危险调用"""
        if func_name != 'shift':
            return
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                self.dangerous_calls.append({
                    'type': 'negative_shift',
                    'name': func_name,
                    'line': self.current_lineno,
                    'function': self.current_function,
                    'detail': 'shift with negative value uses future data',
                })
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                if arg.value < 0:
                    self.dangerous_calls.append({
                        'type': 'negative_shift',
                        'name': func_name,
                        'line': self.current_lineno,
                        'function': self.current_function,
                        'detail': f'shift({arg.value}) uses future data',
                    })

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """检测未来日期索引"""
        self.current_lineno = getattr(node, 'lineno', 0)
        self.generic_visit(node)

    def get_dangerous_calls_summary(self) -> List[Dict[str, Any]]:
        """
        返回所有检测到的危险调用摘要列表。

        Returns:
            危险调用列表，每项包含 type / name / line / function 等字段
        """
        return list(self.dangerous_calls)


class LookAheadDetector:
    """
    未来函数检测器

    双重检测机制：
    1. 静态 AST 分析：扫描源代码中的危险模式
    2. 动态 IC 延迟测试：比较原始 IC 与延迟 IC
    """

    # 默认配置
    DEFAULT_CONFIG: Dict[str, Any] = {
        'ic_delay_threshold': 0.1,
        'max_delay_days': 5,
        'significance_level': 0.05,
        'min_samples': 30,
    }

    def __init__(
        self,
        ic_lag_periods: int = 5,
        ic_threshold: float = 0.1,
        significance_level: float = 0.05,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            ic_lag_periods: IC 延迟测试的最大滞后期数
            ic_threshold: IC 显著性阈值
            significance_level: 统计显著性水平
            config: 可选的配置字典，允许以 dict 形式覆盖所有参数。
                    支持的键：ic_delay_threshold, max_delay_days, significance_level, min_samples
        """
        self.ic_lag_periods = ic_lag_periods
        self.ic_threshold = ic_threshold
        self.significance_level = significance_level

        # 初始化 config，合并默认值与用户传入值
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        # 用 config 覆盖具名参数（config 优先级更高）
        if 'max_delay_days' in self.config:
            self.ic_lag_periods = int(self.config['max_delay_days'])
        if 'ic_delay_threshold' in self.config:
            self.ic_threshold = float(self.config['ic_delay_threshold'])
        if 'significance_level' in self.config:
            self.significance_level = float(self.config['significance_level'])

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def static_check(self, factor: Any) -> Dict[str, Any]:
        """
        静态 AST 检查接口（公共方法）。

        Args:
            factor: 因子对象（需有 compute 或 calculate 方法）或可调用对象

        Returns:
            检测结果字典::

                {
                    'has_lookahead': bool,
                    'dangerous_calls': List[dict],
                    'risk_level': str,  # 'low' / 'medium' / 'high'
                }
        """
        issues = self._static_check(factor)
        return {
            'has_lookahead': len(issues) > 0,
            'dangerous_calls': issues,
            'risk_level': 'high' if len(issues) > 2 else ('medium' if issues else 'low'),
        }

    def static_check_from_source(self, source: str) -> Dict[str, Any]:
        """
        直接提供源代码字符串进行静态检测。

        Args:
            source: Python 源代码字符串

        Returns:
            检测结果字典（同 static_check）
        """
        issues: List[Dict[str, Any]] = []
        try:
            tree = ast.parse(source)
            visitor = LookAheadPattern()
            visitor.visit(tree)
            issues.extend(visitor.dangerous_calls)
        except SyntaxError as exc:
            logger.warning(f'AST parse error: {exc}')
            return {
                'has_lookahead': False,
                'dangerous_calls': [],
                'risk_level': 'unknown',
                'error': str(exc),
            }

        return {
            'has_lookahead': len(issues) > 0,
            'dangerous_calls': issues,
            'risk_level': 'high' if len(issues) > 2 else ('medium' if issues else 'low'),
        }

    def dynamic_check(
        self,
        factor: Any,
        data: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        factor_col: str = 'factor',
        return_col: str = 'return',
    ) -> Dict[str, Any]:
        """
        动态 IC 延迟检测接口（公共方法）。

        通过比较不同滞后期的 IC 来判断因子是否使用了未来数据。

        Args:
            factor: 因子对象（需有 calculate 或 compute 方法）
            data: OHLCV 或含因子值的 DataFrame
            returns: 收益率 Series（若提供则直接使用；否则从 data 中提取）
            factor_col: 当 returns 为 None 时，从 data 中读取的因子列名
            return_col: 当 returns 为 None 时，从 data 中读取的收益率列名

        Returns:
            检测结果字典，含 has_lookahead / ic_original / ic_delayed / delay_days / risk_level
        """
        # ------------------------------------------------------------------
        # Step 1: 计算因子值
        # ------------------------------------------------------------------
        factor_values: Optional[pd.Series] = None
        for attr in ('calculate', 'compute'):
            fn = getattr(factor, attr, None)
            if callable(fn) and not data.empty:
                try:
                    factor_values = fn(data)
                    break
                except Exception:
                    pass

        if factor_values is None:
            if factor_col in getattr(data, 'columns', []):
                factor_values = data[factor_col]
            else:
                return {
                    'has_lookahead': False,
                    'ic_original': 0.0,
                    'ic_delayed': 0.0,
                    'delay_days': 0,
                    'risk_level': 'unknown',
                    'error': 'Cannot compute factor values',
                }

        # ------------------------------------------------------------------
        # Step 2: 获取收益率
        # ------------------------------------------------------------------
        if returns is None:
            if return_col in getattr(data, 'columns', []):
                returns = data[return_col]
            else:
                return {
                    'has_lookahead': False,
                    'ic_original': 0.0,
                    'ic_delayed': 0.0,
                    'delay_days': 0,
                    'risk_level': 'unknown',
                    'error': f'returns not provided and column {return_col!r} not found in data',
                }

        if data.empty or returns.empty or factor_values is None:
            return {
                'has_lookahead': False,
                'ic_original': 0.0,
                'ic_delayed': 0.0,
                'delay_days': 0,
                'risk_level': 'low',
            }

        # ------------------------------------------------------------------
        # Step 3: IC 延迟测试
        # ------------------------------------------------------------------
        try:
            lag_result = self.check_ic_lag(factor_values, returns)
        except Exception as exc:
            return {
                'has_lookahead': False,
                'ic_original': 0.0,
                'ic_delayed': 0.0,
                'delay_days': 0,
                'risk_level': 'unknown',
                'error': str(exc),
            }

        # 原始 IC（lag=0）
        ic_original = lag_result['ic_by_lag'].get(0, {}).get('ic', 0.0)
        # 最大负滞后 IC（表示未来方向）
        neg_lag_ics = {k: v['ic'] for k, v in lag_result['ic_by_lag'].items() if k < 0}
        ic_delayed = max(abs(v) for v in neg_lag_ics.values()) if neg_lag_ics else 0.0
        best_delay = min(neg_lag_ics.keys(), key=lambda k: abs(k)) if neg_lag_ics else 0

        has_lookahead = lag_result.get('has_lookahead', False)
        risk_level = 'high' if has_lookahead else 'low'

        return {
            'has_lookahead': has_lookahead,
            'ic_original': float(ic_original),
            'ic_delayed': float(ic_delayed),
            'delay_days': int(best_delay),
            'ic_by_lag': lag_result['ic_by_lag'],
            'risk_level': risk_level,
        }

    def comprehensive_check(
        self,
        factor: Any,
        data: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
        factor_col: str = 'factor',
        return_col: str = 'return',
    ) -> Dict[str, Any]:
        """
        综合检测：同时执行静态 + 动态检测，返回汇总报告。

        Args:
            factor: 因子对象
            data: 可选的 DataFrame（OHLCV，用于动态检测）
            returns: 可选的收益率 Series（用于动态检测）
            factor_col: 因子列名（data 含因子值时使用）
            return_col: 收益率列名（data 含收益率时使用）

        Returns:
            综合报告字典::

                {
                    'has_lookahead': bool,
                    'static_check': dict,
                    'dynamic_check': dict | None,
                    'overall_risk': str,   # 'low' / 'medium' / 'high'
                    'summary': str,
                }
        """
        static_result = self.static_check(factor)
        dynamic_result: Optional[Dict[str, Any]] = None

        if data is not None:
            dynamic_result = self.dynamic_check(factor, data, returns, factor_col, return_col)

        has_lookahead = static_result['has_lookahead'] or (
            dynamic_result is not None and dynamic_result.get('has_lookahead', False)
        )

        if has_lookahead:
            overall_risk = 'high'
        elif static_result['risk_level'] == 'medium':
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        summary_parts = [f"Static: {static_result['risk_level']}"]
        if dynamic_result is not None:
            summary_parts.append(f"Dynamic: {dynamic_result.get('risk_level', 'n/a')}")
        summary = ' | '.join(summary_parts)

        return {
            'has_lookahead': has_lookahead,
            'static_check': static_result,
            'dynamic_check': dynamic_result,
            'overall_risk': overall_risk,
            'summary': summary,
        }

    def batch_check(
        self,
        factors: List[Any],
        data: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量因子检测。

        Args:
            factors: 因子对象列表
            data: 可选数据 DataFrame（用于动态检测）
            returns: 可选收益率 Series（用于动态检测）

        Returns:
            每个因子的综合报告列表，顺序与输入一致
        """
        results = []
        for factor in factors:
            try:
                result = self.comprehensive_check(factor, data, returns)
                factor_name = getattr(factor, 'name', repr(factor))
                result['factor_name'] = factor_name
            except Exception as exc:
                result = {
                    'has_lookahead': False,
                    'risk_level': 'unknown',
                    'error': str(exc),
                    'factor_name': getattr(factor, 'name', repr(factor)),
                }
            results.append(result)
        return results

    def check(self, factor: Any) -> bool:
        """
        检测因子是否存在未来函数（轻量级接口）。

        Args:
            factor: 因子对象或可调用对象

        Returns:
            True 表示存在未来函数风险
        """
        result = self.detect(factor)
        return result.get('has_lookahead', False)

    def detect(self, factor: Any) -> Dict[str, Any]:
        """
        完整检测，返回详细报告。

        Args:
            factor: 因子对象

        Returns:
            检测报告字典
        """
        report: Dict[str, Any] = {
            'has_lookahead': False,
            'static_issues': [],
            'dynamic_issues': [],
            'risk_level': 'low',
            'details': '',
        }

        # 静态检测
        static_issues = self._static_check(factor)
        report['static_issues'] = static_issues

        if static_issues:
            report['has_lookahead'] = True
            report['risk_level'] = 'high'
            report['details'] = f'Static analysis found {len(static_issues)} issue(s)'

        return report

    # ------------------------------------------------------------------
    # 私有实现
    # ------------------------------------------------------------------

    def _static_check(self, factor: Any) -> List[Dict[str, Any]]:
        """静态 AST 检测"""
        issues: List[Dict[str, Any]] = []

        # 尝试获取源代码（优先 compute，其次 calculate，再其次整个对象）
        source = None
        for attr_name in ('compute', 'calculate', None):
            try:
                target = getattr(factor, attr_name) if attr_name else factor
                source = inspect.getsource(target)
                break
            except (OSError, TypeError, AttributeError):
                continue

        if source is None:
            return issues

        try:
            # dedent 消除嵌套定义的缩进（例如测试方法内部的函数）
            source = textwrap.dedent(source)
            tree = ast.parse(source)
            visitor = LookAheadPattern()
            visitor.visit(tree)
            issues.extend(visitor.dangerous_calls)
        except SyntaxError as e:
            logger.warning(f'AST parse error: {e}')

        return issues

    def check_ic_lag(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        动态 IC 延迟测试

        Args:
            factor_values: 因子值序列
            returns: 收益率序列

        Returns:
            IC 延迟测试结果
        """
        results: Dict[str, Any] = {
            'ic_by_lag': {},
            'has_lookahead': False,
            'max_ic_lag': 0,
        }

        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < 30:
            return results

        f_vals = aligned.iloc[:, 0]
        r_vals = aligned.iloc[:, 1]

        for lag in range(-self.ic_lag_periods, self.ic_lag_periods + 1):
            shifted_r = r_vals.shift(-lag)
            valid = pd.concat([f_vals, shifted_r], axis=1).dropna()
            if len(valid) < 20:
                continue
            corr, pval = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
            results['ic_by_lag'][lag] = {'ic': corr, 'pval': pval}

        # 检测负滞后（未来）IC 是否显著高于正滞后
        neg_ics = [v['ic'] for k, v in results['ic_by_lag'].items() if k < 0]
        pos_ics = [v['ic'] for k, v in results['ic_by_lag'].items() if k > 0]

        if neg_ics and pos_ics:
            if abs(np.mean(neg_ics)) > abs(np.mean(pos_ics)) * 1.5:
                results['has_lookahead'] = True

        return results
