"""
未来函数检测器 (Look-Ahead Bias Detector)

检测因子计算中是否存在未来函数（look-ahead bias），包括：
- 静态代码检测：使用 AST 分析源代码中的危险模式
- 动态数据检测：比较原始 IC 与延迟 IC，检测信息泄露
"""

import ast
import inspect
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

    def __init__(self) -> None:
        self.dangerous_calls: List[Dict[str, Any]] = []
        self.suspicious_names: Set[str] = set()
        self.current_function: str = ''
        self.current_lineno: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old = self.current_function
        self.current_function = node.name
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


class LookAheadDetector:
    """
    未来函数检测器

    双重检测机制：
    1. 静态 AST 分析：扫描源代码中的危险模式
    2. 动态 IC 延迟测试：比较原始 IC 与延迟 IC
    """

    def __init__(
        self,
        ic_lag_periods: int = 5,
        ic_threshold: float = 0.1,
        significance_level: float = 0.05,
    ) -> None:
        """
        Args:
            ic_lag_periods: IC 延迟测试的最大滞后期数
            ic_threshold: IC 显著性阈值
            significance_level: 统计显著性水平
        """
        self.ic_lag_periods = ic_lag_periods
        self.ic_threshold = ic_threshold
        self.significance_level = significance_level

    def check(self, factor: Any) -> bool:
        """
        检测因子是否存在未来函数

        Args:
            factor: 因子对象或可调用对象

        Returns:
            True 表示存在未来函数风险
        """
        result = self.detect(factor)
        return result.get('has_lookahead', False)

    def detect(self, factor: Any) -> Dict[str, Any]:
        """
        完整检测，返回详细报告

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

    def _static_check(self, factor: Any) -> List[Dict[str, Any]]:
        """静态 AST 检测"""
        issues: List[Dict[str, Any]] = []

        # 尝试获取源代码
        source = None
        if callable(factor):
            try:
                source = inspect.getsource(factor)
            except (OSError, TypeError):
                pass
        elif hasattr(factor, 'compute'):
            try:
                source = inspect.getsource(factor.compute)
            except (OSError, TypeError):
                pass

        if source is None:
            return issues

        try:
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
