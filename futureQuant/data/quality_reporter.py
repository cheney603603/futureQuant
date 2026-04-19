"""
数据质量报告生成器 - 自动数据质量评估

P2.3 实现：
- 自动生成数据质量摘要
- 新鲜度、缺失率、异常值检测
- 支持 Markdown/HTML 报告输出
- 可集成到数据获取流程

Author: futureQuant Team
Date: 2026-04-19
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger

logger = get_logger('data.quality')


@dataclass
class QualityMetrics:
    """数据质量指标"""
    # 基础信息
    data_type: str
    variety: str
    start_date: str
    end_date: str
    row_count: int
    
    # 新鲜度
    latest_date: Optional[str] = None
    days_since_latest: Optional[int] = None
    freshness_score: float = 0.0  # 0-1
    
    # 完整性
    expected_rows: Optional[int] = None
    completeness_rate: float = 1.0  # 0-1
    missing_columns: List[str] = None
    
    # 异常值
    outlier_count: int = 0
    outlier_rate: float = 0.0
    price_anomalies: List[Dict] = None
    
    # 统计
    price_stats: Dict[str, float] = None
    
    # 总体评分
    overall_score: float = 0.0
    quality_level: str = 'unknown'  # excellent/good/fair/poor
    issues: List[str] = None
    
    def __post_init__(self):
        if self.missing_columns is None:
            self.missing_columns = []
        if self.price_anomalies is None:
            self.price_anomalies = []
        if self.price_stats is None:
            self.price_stats = {}
        if self.issues is None:
            self.issues = []


class DataQualityReporter:
    """
    数据质量报告生成器
    
    使用示例：
        reporter = DataQualityReporter()
        
        # 生成报告
        report = reporter.generate_report(df, 'price', 'RB', '2026-01-01', '2026-04-01')
        
        # 保存为 Markdown
        reporter.save_markdown_report(report, 'docs/reports/data_quality_RB.md')
    """
    
    # 质量等级阈值
    QUALITY_LEVELS = {
        'excellent': (0.9, 1.0),
        'good': (0.75, 0.9),
        'fair': (0.6, 0.75),
        'poor': (0.0, 0.6),
    }
    
    # 新鲜度评分阈值（天）
    FRESHNESS_THRESHOLDS = {
        'price': 3,        # 价格数据3天内算新鲜
        'fundamental': 7,  # 基本面数据7天内算新鲜
        'inventory': 5,
        'basis': 3,
    }
    
    def __init__(self):
        """初始化报告生成器"""
        self.reports: List[QualityMetrics] = []
    
    def analyze(
        self,
        data: pd.DataFrame,
        data_type: str,
        variety: str,
        start_date: str,
        end_date: str,
        expected_columns: Optional[List[str]] = None
    ) -> QualityMetrics:
        """
        分析数据质量
        
        Args:
            data: 要分析的数据
            data_type: 数据类型
            variety: 品种代码
            start_date: 请求开始日期
            end_date: 请求结束日期
            expected_columns: 期望的列名
            
        Returns:
            QualityMetrics
        """
        metrics = QualityMetrics(
            data_type=data_type,
            variety=variety,
            start_date=start_date,
            end_date=end_date,
            row_count=len(data),
        )
        
        if data.empty:
            metrics.issues.append("数据为空")
            metrics.quality_level = 'poor'
            return metrics
        
        # 1. 新鲜度分析
        self._analyze_freshness(data, metrics, data_type)
        
        # 2. 完整性分析
        self._analyze_completeness(data, metrics, start_date, end_date, expected_columns)
        
        # 3. 异常值分析
        self._analyze_outliers(data, metrics, variety)
        
        # 4. 统计信息
        self._calculate_stats(data, metrics)
        
        # 5. 计算总体评分
        self._calculate_overall_score(metrics)
        
        return metrics
    
    def _analyze_freshness(
        self,
        data: pd.DataFrame,
        metrics: QualityMetrics,
        data_type: str
    ):
        """分析数据新鲜度"""
        if 'date' not in data.columns:
            return
        
        latest_date = data['date'].max()
        metrics.latest_date = str(latest_date)[:10]
        
        if isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date)
        
        days_old = (datetime.now() - pd.to_datetime(latest_date)).days
        metrics.days_since_latest = days_old
        
        # 计算新鲜度评分
        threshold = self.FRESHNESS_THRESHOLDS.get(data_type, 7)
        
        if days_old <= threshold:
            metrics.freshness_score = 1.0
        elif days_old <= threshold * 2:
            metrics.freshness_score = 0.7
        elif days_old <= threshold * 3:
            metrics.freshness_score = 0.4
        else:
            metrics.freshness_score = 0.0
            metrics.issues.append(f"数据过时，最新数据 {days_old} 天前")
    
    def _analyze_completeness(
        self,
        data: pd.DataFrame,
        metrics: QualityMetrics,
        start_date: str,
        end_date: str,
        expected_columns: Optional[List[str]]
    ):
        """分析数据完整性"""
        # 检查期望列
        if expected_columns:
            missing = [col for col in expected_columns if col not in data.columns]
            metrics.missing_columns = missing
            if missing:
                metrics.issues.append(f"缺少列: {missing}")
        
        # 计算期望行数（交易日）
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            # 粗略估计：约 250 个交易日/年
            days = (end - start).days
            expected_rows = int(days * 250 / 365)
            metrics.expected_rows = expected_rows
            
            if expected_rows > 0:
                metrics.completeness_rate = min(len(data) / expected_rows, 1.0)
            
            if metrics.completeness_rate < 0.8:
                metrics.issues.append(
                    f"数据不完整: {len(data)}/{expected_rows} "
                    f"({metrics.completeness_rate*100:.1f}%)"
                )
        except Exception:
            pass
    
    def _analyze_outliers(
        self,
        data: pd.DataFrame,
        metrics: QualityMetrics,
        variety: str
    ):
        """分析异常值"""
        if 'close' not in data.columns:
            return
        
        # 价格范围检查
        from futureQuant.data.validator import DataValidator
        validator = DataValidator(variety=variety)
        
        price_range = validator.PRICE_RANGES.get(variety, validator.PRICE_RANGES['DEFAULT'])
        min_price, max_price = price_range
        
        outliers = data[
            (data['close'] < min_price) | (data['close'] > max_price)
        ]
        
        metrics.outlier_count = len(outliers)
        metrics.outlier_rate = len(outliers) / len(data) if len(data) > 0 else 0
        
        if len(outliers) > 0:
            metrics.issues.append(
                f"发现 {len(outliers)} 个价格异常值 "
                f"({metrics.outlier_rate*100:.2f}%)"
            )
            
            # 记录前3个异常
            for _, row in outliers.head(3).iterrows():
                metrics.price_anomalies.append({
                    'date': str(row['date'])[:10] if 'date' in row else None,
                    'price': float(row['close']),
                    'expected_range': [min_price, max_price],
                })
        
        # 日收益率异常
        if 'close' in data.columns and len(data) > 1:
            data_sorted = data.sort_values('date') if 'date' in data.columns else data
            returns = data_sorted['close'].pct_change().abs()
            large_moves = returns[returns > 0.1]  # 超过10%
            
            if len(large_moves) > 0:
                metrics.issues.append(
                    f"发现 {len(large_moves)} 次日收益率超过10%"
                )
    
    def _calculate_stats(self, data: pd.DataFrame, metrics: QualityMetrics):
        """计算统计信息"""
        if 'close' in data.columns:
            metrics.price_stats = {
                'min': float(data['close'].min()),
                'max': float(data['close'].max()),
                'mean': float(data['close'].mean()),
                'std': float(data['close'].std()),
                'latest': float(data['close'].iloc[-1]),
            }
    
    def _calculate_overall_score(self, metrics: QualityMetrics):
        """计算总体质量评分"""
        # 权重
        weights = {
            'freshness': 0.4,
            'completeness': 0.4,
            'outliers': 0.2,
        }
        
        # 异常值扣分
        outlier_score = max(0, 1 - metrics.outlier_rate * 10)
        
        # 计算加权分数
        score = (
            weights['freshness'] * metrics.freshness_score +
            weights['completeness'] * metrics.completeness_rate +
            weights['outliers'] * outlier_score
        )
        
        metrics.overall_score = round(score, 2)
        
        # 确定质量等级
        for level, (min_score, max_score) in self.QUALITY_LEVELS.items():
            if min_score <= score <= max_score:
                metrics.quality_level = level
                break
    
    def generate_report(
        self,
        data: pd.DataFrame,
        data_type: str,
        variety: str,
        start_date: str,
        end_date: str,
        expected_columns: Optional[List[str]] = None
    ) -> QualityMetrics:
        """
        生成数据质量报告
        
        Args:
            data: 数据
            data_type: 数据类型
            variety: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            expected_columns: 期望列
            
        Returns:
            QualityMetrics
        """
        metrics = self.analyze(data, data_type, variety, start_date, end_date, expected_columns)
        self.reports.append(metrics)
        return metrics
    
    def to_markdown(self, metrics: QualityMetrics) -> str:
        """转换为 Markdown 格式报告"""
        lines = [
            f"# 数据质量报告: {metrics.variety} {metrics.data_type}",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**数据期间**: {metrics.start_date} ~ {metrics.end_date}",
            "",
            "## 总体评估",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 质量评分 | {metrics.overall_score:.2f} / 1.0 |",
            f"| 质量等级 | {self._get_level_emoji(metrics.quality_level)} {metrics.quality_level.upper()} |",
            f"| 数据行数 | {metrics.row_count} |",
            "",
            "## 详细指标",
            "",
            "### 新鲜度",
            f"- 最新数据日期: {metrics.latest_date or 'N/A'}",
            f"- 距今: {metrics.days_since_latest or 'N/A'} 天",
            f"- 新鲜度评分: {metrics.freshness_score:.2f}",
            "",
            "### 完整性",
            f"- 期望行数: {metrics.expected_rows or 'N/A'}",
            f"- 完整率: {metrics.completeness_rate*100:.1f}%",
            "",
            "### 异常值",
            f"- 异常值数量: {metrics.outlier_count}",
            f"- 异常率: {metrics.outlier_rate*100:.2f}%",
            "",
        ]
        
        if metrics.price_anomalies:
            lines.extend([
                "#### 价格异常详情",
                "",
                "| 日期 | 价格 | 期望范围 |",
                "|------|------|----------|",
            ])
            for anomaly in metrics.price_anomalies:
                lines.append(
                    f"| {anomaly['date']} | {anomaly['price']:.2f} | "
                    f"{anomaly['expected_range'][0]:.0f}~{anomaly['expected_range'][1]:.0f} |"
                )
            lines.append("")
        
        if metrics.price_stats:
            lines.extend([
                "## 价格统计",
                "",
                f"- 最低价: {metrics.price_stats['min']:.2f}",
                f"- 最高价: {metrics.price_stats['max']:.2f}",
                f"- 平均价: {metrics.price_stats['mean']:.2f}",
                f"- 标准差: {metrics.price_stats['std']:.2f}",
                f"- 最新价: {metrics.price_stats['latest']:.2f}",
                "",
            ])
        
        if metrics.issues:
            lines.extend([
                "## ⚠️ 发现的问题",
                "",
            ])
            for issue in metrics.issues:
                lines.append(f"- {issue}")
            lines.append("")
        else:
            lines.extend([
                "## ✅ 数据质量良好",
                "",
                "未发现明显问题。",
                "",
            ])
        
        return '\n'.join(lines)
    
    def _get_level_emoji(self, level: str) -> str:
        """获取质量等级表情"""
        emojis = {
            'excellent': '🟢',
            'good': '🟡',
            'fair': '🟠',
            'poor': '🔴',
            'unknown': '⚪',
        }
        return emojis.get(level, '⚪')
    
    def save_markdown_report(
        self,
        metrics: QualityMetrics,
        output_path: str
    ):
        """保存 Markdown 报告"""
        content = self.to_markdown(metrics)
        Path(output_path).write_text(content, encoding='utf-8')
        logger.info(f"Saved quality report to {output_path}")
    
    def to_dict(self, metrics: QualityMetrics) -> Dict:
        """转换为字典"""
        return asdict(metrics)
    
    def to_json(self, metrics: QualityMetrics, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(metrics), indent=indent, ensure_ascii=False, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取所有报告的汇总"""
        if not self.reports:
            return {}
        
        total = len(self.reports)
        avg_score = sum(r.overall_score for r in self.reports) / total
        
        level_counts = {}
        for r in self.reports:
            level_counts[r.quality_level] = level_counts.get(r.quality_level, 0) + 1
        
        return {
            'total_reports': total,
            'average_score': round(avg_score, 2),
            'level_distribution': level_counts,
            'reports': [self.to_dict(r) for r in self.reports],
        }


# 便捷函数

def check_data_quality(
    data: pd.DataFrame,
    data_type: str,
    variety: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> QualityMetrics:
    """快速检查数据质量"""
    reporter = DataQualityReporter()
    return reporter.generate_report(data, data_type, variety, start_date, end_date, **kwargs)
