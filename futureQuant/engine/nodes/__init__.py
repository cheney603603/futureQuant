"""
因子研究节点目录和管理工具。

提供因子研究工作流中所有可用步骤的元数据，
帮助 LLM 理解每个步骤的配置格式、输入输出结构，
从而正确构建研究流水线。

新增功能：
- SmartRecommender: 智能推荐下一步操作
- ConfigTemplateGenerator: 配置模板生成器
- PipelineQualityChecker: 流水线质量检查器
"""

from .factor_catalog import get_catalog, get_details, FactorCatalog
from .pipeline_builder import (
    FactorPipelineBuilder,
    ExecutionContext,
    StepExecutorRegistry,
    bind_builder,
    execute_tool,
)
from .smart_recommender import SmartRecommender, Recommendation
from .config_template_generator import ConfigTemplateGenerator
from .pipeline_quality_checker import (
    PipelineQualityChecker,
    check_pipeline_quality,
    quick_validate_pipeline,
)

__all__ = [
    # 目录相关
    "get_catalog",
    "get_details",
    "FactorCatalog",
    # 流水线构建
    "FactorPipelineBuilder",
    "ExecutionContext",
    "StepExecutorRegistry",
    "bind_builder",
    "execute_tool",
    # 智能推荐
    "SmartRecommender",
    "Recommendation",
    # 配置模板
    "ConfigTemplateGenerator",
    # 质量检查
    "PipelineQualityChecker",
    "check_pipeline_quality",
    "quick_validate_pipeline",
]
