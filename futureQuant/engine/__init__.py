"""
futureQuant Engine - 因子研究执行引擎

提供流水线执行、节点注册、步骤编排等核心功能。

模块结构:
- nodes: 节点目录、流水线构建器、智能推荐、配置模板、质量检查
"""

from .nodes.factor_catalog import (
    get_catalog,
    get_details,
    FactorCatalog,
)
from .nodes.pipeline_builder import (
    FactorPipelineBuilder,
    ExecutionContext,
    StepExecutorRegistry,
    bind_builder,
    execute_tool,
)
from .nodes.smart_recommender import SmartRecommender, Recommendation
from .nodes.config_template_generator import ConfigTemplateGenerator
from .nodes.pipeline_quality_checker import (
    PipelineQualityChecker,
    check_pipeline_quality,
    quick_validate_pipeline,
)

__all__ = [
    # 目录
    "get_catalog",
    "get_details",
    "FactorCatalog",
    # 流水线
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
