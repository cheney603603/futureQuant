"""
配置管理 - 使用Pydantic进行配置验证
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DataConfig(BaseModel):
    """数据配置"""
    cache_dir: str = "./data_cache"
    db_path: str = "./data_cache/futures.db"
    update_time: str = "20:00"  # 每日更新时间
    
    # akshare配置
    akshare_timeout: int = 30
    akshare_retry: int = 3
    
    # 爬虫配置
    crawler_delay: float = 5.0  # 请求间隔（秒）
    crawler_timeout: int = 30
    crawler_retry: int = 3


class BacktestConfig(BaseModel):
    """回测配置"""
    initial_capital: float = 1_000_000
    commission: float = 0.0001  # 万1
    slippage: float = 1.0  # 滑点跳数
    margin_rate: float = 0.1  # 保证金率
    
    # 风控参数
    max_position_ratio: float = 0.8  # 最大仓位
    max_drawdown_limit: float = 0.2  # 最大回撤限制
    
    # 交易规则
    allow_short: bool = True  # 允许做空
    trade_on_close: bool = False  # 收盘价成交（否则开盘价）


class FactorConfig(BaseModel):
    """因子配置"""
    # 标准化方法
    normalize_method: str = "zscore"  # zscore, rank, minmax
    
    # 中性化
    industry_neutral: bool = False
    market_neutral: bool = True
    
    # 评估参数
    ic_lookback: int = 20
    quantile_groups: int = 5


class ModelConfig(BaseModel):
    """模型配置"""
    # 训练参数
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 时序交叉验证
    n_splits: int = 5
    
    # 随机种子
    random_seed: int = 42


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "./logs"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config(BaseSettings):
    """全局配置"""
    
    # 项目信息
    project_name: str = "futureQuant"
    version: str = "0.1.0"
    
    # 子配置
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    factor: FactorConfig = Field(default_factory=FactorConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # 交易品种列表
    varieties: List[str] = Field(default_factory=lambda: [
        "RB", "HC", "I", "J", "JM",  # 黑色
        "CU", "AL", "ZN", "NI", "SN",  # 有色
        "TA", "MA", "PP", "L", "PVC", "EG",  # 化工
        "M", "Y", "P", "OI", "RM",  # 油脂
        "CF", "SR", "RU", "AU", "AG",  # 其他
    ])
    
    class Config:
        env_prefix = "FQ_"  # 环境变量前缀
        
    @validator('varieties', pre=True)
    def parse_varieties(cls, v):
        if isinstance(v, str):
            return v.split(',')
        return v


# 全局配置实例
_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config实例
    """
    global _config
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        _config = Config(**yaml_config)
    else:
        _config = Config()
    
    return _config


def get_config() -> Config:
    """
    获取当前配置
    
    Returns:
        Config实例
    """
    global _config
    if _config is None:
        # 尝试加载默认配置文件
        default_config = Path(__file__).parent.parent / 'config' / 'settings.yaml'
        if default_config.exists():
            _config = load_config(str(default_config))
        else:
            _config = Config()
    return _config


def reload_config(config_path: Optional[str] = None):
    """重新加载配置"""
    global _config
    _config = load_config(config_path)
