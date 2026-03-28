"""
日志系统 - 统一的日志管理
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 保存原始levelname
        original_levelname = record.levelname
        
        # 添加颜色
        if sys.platform != 'win32' or 'ANSICON' in os.environ:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        result = super().format(record)
        record.levelname = original_levelname
        return result


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录，默认为 ./logs
        log_level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数
        
    Returns:
        配置好的logger
    """
    # 创建logger
    logger = logging.getLogger('futureQuant')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # 清除已有handler
    
    # 日志格式
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    
    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(console_format, datefmt=datefmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / 'logs'
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 按日期命名日志文件
        log_file = log_dir / f"futureQuant_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(file_format, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称，为None时返回根logger
        
    Returns:
        logger实例
    """
    if name:
        return logging.getLogger(f'futureQuant.{name}')
    return logging.getLogger('futureQuant')
