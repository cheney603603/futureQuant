"""
数据更新调度器 - 定时更新数据

支持：
- 定时任务调度
- 增量更新
- 更新日志记录
"""

import time
import schedule
from datetime import datetime, timedelta
from typing import List, Optional, Callable
from pathlib import Path
import threading

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.exceptions import DataError

logger = get_logger('data.storage.updater')


class DataUpdater:
    """数据更新调度器"""
    
    def __init__(self, data_manager=None):
        """
        初始化更新调度器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.config = get_config()
        self.data_manager = data_manager
        self.jobs: List[schedule.Job] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def register_daily_update(
        self, 
        varieties: Optional[List[str]] = None,
        update_time: str = "20:00",
        callback: Optional[Callable] = None
    ):
        """
        注册每日数据更新任务
        
        Args:
            varieties: 品种列表，为None时更新所有配置的品种
            update_time: 更新时间，格式 "HH:MM"
            callback: 更新完成后的回调函数
        """
        varieties = varieties or self.config.varieties
        
        def job():
            logger.info(f"Starting scheduled daily update at {datetime.now()}")
            try:
                if self.data_manager:
                    self.data_manager.update_all_data(varieties)
                
                if callback:
                    callback()
                    
                logger.info("Scheduled daily update completed")
            except Exception as e:
                logger.error(f"Scheduled daily update failed: {e}")
        
        # 解析时间
        hour, minute = map(int, update_time.split(':'))
        
        # 注册定时任务
        job_instance = schedule.every().day.at(update_time).do(job)
        self.jobs.append(job_instance)
        
        logger.info(f"Registered daily update at {update_time} for {len(varieties)} varieties")
    
    def register_custom_job(
        self, 
        job_func: Callable,
        schedule_type: str = 'daily',
        **schedule_kwargs
    ):
        """
        注册自定义更新任务
        
        Args:
            job_func: 任务函数
            schedule_type: 调度类型，'daily', 'hourly', 'weekly'
            **schedule_kwargs: 调度参数
        """
        if schedule_type == 'daily':
            time_str = schedule_kwargs.get('at', '20:00')
            job_instance = schedule.every().day.at(time_str).do(job_func)
        elif schedule_type == 'hourly':
            job_instance = schedule.every().hour.do(job_func)
        elif schedule_type == 'weekly':
            day = schedule_kwargs.get('day', 'monday')
            time_str = schedule_kwargs.get('at', '20:00')
            job_instance = getattr(schedule.every(), day).at(time_str).do(job_func)
        else:
            raise DataError(f"Unknown schedule type: {schedule_type}")
        
        self.jobs.append(job_instance)
        logger.info(f"Registered {schedule_type} job")
    
    def start(self, blocking: bool = False):
        """
        启动调度器
        
        Args:
            blocking: 是否阻塞主线程
        """
        if self._running:
            logger.warning("Updater is already running")
            return
        
        self._running = True
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Updater started in background thread")
    
    def _run_loop(self):
        """运行调度循环"""
        logger.info("Updater loop started")
        
        while self._running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"Error in updater loop: {e}")
                time.sleep(60)
        
        logger.info("Updater loop stopped")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Updater stopped")
    
    def clear_jobs(self):
        """清除所有任务"""
        schedule.clear()
        self.jobs = []
        logger.info("All jobs cleared")
    
    def run_once(self, varieties: Optional[List[str]] = None):
        """
        立即执行一次更新
        
        Args:
            varieties: 品种列表
        """
        varieties = varieties or self.config.varieties
        
        logger.info(f"Running one-time update for {len(varieties)} varieties")
        
        if self.data_manager:
            self.data_manager.update_all_data(varieties)
        else:
            logger.warning("No data manager available")
    
    def get_next_run_time(self) -> Optional[datetime]:
        """获取下次运行时间"""
        next_runs = [job.next_run for job in self.jobs if job.next_run]
        if next_runs:
            return min(next_runs)
        return None
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        return {
            'running': self._running,
            'jobs_count': len(self.jobs),
            'next_run': self.get_next_run_time(),
        }


class IncrementalUpdater:
    """增量更新器"""
    
    def __init__(self, db_manager):
        """
        初始化增量更新器
        
        Args:
            db_manager: 数据库管理器
        """
        self.db = db_manager
    
    def get_update_range(self, symbol: str) -> tuple:
        """
        获取需要更新的日期范围
        
        Args:
            symbol: 合约代码
            
        Returns:
            (start_date, end_date) 或 (None, None) 如果不需要更新
        """
        # 获取最后更新日期
        history = self.db.get_update_history(symbol=symbol, limit=1)
        
        if history.empty:
            # 从未更新过，获取全部历史
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        else:
            # 从上次的结束日期开始
            last_end = history.iloc[0]['end_date']
            start_date = (datetime.strptime(last_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 检查是否需要更新
        if start_date > end_date:
            return None, None
        
        return start_date, end_date
    
    def should_update(self, symbol: str, max_age_hours: int = 6) -> bool:
        """
        判断是否需要更新
        
        Args:
            symbol: 合约代码
            max_age_hours: 最大数据年龄（小时）
            
        Returns:
            是否需要更新
        """
        history = self.db.get_update_history(symbol=symbol, limit=1)
        
        if history.empty:
            return True
        
        last_update = history.iloc[0]['created_at']
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
        
        age = datetime.now() - last_update.replace(tzinfo=None)
        return age > timedelta(hours=max_age_hours)
