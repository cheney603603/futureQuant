"""
期货交易日历 - 管理期货市场的交易日、节假日

中国期货市场交易时间：
- 日盘：9:00-10:15, 10:30-11:30, 13:30-15:00
- 夜盘：21:00-23:00（部分品种到次日1:00或2:30）
"""

from datetime import datetime, timedelta
from typing import List, Optional, Union
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('data.processor.calendar')


class FuturesCalendar:
    """期货交易日历"""
    
    # 中国期货市场节假日（需要每年更新）
    # 这里提供基础框架，实际使用时可以从交易所官网获取
    DEFAULT_HOLIDAYS = [
        # 2024年节假日
        '2024-01-01',  # 元旦
        '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16',  # 春节
        '2024-04-04', '2024-04-05', '2024-04-06',  # 清明
        '2024-05-01', '2024-05-02', '2024-05-03',  # 劳动节
        '2024-06-10',  # 端午
        '2024-09-17',  # 中秋
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07',  # 国庆
        # 2025年节假日（预估）
        '2025-01-01',  # 元旦
        '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',  # 春节
    ]
    
    # 品种夜盘时间配置
    NIGHT_SESSION = {
        '21:00-23:00': ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG', 'RB', 'HC', 'BU', 'RU', 'NR'],
        '21:00-01:00': ['SC', 'LU', 'BC', 'EC'],
        '21:00-02:30': ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'AU', 'AG'],  # 部分品种夜盘延长
    }
    
    def __init__(self, holidays: Optional[List[str]] = None):
        """
        初始化交易日历
        
        Args:
            holidays: 节假日列表，为None时使用默认节假日
        """
        self.holidays = set(holidays or self.DEFAULT_HOLIDAYS)
        self.holidays = {pd.Timestamp(h) for h in self.holidays}
    
    def is_trading_day(self, date: Union[str, datetime, pd.Timestamp]) -> bool:
        """
        判断是否为交易日
        
        Args:
            date: 日期
            
        Returns:
            是否为交易日
        """
        date = pd.Timestamp(date)
        
        # 周末不是交易日
        if date.weekday() >= 5:  # 5=周六, 6=周日
            return False
        
        # 节假日不是交易日
        if date in self.holidays:
            return False
        
        return True
    
    def get_trading_days(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """
        获取日期范围内的所有交易日
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日索引
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # 生成所有日期
        all_dates = pd.date_range(start=start, end=end, freq='D')
        
        # 过滤交易日
        trading_days = [d for d in all_dates if self.is_trading_day(d)]
        
        return pd.DatetimeIndex(trading_days)
    
    def get_previous_trading_day(
        self, 
        date: Union[str, datetime],
        n: int = 1
    ) -> pd.Timestamp:
        """
        获取前N个交易日
        
        Args:
            date: 日期
            n: 往前推N个交易日
            
        Returns:
            前N个交易日
        """
        date = pd.Timestamp(date)
        
        count = 0
        current = date - timedelta(days=1)
        
        while count < n:
            if self.is_trading_day(current):
                count += 1
            if count < n:
                current -= timedelta(days=1)
        
        return current
    
    def get_next_trading_day(
        self, 
        date: Union[str, datetime],
        n: int = 1
    ) -> pd.Timestamp:
        """
        获取后N个交易日
        
        Args:
            date: 日期
            n: 往后推N个交易日
            
        Returns:
            后N个交易日
        """
        date = pd.Timestamp(date)
        
        count = 0
        current = date + timedelta(days=1)
        
        while count < n:
            if self.is_trading_day(current):
                count += 1
            if count < n:
                current += timedelta(days=1)
        
        return current
    
    def get_trading_days_between(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> int:
        """
        获取两个日期之间的交易日数量
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日数量
        """
        trading_days = self.get_trading_days(start_date, end_date)
        return len(trading_days)
    
    def has_night_session(self, variety: str) -> bool:
        """
        判断品种是否有夜盘
        
        Args:
            variety: 品种代码
            
        Returns:
            是否有夜盘
        """
        variety = variety.upper()
        for session_varieties in self.NIGHT_SESSION.values():
            if variety in session_varieties:
                return True
        return False
    
    def get_night_session_time(self, variety: str) -> Optional[str]:
        """
        获取品种夜盘时间
        
        Args:
            variety: 品种代码
            
        Returns:
            夜盘时间字符串，如 '21:00-23:00'
        """
        variety = variety.upper()
        for time_range, varieties in self.NIGHT_SESSION.items():
            if variety in varieties:
                return time_range
        return None
    
    def to_dataframe(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        生成交易日历DataFrame
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日历DataFrame
        """
        trading_days = self.get_trading_days(start_date, end_date)
        
        df = pd.DataFrame({
            'date': trading_days,
            'year': trading_days.year,
            'month': trading_days.month,
            'day': trading_days.day,
            'weekday': trading_days.weekday,
            'weekday_name': trading_days.strftime('%A'),
        })
        
        return df
    
    def add_holidays(self, holidays: List[str]):
        """添加节假日"""
        for h in holidays:
            self.holidays.add(pd.Timestamp(h))
    
    def remove_holidays(self, holidays: List[str]):
        """移除节假日"""
        for h in holidays:
            self.holidays.discard(pd.Timestamp(h))
    
    @classmethod
    def from_exchange(cls, exchange: str, year: int) -> 'FuturesCalendar':
        """
        从交易所官网获取交易日历
        
        Args:
            exchange: 交易所代码，'SHFE', 'DCE', 'CZCE', 'INE', 'GFEX', 'CFFEX'
            year: 年份
            
        Returns:
            FuturesCalendar实例
        """
        # 这里可以实现从交易所官网爬取节假日信息
        # 目前返回默认日历
        logger.info(f"Using default calendar for {exchange} {year}")
        return cls()
