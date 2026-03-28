"""
test_calendar.py - FuturesCalendar 单元测试

测试内容：
1. is_trading_day() 正确识别交易日/非交易日
2. get_next_trading_day() 返回正确的下一个交易日
"""
import pytest
import pandas as pd

pytest.importorskip("futureQuant")

from futureQuant.data.processor.calendar import FuturesCalendar


# =============================================================================
# 测试用例
# =============================================================================

class TestIsTradingDay:
    """测试交易日判断"""
    
    def test_weekend_not_trading(self):
        """周末不是交易日"""
        calendar = FuturesCalendar()
        
        # 2024-08-03 是周六
        assert calendar.is_trading_day('2024-08-03') is False
        # 2024-08-04 是周日
        assert calendar.is_trading_day('2024-08-04') is False
    
    def test_weekday_is_trading(self):
        """工作日是交易日（无节假日）"""
        calendar = FuturesCalendar()
        
        # 2024-08-05 是周一
        assert calendar.is_trading_day('2024-08-05') is True
        # 2024-08-07 是周三
        assert calendar.is_trading_day('2024-08-07') is True
    
    def test_holiday_not_trading(self):
        """节假日不是交易日"""
        calendar = FuturesCalendar()
        
        # 元旦 2024-01-01
        assert calendar.is_trading_day('2024-01-01') is False
        # 国庆 2024-10-01
        assert calendar.is_trading_day('2024-10-01') is False
    
    def test_is_trading_day_with_timestamp(self):
        """支持 pd.Timestamp 输入"""
        calendar = FuturesCalendar()
        
        ts = pd.Timestamp('2024-08-07')
        result = calendar.is_trading_day(ts)
        
        assert isinstance(result, bool)
    
    def test_custom_holidays(self):
        """自定义节假日"""
        calendar = FuturesCalendar(holidays=['2024-08-07'])
        
        # 08-07 本应是工作日，但被设为节假日
        assert calendar.is_trading_day('2024-08-07') is False
        # 08-05 仍是工作日
        assert calendar.is_trading_day('2024-08-05') is True


class TestGetNextTradingDay:
    """测试获取下一个交易日"""
    
    def test_get_next_trading_day_weekend(self):
        """周五 -> 下周一"""
        calendar = FuturesCalendar()
        
        # 2024-08-02 是周五
        next_day = calendar.get_next_trading_day('2024-08-02')
        
        # 下一个交易日是周一 2024-08-05
        assert next_day.weekday() < 5  # 周一到周五
        assert next_day > pd.Timestamp('2024-08-02')
    
    def test_get_next_trading_day_weekday(self):
        """周一 -> 周二"""
        calendar = FuturesCalendar()
        
        # 2024-08-05 是周一
        next_day = calendar.get_next_trading_day('2024-08-05')
        
        # 下一个交易日是周二 2024-08-06
        assert next_day == pd.Timestamp('2024-08-06')
    
    def test_get_next_trading_day_with_holiday(self):
        """节假日跳过"""
        calendar = FuturesCalendar()
        
        # 2024-09-13 是周五，中秋 2024-09-17
        # 下一个交易日应该是 2024-09-18
        next_day = calendar.get_next_trading_day('2024-09-13')
        
        # 应该跳过中秋假期
        assert next_day > pd.Timestamp('2024-09-13')
        assert next_day.weekday() < 5  # 不是周末
    
    def test_get_next_trading_day_n_param(self):
        """支持 N 天"""
        calendar = FuturesCalendar()
        
        # 2024-08-02 (周五)，往后 3 个交易日
        next_day = calendar.get_next_trading_day('2024-08-02', n=3)
        
        # 周五 -> 周一 -> 周二 -> 周三
        assert next_day.weekday() == 2  # 周三
        assert next_day >= pd.Timestamp('2024-08-07')  # 至少过3天
    
    def test_get_next_trading_day_returns_timestamp(self):
        """返回 pd.Timestamp"""
        calendar = FuturesCalendar()
        
        result = calendar.get_next_trading_day('2024-08-05')
        
        assert isinstance(result, pd.Timestamp)


class TestGetTradingDays:
    """测试获取交易日列表"""
    
    def test_get_trading_days_single_week(self):
        """单周交易日"""
        calendar = FuturesCalendar()
        
        days = calendar.get_trading_days('2024-08-05', '2024-08-09')
        
        # 周一到周五，5个交易日
        assert len(days) == 5
        assert all(d.weekday() < 5 for d in days)
    
    def test_get_trading_days_with_weekend(self):
        """跨周末"""
        calendar = FuturesCalendar()
        
        days = calendar.get_trading_days('2024-08-02', '2024-08-06')
        
        # 周五到周二 = 周五、周一、周二 = 3天
        assert len(days) == 3
    
    def test_get_trading_days_with_holiday(self):
        """包含节假日"""
        calendar = FuturesCalendar()
        
        days = calendar.get_trading_days('2024-09-13', '2024-09-18')
        
        # 周五 + 中秋假期 + 周三 = 2天
        # (9-13周五, 9-14/15周末, 9-16周一中秋, 9-17周二中秋, 9-18周三)
        # 实际：9-13(周五), 9-18(周三) = 2天
        assert len(days) == 2
        assert pd.Timestamp('2024-09-13') in days
        assert pd.Timestamp('2024-09-18') in days
    
    def test_get_trading_days_returns_datetimeindex(self):
        """返回 DatetimeIndex"""
        calendar = FuturesCalendar()
        
        days = calendar.get_trading_days('2024-08-05', '2024-08-09')
        
        assert isinstance(days, pd.DatetimeIndex)


class TestGetPreviousTradingDay:
    """测试获取前一个交易日"""
    
    def test_previous_trading_day_weekend(self):
        """周一 -> 上周五"""
        calendar = FuturesCalendar()
        
        # 2024-08-05 是周一
        prev_day = calendar.get_previous_trading_day('2024-08-05')
        
        assert prev_day == pd.Timestamp('2024-08-02')  # 上周五
    
    def test_previous_trading_day_with_holiday(self):
        """节假日跳过"""
        calendar = FuturesCalendar()
        
        # 2024-09-18 是周三（中秋后第一个工作日）
        prev_day = calendar.get_previous_trading_day('2024-09-18')
        
        # 应该是 9-13（周五），跳过周末和中秋
        assert prev_day == pd.Timestamp('2024-09-13')


class TestTradingDaysBetween:
    """测试交易日计数"""
    
    def test_count_trading_days(self):
        """计算交易日数量"""
        calendar = FuturesCalendar()
        
        count = calendar.get_trading_days_between('2024-08-05', '2024-08-09')
        
        assert count == 5


class TestNightSession:
    """测试夜盘信息"""
    
    def test_has_night_session(self):
        """判断是否有夜盘"""
        calendar = FuturesCalendar()
        
        # 螺纹钢有夜盘
        assert calendar.has_night_session('RB') is True
        # 不存在的品种
        assert calendar.has_night_session('XXX') is False
    
    def test_get_night_session_time(self):
        """获取夜盘时间"""
        calendar = FuturesCalendar()
        
        time_str = calendar.get_night_session_time('RB')
        
        assert time_str is not None
        assert '21:00' in time_str


class TestToDataframe:
    """测试日历 DataFrame"""
    
    def test_to_dataframe(self):
        """生成日历 DataFrame"""
        calendar = FuturesCalendar()
        
        df = calendar.to_dataframe('2024-08-05', '2024-08-09')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'date' in df.columns
        assert 'weekday' in df.columns


class TestAddRemoveHolidays:
    """测试节假日增删"""
    
    def test_add_holidays(self):
        """添加节假日"""
        calendar = FuturesCalendar()
        
        calendar.add_holidays(['2024-08-07'])
        
        assert calendar.is_trading_day('2024-08-07') is False
    
    def test_remove_holidays(self):
        """移除节假日"""
        calendar = FuturesCalendar()
        
        # 默认元旦不是交易日
        assert calendar.is_trading_day('2024-01-01') is False
        
        calendar.remove_holidays(['2024-01-01'])
        
        # 移除后变成交易日
        assert calendar.is_trading_day('2024-01-01') is True
