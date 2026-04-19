"""
快速数据验证测试

测试修复后的数据验证功能
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from futureQuant.core.logger import get_logger
from futureQuant.data.fetcher.akshare_fetcher import AKShareFetcher
from futureQuant.data.validator import DataValidator, validate_fetched_data

logger = get_logger('data_test')


def test_rb_data():
    """测试 RB 螺纹钢数据"""
    print("=" * 60)
    print("测试 RB 螺纹钢数据验证")
    print("=" * 60)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"\n请求日期范围: {start_date} ~ {end_date}")
    
    # 获取数据
    fetcher = AKShareFetcher()
    df = fetcher.fetch_daily('RB', start_date, end_date)
    
    if df.empty:
        print("获取数据失败!")
        return False
    
    print(f"原始数据: {len(df)} 条记录")
    
    # 使用 DataValidator 验证
    validator = DataValidator(variety='RB')
    result = validator.validate_all(
        df,
        start_date=start_date,
        end_date=end_date,
        variety='RB',
        strict=False
    )
    
    # 打印结果
    print("\n" + validator.get_validation_summary(result))
    
    return result['overall_valid']


def test_fg_data():
    """测试 FG 玻璃数据"""
    print("\n" + "=" * 60)
    print("测试 FG 玻璃数据验证")
    print("=" * 60)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"\n请求日期范围: {start_date} ~ {end_date}")
    
    # 获取数据
    fetcher = AKShareFetcher()
    df = fetcher.fetch_daily('FG', start_date, end_date)
    
    if df.empty:
        print("获取数据失败!")
        return False
    
    print(f"原始数据: {len(df)} 条记录")
    
    # 使用 DataValidator 验证
    validator = DataValidator(variety='FG')
    result = validator.validate_all(
        df,
        start_date=start_date,
        end_date=end_date,
        variety='FG',
        strict=False
    )
    
    # 打印结果
    print("\n" + validator.get_validation_summary(result))
    
    return result['overall_valid']


def test_price_range():
    """测试价格合理性检查"""
    print("\n" + "=" * 60)
    print("测试价格合理性检查")
    print("=" * 60)
    
    # 创建异常价格数据
    test_data = pd.DataFrame({
        'date': pd.date_range('2026-04-01', periods=5),
        'close': [1000, 1050, 50, 1100, 1150],  # 50 是异常值
        'open': [1000, 1040, 60, 1080, 1140],
        'high': [1020, 1060, 70, 1120, 1160],
        'low': [980, 1020, 40, 1060, 1120],
    })
    
    print("\n原始数据:")
    print(test_data)
    
    # 非严格模式
    validator = DataValidator(variety='RB')
    result_df = validator.validate_price_data(test_data, strict=False)
    
    print("\n非严格模式验证结果:")
    print(f"  警告: {validator.validation_warnings}")
    print(f"  数据行数: {len(result_df)}")
    
    # 严格模式
    validator2 = DataValidator(variety='RB')
    result_df_strict = validator2.validate_price_data(test_data, strict=True)
    
    print("\n严格模式验证结果:")
    print(f"  警告: {validator2.validation_warnings}")
    print(f"  数据行数: {len(result_df_strict)}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("数据验证测试")
    print("=" * 60)
    
    # 测试 RB 数据
    rb_ok = test_rb_data()
    
    # 测试 FG 数据
    fg_ok = test_fg_data()
    
    # 测试价格范围检查
    test_price_range()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"RB 数据验证: {'通过' if rb_ok else '失败'}")
    print(f"FG 数据验证: {'通过' if fg_ok else '失败'}")
    
    return rb_ok and fg_ok


if __name__ == '__main__':
    main()
