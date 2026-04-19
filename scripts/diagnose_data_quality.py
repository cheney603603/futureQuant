"""
数据质量诊断与修复方案

问题：数据时间戳可能不正确，导致使用错误年份的数据

本脚本用于：
1. 诊断当前数据的时间戳问题
2. 验证数据时间范围（使用新的 DataValidator）
3. 提供修复方案

Author: futureQuant Team
Date: 2026-04-19
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.data.fetcher.akshare_fetcher import AKShareFetcher
from futureQuant.data.fetcher.fundamental_fetcher import FundamentalFetcher
from futureQuant.data.validator import DataValidator

logger = get_logger('data_quality_check')


def diagnose_price_data(symbol='RB', days=30):
    """诊断价格数据时间戳问题"""
    logger.info("=" * 70)
    logger.info(f"诊断 {symbol} 价格数据时间戳")
    logger.info("=" * 70)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    logger.info(f"请求日期范围: {start_date} ~ {end_date}")
    
    try:
        fetcher = AKShareFetcher()
        df = fetcher.fetch_daily(symbol, start_date, end_date)
        
        if df.empty:
            logger.error("未获取到数据")
            return None
        
        # 使用新的 DataValidator 进行验证
        variety = symbol[:2].upper()
        validator = DataValidator(variety=variety)
        
        # 完整验证
        result = validator.validate_all(
            df,
            start_date=start_date,
            end_date=end_date,
            variety=variety,
            strict=False
        )
        
        # 输出验证摘要
        summary = validator.get_validation_summary(result)
        logger.info("\n" + summary)
        
        return result.get('cleaned_df', df)
        
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def diagnose_fundamental_data(variety='RB'):
    """诊断基本面数据时间戳问题"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"诊断 {variety} 基本面数据时间戳")
    logger.info("=" * 70)
    
    try:
        fetcher = FundamentalFetcher()
        
        # 诊断库存数据
        logger.info("\n[库存数据]")
        inv_df = fetcher.fetch_inventory(variety)
        
        if inv_df.empty:
            logger.warning("库存数据为空")
        else:
            if 'date' in inv_df.columns:
                min_date = inv_df['date'].min()
                max_date = inv_df['date'].max()
                logger.info(f"  日期范围: {min_date} ~ {max_date}")
                
                now = datetime.now()
                if isinstance(max_date, str):
                    max_date = pd.to_datetime(max_date)
                days_from_now = (now - max_date).days
                logger.info(f"  最新数据距今: {days_from_now} 天")
                
                if days_from_now > 30:
                    logger.warning(f"  警告: 库存数据可能过时")
        
        # 诊断基差数据
        logger.info("\n[基差数据]")
        basis_df = fetcher.fetch_basis(variety)
        
        if basis_df.empty:
            logger.warning("基差数据为空")
        else:
            if 'date' in basis_df.columns:
                min_date = basis_df['date'].min()
                max_date = basis_df['date'].max()
                logger.info(f"  日期范围: {min_date} ~ {max_date}")
                
                now = datetime.now()
                if isinstance(max_date, str):
                    max_date = pd.to_datetime(max_date)
                days_from_now = (now - max_date).days
                logger.info(f"  最新数据距今: {days_from_now} 天")
        
    except Exception as e:
        logger.error(f"基本面数据诊断失败: {e}")


def check_data_alignment(price_df, variety='RB'):
    """检查价格数据与基本面数据的时间对齐"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("检查价格数据与基本面数据的时间对齐")
    logger.info("=" * 70)
    
    if price_df is None or price_df.empty:
        logger.error("无价格数据")
        return
    
    try:
        fetcher = FundamentalFetcher()
        
        # 获取基本面数据
        inv_df = fetcher.fetch_inventory(variety)
        
        if inv_df.empty or 'date' not in inv_df.columns:
            logger.warning("无法获取库存数据日期")
            return
        
        # 价格数据日期范围
        price_min = pd.to_datetime(price_df['date'].min())
        price_max = pd.to_datetime(price_df['date'].max())
        
        # 库存数据日期范围
        inv_min = pd.to_datetime(inv_df['date'].min())
        inv_max = pd.to_datetime(inv_df['date'].max())
        
        logger.info(f"价格数据日期: {price_min.date()} ~ {price_max.date()}")
        logger.info(f"库存数据日期: {inv_min.date()} ~ {inv_max.date()}")
        
        # 检查重叠
        overlap_start = max(price_min, inv_min)
        overlap_end = min(price_max, inv_max)
        
        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            logger.info(f"OK 日期重叠: {overlap_start.date()} ~ {overlap_end.date()} ({overlap_days} 天)")
        else:
            gap_days = (overlap_start - overlap_end).days
            logger.warning(f"警告: 数据日期不重叠，间隔 {gap_days} 天")
            logger.warning(f"   这会导致基本面分析使用错误时间的数据！")
        
    except Exception as e:
        logger.error(f"对齐检查失败: {e}")


def generate_diagnosis_report():
    """生成诊断报告"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("数据质量诊断报告")
    logger.info("=" * 70)
    
    # 诊断多个品种
    varieties = ['RB', 'FG', 'CU', 'AL']
    
    results = {}
    for variety in varieties:
        logger.info(f"\n{'='*70}")
        logger.info(f"品种: {variety}")
        logger.info('='*70)
        
        # 价格数据诊断（使用 DataValidator）
        price_df = diagnose_price_data(variety, days=30)
        results[variety] = {'price': price_df}
        
        # 基本面数据诊断
        diagnose_fundamental_data(variety)
        
        # 对齐检查
        if price_df is not None:
            check_data_alignment(price_df, variety)
    
    # 总结
    logger.info("")
    logger.info("=" * 70)
    logger.info("诊断总结")
    logger.info("=" * 70)
    
    issues_found = []
    
    for variety, data in results.items():
        if data['price'] is None:
            issues_found.append(f"{variety}: 价格数据获取失败")
        elif len(data['price']) == 0:
            issues_found.append(f"{variety}: 价格数据为空")
    
    if issues_found:
        logger.warning("发现以下问题:")
        for issue in issues_found:
            logger.warning(f"  - {issue}")
    else:
        logger.info("OK 基础数据检查通过")
    
    logger.info("")
    logger.info("Phase 1 修复已完成:")
    logger.info("1. 添加 DataValidator 数据验证模块")
    logger.info("2. 修复 akshare_fetcher 日期过滤")
    logger.info("3. 添加 FG 玻璃品种映射")
    logger.info("4. 添加价格合理性检查")
    logger.info("")
    logger.info("如需进一步优化，可实施 Phase 2:")


def main():
    """主函数"""
    generate_diagnosis_report()


if __name__ == '__main__':
    main()
