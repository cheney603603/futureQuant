"""
合约管理器 - 处理主力合约切换、连续合约合成

主力合约切换规则：
1. 持仓量最大法：每日持仓量最大的合约
2. 成交量最大法：每日成交量最大的合约
3. 固定切换法：按固定日期切换（如1月、5月、9月合约）

连续合约合成：
1. 后复权：保持最新合约价格不变，向前调整历史价格
2. 前复权：保持最早合约价格不变，向后调整历史价格
3. 不复权：直接拼接，保留跳空
"""

from typing import List, Dict, Optional, Literal
from datetime import datetime
import pandas as pd
import numpy as np

from ...core.logger import get_logger
from ...core.exceptions import ProcessingError

logger = get_logger('data.processor.contract')


class ContractManager:
    """合约管理器"""
    
    # 主力合约切换规则
    ROLLOVER_METHODS = ['volume', 'open_interest', 'fixed']
    
    # 复权方式
    ADJUST_METHODS = ['backward', 'forward', 'none']
    
    def __init__(self):
        self.contracts_data: Dict[str, pd.DataFrame] = {}
    
    def add_contract_data(self, symbol: str, df: pd.DataFrame):
        """添加合约数据"""
        self.contracts_data[symbol] = df.copy()
    
    def identify_main_contract(
        self, 
        contracts_df: pd.DataFrame,
        method: Literal['volume', 'open_interest'] = 'open_interest'
    ) -> pd.DataFrame:
        """
        识别每日主力合约
        
        Args:
            contracts_df: 包含多合约的数据，格式为 [date, symbol, volume, open_interest, close]
            method: 识别方法，'volume' 或 'open_interest'
            
        Returns:
            DataFrame with columns: [date, main_contract, volume, open_interest, close]
        """
        if method not in ['volume', 'open_interest']:
            raise ProcessingError(f"Unknown method: {method}")
        
        # 按日期分组，找出每日主力合约
        main_contracts = []
        
        for date, group in contracts_df.groupby('date'):
            if method == 'volume':
                main_row = group.loc[group['volume'].idxmax()]
            else:
                main_row = group.loc[group['open_interest'].idxmax()]
            
            main_contracts.append({
                'date': date,
                'main_contract': main_row['symbol'],
                'volume': main_row['volume'],
                'open_interest': main_row['open_interest'],
                'close': main_row['close'],
            })
        
        return pd.DataFrame(main_contracts)
    
    def create_continuous_contract(
        self,
        variety: str,
        rollover_days: int = 3,
        adjust_method: Literal['backward', 'forward', 'none'] = 'backward',
        rollover_method: Literal['volume', 'open_interest'] = 'open_interest'
    ) -> pd.DataFrame:
        """
        创建连续合约
        
        Args:
            variety: 品种代码
            rollover_days: 提前切换天数（在主力合约切换前N天开始切换）
            adjust_method: 复权方式
            rollover_method: 主力合约识别方法
            
        Returns:
            连续合约DataFrame
        """
        if not self.contracts_data:
            raise ProcessingError("No contract data available")
        
        # 1. 合并所有合约数据
        all_data = []
        for symbol, df in self.contracts_data.items():
            df = df.copy()
            df['symbol'] = symbol
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values(['date', 'symbol'])
        
        # 2. 识别每日主力合约
        main_df = self.identify_main_contract(combined, method=rollover_method)
        main_df['date'] = pd.to_datetime(main_df['date'])
        main_df = main_df.sort_values('date')
        
        # 3. 确定切换点
        main_df['prev_contract'] = main_df['main_contract'].shift(1)
        main_df['is_rollover'] = main_df['main_contract'] != main_df['prev_contract']
        
        # 4. 构建连续合约
        continuous_data = []
        current_contract = None
        
        for _, row in main_df.iterrows():
            date = row['date']
            target_contract = row['main_contract']
            
            # 获取当日该合约的数据
            contract_data = combined[
                (combined['date'] == date) & 
                (combined['symbol'] == target_contract)
            ]
            
            if not contract_data.empty:
                data = contract_data.iloc[0].to_dict()
                data['original_contract'] = target_contract
                data['is_main'] = True
                continuous_data.append(data)
                current_contract = target_contract
        
        continuous_df = pd.DataFrame(continuous_data)
        
        # 5. 复权处理
        if adjust_method != 'none':
            continuous_df = self._adjust_prices(continuous_df, method=adjust_method)
        
        continuous_df['variety'] = variety
        logger.info(f"Created continuous contract for {variety} with {len(continuous_df)} records")
        
        return continuous_df
    
    def _adjust_prices(
        self, 
        df: pd.DataFrame, 
        method: Literal['backward', 'forward'] = 'backward'
    ) -> pd.DataFrame:
        """
        价格复权
        
        Args:
            df: 连续合约数据
            method: 'backward'(后复权) 或 'forward'(前复权)
            
        Returns:
            复权后的数据
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # 找出切换点
        df['prev_contract'] = df['original_contract'].shift(1)
        rollover_points = df[df['original_contract'] != df['prev_contract']].index.tolist()
        
        if len(rollover_points) <= 1:
            return df
        
        # 计算调整因子
        adjust_factors = []
        
        for i in range(1, len(rollover_points)):
            idx = rollover_points[i]
            prev_idx = rollover_points[i - 1]
            
            # 切换前后的价格
            if idx > 0 and prev_idx < len(df):
                prev_close = df.loc[prev_idx, 'close']
                curr_close = df.loc[idx, 'close']
                
                if prev_close > 0 and curr_close > 0:
                    factor = prev_close / curr_close
                    adjust_factors.append((idx, factor))
        
        # 应用调整因子
        price_cols = ['open', 'high', 'low', 'close']
        
        if method == 'backward':
            # 后复权：从最新向前调整
            cumulative_factor = 1.0
            for idx, factor in reversed(adjust_factors):
                cumulative_factor *= factor
                mask = df.index < idx
                for col in price_cols:
                    if col in df.columns:
                        df.loc[mask, col] *= cumulative_factor
                        
        elif method == 'forward':
            # 前复权：从最早向后调整
            cumulative_factor = 1.0
            for idx, factor in adjust_factors:
                cumulative_factor *= factor
                mask = df.index >= idx
                for col in price_cols:
                    if col in df.columns:
                        df.loc[mask, col] *= cumulative_factor
        
        return df
    
    def get_contract_info(self, symbol: str) -> Dict:
        """
        解析合约信息
        
        Args:
            symbol: 合约代码，如 'RB2501'
            
        Returns:
            合约信息字典
        """
        # 提取品种和到期月份
        import re
        
        match = re.match(r'^([A-Za-z]+)(\d+)$', symbol)
        if not match:
            return {}
        
        variety = match.group(1).upper()
        year_month = match.group(2)
        
        # 解析年份和月份
        if len(year_month) == 4:
            year = int('20' + year_month[:2])
            month = int(year_month[2:])
        elif len(year_month) == 3:
            year = int('202' + year_month[0])  # 假设2020年代
            month = int(year_month[1:])
        else:
            return {'variety': variety}
        
        # 计算到期日（通常是合约月份第10个交易日）
        # 这里简化处理，实际应根据交易所规则
        expire_date = datetime(year, month, 15)
        
        return {
            'symbol': symbol,
            'variety': variety,
            'year': year,
            'month': month,
            'expire_date': expire_date.strftime('%Y-%m-%d'),
        }
    
    def get_rollover_calendar(
        self, 
        variety: str, 
        year: int,
        contract_months: List[int]
    ) -> pd.DataFrame:
        """
        生成合约切换日历
        
        Args:
            variety: 品种代码
            year: 年份
            contract_months: 合约月份列表，如 [1, 5, 9]
            
        Returns:
            切换日历DataFrame
        """
        calendar = []
        
        for month in contract_months:
            # 生成合约代码
            contract_code = f"{variety}{str(year)[2:]}{month:02d}"
            
            # 假设到期日为该月15日
            expire_date = datetime(year, month, 15)
            
            # 切换日期（到期前1个月）
            if month == 1:
                rollover_month = 12
                rollover_year = year - 1
            else:
                rollover_month = month - 1
                rollover_year = year
            
            rollover_date = datetime(rollover_year, rollover_month, 15)
            
            calendar.append({
                'contract': contract_code,
                'expire_date': expire_date,
                'rollover_date': rollover_date,
                'month': month,
            })
        
        return pd.DataFrame(calendar)
