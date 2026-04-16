# -*- coding: utf-8 -*-
"""系统性探索遗漏的 akshare API"""
import akshare as ak
import pandas as pd
import time

def explore():
    # 1. COMEX 库存
    print('=' * 60)
    print('  COMEX Inventory')
    print('=' * 60)
    for sym in ['黄金', '白银', '铜']:
        try:
            df = ak.futures_comex_inventory(symbol=sym)
            print('[OK] {} : {} rows, {}~{}'.format(
                sym, len(df),
                str(df.iloc[:, 1].min())[:10],
                str(df.iloc[:, 1].max())[:10]))
            print('  cols:', list(df.columns))
            print(df.tail(2).to_string())
        except Exception as e:
            print('[FAIL] {} : {}'.format(sym, str(e)[:60]))
        print()

    # 2. SHFE 仓单 - 金属品种
    print('=' * 60)
    print('  SHFE Warehouse Receipt (metals)')
    print('=' * 60)
    d = ak.futures_shfe_warehouse_receipt(date='20240102')
    # 找铝、铜、锌、铅、金、锡、镍
    target_names = ['铝', '铜', '锌', '铅', '金', '锡', '镍', '螺纹']
    for key, sub_df in d.items():
        if not isinstance(sub_df, pd.DataFrame) or sub_df.empty:
            continue
        if 'VARNAME' not in sub_df.columns:
            continue
        varname = str(sub_df['VARNAME'].iloc[0])
        for t in target_names:
            if t in varname:
                print('[FOUND] key={} VARNAME={} : {} rows'.format(
                    key, varname, len(sub_df)))
                print('  cols:', list(sub_df.columns))
                print(sub_df.head(2).to_string())
                print()
                break

    # 3. futures_spot_price_daily (日频基差)
    print('=' * 60)
    print('  Daily Spot Price (basis)')
    print('=' * 60)
    try:
        df = ak.futures_spot_price_daily(start_day='2024-01-02', end_day='2024-01-10')
        print('[OK] futures_spot_price_daily:', df.shape)
        print('  cols:', list(df.columns))
        print(df.head(5).to_string())
    except Exception as e:
        print('[FAIL]:', str(e)[:80])

    # 4. futures_stock_shfe_js (SHFE 周度库存)
    print()
    print('=' * 60)
    print('  SHFE Weekly Stock (jin10)')
    print('=' * 60)
    for date in ['20240419', '20240412', '20240405', '20240329']:
        try:
            df = ak.futures_stock_shfe_js(date=date)
            if not df.empty:
                print('[OK] date={}: {} rows'.format(date, len(df)))
                print('  cols:', list(df.columns)[:8])
                print(df.head(3).to_string())
                break
            else:
                print('[EMPTY] date={}'.format(date))
        except Exception as e:
            print('[FAIL] date={}: {}'.format(date, str(e)[:40]))

    # 5. inventory_99
    print()
    print('=' * 60)
    print('  Inventory 99')
    print('=' * 60)
    for sym in ['沪铝', '沪铜', '沪金', '甲醇', '苯乙烯', '螺纹钢', '铝', '铜', '黄金']:
        try:
            df = ak.futures_inventory_99(symbol=sym)
            print('[OK] {} : {} rows'.format(sym, len(df)))
            print(df.head(2).to_string())
            break
        except Exception as e:
            print('[FAIL] {} : {}'.format(sym, str(e)[:40]))


if __name__ == '__main__':
    explore()
