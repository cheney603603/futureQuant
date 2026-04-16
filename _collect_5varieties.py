# -*- coding: utf-8 -*-
"""
5品种 × 4类数据 批量采集脚本

品种: 沪铝(AL)、沪金(AU)、沪铜(CU)、甲醇(MA)、苯乙烯(SM)
数据类型: 日频行情、库存、仓单、基差

运行:
    python _collect_5varieties.py
"""

from __future__ import unicode_literals
import os
import time
import json
import tempfile

import pandas as pd

# 项目路径
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data', 'collected')
os.makedirs(DATA_DIR, exist_ok=True)

# 品种配置
VARIETIES = {
    'AL': {'name': '\u6d59\u94dd', 'exchange': 'SHFE', 'inv_name': '\u6d59\u94dd'},
    'AU': {'name': '\u6d59\u91d1', 'exchange': 'SHFE', 'inv_name': '\u6d59\u91d1'},
    'CU': {'name': '\u6d59\u94dc', 'exchange': 'SHFE', 'inv_name': '\u6d59\u94dc'},
    'MA': {'name': '\u7532\u9187', 'exchange': 'CZCE', 'inv_name': '\u7532\u9187'},
    'SM': {'name': '\u82ef\u4e59\u70ef', 'exchange': 'CZCE', 'inv_name': '\u82ef\u4e59\u70ef'},
}

START_DATE = '2023-01-03'   # 2023年第一个交易日
END_DATE   = '2024-12-31'
REPORT_START = '2024-01-02'  # 报告期起点
REPORT_END   = '2024-04-10'  # 报告期终点


# =====================================================================
# 数据拉取
# =====================================================================

def collect_daily(variety_code: str, sd: str, ed: str) -> pd.DataFrame:
    """日频行情"""
    from futureQuant.data.fetcher.akshare_fetcher import AKShareFetcher
    fetcher = AKShareFetcher(delay=0.5)
    df = fetcher.fetch_daily(variety_code, start_date=sd, end_date=ed)
    if not df.empty:
        df['variety'] = variety_code
    return df


def collect_inventory(name_cn: str, sd: str, ed: str) -> pd.DataFrame:
    """库存数据 (东方财富)"""
    import akshare as ak
    df = ak.futures_inventory_em(symbol=name_cn)
    # 列名是中文: 日期, 品种, 库存, 变化
    col_map = {}
    for c in df.columns:
        cs = c.strip()
        if cs in ('\u65e5\u671f', 'date'):
            col_map[c] = 'date'
        elif cs in ('\u54c1\u79cd', 'symbol', 'variety'):
            col_map[c] = 'variety'
        elif cs in ('\u5e93\u5b58', 'inventory', 'stock'):
            col_map[c] = 'inventory'
        elif cs in ('\u53d8\u5316', 'change', '\u589e\u51cf\u91cf'):
            col_map[c] = 'change'
    df = df.rename(columns=col_map)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[(df['date'] >= sd) & (df['date'] <= ed)]
    if 'inventory' in df.columns:
        df['inventory'] = pd.to_numeric(df['inventory'], errors='coerce')
    if 'change' in df.columns:
        df['change'] = pd.to_numeric(df['change'], errors='coerce')
    return df.dropna(subset=['date']).reset_index(drop=True)


def collect_shfe_warehouse_receipt(sd: str, ed: str) -> pd.DataFrame:
    """SHFE 仓单 (上海期货交易所)"""
    import akshare as ak
    all_records = []
    dates = pd.date_range(sd, ed, freq='QE')  # 每季度取一次（仓单数据更新慢）
    for d in dates:
        date_str = d.strftime('%Y%m%d')
        try:
            ddata = ak.futures_shfe_warehouse_receipt(date=date_str)
            if isinstance(ddata, dict):
                for variety_name, sub_df in ddata.items():
                    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                        sub_df = sub_df.copy()
                        # 找到仓单数量列
                        amount_col = None
                        for c in sub_df.columns:
                            cs = str(c).upper()
                            if 'WRTAMOUNT' in cs or '\u4ed3\u5355\u6570\u91cf' in str(c) or '\u4ed3\u5355' in str(c):
                                amount_col = c
                                break
                        if amount_col is None:
                            amount_col = sub_df.columns[-1]
                        sub_df = sub_df[['VARNAME', amount_col]].copy()
                        sub_df.columns = ['warehouse', 'receipt']
                        sub_df['date'] = d.strftime('%Y-%m-%d')
                        all_records.append(sub_df)
            time.sleep(0.3)
        except Exception:
            pass
    if all_records:
        return pd.concat(all_records, ignore_index=True)
    return pd.DataFrame()


def collect_czce_warehouse_receipt(sd: str, ed: str) -> pd.DataFrame:
    """CZCE 仓单 (郑州商品交易所)"""
    import akshare as ak
    all_records = []
    dates = pd.date_range(sd, ed, freq='QE')  # 每季度取一次
    for d in dates:
        date_str = d.strftime('%Y%m%d')
        try:
            ddata = ak.futures_warehouse_receipt_czce(date=date_str)
            if isinstance(ddata, dict):
                for variety_code, sub_df in ddata.items():
                    if isinstance(sub_df, pd.DataFrame) and not sub_df.empty:
                        sub_df = sub_df.copy()
                        # 找仓单数列
                        amt_col = None
                        for c in sub_df.columns:
                            cs = str(c)
                            if '\u4ed3\u5355' in cs or '\u4ed3\u5e55\u91cf' in cs:
                                amt_col = c
                                break
                        if amt_col is None:
                            amt_col = sub_df.columns[2] if len(sub_df.columns) > 2 else sub_df.columns[-1]
                        cols = ['\u4ed3\u5e95\u5355\u4f4d', '\u54c1\u79cd', amt_col]
                        sub_df = sub_df[[c for c in cols if c in sub_df.columns]].copy()
                        sub_df.columns = ['warehouse', 'variety', 'receipt']
                        sub_df['date'] = d.strftime('%Y-%m-%d')
                        all_records.append(sub_df)
            time.sleep(0.3)
        except Exception:
            pass
    if all_records:
        return pd.concat(all_records, ignore_index=True)
    return pd.DataFrame()


def collect_basis(sd: str, ed: str) -> pd.DataFrame:
    """基差数据"""
    import akshare as ak
    all_records = []
    dates = pd.date_range(sd, ed, freq='ME')  # 每月一次
    for d in dates:
        date_str = d.strftime('%Y%m%d')
        try:
            df = ak.futures_spot_price(date=date_str)
            if not df.empty:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                all_records.append(df)
            time.sleep(0.3)
        except Exception:
            pass
    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        # 标准化列名
        rename = {
            'near_basis': 'near_basis',
            'near_basis_rate': 'near_basis_rate',
            'near_contract_price': 'near_price',
            'spot_price': 'spot_price',
        }
        return combined.dropna(subset=['date'])
    return pd.DataFrame()


# =====================================================================
# 主程序
# =====================================================================

def main():
    print('=' * 60)
    print('  5品种 x 4类数据 批量采集')
    print('  品种:', ', '.join('{} ({})'.format(v['name'], k) for k, v in VARIETIES.items()))
    print('  报告期: {} ~ {}'.format(REPORT_START, REPORT_END))
    print('=' * 60)

    # 清理旧链路库（用于本次测试）
    tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')
    tmp_path = tmp.name
    tmp.close()

    from futureQuant.agent.data_collector import (
        ReliablePathManager, PathDiscovery, DataQuerySkill
    )
    pm = ReliablePathManager(path_file=tmp_path)

    results = {}  # (variety, data_type) -> info

    for code, info in VARIETIES.items():
        print()
        print('=' * 50)
        print('  {} ({})'.format(info['name'], code))
        print('=' * 50)

        # ---- 1. 日频行情 ----
        print()
        print('[1/4] 日频行情...')
        try:
            disc = PathDiscovery(path_manager=pm)
            r = disc.discover(data_type='daily', symbol=code,
                             start_date=REPORT_START, end_date=REPORT_END)
            if r.success and r.data is not None:
                df = r.data.copy()
                out_path = os.path.join(DATA_DIR, 'daily', '{}_daily_{}_{}.parquet'.format(
                    code, REPORT_START.replace('-',''), REPORT_END.replace('-','')))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                df.to_parquet(out_path, index=False)
                print('    OK: {} 条 -> {}'.format(len(df), out_path))
                results[(code, 'daily')] = {'ok': True, 'records': len(df), 'path': out_path}
            else:
                print('    FAIL:', r.message)
                results[(code, 'daily')] = {'ok': False, 'msg': r.message}
        except Exception as e:
            print('    EXCEPTION:', str(e)[:80])
            results[(code, 'daily')] = {'ok': False, 'msg': str(e)}

        time.sleep(1)

        # ---- 2. 库存 ----
        print('[2/4] 库存数据...')
        try:
            df_inv = collect_inventory(info['inv_name'], REPORT_START, REPORT_END)
            if not df_inv.empty:
                out_path = os.path.join(DATA_DIR, 'inventory', '{}_inventory_{}_{}.csv'.format(
                    code, REPORT_START.replace('-',''), REPORT_END.replace('-','')))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                df_inv.to_csv(out_path, index=False, encoding='utf-8-sig')
                print('    OK: {} 条 -> {}'.format(len(df_inv), out_path))
                print(df_inv.tail(5).to_string())
                results[(code, 'inventory')] = {'ok': True, 'records': len(df_inv), 'path': out_path}
            else:
                print('    FAIL: empty')
                results[(code, 'inventory')] = {'ok': False, 'msg': 'empty'}
        except Exception as e:
            print('    EXCEPTION:', str(e)[:80])
            results[(code, 'inventory')] = {'ok': False, 'msg': str(e)}

        time.sleep(0.5)

        # ---- 3. 仓单 ----
        print('[3/4] 仓单数据...')
        try:
            if info['exchange'] == 'SHFE':
                df_wr = collect_shfe_warehouse_receipt(REPORT_START, REPORT_END)
                # 过滤本品种
                if not df_wr.empty:
                    # VARNAME 列包含品种名
                    target = info['name']
                    df_wr = df_wr[df_wr['warehouse'].str.contains(target, na=False)]
            else:
                df_wr = collect_czce_warehouse_receipt(REPORT_START, REPORT_END)
                if not df_wr.empty:
                    df_wr = df_wr[df_wr['variety'].str.upper() == code]
            if not df_wr.empty:
                out_path = os.path.join(DATA_DIR, 'warehouse_receipt', '{}_receipt_{}_{}.csv'.format(
                    code, REPORT_START.replace('-',''), REPORT_END.replace('-','')))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                df_wr.to_csv(out_path, index=False, encoding='utf-8-sig')
                print('    OK: {} 条 -> {}'.format(len(df_wr), out_path))
                print(df_wr.tail(5).to_string())
                results[(code, 'warehouse_receipt')] = {'ok': True, 'records': len(df_wr), 'path': out_path}
            else:
                print('    FAIL: empty')
                results[(code, 'warehouse_receipt')] = {'ok': False, 'msg': 'empty'}
        except Exception as e:
            print('    EXCEPTION:', str(e)[:80])
            results[(code, 'warehouse_receipt')] = {'ok': False, 'msg': str(e)}

        time.sleep(0.5)

        # ---- 4. 基差 ----
        print('[4/4] 基差数据...')
        try:
            df_basis = collect_basis(REPORT_START, REPORT_END)
            if not df_basis.empty:
                # 过滤本品种
                df_basis = df_basis[df_basis['symbol'].str.upper() == code]
                if not df_basis.empty:
                    cols = ['date', 'symbol', 'spot_price', 'near_contract_price',
                            'near_basis', 'near_basis_rate', 'dom_basis_rate']
                    cols = [c for c in cols if c in df_basis.columns]
                    df_basis = df_basis[cols].sort_values('date')
                    out_path = os.path.join(DATA_DIR, 'basis', '{}_basis_{}_{}.csv'.format(
                        code, REPORT_START.replace('-',''), REPORT_END.replace('-','')))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    df_basis.to_csv(out_path, index=False, encoding='utf-8-sig')
                    print('    OK: {} 条 -> {}'.format(len(df_basis), out_path))
                    print(df_basis.tail(5).to_string())
                    results[(code, 'basis')] = {'ok': True, 'records': len(df_basis), 'path': out_path}
                else:
                    print('    FAIL: no data for', code)
                    results[(code, 'basis')] = {'ok': False, 'msg': 'no data for ' + code}
            else:
                print('    FAIL: empty')
                results[(code, 'basis')] = {'ok': False, 'msg': 'empty'}
        except Exception as e:
            print('    EXCEPTION:', str(e)[:80])
            results[(code, 'basis')] = {'ok': False, 'msg': str(e)}

        time.sleep(1)

    # =================================================================
    # 汇总
    # =================================================================
    print()
    print('=' * 60)
    print('  采集结果汇总')
    print('=' * 60)
    print()
    header = '{:<6} {:<10} {:>10} {:>8} {}'.format('品种', '数据类型', '记录数', '状态', '路径')
    print(header)
    print('-' * 80)
    for code, info in VARIETIES.items():
        for dtype in ['daily', 'inventory', 'warehouse_receipt', 'basis']:
            key = (code, dtype)
            r = results.get(key, {})
            status = 'OK' if r.get('ok') else 'FAIL'
            records = r.get('records', '-')
            path = r.get('path', '')
            print('{:<6} {:<10} {:>10} {:>8} {}'.format(code, dtype, records, status, os.path.basename(path)))

    print()
    print('=' * 60)
    print('  可靠链路库')
    print('=' * 60)
    stats = pm.get_stats()
    print('  total: {}  active: {}  degraded: {}'.format(
        stats['total'], stats['active'], stats['degraded']))
    for p in pm.get_all_paths():
        print('  [{}] {} rate={:.0%} ms={:.0f} quality={:.2f}'.format(
            p.status, p.path_id, p.success_rate, p.avg_response_ms, p.quality_score))

    os.unlink(tmp_path)


if __name__ == '__main__':
    main()
