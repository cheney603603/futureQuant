# -*- coding: utf-8 -*-
"""
测试数据收集 Agent 的三个技能

测试覆盖：
1. ReliablePathManager  - 可靠链路管理
2. NLQueryParser       - 自然语言解析
3. DataQuerySkill      - 数据查询（自然语言 + JSON）
4. PathDiscovery        - 新路径探测
5. Integration          - 三技能集成

运行：
    pytest tests/unit/test_data_collector_skills.py -v
"""

from __future__ import unicode_literals

import os
import sys
import tempfile

# 确保项目路径
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
os.chdir(_root)


# ======================================================================
# Test 1: ReliablePathManager - 可靠链路管理
# ======================================================================

def test_rpm_register_path():
    from futureQuant.agent.data_collector import ReliablePathManager

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)
        path, need_confirm = pm.register_path(
            source="akshare",
            data_type="daily",
            symbol_pattern="RB*",
            params={"variety": "RB"},
            tags=["螺纹钢", "daily"],
            ask_user=True,
        )
        assert path is not None
        assert path.path_id.startswith("akshare_daily_")
        assert path.status == "untested"
        print("[OK] register path: " + path.path_id)
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


def test_rpm_confirm_success():
    from futureQuant.agent.data_collector import ReliablePathManager

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)
        path, _ = pm.register_path("akshare", "daily", "RB*", ask_user=False)
        ok = pm.confirm_path(path.path_id, success=True, response_ms=850.0, records=242)
        assert ok is True
        assert path.status == "active"
        assert path.success_count == 1
        assert path.avg_response_ms == 850.0
        assert path.success_rate == 1.0
        print("[OK] confirm success: rate=" + str(path.success_rate) + ", response=" + str(path.avg_response_ms) + "ms")
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


def test_rpm_failure_degrades():
    from futureQuant.agent.data_collector import ReliablePathManager

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)
        path, _ = pm.register_path("akshare", "minute", "AU*", ask_user=False)
        for _ in range(3):
            pm.confirm_path(path.path_id, success=False)
        assert path.failure_count == 3
        assert path.status == "degraded"
        print("[OK] degraded after 3 failures: status=" + path.status)
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


def test_rpm_sorted_by_score():
    from futureQuant.agent.data_collector import ReliablePathManager

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)
        p1, _ = pm.register_path("akshare", "daily", "RB*", ask_user=False)
        p2, _ = pm.register_path("akshare", "daily", "HC*", ask_user=False)
        pm.confirm_path(p1.path_id, success=True, response_ms=500)
        pm.confirm_path(p2.path_id, success=True, response_ms=2000)

        paths = pm.get_reliable_paths(data_type='daily', min_success_rate=0.5)
        assert len(paths) == 2
        assert paths[0].avg_response_ms <= paths[1].avg_response_ms
        print("[OK] sorted paths by score: " + str([p.avg_response_ms for p in paths]))
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


def test_rpm_stats():
    from futureQuant.agent.data_collector import ReliablePathManager

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)
        pm.register_path("akshare", "daily", "RB*", ask_user=False)
        pm.register_path("baostock", "daily", "CU*", ask_user=False)
        paths = pm.get_all_paths()
        pm.confirm_path(paths[0].path_id, success=True)

        stats = pm.get_stats()
        assert stats['total'] == 2
        assert stats['active'] == 1
        print("[OK] stats: total=" + str(stats['total']) + ", active=" + str(stats['active']))
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


# ======================================================================
# Test 2: NLQueryParser - 自然语言解析
# ======================================================================

def test_nl_parse_simple():
    """中文品种名 + 数据类型"""
    from futureQuant.agent.data_collector import NLQueryParser

    parser = NLQueryParser()
    spec = parser.parse("\u87ba\u7eb9\u94a2\u7684\u65e5\u7ebf\u6570\u636e")
    assert spec.symbols == ['RB']
    assert spec.data_type == 'daily'
    print("[OK] parse: symbols=" + str(spec.symbols) + ", dtype=" + spec.data_type)


def test_nl_parse_recent_days():
    """解析最近N天"""
    from futureQuant.agent.data_collector import NLQueryParser

    parser = NLQueryParser()
    spec = parser.parse("\u87ba\u7eb9\u94a2\u6700\u8fd130\u5929\u7684\u6570\u636e")
    assert 'RB' in spec.symbols
    assert spec.data_type == 'daily'
    assert spec.start_date is not None
    assert spec.end_date is not None
    print("[OK] recent 30 days: " + spec.start_date + " ~ " + spec.end_date)


def test_nl_parse_date_range():
    """解析显式日期范围"""
    from futureQuant.agent.data_collector import NLQueryParser

    parser = NLQueryParser()
    # 用 slash 分隔的日期范围（支持 slash / hyphen / 年月日）
    spec = parser.parse("RB 2024/01/01~2024/03/01 daily")
    assert spec.start_date == '2024-01-01'
    assert spec.end_date == '2024-03-01'
    print("[OK] date range: " + spec.start_date + " ~ " + spec.end_date)


def test_nl_parse_contract_code():
    """解析合约代码"""
    from futureQuant.agent.data_collector import NLQueryParser

    parser = NLQueryParser()
    spec = parser.parse("RB2405 \u6700\u8fd110\u5929\u7684\u65e5\u7ebf")
    # RB2405 contains RB
    assert any('RB' in s for s in spec.symbols)
    print("[OK] contract: symbols=" + str(spec.symbols))


def test_nl_parse_inventory():
    """解析库存数据类型"""
    from futureQuant.agent.data_collector import NLQueryParser

    parser = NLQueryParser()
    spec = parser.parse("\u87ba\u7eb9\u94a2\u7684\u5e93\u5b58\u6570\u636e")
    assert spec.data_type == 'inventory'
    print("[OK] inventory dtype: " + spec.data_type)


def test_nl_parse_au():
    """解析黄金"""
    from futureQuant.agent.data_collector import NLQueryParser

    parser = NLQueryParser()
    spec = parser.parse("\u9ec4\u91d1\u7684\u65e5\u7ebf\u6570\u636e")
    assert 'AU' in spec.symbols
    assert spec.data_type == 'daily'
    print("[OK] AU: symbols=" + str(spec.symbols))


# ======================================================================
# Test 3: DataQuerySkill - 数据查询（解析层）
# ======================================================================

def test_query_json_parse():
    """JSON 查询解析"""
    from futureQuant.agent.data_collector import DataQuerySkill

    skill = DataQuerySkill()
    json_query = '{"symbols": ["RB"], "data_type": "daily", "start_date": "2024-01-01", "end_date": "2024-03-01"}'
    result = skill.query(json_query)
    assert result.query_spec is not None
    assert result.query_spec.symbols == ['RB']
    assert result.query_spec.start_date == '2024-01-01'
    print("[OK] json parse: symbols=" + str(result.query_spec.symbols))


def test_query_spec_builder():
    """QuerySpec 构建"""
    from futureQuant.agent.data_collector import QuerySpec

    spec = QuerySpec(
        symbols=['RB', 'HC'],
        data_type='daily',
        start_date='2024-01-01',
        end_date='2024-03-01',
    )
    assert spec.symbols == ['RB', 'HC']
    assert spec.limit == 0
    print("[OK] QuerySpec builder: symbols=" + str(spec.symbols))


# ======================================================================
# Test 4: PathDiscovery - 新路径探测
# ======================================================================

def test_pd_fallback_plans():
    """后备方案生成"""
    from futureQuant.agent.data_collector import PathDiscovery, ReliablePathManager

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)
        pd_engine = PathDiscovery(path_manager=pm)
        plans = pd_engine._fallback_plans('daily', 'RB')
        assert len(plans) >= 1
        assert any('akshare' in p['source'] for p in plans)
        print("[OK] fallback plans: " + str([p['source'] for p in plans]))
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


def test_pd_quality_assessment():
    """质量评估"""
    from futureQuant.agent.data_collector import PathDiscovery
    import pandas as pd

    pd_engine = PathDiscovery()
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'open': [4000] * 30,
        'high': [4050] * 30,
        'low': [3950] * 30,
        'close': [4000] * 30,
        'volume': [10000] * 30,
    })
    quality = pd_engine._assess_quality(df, 'daily')
    assert 0 <= quality <= 1
    print("[OK] quality score: " + str(round(quality, 3)))


def test_pd_search_query_building():
    """搜索查询构建"""
    from futureQuant.agent.data_collector import PathDiscovery

    pd_engine = PathDiscovery()
    q = pd_engine._build_search_query('daily', 'RB')
    assert len(q) > 0
    print("[OK] search query: " + q[:50])

    q2 = pd_engine._build_search_query('inventory', None)
    # Chinese text should contain inventory-related keywords
    assert len(q2) > 0
    print("[OK] inventory query: " + q2[:50])


# ======================================================================
# Test 5: Integration - 三技能集成
# ======================================================================

def test_full_workflow():
    """完整工作流"""
    from futureQuant.agent.data_collector import (
        ReliablePathManager,
        DataQuerySkill,
    )

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        fpath = f.name

    try:
        pm = ReliablePathManager(path_file=fpath)

        # Step 1: 注册链路
        p1, _ = pm.register_path("akshare", "daily", "RB*", ask_user=False)
        p2, _ = pm.register_path("akshare", "daily", "HC*", ask_user=False)
        pm.confirm_path(p1.path_id, success=True, response_ms=620.0, records=242)
        pm.confirm_path(p2.path_id, success=True, response_ms=890.0, records=240)
        pm.register_path("baostock", "daily", "CU*", ask_user=False)  # 未测试

        # Step 2: 按评分查询
        paths = pm.get_reliable_paths(data_type='daily')
        assert len(paths) >= 1
        # 响应620ms < 890ms，所以p1排第一
        assert paths[0].path_id == p1.path_id
        print("[Step2] sorted: " + str([p.path_id for p in paths]))

        # Step 3: 自然语言解析
        skill = DataQuerySkill()
        skill._engine._pm = pm
        nl_text = "\u87ba\u7eb9\u94a2\u6700\u8fd130\u5929\u7684\u65e5\u7ebf\u6570\u636e"
        result = skill.query_nl(nl_text)
        assert result.query_spec.symbols == ['RB']
        assert result.query_spec.data_type == 'daily'
        print("[Step3] nl parsed: " + str(result.query_spec.symbols) + ", " + result.query_spec.data_type)

        # Step 4: 链路统计
        stats = pm.get_stats()
        assert stats['active'] == 2
        print("[Step4] stats: active=" + str(stats['active']) + ", rate=" + str(round(stats['avg_success_rate'], 2)))

        print("[OK] full workflow passed")
    finally:
        try:
            os.unlink(fpath)
        except Exception:
            pass


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-x"]))
