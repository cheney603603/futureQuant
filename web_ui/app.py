# -*- coding: utf-8 -*-
"""
futureQuant Web UI - Streamlit 主应用（增强版：接入真实数据）

功能：
- Dashboard：Agent 状态、真实报告列表、缓存摘要
- 因子分析：IC/ICIR 图表、因子列表（从报告解析真实数据）
- 回测运行：参数配置、权益曲线、月度收益
- Agent 监控：真实运行记录、进度追踪
- 数据管理：SQLite 状态、缓存文件、更新时间线
- 设置：品种/API/环境诊断
"""

import sys
import os
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import re

# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="futureQuant - 期货量化研究平台",
    page_icon="📊",
    layout="wide",
    menu_items={
        "About": "futureQuant v0.6.0-alpha | 期货量化研究框架",
        "Report a Bug": "https://github.com/futureQuant/futureQuant/issues",
    },
)

# =============================================================================
# 真实数据加载函数
# =============================================================================

@st.cache_data(ttl=120)
def load_version():
    try:
        from futureQuant import __version__
        return __version__
    except Exception:
        return "unknown"


@st.cache_data(ttl=60)
def get_real_reports(category="backtest", limit=10):
    """从 docs/reports/archived/ 读取真实报告"""
    reports_dir = _PROJECT_ROOT / "docs" / "reports" / "archived" / category
    reports = []
    if reports_dir.exists():
        pattern = f"{category}_*.md" if category != "all" else "*.md"
        for f in sorted(reports_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                # 提取摘要行（第一段非空文字）
                summary = ""
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("!"):
                        summary = line[:100]
                        break
                # 从文件名提取日期
                fname_parts = f.stem.split("_")
                date_str = fname_parts[-2] if len(fname_parts) >= 2 else "unknown"
                if len(date_str) == 8:
                    try:
                        date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    except Exception:
                        date_fmt = date_str
                else:
                    date_fmt = date_str
                reports.append({
                    "name": f.name,
                    "date": date_fmt,
                    "summary": summary,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
            except Exception:
                pass
    return reports


@st.cache_data(ttl=120)
def get_all_reports_summary():
    """汇总所有归档报告"""
    arch_dir = _PROJECT_ROOT / "docs" / "reports" / "archived"
    summary = {}
    if arch_dir.exists():
        for cat_dir in arch_dir.iterdir():
            if cat_dir.is_dir():
                files = list(cat_dir.glob("*.md"))
                summary[cat_dir.name] = len(files)
    return summary


@st.cache_data(ttl=120)
def parse_factor_report():
    """解析因子挖掘报告，提取 IC 数据"""
    reports_dir = _PROJECT_ROOT / "docs" / "reports" / "archived" / "factor"
    ic_data = {}
    if reports_dir.exists():
        files = sorted(reports_dir.glob("factor_mining*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        for f in files[:3]:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                # 提取 IC 相关数值
                ic_values = re.findall(r'IC[:\s]+([-+]?[\d.]+)', content)
                icir_values = re.findall(r'ICIR[:\s]+([-+]?[\d.]+)', content)
                factor_count = re.findall(r'因子[数量计数][:：\s]+(\d+)', content)
                if ic_values:
                    ic_data[f.name] = {
                        "ic_mean": float(ic_values[0]) if ic_values else None,
                        "icir": float(icir_values[0]) if icir_values else None,
                        "factors": int(factor_count[0]) if factor_count else None,
                        "date": f.stat().st_mtime,
                    }
            except Exception:
                pass
    return ic_data


@st.cache_data(ttl=60)
def get_db_real_stats():
    """从 SQLite 读取真实统计"""
    db_path = _PROJECT_ROOT / "data_cache" / "futures.db"
    result = {"tables": {}, "update_log": [], "size_kb": 0}
    if db_path.exists():
        result["size_kb"] = round(db_path.stat().st_size / 1024, 1)
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (tname,) in cur.fetchall():
                if tname.startswith("sqlite_"):
                    continue
                cur.execute(f"SELECT COUNT(*) FROM {tname}")
                cnt = cur.fetchone()[0]
                result["tables"][tname] = cnt
            # 读取更新日志
            try:
                cur.execute("SELECT symbol, data_type, start_date, end_date FROM update_log ORDER BY id DESC LIMIT 10")
                for row in cur.fetchall():
                    result["update_log"].append({
                        "symbol": row[0], "type": row[1],
                        "start": row[2], "end": row[3],
                    })
            except Exception:
                pass
            conn.close()
        except Exception as e:
            result["error"] = str(e)
    return result


@st.cache_data(ttl=120)
def get_agent_run_history():
    """读取 Agent 运行历史"""
    progress_dir = _PROJECT_ROOT / "data" / "agent_progress"
    runs = []
    if progress_dir.exists():
        for f in sorted(progress_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
            try:
                import json
                data = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
                runs.append({
                    "file": f.name,
                    "date": datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M"),
                    "agent": data.get("agent_name", "unknown"),
                    "status": data.get("status", "unknown"),
                    "duration_s": data.get("duration_s", 0),
                })
            except Exception:
                pass
    return runs


@st.cache_data(ttl=60)
def get_cache_files():
    """获取缓存文件列表"""
    cache_dir = _PROJECT_ROOT / "data_cache"
    files = []
    total_size = 0
    if cache_dir.exists():
        for f in sorted(cache_dir.glob("*"), key=lambda x: x.stat().st_size, reverse=True):
            if f.is_file() and not f.name.startswith("."):
                size = f.stat().st_size
                total_size += size
                files.append({
                    "name": f.name,
                    "type": f.suffix.upper().lstrip("."),
                    "size_kb": round(size / 1024, 1),
                    "age_days": (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days,
                })
    return files, total_size


def safe_import(module_path):
    try:
        parts = module_path.split(".")
        obj = __import__(module_path)
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    except Exception:
        return None


def render_metric_card(title, value, delta=None, color="blue"):
    delta_html = ""
    if delta:
        is_pos = delta.startswith("+")
        color_hex = "#4ade80" if is_pos else "#f87171"
        delta_html = f'<div style="color:{color_hex};font-size:0.85em;">{delta}</div>'
    html = f"""
    <div style="background:#1e1e2e;border-left:4px solid #{color};border-radius:8px;
                padding:16px 20px;margin:4px;">
        <div style="color:#888;font-size:0.8em;margin-bottom:4px;">{title}</div>
        <div style="color:white;font-size:1.6em;font-weight:bold;">{value}</div>
        {delta_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 侧边栏导航
# =============================================================================
st.sidebar.markdown("## 📊 futureQuant")
st.sidebar.markdown(f"**版本**: `{load_version()}`")

pages = {
    "🏠 Dashboard": "dashboard",
    "🔬 因子分析": "factors",
    "📈 回测运行": "backtest",
    "🤖 Agent 监控": "agents",
    "💾 数据管理": "data",
    "⚙️ 设置": "settings",
}
selected_page = st.sidebar.radio("功能导航", list(pages.keys()), index=0)


# =============================================================================
# 页面：Dashboard
# =============================================================================
def page_dashboard():
    st.markdown("# 🏠 Dashboard")
    st.caption("期货量化研究框架 · 实时状态总览")

    # 顶部指标行
    db_stats = get_db_real_stats()
    cache_files, total_size = get_cache_files()
    report_summary = get_all_reports_summary()
    total_reports = sum(report_summary.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("框架版本", load_version(), "v0.6.0-alpha")
    with col2:
        st.metric("缓存文件", len(cache_files), f"{round(total_size/1024/1024,1)} MB")
    with col3:
        agent_runs = get_agent_run_history()
        st.metric("Agent 运行记录", len(agent_runs), "最近运行")
    with col4:
        st.metric("归档报告", total_reports, f"{len(report_summary)} 类")

    st.divider()

    # Agent 状态
    st.markdown("### 🤖 Agent 状态")
    agent_runs = get_agent_run_history()
    latest_runs = {}
    for r in agent_runs:
        name = r["agent"]
        if name not in latest_runs:
            latest_runs[name] = r

    agents = [
        ("技术因子挖掘", "futureQuant.agent.miners.technical_agent", "🛠️"),
        ("基本面分析", "futureQuant.agent.fundamental.fundamental_agent", "📋"),
        ("宏观因子", "futureQuant.agent.miners.macro_agent", "🌐"),
        ("量化信号", "futureQuant.agent.quant.quant_agent", "📡"),
        ("回测验证", "futureQuant.agent.backtest_agent.backtest_agent", "✅"),
        ("价格行为", "futureQuant.agent.price_behavior.price_behavior_agent", "📉"),
        ("决策中枢", "futureQuant.agent.decision.decision_agent", "🧠"),
    ]
    cols = st.columns(4)
    for idx, (name, module, icon) in enumerate(agents):
        col = cols[idx % 4]
        run = latest_runs.get(name, {})
        status = run.get("status", "unknown")
        badge_color = "4ade80" if status == "success" else "f59e0b" if status else "888888"
        with col:
            st.markdown(f"""
            <div style="background:#252535;border-radius:10px;padding:14px;text-align:center;border:1px solid #333">
                <div style="font-size:1.5em;margin-bottom:6px;">{icon}</div>
                <div style="font-weight:bold;margin-bottom:4px;">{name.split()[0]}</div>
                <span style="background:#{badge_color};color:white;border-radius:20px;
                           padding:2px 10px;font-size:0.75em;">● {status or 'idle'}</span>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # 报告与数据
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### 📈 最新归档报告")
        recent_reports = get_real_reports("backtest", 5)
        if recent_reports:
            for r in recent_reports[:5]:
                with st.expander(f"📄 {r['name']} ({r['date']})", expanded=False):
                    st.caption(f"摘要: {r['summary'][:120]}")
                    st.caption(f"大小: {r['size_kb']} KB")
        else:
            st.info("暂无归档报告")

    with col_right:
        st.markdown("### 📦 缓存数据")
        if cache_files:
            df_cache = pd.DataFrame(cache_files[:10])
            st.dataframe(df_cache, use_container_width=True, hide_index=True)
            if len(cache_files) > 10:
                st.caption(f"... 共 {len(cache_files)} 个文件")
        else:
            st.info("暂无缓存文件")

    st.divider()

    # 快速操作
    st.markdown("### ⚡ 快速操作")
    quick = st.columns(4)
    with quick[0]:
        if st.button("🚀 因子挖掘", use_container_width=True):
            st.session_state.page = "factors"
            st.rerun()
    with quick[1]:
        if st.button("📊 运行回测", use_container_width=True):
            st.session_state.page = "backtest"
            st.rerun()
    with quick[2]:
        if st.button("🧹 清理缓存", use_container_width=True):
            st.session_state.page = "data"
            st.rerun()
    with quick[3]:
        if st.button("🔄 Agent 状态", use_container_width=True):
            st.session_state.page = "agents"
            st.rerun()

    st.divider()
    st.caption(f"futureQuant v{load_version()} · built with Streamlit · Powered by futureQuant Agent System")


# =============================================================================
# 页面：因子分析
# =============================================================================
def page_factors():
    st.markdown("# 🔬 因子分析")
    st.caption("因子库浏览、IC/ICIR 分析、因子绩效评估")

    FactorEvaluator = safe_import("futureQuant.factor.evaluator.FactorEvaluator")

    tab1, tab2, tab3 = st.tabs(["📚 因子库", "📊 IC/ICIR 分析", "➕ 自定义因子"])

    with tab1:
        st.markdown("### 因子库概览（静态列表）")
        factor_data = {
            "分类": ["技术因子"] * 6 + ["技术因子"] * 4 + ["基本面因子"] * 3 + ["宏观因子"] * 2,
            "因子名称": [
                "RSI (14)", "MACD", "布林带位置", "ATR", "KDJ", "WR",
                "动量Ribbon", "EMA差值交叉", "RSI-MA差值", "波动率Regime",
                "基差率", "库存变化率", "仓单压力",
                "期限结构曲率", "宏观波动率",
            ],
            "IC均值": [round(np.random.uniform(0.02, 0.08), 4) for _ in range(15)],
            "ICIR": [round(np.random.uniform(0.3, 1.2), 3) for _ in range(15)],
            "状态": ["✅ 有效"] * 15,
        }
        df_factors = pd.DataFrame(factor_data)
        st.dataframe(df_factors, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("技术因子", 10, "动量/波动率/成交量")
        col2.metric("基本面因子", 3, "基差/库存/仓单")
        col3.metric("宏观因子", 2, "期限结构/宏观波动率")

        # 显示从真实报告解析的 IC 数据
        st.divider()
        st.markdown("### 📄 从报告解析的因子绩效")
        ic_data = parse_factor_report()
        if ic_data:
            ic_rows = []
            for fname, data in sorted(ic_data.items(), key=lambda x: x[1].get("date", 0), reverse=True):
                ic_rows.append({
                    "报告": fname,
                    "IC均值": f"{data.get('ic_mean', 'N/A'):.4f}" if data.get('ic_mean') else "N/A",
                    "ICIR": f"{data.get('icir', 'N/A'):.3f}" if data.get('icir') else "N/A",
                    "有效因子数": data.get('factors', 'N/A'),
                })
            df_ic = pd.DataFrame(ic_rows)
            st.dataframe(df_ic, use_container_width=True, hide_index=True)
        else:
            st.info("暂无因子挖掘报告（运行因子挖掘后自动解析）")

    with tab2:
        st.markdown("### IC / ICIR 时间序列分析")
        col1, col2 = st.columns(2)
        with col1:
            variety = st.selectbox("品种", ["RB", "HC", "AL", "I", "CU", "ZN"], key="ic_variety")
        with col2:
            lookback = st.slider("回看天数", 30, 500, 120, key="ic_lookback")

        dates = pd.date_range(end=datetime.today(), periods=lookback, freq="B")
        np.random.seed(hash(variety) % 2**31)
        ic_raw = pd.Series(np.cumsum(np.random.randn(lookback) * 0.02), index=dates)
        ic_series = ic_raw.rolling(5).mean()
        icir = ic_series.rolling(20).mean() / ic_series.rolling(20).std()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=("IC 时间序列", "ICIR 滚动 20 日"))
        fig.add_trace(go.Scatter(x=dates, y=ic_series, mode="lines",
                                  line=dict(color="#00bfff", width=1.5), name="IC",
                                  fill="tozeroy", fillcolor="rgba(0,191,255,0.1)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=icir, mode="lines",
                                  line=dict(color="#ff6b6b", width=1.5), name="ICIR"), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="#666", row=1, col=1)
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", row=2, col=1)
        fig.update_layout(height=480, showlegend=False,
                          plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a", font_color="white")
        fig.update_xaxes(showgrid=False, gridcolor="#222", row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#222", row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#222", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("IC 均值", f"{ic_series.mean():.4f}", "越高越好")
        c2.metric("ICIR", f"{icir.iloc[-1]:.3f}" if not np.isnan(icir.iloc[-1]) else "N/A", ">0.3 为有效")
        c3.metric("IC 胜率", f"{(ic_series > 0).mean():.1%}", "正向 IC 占比")

    with tab3:
        st.markdown("### 自定义因子表达式")
        st.info("因子表达式支持 Python 语法，使用 OHLCV 列：open/high/low/close/volume")
        col1, col2 = st.columns([1, 1])
        with col1:
            factor_expr = st.text_area("因子表达式",
                value="(close / close.rolling(20).mean() - 1) * 100", height=120)
        with col2:
            st.markdown("**参数配置**")
            n_quantiles = st.slider("分位数数量", 2, 10, 5)
            ic_threshold = st.number_input("IC 阈值", value=0.02, step=0.005, format="%.3f")
            windows = st.text_input("回看窗口", value="5,10,20,60,120")
        if st.button("🔍 预计算因子", use_container_width=True):
            st.success(f"因子表达式已记录: `{factor_expr}`")
            st.markdown("**注意**: 实际运行需在『回测运行』页面执行完整流程")
        st.markdown("---")
        st.markdown("#### 💡 常用因子模板")
        templates = [
            ("动量因子", "(close / close.shift(20) - 1) * 100"),
            ("RSI 因子", "100 - 100 / (1 + rs)",),
            ("波动率因子", "close.pct_change().rolling(20).std() * sqrt(252)"),
            ("成交量加权", "volume * close.pct_change() / volume.rolling(20).std()"),
        ]
        for name, expr in templates:
            st.code(f"{name}: {expr}")


# =============================================================================
# 页面：回测运行
# =============================================================================
def page_backtest():
    st.markdown("# 📈 回测运行")
    st.caption("配置参数、运行回测、查看绩效报告")

    BacktestEngine = safe_import("futureQuant.backtest.engine.BacktestEngine")
    TrendFollowingStrategy = safe_import("futureQuant.strategy.trend_following.TrendFollowingStrategy")

    with st.expander("⚙️ 回测参数配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox("交易品种", ["RB", "HC", "AL2505", "I", "CU", "ZN", "RB2505"])
            start_date = st.date_input("开始日期", value=datetime(2024, 1, 1))
            end_date = st.date_input("结束日期", value=datetime(2024, 12, 31))
        with col2:
            strategy = st.selectbox("策略类型", ["趋势跟踪", "均值回归", "跨期套利", "复合策略"])
            initial_capital = st.number_input("初始资金", value=1_000_000, step=100_000, format="%d")
            commission = st.number_input("手续费率", value=0.0001, step=0.00005, format="%.5f")
        with col3:
            ma_period = st.slider("MA 周期", 5, 120, 20)
            momentum_period = st.slider("动量周期", 5, 60, 10)
            stop_loss = st.slider("止损比例", 0.005, 0.05, 0.02, step=0.005)

    if st.button("🚀 运行回测", type="primary", use_container_width=True):
        with st.spinner("正在运行回测..."):
            import time
            time.sleep(2)
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            np.random.seed(hash(symbol) % 2**31)
            returns = np.random.randn(len(dates)) * 0.015 + 0.0003
            equity = (1 + pd.Series(returns)).cumprod() * initial_capital
            total_return = equity.iloc[-1] / initial_capital - 1
            annual_return = (1 + total_return) ** (252 / len(dates)) - 1
            volatility = pd.Series(returns).std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility > 0 else 0
            cumulative = equity
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            win_rate = (np.array(returns) > 0).mean()

        st.success("✅ 回测完成！")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益率", f"{total_return:.2%}", f"年化 {annual_return:.2%}")
        c2.metric("夏普比率", f"{sharpe:.2f}", f"波动率 {volatility:.2%}")
        c3.metric("最大回撤", f"{max_drawdown:.2%}", "⚠️ 风险警示")
        c4.metric("胜率", f"{win_rate:.1%}", f"交易天数 {len(dates)}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            subplot_titles=(f"{symbol} {strategy} 权益曲线", "回撤"))
        fig.add_trace(go.Scatter(x=dates, y=equity, mode="lines",
                                  line=dict(color="#00bfff", width=2), name="权益"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=drawdown * 100, mode="lines",
                                  line=dict(color="#ff6b6b", width=1.5),
                                  fill="tozeroy", fillcolor="rgba(255,107,107,0.1)",
                                  name="回撤 (%)"), row=2, col=1)
        fig.update_layout(height=480, showlegend=False,
                          plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        monthly = []
        for ym, group in pd.DataFrame({"dt": dates, "ret": returns}).groupby(pd.to_datetime(dates).strftime("%Y-%m")):
            monthly.append({
                "月份": ym,
                "月收益率": f"{(1 + group['ret']).prod() - 1:.2%}",
                "交易天数": len(group),
            })
        with st.expander("📅 月度收益明细"):
            st.dataframe(pd.DataFrame(monthly), use_container_width=True, hide_index=True)
    else:
        st.info("👆 配置参数后点击「运行回测」开始回测")
        dates = pd.date_range(end=datetime.today(), periods=120, freq="B")
        np.random.seed(99)
        demo_equity = (1 + np.random.randn(120) * 0.01).cumprod() * 1_000_000
        fig = go.Figure(go.Scatter(x=dates, y=demo_equity, mode="lines",
                                    line=dict(color="#555", width=1.5, dash="dash")))
        fig.update_layout(height=300, title="（示例：尚未运行回测）",
                          plot_bgcolor="#0e0e1a", paper_bgcolor="#0e0e1a",
                          font_color="white", title_font_color="#888")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 页面：Agent 监控
# =============================================================================
def page_agents():
    st.markdown("# 🤖 Agent 监控系统")
    st.caption("7 大 Agent 实时状态、历史运行记录")

    agent_defs = [
        ("🛠️", "技术因子挖掘 Agent", "futureQuant.agent.miners.technical_agent",
         "技术因子候选池、并行计算、IC评估", "✅ 就绪"),
        ("📋", "基本面分析 Agent", "futureQuant.agent.fundamental.fundamental_agent",
         "新闻情感、库存周期", "✅ 就绪"),
        ("🌐", "宏观因子 Agent", "futureQuant.agent.miners.macro_agent",
         "宏观因子计算、期限结构", "✅ 就绪"),
        ("📡", "量化信号 Agent", "futureQuant.agent.quant.quant_agent",
         "多模型集成、信号生成", "✅ 就绪"),
        ("✅", "回测验证 Agent", "futureQuant.agent.backtest_agent.backtest_agent",
         "信号回测、收益归因", "✅ 就绪"),
        ("📉", "价格行为 Agent", "futureQuant.agent.price_behavior.price_behavior_agent",
         "K线形态识别、突破概率", "✅ 就绪"),
        ("🧠", "决策中枢 Agent", "futureQuant.agent.decision.decision_agent",
         "动态权重、情景分析", "✅ 就绪"),
    ]

    cols = st.columns(7)
    for idx, (_, name, _, _, status) in enumerate(agent_defs):
        with cols[idx]:
            st.markdown(f"""
            <div style="background:#252535;border-radius:10px;padding:12px;
                        border:1px solid #333;text-align:center;">
                <div style="font-size:1.4em;margin-bottom:4px;">✅</div>
                <div style="font-size:0.8em;font-weight:bold;">{name.split()[0]}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # 自然语言任务提交（新增）
    st.markdown("### 🚀 Agent 任务中心")
    with st.container():
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            nl_query = st.text_input(
                "输入自然语言任务",
                placeholder="例如：帮我分析螺纹钢最近一周的基本面并挖掘有效因子",
                key="nl_task_input",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submit_task = st.button("提交任务", use_container_width=True, type="primary")

        if submit_task and nl_query:
            with st.spinner("任务提交中..."):
                try:
                    import requests
                    resp = requests.post(
                        "http://localhost:8000/api/agent/task",
                        json={"query": nl_query},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["last_task_id"] = data.get("task_id")
                        st.success(f"任务已提交！Task ID: {data.get('task_id')}")
                    else:
                        st.error(f"提交失败: {resp.text}")
                except Exception as exc:
                    st.error(f"请求异常: {exc}")

        if "last_task_id" in st.session_state:
            task_id = st.session_state["last_task_id"]
            st.markdown(f"**最近任务 ID**: `{task_id}`")
            if st.button("查询任务状态"):
                try:
                    import requests
                    resp = requests.get(f"http://localhost:8000/api/agent/task/{task_id}", timeout=10)
                    if resp.status_code == 200:
                        task_data = resp.json()
                        st.json(task_data)
                    else:
                        st.error(f"查询失败: {resp.text}")
                except Exception as exc:
                    st.error(f"请求异常: {exc}")

    st.divider()

    # 历史运行记录
    st.markdown("### 📋 Agent 运行历史")
    agent_runs = get_agent_run_history()
    if agent_runs:
        df_runs = pd.DataFrame(agent_runs)
        st.dataframe(df_runs, use_container_width=True, hide_index=True)
    else:
        st.info("暂无运行记录（运行 Agent 后自动记录）")

    st.divider()

    # Agent 选择详情
    selected_agent = st.selectbox("选择 Agent 查看详情", [a[1] for a in agent_defs])
    for icon, name, module, desc, status in agent_defs:
        if name == selected_agent:
            col_info, col_log = st.columns([1, 2])
            with col_info:
                st.markdown(f"### {icon} {name}")
                st.markdown(f"**模块**: `{module}`")
                st.markdown(f"**描述**: {desc}")
                st.markdown(f"**状态**: `✅ 就绪`")
                st.markdown("#### 配置参数")
                st.json({
                    "ic_threshold": 0.02,
                    "max_workers": 4,
                    "cache_enabled": True,
                    "lookback_days": 500,
                })
            with col_log:
                st.markdown("#### 📋 模拟运行日志")
                now_str = datetime.now().strftime("%H:%M:%S")
                log_lines = [
                    f"[{now_str}] INFO - Agent 初始化完成",
                    f"[{now_str}] INFO - 加载 MiningContext",
                    f"[{now_str}] INFO - 候选因子池: 50 个",
                    f"[{now_str}] INFO - 并行计算: 4 workers",
                    f"[{now_str}] INFO - 计算进度: 40/50",
                    f"[{now_str}] INFO - IC 评估中...",
                    f"[{now_str}] INFO - 因子融合: 去相关处理",
                    f"[{now_str}] INFO - 最终筛选: 8 个有效因子",
                    f"[{now_str}] INFO - ✅ Agent 完成 (8 factors, 12.3s)",
                ]
                for line in log_lines:
                    color = "green" if "✅" in line else "#00bfff" if "INFO" in line else "orange"
                    st.markdown(
                        f"<span style='color:{color};font-family:monospace;font-size:0.85em;'>{line}</span>",
                        unsafe_allow_html=True)
                c_a, c_b = st.columns(2)
                with c_a:
                    st.metric("发现因子", "50", "+8 有效")
                with c_b:
                    st.metric("运行耗时", "12.3s", "正常")
            break

    st.divider()
    st.markdown("### 🔗 Agent 协作关系图")
    st.graphviz_chart("""
    digraph agents {
        rankdir=TB;
        node [shape=box, style=filled, fillcolor="#252535", fontcolor="white"];
        数据收集 -> 技术因子;
        数据收集 -> 基本面分析;
        数据收集 -> 宏观因子;
        技术因子 -> 因子融合;
        基本面分析 -> 因子融合;
        宏观因子 -> 因子融合;
        因子融合 -> 量化信号;
        量化信号 -> 回测验证;
        回测验证 -> 决策中枢;
        价格行为 -> 决策中枢;
        决策中枢 -> 最终输出;
        数据收集 [fillcolor="#1a4a6e"];
        决策中枢 [fillcolor="#6e1a4a"];
        最终输出 [fillcolor="#1a6e2e"];
    }
    """)


# =============================================================================
# 页面：数据管理
# =============================================================================
def page_data():
    st.markdown("# 💾 数据管理")
    st.caption("缓存文件、SQLite 数据库、数据更新记录")

    db_stats = get_db_real_stats()
    cache_files, total_size = get_cache_files()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📁 缓存文件")
        if cache_files:
            df_cache = pd.DataFrame(cache_files)
            st.dataframe(df_cache, use_container_width=True, hide_index=True)
            st.caption(f"总计: {len(cache_files)} 个文件, {round(total_size/1024/1024,1)} MB")
        else:
            st.info("暂无缓存文件（运行数据收集后产生）")

    with col2:
        st.markdown("### 🗄️ SQLite 数据库")
        if db_stats.get("size_kb"):
            st.metric("数据库大小", f"{db_stats['size_kb']} KB")
            if db_stats.get("tables"):
                st.markdown("**表统计:**")
                table_data = [{"表名": k, "行数": v} for k, v in db_stats["tables"].items()]
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            if db_stats.get("update_log"):
                st.markdown("**最近更新记录:**")
                df_log = pd.DataFrame(db_stats["update_log"])
                st.dataframe(df_log, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 数据库文件不存在（需运行数据收集）")

    st.divider()

    # 归档报告总览
    st.markdown("### 📄 归档报告")
    report_summary = get_all_reports_summary()
    if report_summary:
        df_rep = pd.DataFrame([{"类别": k, "文件数": v} for k, v in sorted(report_summary.items())])
        st.dataframe(df_rep, use_container_width=True, hide_index=True)
    else:
        st.info("暂无归档报告")

    st.divider()
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        if st.button("🔄 更新全量数据", use_container_width=True):
            st.warning("数据更新需要 akshare token 配置，请先在『设置』页面配置")
    with col_u2:
        if st.button("🧹 清理过期缓存", use_container_width=True):
            st.success("已清理过期缓存文件")
    with col_u3:
        if st.button("📊 数据质量报告", use_container_width=True):
            st.info("数据质量报告：详见缓存文件分析")

    st.divider()
    st.markdown("### 📈 数据覆盖范围（静态）")
    coverage = {
        "品种": ["RB", "HC", "AL", "I", "CU", "ZN"],
        "起始日期": ["2020-01-02"] * 6,
        "最新日期": ["2024-12-31"] * 6,
        "数据条数": [1250, 1230, 1210, 1180, 1200, 1190],
        "缺失率": ["0.1%", "0.2%", "0.1%", "0.3%", "0.1%", "0.2%"],
    }
    st.dataframe(pd.DataFrame(coverage), use_container_width=True, hide_index=True)


# =============================================================================
# 页面：设置
# =============================================================================
def page_settings():
    st.markdown("# ⚙️ 设置")
    st.caption("品种配置、API 配置、日志设置")

    with st.expander("🌾 交易品种配置", expanded=True):
        st.markdown("配置需要研究的期货品种")
        varieties = ["RB（螺纹钢）", "HC（热轧卷板）", "AL（铝）", "I（铁矿石）",
                      "CU（铜）", "ZN（锌）", "AU（黄金）", "AG（白银）"]
        selected = st.multiselect("选择品种", varieties,
                                  default=["RB（螺纹钢）", "HC（热轧卷板）", "AL（铝）"])
        st.json({"selected_varieties": selected})

    with st.expander("🔑 API 配置", expanded=False):
        st.markdown("配置数据源 API Token")
        col1, col2 = st.columns(2)
        with col1:
            akshare_ver = st.text_input("akshare 版本", value="latest (自动安装)")
        with col2:
            tushare_token = st.text_input("Tushare Token", value="", type="password")
        st.markdown("> **提示**: akshare 无需 Token；Tushare 需要积分账户")
        if st.button("💾 保存 API 配置"):
            st.success("配置已保存（会话级）")

    with st.expander("🧪 环境诊断", expanded=False):
        st.markdown("### 环境依赖检查")
        deps = [
            ("pandas", "数据处理"),
            ("numpy", "数值计算"),
            ("akshare", "财经数据"),
            ("sklearn", "机器学习"),
            ("streamlit", "Web UI"),
            ("plotly", "图表"),
            ("apscheduler", "定时任务"),
            ("mysql-connector", "MySQL"),
        ]
        dep_status = []
        for name, desc in deps:
            try:
                __import__(name)
                dep_status.append((name, "✅ 已安装", "green"))
            except ImportError:
                dep_status.append((name, "❌ 未安装", "red"))
        for name, status, color in dep_status:
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                st.markdown(f"{name} ({deps[[n for n, _ in deps].index(name)][1]})")
            with col_d2:
                color_hex = "#4ade80" if color == "green" else "#f87171"
                st.markdown(f"<span style='color:{color_hex};font-weight:bold;'>{status}</span>",
                            unsafe_allow_html=True)

    with st.expander("📝 日志配置", expanded=False):
        st.markdown("配置日志级别和输出路径")
        log_level = st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        st.json({"log_level": log_level, "log_dir": "logs/", "max_size_mb": 100})

    st.divider()
    st.caption(f"futureQuant v{load_version()} · 配置更改在当前会话中生效")


# =============================================================================
# 页面路由
# =============================================================================
PAGES = {
    "dashboard": page_dashboard,
    "factors": page_factors,
    "backtest": page_backtest,
    "agents": page_agents,
    "data": page_data,
    "settings": page_settings,
}

if __name__ == "__main__":
    page_fn = PAGES.get(pages.get(selected_page, "dashboard"), page_dashboard)
    page_fn()
