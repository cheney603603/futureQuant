# -*- coding: utf-8 -*-
"""
futureQuant REST API - FastAPI 应用

功能：
- 因子分析：IC/ICIR 计算、分层回测
- 回测运行：参数配置、结果查询
- 数据查询：品种、合约、日线数据
- 日历：交易日判断
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到 path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import uuid
import json
from pathlib import Path as FilePath

# =============================================================================
# FastAPI 应用初始化
# =============================================================================
app = FastAPI(
    title="futureQuant API",
    description="期货量化研究框架 - REST API",
    version="0.6.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pydantic 模型定义
# =============================================================================

class FactorICRequest(BaseModel):
    """IC 计算请求"""
    factor_data: Dict[str, List[float]] = Field(description="因子数据，key为日期")
    returns_data: Dict[str, float] = Field(description="收益率数据，key为日期")
    method: str = Field(default="spearman", description="计算方法: spearman/pearson")


class QuantileBacktestRequest(BaseModel):
    """分层回测请求"""
    factor_data: Dict[str, List[float]] = Field(description="因子数据")
    returns_data: Dict[str, float] = Field(description="收益率数据")
    n_quantiles: int = Field(default=5, ge=2, le=10, description="分位数数量")
    long_short: bool = Field(default=True, description="是否计算多空组合")


class BacktestRequest(BaseModel):
    """回测请求"""
    symbol: str = Field(description="交易品种")
    start_date: str = Field(description="开始日期 YYYY-MM-DD")
    end_date: str = Field(description="结束日期 YYYY-MM-DD")
    strategy_type: str = Field(default="trend_following", description="策略类型")
    initial_capital: float = Field(default=1_000_000, description="初始资金")
    commission: float = Field(default=0.0001, description="手续费率")
    ma_period: int = Field(default=20, description="MA 周期")
    stop_loss: float = Field(default=0.02, description="止损比例")


class BacktestResponse(BaseModel):
    """回测响应"""
    backtest_id: str
    status: str
    message: str


class ICResponse(BaseModel):
    """IC 计算响应"""
    ic_mean: float
    ic_std: float
    icir: float
    ic_win_rate: float
    n_samples: int


class QuantileResponse(BaseModel):
    """分层回测响应"""
    quantile_returns: Dict[str, Any]
    n_periods: int


class VarietyInfo(BaseModel):
    """品种信息"""
    code: str
    name: str
    exchange: str


class ContractInfo(BaseModel):
    """合约信息"""
    symbol: str
    variety: str
    start_date: str
    end_date: str


class CalendarResponse(BaseModel):
    """日历响应"""
    dates: List[str]
    count: int


# =============================================================================
# 辅助函数
# =============================================================================

def safe_import(module_path: str):
    """安全导入模块"""
    try:
        parts = module_path.split(".")
        obj = __import__(module_path)
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    except Exception as e:
        print(f"Import error: {module_path} - {e}")
        return None


def dict_to_series(data: Dict) -> pd.Series:
    """字典转 Series"""
    return pd.Series(list(data.values()), index=list(data.keys()))


# =============================================================================
# 因子分析 API
# =============================================================================

@app.post("/api/factor/ic", response_model=ICResponse)
async def calculate_ic(request: FactorICRequest):
    """计算 IC / ICIR"""
    try:
        FactorEvaluator = safe_import("futureQuant.factor.evaluator.FactorEvaluator")
        if not FactorEvaluator:
            raise HTTPException(status_code=500, detail="FactorEvaluator 模块加载失败")

        # 转换输入数据
        factor_df = pd.DataFrame({"factor": dict_to_series(request.factor_data)})
        returns = pd.Series(
            list(request.returns_data.values()),
            index=list(request.returns_data.keys()),
            name="returns"
        )

        # 计算 IC
        evaluator = FactorEvaluator()
        ic_series = evaluator.calculate_ic(factor_df, returns, request.method)
        ic_stats = evaluator.calculate_icir(ic_series)

        return ICResponse(
            ic_mean=ic_stats["ic_mean"],
            ic_std=ic_stats["ic_std"],
            icir=ic_stats["annual_icir"],
            ic_win_rate=ic_stats["ic_win_rate"],
            n_samples=ic_stats["n_samples"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IC 计算失败: {str(e)}")


@app.post("/api/factor/quantile", response_model=QuantileResponse)
async def quantile_backtest(request: QuantileBacktestRequest):
    """分层回测"""
    try:
        FactorEvaluator = safe_import("futureQuant.factor.evaluator.FactorEvaluator")
        if not FactorEvaluator:
            raise HTTPException(status_code=500, detail="FactorEvaluator 模块加载失败")

        # 转换输入数据
        factor_df = pd.DataFrame({"factor": dict_to_series(request.factor_data)})
        returns = pd.Series(
            list(request.returns_data.values()),
            index=list(request.returns_data.keys()),
            name="returns"
        )

        # 执行分层回测
        evaluator = FactorEvaluator()
        result = evaluator.quantile_backtest(
            factor_df, returns,
            n_quantiles=request.n_quantiles,
            long_short=request.long_short
        )

        return QuantileResponse(
            quantile_returns=result.to_dict(),
            n_periods=len(result),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分层回测失败: {str(e)}")


@app.get("/api/factor/list")
async def list_factors():
    """因子列表"""
    # 返回预定义因子
    factors = {
        "技术因子": [
            {"name": "RSI_14", "desc": "14日相对强弱指标"},
            {"name": "MACD", "desc": "指数移动平均线差离值"},
            {"name": "BOLL", "desc": "布林带位置"},
            {"name": "ATR", "desc": "平均真实波幅"},
            {"name": "KDJ", "desc": "随机指标"},
            {"name": "WR", "desc": "威廉指标"},
            {"name": "Momentum_20", "desc": "20日动量"},
            {"name": "EMA_diff", "desc": "EMA差值交叉"},
            {"name": "Volatility_20", "desc": "20日波动率"},
        ],
        "基本面因子": [
            {"name": "Basis_rate", "desc": "基差率"},
            {"name": "Inventory_change", "desc": "库存变化率"},
            {"name": "Warehouse_pressure", "desc": "仓单压力"},
        ],
        "宏观因子": [
            {"name": "Term_structure", "desc": "期限结构曲率"},
            {"name": "Macro_volatility", "desc": "宏观波动率"},
        ],
    }
    return {"factors": factors, "total": sum(len(v) for v in factors.values())}


# =============================================================================
# 回测 API
# =============================================================================

# 回测结果存储（内存，生产环境用数据库）
_backtest_results: Dict[str, Dict] = {}


@app.post("/api/backtest/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """运行回测"""
    backtest_id = str(uuid.uuid4())[:8]
    
    # 返回立即响应，回测在后台运行
    _backtest_results[backtest_id] = {
        "status": "running",
        "request": request.dict(),
        "started_at": datetime.now().isoformat(),
    }
    
    # 后台执行回测
    background_tasks.add_task(execute_backtest, backtest_id, request)
    
    return BacktestResponse(
        backtest_id=backtest_id,
        status="running",
        message="回测已启动，请通过 /api/backtest/{id} 查询结果"
    )


def execute_backtest(backtest_id: str, request: BacktestRequest):
    """执行回测（后台任务）"""
    try:
        # 模拟回测结果（生产环境调用真实 BacktestEngine）
        dates = pd.date_range(start=request.start_date, end=request.end_date, freq="B")
        np.random.seed(hash(backtest_id) % 2**31)
        returns = np.random.randn(len(dates)) * 0.015 + 0.0003
        equity = (1 + pd.Series(returns)).cumprod() * request.initial_capital
        
        total_return = equity.iloc[-1] / request.initial_capital - 1
        annual_return = (1 + total_return) ** (252 / len(dates)) - 1
        volatility = pd.Series(returns).std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        drawdown = ((equity - equity.expanding().max()) / equity.expanding().max()).min()
        win_rate = (np.array(returns) > 0).mean()
        
        _backtest_results[backtest_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": {
                "total_return": float(total_return),
                "annual_return": float(annual_return),
                "sharpe_ratio": float(sharpe),
                "max_drawdown": float(drawdown),
                "win_rate": float(win_rate),
                "n_trading_days": len(dates),
                "equity_curve": equity.to_dict(),
            }
        })
    except Exception as e:
        _backtest_results[backtest_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })


@app.get("/api/backtest/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """获取回测结果"""
    if backtest_id not in _backtest_results:
        raise HTTPException(status_code=404, detail="回测记录不存在")
    return _backtest_results[backtest_id]


@app.get("/api/backtest/history")
async def list_backtests(limit: int = 10):
    """回测历史列表"""
    items = [
        {"backtest_id": k, **v}
        for k, v in sorted(
            _backtest_results.items(),
            key=lambda x: x[1].get("started_at", ""),
            reverse=True
        )[:limit]
    ]
    return {"backtests": items, "total": len(_backtest_results)}


# =============================================================================
# 数据 API
# =============================================================================

@app.get("/api/data/varieties", response_model=List[VarietyInfo])
async def list_varieties():
    """品种列表"""
    # 预定义期货品种
    varieties = [
        {"code": "RB", "name": "螺纹钢", "exchange": "SHFE"},
        {"code": "HC", "name": "热轧卷板", "exchange": "SHFE"},
        {"code": "AL", "name": "铝", "exchange": "SHFE"},
        {"code": "I", "name": "铁矿石", "exchange": "DCE"},
        {"code": "CU", "name": "铜", "exchange": "SHFE"},
        {"code": "ZN", "name": "锌", "exchange": "SHFE"},
        {"code": "AU", "name": "黄金", "exchange": "SHFE"},
        {"code": "AG", "name": "白银", "exchange": "SHFE"},
        {"code": "J", "name": "焦炭", "exchange": "DCE"},
        {"code": "JM", "name": "焦煤", "exchange": "DCE"},
        {"code": "TA", "name": "PTA", "exchange": "CZCE"},
        {"code": "MA", "name": "甲醇", "exchange": "CZCE"},
        {"code": "RU", "name": "橡胶", "exchange": "SHFE"},
        {"code": "FU", "name": "燃料油", "exchange": "SHFE"},
    ]
    return varieties


@app.get("/api/data/contracts/{variety}", response_model=List[ContractInfo])
async def list_contracts(variety: str):
    """合约列表"""
    # 简化实现：返回主力合约
    year = datetime.now().year
    contracts = [
        {
            "symbol": f"{variety}{year}01",
            "variety": variety,
            "start_date": f"{year-1}-01-01",
            "end_date": f"{year}-12-31",
        },
        {
            "symbol": f"{variety}{year}05",
            "variety": variety,
            "start_date": f"{year-1}-01-01",
            "end_date": f"{year}-12-31",
        },
    ]
    return contracts


@app.get("/api/data/daily/{variety}")
async def get_daily_data(
    variety: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
):
    """日线数据"""
    # 简化实现：返回模拟数据
    # 生产环境应从数据库或 akshare 获取
    if not start_date:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    np.random.seed(hash(variety) % 2**31)
    
    base_price = 3000 if variety in ["RB", "HC", "I"] else 50000 if variety == "AU" else 10000
    close = base_price + np.cumsum(np.random.randn(len(dates)) * base_price * 0.02)
    
    data = []
    for i, d in enumerate(dates):
        o = close[i] * (1 + np.random.randn() * 0.005)
        h = max(close[i], o) * (1 + abs(np.random.randn()) * 0.01)
        l = min(close[i], o) * (1 - abs(np.random.randn()) * 0.01)
        v = np.random.randint(10000, 100000)
        data.append({
            "date": d.strftime("%Y-%m-%d"),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(close[i], 2),
            "volume": v,
        })
    
    return {
        "variety": variety,
        "count": len(data),
        "data": data[-limit:],
    }


# =============================================================================
# 日历 API
# =============================================================================

@app.get("/api/calendar/trading-days", response_model=CalendarResponse)
async def get_trading_days(
    start_date: str,
    end_date: str,
    exchange: str = "SHFE"
):
    """交易日列表"""
    try:
        FuturesCalendar = safe_import("futureQuant.data.processor.calendar.FuturesCalendar")
        if not FuturesCalendar:
            # 回退：使用简单实现
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            return CalendarResponse(
                dates=[d.strftime("%Y-%m-%d") for d in dates],
                count=len(dates)
            )
        
        cal = FuturesCalendar()
        dates = cal.get_trading_days(start_date, end_date, exchange)
        return CalendarResponse(
            dates=[d.strftime("%Y-%m-%d") for d in dates],
            count=len(dates)
        )
    except Exception as e:
        # 回退到简单实现
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        return CalendarResponse(
            dates=[d.strftime("%Y-%m-%d") for d in dates],
            count=len(dates)
        )


@app.get("/api/calendar/next")
async def get_next_trading_day(date: str, exchange: str = "SHFE"):
    """下一交易日"""
    try:
        FuturesCalendar = safe_import("futureQuant.data.processor.calendar.FuturesCalendar")
        cal = FuturesCalendar()
        next_date = cal.get_next_trading_day(date, exchange)
        return {"date": date, "next_trading_day": next_date}
    except Exception:
        # 回退
        d = pd.Timestamp(date) + pd.Timedelta(days=1)
        return {"date": date, "next_trading_day": d.strftime("%Y-%m-%d")}


# =============================================================================
# Agent 任务 API
# =============================================================================

_agent_tasks: Dict[str, Dict[str, Any]] = {}


class AgentTaskRequest(BaseModel):
    """Agent 任务提交请求"""
    query: str = Field(description="用户自然语言需求")


class AgentTaskResponse(BaseModel):
    """Agent 任务提交响应"""
    task_id: str
    status: str
    message: str


@app.post("/api/agent/task", response_model=AgentTaskResponse)
async def submit_agent_task(request: AgentTaskRequest, background_tasks: BackgroundTasks):
    """提交自然语言 Agent 任务"""
    task_id = str(uuid.uuid4())[:8]
    _agent_tasks[task_id] = {"status": "running", "query": request.query, "result": None}
    background_tasks.add_task(_execute_agent_task, task_id, request.query)
    return AgentTaskResponse(
        task_id=task_id,
        status="running",
        message="任务已提交，请通过 /api/agent/task/{task_id} 查询结果",
    )


def _execute_agent_task(task_id: str, query: str):
    """后台执行 Agent 任务"""
    try:
        from futureQuant.agent.orchestrator import NaturalLanguageTaskRunner
        runner = NaturalLanguageTaskRunner()
        result = runner.run(query)
        _agent_tasks[task_id]["status"] = result.get("status", "unknown")
        _agent_tasks[task_id]["result"] = result
    except Exception as exc:
        _agent_tasks[task_id]["status"] = "failed"
        _agent_tasks[task_id]["result"] = {"error": str(exc)}


@app.get("/api/agent/task/{task_id}")
async def get_agent_task(task_id: str):
    """查询 Agent 任务状态与结果"""
    if task_id not in _agent_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _agent_tasks[task_id]


@app.post("/api/agent/intervention/{request_id}")
async def respond_intervention(request_id: str, response: Dict[str, Any]):
    """响应人工介入请求（预留接口）"""
    return {"request_id": request_id, "received": True, "response": response}


# =============================================================================
# 健康检查
# =============================================================================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "0.6.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "futureQuant API",
        "version": "0.6.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
