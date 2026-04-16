# futureQuant REST API

FastAPI 实现的 REST API 服务。

## 快速启动

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/factor/ic` | POST | IC/ICIR 计算 |
| `/api/factor/quantile` | POST | 分层回测 |
| `/api/factor/list` | GET | 因子列表 |
| `/api/backtest/run` | POST | 运行回测 |
| `/api/backtest/{id}` | GET | 回测结果 |
| `/api/data/varieties` | GET | 品种列表 |
| `/api/data/daily/{variety}` | GET | 日线数据 |
| `/api/calendar/trading-days` | GET | 交易日列表 |

## 示例请求

```bash
# 健康检查
curl http://localhost:8000/api/health

# 品种列表
curl http://localhost:8000/api/data/varieties

# IC 计算
curl -X POST http://localhost:8000/api/factor/ic \
  -H "Content-Type: application/json" \
  -d '{"factor_data": {"2024-01-01": [0.1, 0.2]}, "returns_data": {"2024-01-01": 0.01}}'
```

## API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
