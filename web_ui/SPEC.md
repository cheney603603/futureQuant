# futureQuant Web UI 规格说明

**版本**: v0.1.0  
**框架**: Streamlit + Plotly + Requests  
**目标**: 为 futureQuant 量化研究框架提供可视化操作界面

---

## 核心功能

### 1. Dashboard 首页
- 显示框架版本、Agent 状态摘要
- 近期回测绩效卡片（年化收益、夏普比率、最大回撤）
- 因子库统计（因子总数、分类分布）
- 快捷入口按钮

### 2. 因子分析
- 因子列表展示（技术/基本面/宏观分类）
- IC / ICIR 时间序列图表
- 因子绩效排名表格
- 单因子详情面板

### 3. 回测运行
- 品种/时间范围/参数配置
- 策略选择（趋势跟踪/均值回归/套利）
- 回测进度实时展示
- 绩效报告（收益曲线、最大回撤、交易统计）

### 4. Agent 监控
- 7 个 Agent 状态一览
- 实时任务日志输出
- 进度追踪面板
- 错误/警告高亮

### 5. 数据管理
- 缓存数据列表与大小
- 数据更新触发
- 数据质量报告

### 6. 设置页
- 品种配置列表
- akshare / tushare token 配置
- 日志级别
- MySQL 连接配置

---

## 技术架构

```
web_ui/
├── app.py              # 主入口，Streamlit 多页面路由
├── config_page.py      # 设置页
├── SPEC.md             # 本文件
├── requirements.txt    # Python 依赖
└── README.md           # 使用说明
```

## 数据来源

通过 `sys.path` 导入 `futureQuant` 本地包，调用：
- `DataManager` 获取数据
- `FactorEvaluator` 计算 IC/ICIR
- `BacktestEngine` 运行回测
- `MultiAgentFactorMiner` 运行因子挖掘
- Agent `run()` 方法获取状态

## 图表

- **Plotly**：交互式 K 线、收益曲线、IC 时间序列
- **Streamlit 原生**：简单指标卡片、表格
- 图表全部内联，无需额外服务器

## 部署方式

```bash
cd futureQuant/web_ui
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

访问 http://localhost:8501
