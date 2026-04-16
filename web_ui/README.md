# futureQuant Web UI

期货量化研究框架的可视化操作界面，基于 Streamlit 构建。

## 功能概览

- **Dashboard** - 框架状态总览、Agent 状态、近期回测报告
- **因子分析** - 因子库浏览、IC/ICIR 图表、自定义因子表达式
- **回测运行** - 参数配置、权益曲线、月度收益表
- **Agent 监控** - 7 大 Agent 状态、日志输出、协作关系图
- **数据管理** - 缓存文件、数据库状态、数据质量报告
- **设置** - 品种配置、API Token、环境诊断

## 安装

```bash
cd futureQuant/web_ui
pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py --server.port 8501
```

访问 http://localhost:8501

## 首次使用

1. 确保 `futureQuant` 包所在目录为 `D:\310Programm\futureQuant`
2. 安装依赖：`pip install -r requirements.txt`
3. 运行：`streamlit run app.py`
4. 在「设置」页面配置品种和数据源 API

## 项目结构

```
web_ui/
├── app.py          # Streamlit 主应用（6 个页面）
├── SPEC.md         # 功能规格说明
├── requirements.txt # Python 依赖
└── README.md       # 本文件
```

## 技术栈

- **Streamlit** - Web 框架
- **Plotly** - 交互式图表
- **Pandas** - 数据处理
- **Numpy** - 数值计算
