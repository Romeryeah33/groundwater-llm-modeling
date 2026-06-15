# 物理约束嵌入LLM的地下水智能建模原型系统

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key
```

### 3. 启动前端
```bash
python -m app.frontend.gradio_app
```

### 4. (可选) 启动后端API
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 项目结构

```
groundwater-llm-modeling/
├── app/
│   ├── frontend/        # Gradio Web界面
│   ├── api/             # FastAPI 路由
│   ├── llm_agent/       # LLM Agent — 意图解析、对话管理
│   ├── physics_engine/   # 物理约束校验引擎
│   ├── modeling/         # FloPy 自动建模
│   ├── analysis/         # 结果分析与报告生成
│   └── knowledge/        # 专业语义库
├── data/                 # 数据文件
├── tests/                # 测试
├── docs/                 # 文档
└── docker/               # Docker 部署
```

## 依赖

- Python 3.11+
- MODFLOW 6 (通过 FloPy 自动下载)
- LLM API Key (DeepSeek / OpenAI / Anthropic)
