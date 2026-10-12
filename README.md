# 物理约束嵌入LLM的地下水智能建模原型系统

> 第一次用、或者只是想把它跑起来？看 [`docs/新手指南.md`](docs/新手指南.md) ——
> 那是写给**使用者**的（怎么启动、怎么填自己的密钥、怎么调接口、有哪些限制），
> 不含开发内容。下面这份是简版。

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

### 3. 启动（网页 + 接口，一个进程）

Windows 直接双击 `启动程序.bat`；命令行：

```bash
python -m app.main
```

- 网页界面：<http://127.0.0.1:7860>
- 接口文档：<http://127.0.0.1:8000/docs>

只想起网页（不起接口）：

```bash
python -m app.frontend.gradio_app
```

只想起接口（不起网页）：

```bash
set GW_WITH_UI=0
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 4. 调用接口

```bash
# 对话（密钥走请求头，不放在 body 里）
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-LLM-API-Key: sk-你的密钥" \
  -d '{"message":"单层网格 100x100x1，纵向弥散度 20 m","session_id":"me-001"}'

# 跑模型（提交即返回，用 status 轮询）
curl -X POST http://127.0.0.1:8000/api/model/run \
  -H "Content-Type: application/json" \
  -d '{"json_ir":{...},"session_id":"me-001"}'
# → {"task_id":"...","status":"queued"}

curl http://127.0.0.1:8000/api/model/status/<task_id>
curl http://127.0.0.1:8000/api/model/result/<task_id>
curl -O -J http://127.0.0.1:8000/api/model/result/<task_id>/files/auto_model/head.png
```

四个端点：`POST /api/chat`、`POST /api/model/run`、`GET /api/model/status/{id}`、
`GET /api/model/result/{id}`（外加一个 `.../files/{name}` 供下载结果文件）。

> **接口不返回可交互的模型对象**（gwf/gwt），只返回结构化摘要与文件路径 ——
> 那些对象是 flopy 的内存实例，跨不了进程。

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
