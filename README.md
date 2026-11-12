# 物理约束嵌入LLM的地下水智能建模系统（v1.0.0）

> 通过自然语言对话构建 MODFLOW 6 水流/溶质运移模型。许可证：MIT。

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥（每人填自己的 Key）

程序调用大模型需要 API Key。**每个使用者填自己的 Key**，不要共用别人的：

```bash
# 1. 复制模板为 .env
cp .env.example .env
# 2. 用文本编辑器（记事本）打开 .env，填你自己的 Key
```

`.env` 里需要填的关键项：

| 配置项 | 说明 | 示例 |
|---|---|---|
| `LLM_API_KEY` | 你的 API 密钥（必填） | DeepSeek 的 `sk-...` |
| `LLM_MODEL` | 模型名 | `deepseek-chat` |
| `LLM_API_BASE` | API 地址（一般不用改） | `https://api.deepseek.com/v1` |

`.env.example` 里还附了 OpenAI、通义千问的填法示例，照抄改 Key 即可。

> **分发给别人用时**：让对方把 `.env.example` 复制成 `.env`，填**对方自己的** API Key，再启动程序。不要把部署方的 Key 直接写进发给别人的包里。

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
├── data/                 # 语义库与示例案例
├── scripts/              # 初始化脚本
├── docker/               # Docker 部署
├── 启动程序.bat          # Windows 一键启动
├── requirements.txt      # 依赖清单
└── LICENSE               # MIT 许可证
```

## 依赖

- Python 3.11+
- MODFLOW 6 (通过 FloPy 自动下载)
- LLM API Key (DeepSeek / OpenAI / Anthropic)
