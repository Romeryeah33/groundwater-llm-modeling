# REST 接口参考

本文是**接口的权威说明**：每个端点的入参、出参、状态码、错误语义，以及启动方式和环境变量。
面向要写程序调这套接口的人（集成方、部署方、脚本作者）；只想点界面用的，看 [`新手指南.md`](新手指南.md)。

**约定**：所有请求与响应体均为 UTF-8 编码的 JSON（下载文件的那条除外）。
下文 `http://127.0.0.1:8000` 是默认地址，实际以你的 `GW_API_HOST` / `GW_API_PORT` 为准。

---

## 一、启动

接口与网页**同进程**启动：

```bash
python -m app.main
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GW_WITH_UI` | `1` | 设 `0` 则**只起接口、不起网页**（纯后端部署用） |
| `GW_API_HOST` | `127.0.0.1` | 接口监听地址 |
| `GW_API_PORT` | `8000` | 接口监听端口 |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | 网页监听地址（`GW_WITH_UI=0` 时忽略） |
| `GRADIO_PORT` | `7860` | 网页监听端口（同上） |

只想跑接口：

```bash
GW_WITH_UI=0 python -m app.main
```

**健康检查**（不需要任何认证）：

```bash
curl http://127.0.0.1:8000/health          # → {"status":"ok"}
curl http://127.0.0.1:8000/               # → 名称/版本/当前 LLM 模型
```

---

## 二、认证

**默认完全开放，不需要任何账号。** 设计前提是"每人自持自己的 LLM 密钥"，账号体系解决不了这个问题，只会增加部署负担。

如果部署方想加一道门，设环境变量 `API_TOKEN` 即启用：

```bash
API_TOKEN=my-secret python -m app.main
```

启用后，**每个请求**都要带请求头 `X-API-Token: my-secret`，否则返回 `401`。不设该变量时这个闸门整体关闭。

---

## 三、LLM 密钥的传递

**密钥走请求头，不放在请求体里。** 请求体常被日志、网关、错误上报顺手记下来；请求头默认不会。

| 请求头 | 必填 | 说明 |
| --- | --- | --- |
| `X-LLM-API-Key` | 否 | 该会话使用的密钥。**带上就为该会话启用**，不带则用服务器 `.env` 里的默认密钥 |
| `X-LLM-Model` | 否 | 模型名，如 `deepseek-chat`、`gpt-4o` |
| `X-LLM-API-Base` | 否 | 兼容 OpenAI 协议的服务地址，如 `https://api.deepseek.com/v1` |

**密钥的生命周期**：只存在服务器内存里，**不写磁盘、不进日志**。服务器重启即失效（客户端重新带一次即可）。同一个 `session_id` 只需在第一次请求时带。

---

## 四、会话编号 `session_id`

**所有需要上下文的接口都要求显式传 `session_id`。** 同一个编号共享同一份对话上下文与参数（JSON IR）。

- 允许的字符：`A-Za-z0-9_-`，**仅此四类**。
- 含其他字符 → **返回 `400`，不会替你改写**。

> 为什么是报错而不是像界面那样默默删掉非法字符：界面的编号是系统自己分配的，删几个字符不影响谁是谁；接口的编号**由调用方给**，静默改写会制造碰撞 —— `../etc` 和 `/etc` 都会变成 `etc`，两个客户端会共用一份会话，而它们自己完全不知道发生了什么。

首次调用可以随便编一个，之后一直用同一个即可。会话与网页界面**共用同一份注册表**，所以网页里聊出来的参数，接口里可以接着用（反过来也一样）。

---

## 五、端点总览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/chat` | 发一句话，拿回回复 + 当前参数 + 还缺什么 |
| `POST` | `/api/model/run` | 提交建模任务，**立即**返回 `task_id` |
| `GET` | `/api/model/status/{task_id}` | 查任务状态 |
| `GET` | `/api/model/result/{task_id}` | 取回结果摘要 + 结果文件清单 |
| `GET` | `/api/model/result/{task_id}/files/{path}` | 下载某个结果文件 |

---

### 5.1 `POST /api/chat` — 对话

与网页对话框走**同一个** `DialogueManager`，因此行为完全一致。

**请求体**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `message` | string | 是 | 用户自然语言输入 |
| `session_id` | string | 是 | 会话编号，见第四节 |

**响应体**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `reply` | string | 系统的自然语言回复 |
| `session_id` | string | 回显（已消毒，正常情况下与传入一致） |
| `json_ir` | object | **当前累积的建模参数**，即标准 JSON IR。可直接喂给 `/api/model/run` |
| `missing_params` | object | 还缺哪些参数，形如 `{"priority_1": [...], "priority_2": [...], "priority_3": [...]}` |
| `phase` | string | 当前对话阶段 |
| `ready_to_run` | bool | 是否已满足建模的最低条件 |

**注意**：这个端点是**同步阻塞**的（LLM 调用本身就是慢的，一秒到几十秒）。超时由 `LLM_REQUEST_TIMEOUT` 控制，默认 `120` 秒。

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-LLM-API-Key: sk-your-key" \
  -d '{"message":"模拟一个 1000×800 米的承压含水层，渗透系数 10 m/d，西侧水头 100 米、东侧 95 米","session_id":"demo-001"}'
```

---

### 5.2 `POST /api/model/run` — 提交建模

**立即返回，不同步等待。** 建模要跑几十分钟，同步返回会被反向代理掐断（nginx 默认 `proxy_read_timeout` 远短于求解器超时）。

**请求体**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `json_ir` | object | 是 | 建模参数，非空。通常直接取 `/api/chat` 返回的 `json_ir` |
| `session_id` | string | 是 | 会话编号；同时决定输出目录 `data/outputs/workspaces/run_{session_id}` |

**响应体**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `task_id` | string | 任务编号，用于后续查询 |
| `status` | string | 固定为 `"queued"` |

提交前会先过**三道前置闸门**（缺失参数 → 物理校验 → 源区物化），与网页界面走的是同一个函数，判定条件不会漂移。任何一道不过，任务会直接进入 `failed`，原因见 `/api/model/status` 的 `stage` 与 `error`。

> ⚠️ **手写 IR 的两个坑（实测踩过）**
>
> 1. **参数必须是"有值"的裸形式。** 缺失检查对**不带 `value` 键的 dict 一律判为缺失** ——
>    规范的 `"stress_periods": {"nper":1, "period_data":[…]}` 会被判成"缺 `stress_periods`"，
>    而顶层放裸整数 `"nper": 1` 反而通过。所以**手写 IR 时把 `nper` 放顶层**，
>    完整的时间信息放进 `transport.stress_periods`（mapper 有回退合并分支）。
> 2. **`domain.x_range` / `domain.y_range` 是必填**（P1），
>    `initial_conditions` 是必填（P2）；后者写 `{"type":"top_elevation"}` 即可，
>    只有显式给 `head` 键时才会覆盖初始水头。
>
> 仓库里 `data/cases/*.json` 是**规范写法**（过不了闸）；`data/cases/*.ui.json` 是
> **过闸兼容版**（`scripts/export_cases.py` 导出，物理设置完全相同）。要拿现成 IR 试接口，
> 用 `*.ui.json` 那份。

```bash
curl -X POST http://127.0.0.1:8000/api/model/run \
  -H "Content-Type: application/json" \
  -H "X-LLM-API-Key: sk-your-key" \
  -d '{"json_ir": { ... 上一步返回的 json_ir ... }, "session_id":"demo-001"}'
# → {"task_id":"3f9c2a1b8e04","status":"queued"}
```

---

### 5.3 `GET /api/model/status/{task_id}` — 查状态

**响应体**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `task_id` | string | 任务编号 |
| `status` | string | `queued` / `running` / `completed` / `failed` |
| `stage` | string \| null | 失败时：卡在哪一道闸门 |
| `error` | string \| null | 失败时：原因 |

```bash
curl http://127.0.0.1:8000/api/model/status/3f9c2a1b8e04
```

> **并发提示**：工作线程只有 **1 个**（原因见第七节）。同时提交多个任务，后面的会排队，`status` 会较长时间停在 `queued`。

---

### 5.4 `GET /api/model/result/{task_id}` — 取结果

**响应体**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `task_id` | string | 任务编号 |
| `status` | string | `"completed"` |
| `summary` | object | 结构化摘要，键见下 |
| `head_range` | array[2] \| null | 水头 `[最小值, 最大值]`（**是数组，不是对象**） |
| `reaction_effective` | object | 实际生效的反应参数 |
| `build_warnings` | array | 建模过程中的警告（如"多组分已按一级降解处理"） |
| `source_zone_rows` | array | 点源被物化成的网格行列表 |
| `boundary_warnings` | array | **定不出位置的边界清单**（CHD/GHB/RIV/DRN）。这些边界**不会写进模型**，但模型照常跑完 —— 只给警告、不阻断建模。每条是可直接展示的中文说明 |
| `files` | array | 结果文件清单，见下 |

> ⚠️ **`boundary_warnings` 非空时，结果的物理意义是不完整的。** 例如某条定水头边界
> 因名字里没有方位词、也没给 `position` 而落不下去时，模型是在**缺少该边界**的条件下
> 跑出来的 —— 可能退化成平凡解（水头=初始值）。请根据提示补上 `position`
> （`west`/`east`/`north`/`south`，或 `西`/`东`/`南`/`北`）后重跑。

`summary` 的键（实测）：

| 键 | 说明 |
| --- | --- |
| `head_min` / `head_max` / `head_mean` | 水头统计 |
| `water_balance_error` | 水均衡误差（越接近 0 越好） |
| `total_flow_in` / `total_flow_out` | 总进出流量 |
| `conc_max` / `conc_min` | 浓度统计（无溶质运移的模型不出现这两个键） |

`files[]` 每项：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | string | 文件名，如 `auto_model_gwt.ucn` |
| `path` | string | 相对工作目录的路径，如 `auto_model/auto_model_gwt.ucn`。**下载时用它** |
| `size` | int | 字节数 |

**接口只跑建模，不产图、不产报告。** 因此 `files` 里通常**只有 mf6 的原始输出**（`.ucn`）。
`.png` / `.docx` 也在白名单里，但那是给"工作目录里恰好已有这些文件"的情况留的 ——
本接口这条路径**不会生成**它们。要看图、要报告，用网页界面跑同一个 IR。
工作目录里还有一堆 mf6 中间文件（`.dis` / `.nam` / `.lst`），刻意不列。

**状态码**

| 码 | 含义 |
| --- | --- |
| `200` | 完成 |
| `404` | 未知 `task_id`，**或任务尚未完成** |
| `409` | 任务失败，响应体里带 `stage` / `error` / `detail` |

> 未完成时返回 `404` 而非"空对象"，是为了避免客户端把空对象误当成"跑完了但没结果"。

---

### 5.5 `GET /api/model/result/{task_id}/files/{path}` — 下载结果文件

`{path}` 取 `/api/model/result` 返回的 `files[].path`（**可以含斜杠**，如 `auto_model/auto_model_gwt.ucn`）。

**安全性**：不是"拼路径再判断"，而是**拿 `path` 去结果清单里比对** —— 清单里没有的一律 `404`。白名单比对从源头上没有目录穿越（`../../.env` 那种）的面。

```bash
# 文件名从 /api/model/result 的 files[].path 里取（别自己拼，拼错就是 404）
curl -O -J http://127.0.0.1:8000/api/model/result/3f9c2a1b8e04/files/auto_model/auto_model_gwt.ucn
```

---

## 六、完整调用示例

```bash
BASE=http://127.0.0.1:8000
SID=demo-001
KEY=sk-your-key

# 1) 对话，把需求说清楚
REPLY=$(curl -s -X POST $BASE/api/chat \
  -H "Content-Type: application/json" -H "X-LLM-API-Key: $KEY" \
  -d "{\"message\":\"模拟一个 1000×800 米的承压含水层，渗透系数 10 m/d，西侧水头 100、东侧 95\",\"session_id\":\"$SID\"}")

# 2) 取出 json_ir，提交建模
IR=$(echo "$REPLY" | python -c "import sys,json;print(json.dumps(json.load(sys.stdin)['json_ir'],ensure_ascii=False))")
TASK=$(curl -s -X POST $BASE/api/model/run \
  -H "Content-Type: application/json" -H "X-LLM-API-Key: $KEY" \
  -d "{\"json_ir\":$IR,\"session_id\":\"$SID\"}" | python -c "import sys,json;print(json.load(sys.stdin)['task_id'])")

# 3) 轮询到完成
while true; do
  ST=$(curl -s $BASE/api/model/status/$TASK | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "状态: $ST"
  [ "$ST" = "completed" ] && break
  [ "$ST" = "failed" ] && { curl -s $BASE/api/model/status/$TASK; exit 1; }
  sleep 10
done

# 4) 取摘要
curl -s $BASE/api/model/result/$TASK
```

---

## 七、已知限制（调之前先知道）

| 限制 | 说明 |
| --- | --- |
| **接口只建模，不产图、不产报告** | `files` 里通常只有 mf6 的 `.ucn` 原始输出。图件与 Word 报告只有网页界面那条路径会生成。要图要报告，用网页界面跑同一个 IR |
| **接口不返回可交互的模型对象** | 只返回结构化摘要 + 文件路径。模型对象（flopy 的 `gwf`/`gwt`）不可序列化，也不该跨进程传。要交互式看图，用网页界面 |
| **工作线程只有 1 个** | 图件样式与结果缓存是**进程级单例**，同时跑两个模型会互相串图。因此接口内部串行；**网页与接口同时跑模型仍有风险**，建议不要同时用两个入口 |
| **任务表不持久化** | 存在进程内存里。服务器重启后，之前提交的 `task_id` 一律失效 |
| **队列无上限** | 没有背压。同时提交很多任务会一直排队，不会拒绝 |
| **没有 WebSocket** | 对话是"一问一答"的同步 HTTP，不是流式推送 |
| **没有账号体系** | 默认开放。需要门禁就设 `API_TOKEN` |

---

## 八、相关环境变量速查

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `GW_WITH_UI` | `1` | `0` = 只起接口 |
| `GW_API_HOST` / `GW_API_PORT` | `127.0.0.1` / `8000` | 接口监听 |
| `API_TOKEN` | 未设（=关闭） | 设了就要求 `X-API-Token` |
| `LLM_REQUEST_TIMEOUT` | `120` | `/api/chat` 的单次 LLM 超时（秒） |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` / `LLM_API_BASE` | 见 `.env.example` | 服务器默认密钥（客户端不带 `X-LLM-*` 时用） |
| `GW_SESSION_SECRET` | 内置常量 | 会话编号的签名密钥。**换部署机器时务必固定**，否则服务器一重启用户会看到"对话莫名没了" |

完整清单见仓库根目录 [`.env.example`](../.env.example)。
