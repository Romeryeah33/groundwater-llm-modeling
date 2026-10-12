# -*- coding: utf-8 -*-
"""REST 接口回归锁（阶段 B）。

锁七件事：

  1. 五个端点**真的挂上了**。此前 `app/api/routes.py` 写好了四个函数，
     但全仓没有一处 `include_router` —— 文档承诺的接口从来没生效过。
     所以第一件事就是断言路由存在，而不是断言函数写得对。
  2. 会话编号非法时**报 400 而不是静默改写** —— 接口的编号由调用方给，
     改写会让 `../etc` 与 `/etc` 撞成同一个会话。
  3. 任务生命周期：提交 → 轮询 → 完成，且失败时**必须落进任务表**
     （不落表的话客户端永远看到 running，那是最难查的一种卡住）。
  4. **两条路径不许漂移**：同一个 IR 走界面和走接口，到达 `build_and_run`
     的 IR 必须逐字节相同。这是本轮抽出 `run_gates.py` 的**唯一目的**，
     也是这个测试文件里最重要的一条。
  5. 结果文件的下载**只认清单里列过的文件**，不接受任意路径。
  6. 密钥经请求头传入、**不进任务表也不进返回体**。
  7. `API_TOKEN` 设了才开门禁，不设完全开放（默认按设计如此）。

脚本风格，不用 pytest：``python tests/test_api_endpoints.py``
（计时 ≈ 20 s，其中一个真实建模算例占大头）
"""
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 测试期间不要拉网页（会占端口、拖慢启动）
import os                                     # noqa: E402
os.environ.setdefault("GW_WITH_UI", "0")
logging.disable(logging.WARNING)

from fastapi.testclient import TestClient     # noqa: E402

from app.main import app                      # noqa: E402
from app.api import routes as routes_mod      # noqa: E402
from app.llm_agent import session_registry     # noqa: E402

_passed = 0
_SENTINEL = "sk-API-SENTINEL-must-not-leak-0123456789"

_PROBE_SIDS = ["__api_a__", "__api_b__", "__drift_ui__", "__drift_api__"]


def _ok(msg):
    global _passed
    _passed += 1
    print(f"   OK: {msg}")


def _cleanup():
    for sid in _PROBE_SIDS:
        session_registry.drop(sid)
        f = ROOT / "data" / "sessions" / f"{sid}.json"
        if f.exists():
            f.unlink()


client = TestClient(app)


# ================================================================
print("\n1. 五个端点真的挂上了...")
# ================================================================
_spec = app.openapi()["paths"]
_expect = {
    ("post", "/api/chat"),
    ("post", "/api/model/run"),
    ("get", "/api/model/status/{task_id}"),
    ("get", "/api/model/result/{task_id}"),
    ("get", "/api/model/result/{task_id}/files/{name}"),
}
_have = {(m, p) for p, ops in _spec.items() for m in ops}
_missing = _expect - _have
assert not _missing, f"这些端点没挂上（缺 include_router？）: {sorted(_missing)}"
_ok(f"五个端点全部注册（OpenAPI 里共 {len(_have)} 条路径）")

# 反向锁：/api 前缀确实生效，没被挂到根路径上
assert "/chat" not in _spec, "端点挂到了根路径 —— 前缀丢了"
_ok("/api 前缀生效（不是挂在根路径）")


# ================================================================
print("\n2. 会话编号：非法就报 400，不静默改写...")
# ================================================================
for bad in ["../../etc", "a/b", "..\\..\\win", ""]:
    r = client.post("/api/chat", json={"message": "hi", "session_id": bad})
    assert r.status_code == 400, f"编号 {bad!r} 应被拒，实际 {r.status_code}"
_ok("穿越型 / 空编号一律 400（不替调用方改写）")

# 关键：两个不同的非法编号不能被改写成同一个合法编号
r1 = client.post("/api/model/run",
                 json={"json_ir": {"project": {"name": "x"}}, "session_id": "../etc"})
r2 = client.post("/api/model/run",
                 json={"json_ir": {"project": {"name": "x"}}, "session_id": "/etc"})
assert r1.status_code == 400 and r2.status_code == 400
_ok("'../etc' 与 '/etc' 都被拒，不会撞成同一个会话")

r = client.post("/api/model/run", json={"json_ir": {}, "session_id": "__api_a__"})
assert r.status_code == 400, "空 IR 应被拒"
_ok("空 json_ir 被拒（不会拿一个空 IR 去建模）")


# ================================================================
print("\n3. 任务生命周期：提交 → 轮询 → 落表...")
# ================================================================
r = client.get("/api/model/status/does-not-exist")
assert r.status_code == 404
r = client.get("/api/model/result/does-not-exist")
assert r.status_code == 404
_ok("未知 task_id → 404（不是 500，也不是空对象）")

# 一个**必然失败**的提交：IR 过不了缺失参数闸
r = client.post("/api/model/run",
                json={"json_ir": {"project": {"name": "api-probe"}},
                      "session_id": "__api_a__"})
assert r.status_code == 200 and r.json()["status"] == "queued"
tid = r.json()["task_id"]
_ok(f"提交返回 task_id={tid}, status=queued（不阻塞等待）")

_st = None
for _ in range(120):
    _st = client.get(f"/api/model/status/{tid}").json()
    if _st["status"] in ("completed", "failed"):
        break
    time.sleep(0.5)
assert _st is not None and _st["status"] != "queued", \
    f"任务卡在 queued 没被线程池捡起: {_st}"
assert _st["status"] == "failed", f"缺参数的 IR 不该跑成功: {_st}"
assert _st["stage"] == "missing", f"应报被闸 1 拦下，实际 stage={_st['stage']}"
assert _st["error"], "失败必须带原因 —— 只说 failed 用户无从下手"
_ok(f"过不了闸的任务落表为 failed（stage={_st['stage']}，带原因）")

r = client.get(f"/api/model/result/{tid}")
assert r.status_code == 409, f"失败的任务取结果应 409，实际 {r.status_code}"
assert r.json()["detail"]["error"]
_ok("失败任务取结果 → 409 + 原因（不是 404「还没好」，语义不混淆）")


# ================================================================
print("\n4. 界面与接口不许漂移（本轮抽 run_gates 的唯一目的）...")
# ================================================================
from app.modeling.model_builder import AutoModelBuilder   # noqa: E402
from app.frontend import gradio_app as ga                 # noqa: E402

_case = ROOT / "data" / "cases" / "case_c_east_texas.ui.json"
assert _case.exists(), f"缺测试算例 {_case}"

_caught = []
_orig = AutoModelBuilder.build_and_run


def _spy(self, ir, sim_name="auto_model"):
    _caught.append(json.dumps(ir, sort_keys=True, ensure_ascii=False))
    raise RuntimeError("__stop__")     # 只验入参，不真跑


AutoModelBuilder.build_and_run = _spy
try:
    _ir = json.loads(_case.read_text(encoding="utf-8"))
    _ir = _ir.get("json_ir", _ir)

    # 界面路径
    try:
        ga.run_model_handler(json.loads(json.dumps(_ir)), "__drift_ui__")
    except RuntimeError as e:
        assert e.args[0] == "__stop__", f"界面路径意外异常: {e}"

    # 接口路径
    routes_mod._run_model_task("drift-probe", json.loads(json.dumps(_ir)), "__drift_api__")
finally:
    AutoModelBuilder.build_and_run = _orig

assert len(_caught) == 2, f"应捕捉到两次调用，实际 {len(_caught)}"
assert _caught[0] == _caught[1], (
    "界面与接口传给 build_and_run 的 IR 不一致 —— 两条路径各有一套闸门逻辑，"
    "判定条件已经漂移。今天只是差一个字段，明天就是「界面拦下的用例接口能跑」。")
_ok("同一 IR 走界面/接口，到达 build_and_run 时逐字节相同（含物化的源区索引）")


# ================================================================
print("\n5. 结果文件下载只认白名单...")
# ================================================================
from app.api.routes import _list_outputs                # noqa: E402

# 造一个假的任务条目。**故意放一个嵌套路径** —— 真实的结果就放在
# `auto_model/auto_model_gwt.ucn` 这种子目录里，只测顶层文件会把
# "路由参数不匹配斜杠"这个坑漏掉（曾真的漏掉过：下载 404）。
_tmp = ROOT / "data" / "outputs" / "__api_test_ws"
(_tmp / "auto_model").mkdir(parents=True, exist_ok=True)
(_tmp / "head.png").write_bytes(b"\x89PNG\r\n\x1a\n")
(_tmp / "auto_model" / "run.ucn").write_bytes(b"UCN" * 100)
(_tmp / "mfsim.lst").write_text("intermediate", encoding="utf-8")

_listed = _list_outputs(_tmp)
_names = {f["path"] for f in _listed}
assert "head.png" in _names, "png 应进清单"
assert "auto_model/run.ucn" in _names, "嵌套路径应进清单（用正斜杠）"
assert "mfsim.lst" not in _names, "mf6 中间文件不该进清单（客户端下载了也没用）"
_ok("清单只收白名单后缀，嵌套路径保留正斜杠（.lst 等被排除）")

with routes_mod._TASK_LOCK:
    routes_mod._TASKS["__api_filetest__"] = {
        "task_id": "__api_filetest__", "status": "completed",
        "workspace": str(_tmp), "files": _listed,
    }
try:
    r = client.get("/api/model/result/__api_filetest__/files/head.png")
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    _ok("清单内的顶层文件可以下载")

    # 这一条是刚才那个坑的锁：路由必须带 {name:path}，否则带斜杠的路径
    # 根本匹配不到路由，直接 404
    r = client.get("/api/model/result/__api_filetest__/files/auto_model/run.ucn")
    assert r.status_code == 200, \
        f"子目录里的文件下载不到（路由缺 {{name:path}}？）实际 {r.status_code}"
    _ok("清单内的**嵌套路径**也能下载（路由带 :path 转换器）")

    # 不在清单里的（哪怕真实存在、哪怕就在工作目录里）必须拒绝
    for evil in ["mfsim.lst", "..%2F..%2F.env", "....//....//.env",
                 "../../.env", "sub/../../.env", "auto_model/../../.env"]:
        rr = client.get(f"/api/model/result/__api_filetest__/files/{evil}")
        assert rr.status_code == 404, \
            f"越权路径 {evil!r} 不该被放行，实际 {rr.status_code}"
    _ok("清单外的路径全部 404（不是拼路径再判断，是白名单比对）")
finally:
    with routes_mod._TASK_LOCK:
        routes_mod._TASKS.pop("__api_filetest__", None)
    import shutil
    shutil.rmtree(_tmp, ignore_errors=True)


# ================================================================
print("\n6. 密钥不从请求头漏进任务表或返回体...")
# ================================================================
r = client.post("/api/chat",
                json={"message": "hi", "session_id": "__api_b__"},
                headers={"X-LLM-API-Key": _SENTINEL})
_body = r.text
assert _SENTINEL not in _body, "密钥出现在聊天接口的返回体里"
_ok("密钥不在 /api/chat 返回体里")

assert session_registry.has_custom_key("__api_b__") is True, \
    "请求头里的密钥没有生效 —— 接口收下了却没用"
_ok("请求头里的密钥确实被该会话启用（has_custom_key 为真）")

with routes_mod._TASK_LOCK:
    _dump = json.dumps(routes_mod._TASKS, ensure_ascii=False, default=str)
assert _SENTINEL not in _dump, "密钥出现在了任务表里"
_ok("密钥不在任务表里（任务表将来要落盘或回传都不怕）")

_blob = json.dumps(session_registry.get("__api_b__", restore=False).state,
                   ensure_ascii=False, default=str)
assert _SENTINEL not in _blob, "密钥进了会被 save_session 落盘的 state"
_ok("密钥不在 DialogueManager.state 里（那是唯一被落盘的东西）")


# ================================================================
print("\n7. API_TOKEN 设了才开门禁...")
# ================================================================
import importlib                                # noqa: E402

_old = os.environ.pop("API_TOKEN", None)
try:
    assert client.get("/api/model/status/nope").status_code == 404, \
        "没设 API_TOKEN 时不该拦（按设计完全开放）"
    _ok("不设 API_TOKEN → 接口开放（404 是路由自身的「未知 task」）")

    os.environ["API_TOKEN"] = "secret-token-probe"
    r = client.get("/api/model/status/nope")
    assert r.status_code == 401, f"设了 API_TOKEN 应拦未带头部者，实际 {r.status_code}"
    r = client.get("/api/model/status/nope", headers={"X-API-Token": "secret-token-probe"})
    assert r.status_code == 404, "带对了口令应放行到路由（404 才是路由自己的回答）"
    _ok("设了 API_TOKEN → 无口令 401、口令正确放行")
finally:
    os.environ.pop("API_TOKEN", None)
    if _old is not None:
        os.environ["API_TOKEN"] = _old


# ================================================================
print("\n8. 那条超时确实传下去了（否则接口线程会被永久占住）...")
# ================================================================
from app.config import settings                         # noqa: E402
from app.llm_agent.llm_client import LLMClient           # noqa: E402

_kw = LLMClient(model="deepseek/deepseek-v4-flash", api_key="sk-x")._build_kwargs()
assert _kw.get("timeout") == settings.llm_request_timeout, (
    f"litellm 没收到 timeout —— 请求会无限等待。_build_kwargs 里没有传 "
    f"settings.llm_request_timeout（{settings.llm_request_timeout}）")
_ok(f"LLM 请求带上了 timeout={settings.llm_request_timeout}s（原先传了 0 次）")


# ================================================================
_cleanup()
print("\n" + "=" * 60)
print(f"REST 接口: {_passed} 项通过")
print("=" * 60)
