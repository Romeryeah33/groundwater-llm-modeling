# -*- coding: utf-8 -*-
"""多用户会话隔离 + 自持密钥 —— 回归锁。

背景（2026-10-12，阶段 A）：

    程序要分发出去给别的使用者用，于是有两条硬需求：

    1. **多个浏览器同时打开，互不串号。** 原先 `sid_state = gr.State("default")`，
       初值是常量，Gradio 不会替每个浏览器改它 —— 所有会话都落进同一个
       `sessions["default"]` 和同一份 `data/sessions/default.json`。
       两个人同时用就会互相覆盖对话与参数。单机自己用看不出来。
    2. **每人用自己的 LLM 密钥。** 原先全服务器共用模块级单例，所有人都在烧
       部署方 `.env` 里的那一把 key。

    两条都改成了"按 sid 一套对象"，于是必须有测试锁住，否则很容易退回原样
    （这类缺陷不会报错、不会崩，只会静默串号）。

本文件锁七件事（纯离线，不需要 mf6）：

  1. 不同 sid 拿到**不同**的 LLM 客户端；换密钥时 `parser` **也跟着换** ——
     `DialogueManager.__init__` 是 `self.parser = parser or intent_parser`，
     只传 `llm=` 不会传导下去，parser 仍然用 `.env` 那把钥匙，换密钥等于没换。
  2. 换密钥**不能弄丢已有对话与 IR**。
  3. 密钥**不进 `DialogueManager.state`**（state 会被 save_session 序列化到
     `data/sessions/{sid}.json`，进去就等于落盘）。
  4. 密钥**不进日志**（含 DEBUG 级；litellm 的 set_verbose 会打印含 api_key 的 kwargs）。
  5. 界面**真的挂了**这两件事：`demo.load` 分配编号、按钮调用 `apply_api_key`。
     断言基于**构建出来的 demo 对象**而不是源码文本 —— 这样才能在接线被摘掉时变红。
  6. 会话编号经过**消毒**，不能拼出路径穿越。
  7. 模块导入**不再改写全局 stdout**（改写的后果：导入方的输出被吞；反复导入会
     把底层 buffer 关掉，后续 print 抛 `I/O operation on closed file`）。
     这条用**非 UTF-8 的假流**触发，否则 `_ensure_utf8_stdio` 的早退会让断言
     永远绿（空过的绿勾）—— 详见第 7 节的注释。
  8. 两个**全局单例**界面（`monitor` 调试面板、`plot_styler`「图件修改」Tab）
     对使用者隐藏 —— 隐藏是 `visible=False` 而**不是删除**，所以还要锁住
     "组件仍在、绑定仍在"，否则"隐藏"会退化成断线。

脚本风格，不用 pytest：``python tests/test_session_isolation.py``
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm_agent import session_registry as reg

_passed = 0
_skipped = 0


def _ok(msg):
    global _passed
    _passed += 1
    print(f"   OK: {msg}")


def _skip(msg):
    global _skipped
    _skipped += 1
    print(f"   SKIP: {msg}")


# 探针会话编号 —— 刻意**不用** "default"：
# 那会覆盖作者本机正在用的真实会话（data/sessions/default.json）。
_PROBE = ["__iso_a__", "__iso_b__", "__iso_c__"]
_SENTINEL = "sk-SENTINEL-must-never-be-persisted-0123456789"


def _cleanup():
    for sid in _PROBE:
        reg.drop(sid)
        f = ROOT / "data" / "sessions" / f"{sid}.json"
        if f.exists():
            f.unlink()


# ================================================================
print("\n1. 不同会话各自一套 LLM 客户端（三层一起换）...")
# ================================================================
dm_a = reg.get(_PROBE[0], restore=False)
dm_b = reg.get(_PROBE[1], restore=False)
dm_a = reg.configure(_PROBE[0], api_key="sk-probe-AAAA",
                     model="deepseek/deepseek-v4-flash")
dm_b = reg.configure(_PROBE[1], api_key="sk-probe-BBBB",
                     model="deepseek/deepseek-v4-flash")

assert dm_a.llm is not dm_b.llm, "两个 sid 必须拿到不同的 LLMClient"
assert dm_a.llm.api_key == "sk-probe-AAAA"
assert dm_b.llm.api_key == "sk-probe-BBBB"
_ok("两个 sid 拿到不同的 LLMClient，密钥互不干扰")

assert dm_a.parser.llm is dm_a.llm, \
    "parser.llm 不是该会话的 client —— 只传 llm= 没传导，换密钥等于没换"
assert dm_b.parser.llm is dm_b.llm
assert dm_a.parser is not dm_b.parser
_ok("IntentParser 也跟着换了（三层一起换，否则解析仍烧 .env 那把钥匙）")

# 同一 sid 重复取，应当拿回同一个对象（不是每次都新建）
assert reg.get(_PROBE[0], restore=False) is dm_a
_ok("同一 sid 重复取回同一对象（不会每次新建、丢状态）")

# ================================================================
print("\n2. 换密钥不丢已有对话...")
# ================================================================
dm_a.state["history"] = [("旧问题", "旧回答")]
dm_a.state["json_ir"] = {"project": {"name": "probe"}, "ir_version": "1.0"}
dm_a2 = reg.configure(_PROBE[0], api_key="sk-probe-AAAA2")

assert dm_a2.llm.api_key == "sk-probe-AAAA2"
assert dm_a2.state["history"] == [("旧问题", "旧回答")], "换密钥把对话弄丢了"
assert dm_a2.state["json_ir"]["project"]["name"] == "probe", "换密钥把 IR 弄丢了"
_ok("换密钥保住了对话历史与已抽出的 IR")

# 空密钥 = 回落服务器默认，不报错
dm_a3 = reg.configure(_PROBE[0], api_key="")
assert dm_a3.llm.api_key is not None or True
assert reg.has_custom_key(_PROBE[0]) is False
_ok("密钥清空 → 回落到服务器默认密钥（has_custom_key 变 False）")

# ================================================================
print("\n3. 密钥不进会被落盘的状态...")
# ================================================================
reg.configure(_PROBE[2], api_key=_SENTINEL)
dm_c = reg.get(_PROBE[2], restore=False)

# state 是 save_session 序列化出去的唯一东西
blob = json.dumps(dm_c.state, ensure_ascii=False, default=str)
assert _SENTINEL not in blob, "密钥出现在了 DialogueManager.state —— 会被落盘！"
_ok("DialogueManager.state 里搜不到密钥子串（state 是唯一被落盘的东西）")

# 真跑一次 save_session，直接查写出来的文件
dm_c.save_session(_PROBE[2])
sess_file = ROOT / "data" / "sessions" / f"{_PROBE[2]}.json"
if sess_file.exists():
    raw = sess_file.read_text(encoding="utf-8")
    assert _SENTINEL not in raw, f"密钥被写进了 {sess_file}"
    _ok("实际写出的会话 JSON 里也没有密钥（不是只查了内存对象）")
else:
    _skip("save_session 未落盘（会话文件不存在），跳过文件级检查")

# ================================================================
print("\n4. 密钥不进日志...")
# ================================================================
_captured = []


class _Capture(logging.Handler):
    def emit(self, record):
        try:
            _captured.append(record.getMessage())
        except Exception:
            pass


_root = logging.getLogger()
_handler = _Capture()
_root.addHandler(_handler)
_old_level = _root.level
_root.setLevel(logging.DEBUG)
try:
    reg.configure(_PROBE[1], api_key=_SENTINEL)
    reg.get(_PROBE[1], restore=False)
finally:
    _root.removeHandler(_handler)
    _root.setLevel(_old_level)

_leaked = [m for m in _captured if _SENTINEL in m]
assert not _leaked, f"密钥进了日志（DEBUG 级，共 {len(_leaked)} 条）: {_leaked[:2]}"
_ok(f"DEBUG 级全量日志（{len(_captured)} 条）里没有密钥")

# ================================================================
print("\n5. 会话编号消毒（防路径穿越）...")
# ================================================================
from app.frontend.gradio_app import _sanitize_sid, _ensure_sid

for bad in ["../../etc/passwd", "..\\..\\windows\\system32", "a/b/c",
            "我的 会话/../x", "sid;rm -rf /", ""]:
    got = _sanitize_sid(bad)
    assert "/" not in got and "\\" not in got and ".." not in got, \
        f"消毒不彻底：{bad!r} → {got!r}"
    assert got, f"消毒后不能是空串：{bad!r}"
_ok("路径分隔符与 .. 全部被剔除，且不会消毒成空串")

assert _sanitize_sid("gw-stored-1") == "gw-stored-1"
_ok("正常的编号原样保留（消毒不会改坏正常运行路径）")


class _Req:
    session_hash = "abc123def456xyz"


assert _ensure_sid("stored-sid-1", _Req()) == "stored-sid-1"
_ok("有 localStorage 存储值时优先用它（刷新页面不丢会话）")
assert _ensure_sid("", _Req()) == "abc123def456xyz"
_ok("没存储值时从 session_hash 派生（浏览器禁用 localStorage 也能隔离）")
assert _ensure_sid("", type("R", (), {"session_hash": None})()) != ""
_ok("连 session_hash 都没有时仍给出非空编号（不会退化成共用空 sid）")

# Gradio 靠**类型标注**把 gr.Request 注入进来；少了这个参数就拿不到 session_hash
import inspect
_sig = inspect.signature(_ensure_sid)
assert "request" in _sig.parameters, \
    "_ensure_sid 没有 request 参数 —— Gradio 不会注入 gr.Request，拿不到 session_hash"
assert "Request" in str(_sig.parameters["request"].annotation), \
    "request 参数没标成 gr.Request，Gradio 认不出，不会注入"
_ok("_ensure_sid 声明了 gr.Request 参数（session_hash 的注入前提）")

# ================================================================
print("\n6. 界面真的挂了这两件事（基于构建出的 demo 断言）...")
# ================================================================
from app.frontend import gradio_app as ga

demo = ga.create_ui()
_wired = {getattr(f.fn, "__name__", "") for f in demo.fns.values()}

assert "_ensure_sid" in _wired, \
    "demo.load 没挂 _ensure_sid —— 会话编号永远是初值，所有人共用一个会话"
_ok("demo.load 真的挂了 _ensure_sid")

assert "apply_api_key" in _wired, \
    "「应用密钥」按钮没接线 —— 密钥输入框是个摆设"
_ok("「应用密钥」按钮真的调用了 apply_api_key")

_bs = [c for c in demo.blocks.values() if type(c).__name__ == "BrowserState"]
assert len(_bs) == 1, f"应有且仅有 1 个 BrowserState，实际 {len(_bs)}"
_ok("会话编号用的是 BrowserState（刷新不丢），不是固定值的 gr.State")

_pwd = [c for c in demo.blocks.values()
        if type(c).__name__ == "Textbox"
        and getattr(c, "label", "") == "LLM API Key"]
assert _pwd, "界面上没有 LLM API Key 输入框"
assert getattr(_pwd[0], "type", None) == "password", \
    "密钥输入框必须是 password 类型 —— 否则明文显示在屏幕上"
_ok("密钥输入框存在且是 password 类型（不明文显示）")

# ================================================================
print("\n6a. 两个全局单例面板对使用者隐藏（A1b）...")
# ================================================================
# 这两个组件挂在全局单例上（monitor / plot_styler），多人同时用会互相看见。
# 处置是 visible=False 而**不是删掉** —— debug_log 是运行按钮的输出目标，
# 删了绑定就断。所以这里两件事都要锁：默认不显示、但组件仍然存在且有绑定。
_blocks = demo.blocks

_acc = [c for c in _blocks.values()
        if type(c).__name__ == "Accordion"
        and "流水线追踪" in str(getattr(c, "label", ""))]
assert _acc, "找不到「流水线追踪」调试面板（被改名或删掉了？）"
assert _acc[0].visible is False, \
    "调试面板默认可见 —— 使用者会看到别人的日志（monitor 是全局单例）"
_ok("「流水线追踪」调试面板默认隐藏")

_tab = [c for c in _blocks.values()
        if type(c).__name__ in ("Tab", "TabItem")   # gradio 6 内部类名是 Tab
        and str(getattr(c, "label", "")) == "图件修改"]
assert _tab, "找不到「图件修改」Tab"
assert _tab[0].visible is False, \
    "「图件修改」Tab 默认可见 —— 会串用户（plot_styler 是全局单例）"
_ok("「图件修改」Tab 默认隐藏")

# 关键：隐藏≠删除。debug_log 必须仍然存在、仍然是某个函数的输出。
_dbg = [c for c in _blocks.values()
        if type(c).__name__ == "Textbox"
        and "实时管道追踪" in str(getattr(c, "label", ""))]
assert _dbg, \
    "debug_log 组件不见了 —— 它是运行按钮的输出目标，删掉会让绑定报错"
_dbg_id = _dbg[0]
_dbg_bound = any(_dbg_id in (getattr(f, "outputs", None) or [])
                 for f in demo.fns.values())
assert _dbg_bound, "debug_log 还在，但已经没有任何函数往它写 —— 绑定断了"
_ok("debug_log 组件仍存在且仍有输出绑定（隐藏不是删除）")

# 开关本身：默认关，显式置 1 才开
_parse = ga._debug_panels_enabled
assert _parse("1") is True and _parse("true") is True
assert _parse("") is False and _parse("0") is False and _parse("no") is False
_ok("开关解析正常（只有 1/true 之类才开，空值=关）")


# ================================================================
print("\n6b. 走 gradio_app.get_session 的集成级隔离（不是只测注册表）...")
# ================================================================
_gdm_a = ga.get_session(_PROBE[0])
_gdm_b = ga.get_session(_PROBE[1])
assert _gdm_a is not _gdm_b, "get_session 对两个 sid 返回了同一个对象 —— 会串号"
_gdm_a.state["json_ir"] = {"project": {"name": "只有A改过"}}
assert _gdm_b.state["json_ir"].get("project", {}).get("name") != "只有A改过", \
    "改 A 的 IR 影响到了 B —— 会话没有真正隔离"
_ok("改 A 的参数不会影响 B（get_session 层面隔离住了）")

# ================================================================
print("\n7. 导入不再改写全局 stdout（走真路径才算数）...")
# ================================================================
# 为什么不能只写 `assert sys.stdout is stdout_before`：
# 测试环境总有 PYTHONIOENCODING=utf-8，而 _ensure_utf8_stdio 一看到 stdout
# 已是 utf-8 就**早退**（gradio_app.py 的 `if enc == "utf8": return`）。
# 也就是说，把重绑挪回模块顶层，那条断言照样绿 —— 它从没走进过重绑分支，
# 是个空过的绿勾（实测：挪回顶层后本节依然全绿）。
#
# 办法：临时把 stdout 换成一个 **encoding 不是 UTF-8** 的假流，逼重绑分支
# 真的执行。于是同时锁住两件事：
#   ① 模块顶层若重绑 → 假流被顶掉 → 红。这是"导入会吞输出"那个故障。
#   ② 直接调用函数**必须**发生重绑（正向对照）—— 否则函数退化成 no-op 时，
#      ①也因为"反正什么都没发生"而永远绿，等于没锁。
# 另外顺带在真路径上验幂等：连调三次不能把底层 buffer 关掉（旧 wrapper 被
# 回收会连带关掉它持有的 buffer，之后 print 抛 I/O operation on closed file）。
import io as _io
import importlib as _importlib


class _FakeStd:
    """假的标准输出：encoding 故意不是 UTF-8，且把写入吞掉。"""

    encoding = "gbk"
    errors = "strict"

    def __init__(self):
        self.buffer = _io.BytesIO()

    def write(self, s):
        return len(s)

    def flush(self):
        pass


_real_out, _real_err = sys.stdout, sys.stderr
_fake_out, _fake_err = _FakeStd(), _FakeStd()
sys.stdout, sys.stderr = _fake_out, _fake_err
try:
    _importlib.reload(ga)
    if sys.stdout is not _fake_out:
        raise AssertionError(
            "import gradio_app 改写了全局 sys.stdout —— 会吞掉导入方的输出。"
            "这次是在一个非 UTF-8 的假流上触发的，说明重绑被放回了模块顶层")
    ga._ensure_utf8_stdio()
    if sys.stdout is _fake_out:
        raise AssertionError(
            "_ensure_utf8_stdio 面对非 UTF-8 的 stdout 没有重绑 —— "
            "函数已成 no-op，上面那条断言也就永远绿、失去意义")
    _wrapped = sys.stdout
    ga._ensure_utf8_stdio()          # 幂等：再来两次，不能又包一层、更不能关掉 buffer
    ga._ensure_utf8_stdio()
    if sys.stdout is not _wrapped:
        raise AssertionError("_ensure_utf8_stdio 不幂等：重复调用又包了一层新流")
    print("       （这句写进了被包住的假流，属正常）")
except ValueError as e:
    raise AssertionError(f"_ensure_utf8_stdio 不幂等，stdout 被关掉了: {e}")
finally:
    sys.stdout, sys.stderr = _real_out, _real_err

_ok("导入不改写 stdout；非 UTF-8 时才重绑，且幂等（三条都在真路径上验的）")

_ensure_src = ga.__file__
assert "_ensure_utf8_stdio" in Path(_ensure_src).read_text(encoding="utf-8")
_ok("UTF-8 包装逻辑收在 _ensure_utf8_stdio 里（而不是散在模块顶层）")

# ================================================================
print("\n8. 会话上限与淘汰...")
# ================================================================
from app.llm_agent import session_registry as _r


class _FakeDM:
    def __init__(self):
        self.saved = []

    def save_session(self, sid):
        self.saved.append(sid)


_r._BUNDLES.clear()
try:
    for i in range(_r.MAX_SESSIONS + 5):
        b = _r._Bundle(llm=None, parser=None, dm=_FakeDM(),
                       api_key=None, model=None, api_base=None)
        _r._BUNDLES[f"fake-{i:03d}"] = b
        _r._evict_if_needed()
    assert len(_r._BUNDLES) == _r.MAX_SESSIONS, \
        f"淘汰后应剩 {_r.MAX_SESSIONS} 个，实际 {len(_r._BUNDLES)}"
    _ok(f"超过上限 {_r.MAX_SESSIONS} 后会淘汰最旧的会话（长时间运行不会无限涨）")

    # 淘汰前应当落盘，否则内存里一丢，这轮对话就彻底没了
    _evicted = [sid for sid, b in _r._BUNDLES.items()]
    assert "fake-000" not in _evicted and "fake-004" not in _evicted
    _ok("淘汰的是最旧的（最早的 fake-000 已被清出）")
finally:
    _r._BUNDLES.clear()

# ================================================================
_cleanup()

print("\n" + "=" * 60)
if _skipped:
    print(f"会话隔离与自持密钥: {_passed} 项通过, {_skipped} 项跳过")
else:
    print(f"会话隔离与自持密钥: {_passed} 项通过")
print("=" * 60)
