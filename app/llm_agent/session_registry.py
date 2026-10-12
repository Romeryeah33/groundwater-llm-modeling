"""
会话注册表 —— 每个会话一套独立的 LLM 三层对象

为什么需要它
------------
原先 `gradio_app.get_session()` 里写的是裸 `DialogueManager()`，于是全服务器
**共用同一份模块级单例**（`llm_client` / `intent_parser` / `dialogue_manager`）。
单机自己用没问题；一旦分发给别人，就分不清"谁在用、用谁的密钥"——
所有人都在烧部署方 `.env` 里的那一把 key。

本模块把"sid → 三件套"收成唯一一份实现，供 **Gradio 界面**与 **REST 接口**共用，
避免两个入口各自维护一套会话（那样会出现"网页里聊过的参数，接口里查不到"）。

为什么必须三层一起换
--------------------
`DialogueManager.__init__` 是 `self.parser = parser or intent_parser`——
**只传 `llm=` 不会传导到 parser**，parser 仍然指着模块级单例、仍然用 `.env` 那把 key。
所以构造顺序必须是 client → parser → manager，三个都显式传入。

密钥去哪儿了（红线）
--------------------
密钥只挂在 `LLMClient.api_key` 上，**绝不写进 `DialogueManager.state`**。
`state` 会被 `save_session` 序列化到 `data/sessions/{sid}.json`，写进去就等于落盘。
本模块不提供任何"把密钥暴露出去"的读取口，只保留一个布尔量 `has_custom_key`，
仅供界面回显"已启用你的密钥 / 用的是服务器默认密钥"，不吐密钥本身。
"""

import collections
import logging
import re
import threading
from typing import Optional

from app.llm_agent.llm_client import LLMClient
from app.llm_agent.intent_parser import IntentParser
from app.llm_agent.dialogue_manager import DialogueManager

logger = logging.getLogger(__name__)


# 会话编号会被拼进两个文件系统路径：`data/sessions/{sid}.json` 与
# `workspaces/run_{sid}`。**不消毒就是路径穿越** —— 一个 `../../` 就能让
# 请求方决定往哪儿写。
_SID_BAD = re.compile(r"[^A-Za-z0-9_-]")
_SID_MAX = 64


def sanitize_sid(raw) -> str:
    """白名单过滤：只留 `[A-Za-z0-9_-]`，其余**删除**。可能返回空串。

    为什么用白名单而不是剔除 `..` 和 `/`：黑名单永远漏（`..\\`、URL 编码、
    全角字符……），而白名单只要"不认识的就不放行"。

    为什么是**删除**而不是替换成 `_`：替换会让 `a/b` 和 `a_b` 撞成同一个
    编号 —— 两个不同的浏览器共用一份会话，正是本轮要修的毛病。删除则
    `__a__` 与 `a` 仍然不同。

    **不替调用方决定空串怎么办**，因为两边答案相反：界面空串要发一个新编号
    （每个匿名访客各自独立），接口空串应当直接报错（否则每次请求都新建一个
    会话，用户永远接不上上一轮）。
    """
    return _SID_BAD.sub("", str(raw or ""))[:_SID_MAX]


def _require_sid(sid) -> str:
    """注册表内部的兜底：消毒后为空就是调用方漏了决策，**要响**不要静默。"""
    s = sanitize_sid(sid)
    if not s:
        raise ValueError(
            f"会话编号 {sid!r} 消毒后为空 —— 调用方必须自己决定空串怎么处理"
            "（界面发新编号 / 接口报 400），不能交给注册表猜。")
    return s


class _Bundle:
    """一个会话的三件套。密钥只在这里，且不参与任何序列化。"""

    __slots__ = ("llm", "parser", "dm", "api_key", "model", "api_base")

    def __init__(self, llm, parser, dm, api_key, model, api_base):
        self.llm = llm
        self.parser = parser
        self.dm = dm
        self.api_key = api_key
        self.model = model
        self.api_base = api_base

    @property
    def has_custom_key(self) -> bool:
        return bool(self.api_key)


# 按最近使用排序的会话表。分发出去以后是长时间运行的服务器，不设上限的话
# 每个开过的浏览器都会留下一套 LLM 对象，内存只增不减。
_BUNDLES: "collections.OrderedDict[str, _Bundle]" = collections.OrderedDict()
_LOCK = threading.RLock()

# 上限取得宽松：正常同时用的人不会有这么多，它拦的是"长期运行慢慢攒"。
MAX_SESSIONS = 64


def _evict_if_needed() -> None:
    """超出上限就淘汰最久没被碰过的会话。调用方必须已持锁。"""
    while len(_BUNDLES) > MAX_SESSIONS:
        old_sid, old = _BUNDLES.popitem(last=False)
        # 淘汰前落盘一次：内存里丢了，磁盘上还在，下次进来还能恢复
        try:
            old.dm.save_session(old_sid)
        except Exception as e:
            logger.warning("会话 %s 淘汰时落盘失败（忽略）: %s", old_sid, e)
        logger.info("会话 %s 已淘汰（超过 %d 个会话上限）", old_sid, MAX_SESSIONS)


def _build(*, api_key: Optional[str] = None,
           model: Optional[str] = None,
           api_base: Optional[str] = None) -> _Bundle:
    """按密钥构造一套全新的三层对象。"""
    # 每次自建客户端都把 litellm 的详细日志关掉：set_verbose=True 时 litellm 会
    # 把含 api_key 的 kwargs 打到日志里（见 llm_client.py 顶部的全局开关）。
    # 这里再显式关一次，保证"用户自己的密钥"不会因为服务器开了 DEBUG 而外泄。
    try:
        import litellm
        if litellm.set_verbose:
            litellm.set_verbose = False
    except Exception:      # litellm 不可用时不影响主流程
        pass

    client = LLMClient(model=model, api_key=api_key, api_base=api_base)
    parser = IntentParser(llm=client)
    dm = DialogueManager(llm=client, parser=parser)
    return _Bundle(llm=client, parser=parser, dm=dm,
                   api_key=api_key, model=model, api_base=api_base)


def _restore(dm: DialogueManager, sid: str) -> None:
    """从磁盘恢复上次会话（重启后无需重新输入）。失败不阻断。"""
    try:
        restored = dm.load_session(sid)
    except Exception as e:
        logger.warning("会话 %s 恢复失败（忽略）: %s", sid, e)
        return
    if restored:
        logger.info("会话 %s: 已从磁盘恢复，历史 %d 轮",
                    sid, len(dm.state.get("history", [])))
        dm._just_restored = True


def get(sid: str, *, restore: bool = True) -> DialogueManager:
    """
    取该会话的 DialogueManager；没有就用服务器默认密钥建一个。

    注意 `restore`：内部测试/探针请传 False，否则会去读真实的
    `data/sessions/{sid}.json`。
    """
    sid = _require_sid(sid)
    with _LOCK:
        b = _BUNDLES.get(sid)
        if b is None:
            b = _build()
            _BUNDLES[sid] = b
            _evict_if_needed()
            if restore:
                _restore(b.dm, sid)
        else:
            _BUNDLES.move_to_end(sid)
        return b.dm


def configure(sid: str, *, api_key: Optional[str] = None,
              model: Optional[str] = None,
              api_base: Optional[str] = None) -> DialogueManager:
    """
    为该会话换上指定的密钥（界面里"应用密钥"按钮 / 接口的 X-LLM-API-Key 走这里）。

    换密钥会重建三层对象，但**已有对话与参数不丢**——旧 state 原样搬过去。
    空字符串（界面输入框被清空）视同"没给"，回落到服务器默认。
    """
    sid = _require_sid(sid)
    api_key = (api_key or "").strip() or None
    model = (model or "").strip() or None
    api_base = (api_base or "").strip() or None

    with _LOCK:
        old = _BUNDLES.get(sid)
        new = _build(api_key=api_key, model=model, api_base=api_base)
        if old is not None:
            # 保住已有对话、已抽出的 IR、阶段等，只换"用哪把钥匙"
            new.dm.state = old.dm.state
            new.dm._just_restored = False
        _BUNDLES[sid] = new
        _BUNDLES.move_to_end(sid)
        _evict_if_needed()
        logger.info("会话 %s: 密钥已%s", sid,
                    "由用户自填" if new.has_custom_key else "回落到服务器默认")
        return new.dm


def has_custom_key(sid: str) -> bool:
    """界面回显用：该会话是否已经用了用户自填的密钥。不吐密钥本身。

    这里**不报错**（与 get/configure 不同）：编号不认识就等于"没有自填密钥"，
    回显一句"用的是服务器默认"即可，没必要为此让页面炸掉。
    """
    s = sanitize_sid(sid)
    with _LOCK:
        b = _BUNDLES.get(s) if s else None
        return bool(b and b.has_custom_key)


def drop(sid: str) -> None:
    """释放一个会话（内存回收）。磁盘上的会话文件不动。

    同样不报错 —— 这是清理路径，编号本来就可能已经不存在了。
    """
    s = sanitize_sid(sid)
    if not s:
        return
    with _LOCK:
        _BUNDLES.pop(s, None)


def active_sids() -> list[str]:
    """当前内存里活跃的会话编号（诊断用）。"""
    with _LOCK:
        return list(_BUNDLES.keys())
