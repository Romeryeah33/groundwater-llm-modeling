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
import threading
from typing import Optional

from app.llm_agent.llm_client import LLMClient
from app.llm_agent.intent_parser import IntentParser
from app.llm_agent.dialogue_manager import DialogueManager

logger = logging.getLogger(__name__)


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
    """界面回显用：该会话是否已经用了用户自填的密钥。不吐密钥本身。"""
    with _LOCK:
        b = _BUNDLES.get(sid)
        return bool(b and b.has_custom_key)


def drop(sid: str) -> None:
    """释放一个会话（内存回收）。磁盘上的会话文件不动。"""
    with _LOCK:
        _BUNDLES.pop(sid, None)


def active_sids() -> list[str]:
    """当前内存里活跃的会话编号（诊断用）。"""
    with _LOCK:
        return list(_BUNDLES.keys())
