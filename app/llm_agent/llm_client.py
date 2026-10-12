"""
LiteLLM 统一调用封装

支持通过 LiteLLM 统一接口调用多种大语言模型：
- DeepSeek (推荐，国内访问快，成本低)
- OpenAI GPT-4/GPT-4o
- Anthropic Claude
- Qwen (通义千问，本地部署备选)
"""

import json
import re
import logging
from typing import Optional, AsyncGenerator

import litellm
from litellm import completion, acompletion

from app.config import settings

logger = logging.getLogger(__name__)

# LiteLLM 全局配置
litellm.drop_params = True  # 自动丢弃不支持的参数
litellm.set_verbose = settings.log_level == "DEBUG"


class LLMClient:
    """
    统一LLM调用客户端

    使用方式：
        client = LLMClient()
        response = client.chat("你好，请帮我建模...")
        # 或指定模型
        response = client.chat("...", model="openai/gpt-4o")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.llm_api_key
        self.api_base = api_base or settings.llm_api_base
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

    def _build_kwargs(self) -> dict:
        """构建LiteLLM调用参数"""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # 请求超时（LLM_REQUEST_TIMEOUT，默认 120 s）。
        # 此前 settings.llm_request_timeout 定义了却**从没传下来**，于是请求无限等待：
        # 对界面而言只是"转圈很久"，对 REST 接口却是**工作线程被永久占住** ——
        # 有一个卡住的请求，后面所有请求都排在它后面。
        kwargs["timeout"] = settings.llm_request_timeout
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    def _build_messages(
        self,
        user_message: str = "",
        system_message: Optional[str] = None,
        messages: Optional[list] = None,
    ) -> list:
        """构建消息列表（chat/chat_async/chat_stream 共用）"""
        if messages is not None:
            return messages
        msgs = []
        if system_message:
            msgs.append({"role": "system", "content": system_message})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        messages: Optional[list] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        同步对话调用

        Args:
            user_message: 用户消息
            system_message: 系统提示词
            messages: 完整消息列表（优先级高于user_message）
            response_format: JSON Schema响应格式（用于结构化输出）

        Returns:
            LLM响应文本
        """
        kwargs = self._build_kwargs()
        kwargs["messages"] = self._build_messages(user_message, system_message, messages)

        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"LLM调用: model={self.model}, messages_count={len(kwargs['messages'])}")

        try:
            response = completion(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"LLM返回: {content[:200]}...")
            return content
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    async def chat_async(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        messages: Optional[list] = None,
    ) -> str:
        """异步对话调用"""
        kwargs = self._build_kwargs()
        kwargs["messages"] = self._build_messages(user_message, system_message, messages)

        try:
            response = await acompletion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM异步调用失败: {e}")
            raise

    async def chat_stream(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        messages: Optional[list] = None,
    ) -> AsyncGenerator[str, None]:
        """流式对话调用（用于WebSocket推送）"""
        kwargs = self._build_kwargs()
        kwargs["messages"] = self._build_messages(user_message, system_message, messages)
        kwargs["stream"] = True

        try:
            response = await acompletion(**kwargs)
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM流式调用失败: {e}")
            raise

    def chat_json(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        json_schema: Optional[dict] = None,
        expect_array: bool = False,
    ) -> dict | list:
        """
        获取JSON结构化输出

        Args:
            user_message: 用户消息
            system_message: 系统提示词
            json_schema: JSON Schema定义
            expect_array: 期望返回JSON数组而非对象

        Returns:
            解析后的JSON字典或列表
        """
        if system_message:
            system_message += (
                "\n\n重要：你的回复必须是合法的JSON格式，严格遵循给定的Schema。"
                "不要包含任何JSON之外的文字说明。"
            )
        else:
            system_message = "你的回复必须是合法的JSON格式。不要包含任何JSON之外的文字说明。"

        # 如果期望数组，不能使用 json_object 格式限制
        response_format = None
        if not expect_array:
            response_format = {"type": "json_object"} if json_schema else None

        raw = self.chat(
            user_message=user_message,
            system_message=system_message,
            response_format=response_format,
        )

        # 尝试提取JSON（处理LLM可能额外输出的markdown代码块）
        raw = raw.strip()
        if raw.startswith("```"):
            # 移除markdown代码块标记
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines)

        try:
            result = json.loads(raw)
            return result
        except json.JSONDecodeError:
            # 尝试提取 JSON 内容
            # 先尝试提取数组
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match and expect_array:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            # 再尝试提取对象
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"无法解析LLM返回的JSON: {raw[:500]}")


# 全局客户端实例
llm_client = LLMClient()
