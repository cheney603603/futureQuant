"""
LLM Client - 统一大语言模型调用接口

支持多后端：
- OpenAI (GPT-4o / GPT-4o-mini / etc.)
- Ollama / vLLM / 其他 OpenAI 兼容接口

特性：
- 统一 chat(messages, tools=None) 接口
- 自动从配置/环境变量读取 API Key
- 内置指数退避重试
- Token 使用量追踪
- 支持 Function Calling schema 传入

使用示例：
    >>> from futureQuant.core.llm_client import LLMClient
    >>> client = LLMClient()
    >>> resp = client.chat([{"role": "user", "content": "Hello"}])
    >>> print(resp.content)
"""

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx

from .config import get_config
from .logger import get_logger

logger = get_logger("core.llm_client")


class LLMProvider(Enum):
    """支持的 LLM 提供商"""

    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMUsage:
    """Token 使用量统计"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """LLM 响应封装"""

    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    model: str = ""
    finish_reason: str = ""
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient:
    """
    统一 LLM 客户端

    自动读取全局配置中的 llm 段落，也可在初始化时显式覆盖。
    """

    def __init__(
        self,
        provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            provider: 提供商名称，默认从配置读取
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            api_key: API 密钥（OpenAI 需要）
            base_url: 自定义 Base URL
            max_retries: 最大重试次数
            timeout: 请求超时（秒）
        """
        config = get_config().llm

        self.provider = LLMProvider(
            (provider or config.provider).lower().strip()
        )
        self.model = model or config.model
        self.temperature = (
            temperature if temperature is not None else config.temperature
        )
        self.max_tokens = max_tokens or config.max_tokens
        self.max_retries = max_retries or config.max_retries
        self.timeout = timeout or config.timeout

        # 解析 api_key / base_url
        if self.provider == LLMProvider.OPENAI:
            self.api_key = api_key or config.openai_api_key or os.getenv("FQ_LLM__OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
            self.base_url = base_url or config.openai_base_url
            if not self.api_key:
                logger.warning("OpenAI API key is empty. Set FQ_LLM__OPENAI_API_KEY or OPENAI_API_KEY env var.")
        elif self.provider == LLMProvider.OLLAMA:
            self.api_key = api_key or "ollama"  # Ollama 不需要真实 key，但 OpenAI 客户端需要非空字符串
            self.base_url = base_url or config.ollama_base_url
            if model is None and config.ollama_model:
                self.model = config.ollama_model
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # 初始化 HTTP 客户端
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

        logger.info(
            f"LLMClient initialized: provider={self.provider.value}, "
            f"model={self.model}, base_url={self.base_url}"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        发送对话请求

        Args:
            messages: OpenAI 格式消息列表
            tools: Function calling schema 列表（可选）
            temperature: 覆盖默认 temperature
            max_tokens: 覆盖默认 max_tokens

        Returns:
            LLMResponse 封装对象
        """
        if self.provider == LLMProvider.OPENAI:
            return self._chat_openai(messages, tools, temperature, max_tokens)
        elif self.provider == LLMProvider.OLLAMA:
            return self._chat_ollama(messages, tools, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """调用 OpenAI 兼容接口（含 Ollama）"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"LLM HTTP error (attempt {attempt + 1}/{self.max_retries}): {e.response.status_code} - {e.response.text[:200]}")
                if e.response.status_code in (429, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"LLM request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2 ** attempt)

        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_error}")

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Ollama 通过 OpenAI 兼容端点 /v1/chat/completions 调用
        因此直接复用 _chat_openai 逻辑
        """
        return self._chat_openai(messages, tools, temperature, max_tokens)

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> LLMResponse:
        """解析 OpenAI 格式响应"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")

        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", ""),
            raw_response=data,
        )

    def close(self):
        """关闭 HTTP 客户端"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def quick_chat(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> LLMResponse:
    """快速发起一次对话（无需显式初始化客户端）"""
    with LLMClient(provider=provider, model=model) as client:
        return client.chat(messages, tools=tools)
