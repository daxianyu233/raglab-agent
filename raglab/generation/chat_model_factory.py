"""聊天模型工厂。

当前支持：

1. DeepSeek API；
2. OpenAI API；
3. Ollama 本地模型。

DeepSeek API 使用 OpenAI 兼容接口，因此底层通过
langchain_openai.ChatOpenAI 接入。

本模块只负责创建聊天模型，不负责：

1. 文档检索；
2. Prompt 构造；
3. RAG 执行；
4. Agent 循环；
5. 会话状态管理。
"""

from __future__ import annotations

import os
from typing import Any


SUPPORTED_PROVIDERS = {
    "deepseek",
    "openai",
    "ollama",
}


def normalize_provider(
    provider: str,
) -> str:
    """规范化并检查模型提供商名称。"""

    if not isinstance(provider, str):
        raise TypeError(
            "provider 必须是字符串，"
            f"实际类型：{type(provider)!r}"
        )

    normalized = provider.strip().lower()

    if not normalized:
        raise ValueError(
            "provider 不能为空。"
        )

    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(
            sorted(SUPPORTED_PROVIDERS)
        )

        raise ValueError(
            f"暂不支持 provider={provider!r}。"
            f"当前支持：{supported}"
        )

    return normalized


def read_api_key(
    environment_name: str,
) -> str:
    """从环境变量读取 API Key。"""

    normalized_name = (
        str(environment_name).strip()
    )

    if not normalized_name:
        raise ValueError(
            "API Key 环境变量名称不能为空。"
        )

    api_key = os.environ.get(
        normalized_name
    )

    if not api_key:
        raise RuntimeError(
            f"没有检测到环境变量 "
            f"{normalized_name}。\n"
            "请先在当前 PowerShell 会话中"
            "设置 API Key。"
        )

    return api_key


def create_deepseek_chat_model(
    *,
    model_name: str = (
        "deepseek-v4-flash"
    ),
    thinking_enabled: bool = False,
    reasoning_effort: str = "high",
    temperature: float = 0.0,
    max_tokens: int | None = 1024,
    timeout: float | None = 120.0,
    max_retries: int = 2,
    api_key_env: str = (
        "DEEPSEEK_API_KEY"
    ),
    base_url: str = (
        "https://api.deepseek.com"
    ),
) -> Any:
    """创建 DeepSeek V4 聊天模型。

    Parameters
    ----------
    model_name:
        DeepSeek 正式模型名称。

        当前支持：
        deepseek-v4-flash
        deepseek-v4-pro

    thinking_enabled:
        是否开启思考模式。

        基础 RAG 默认关闭，降低响应时间和费用。
        后续复杂 Agent 决策可以开启。

    reasoning_effort:
        思考强度。

        仅在 thinking_enabled=True 时生效。
        可用值：

        high
        max

    temperature:
        非思考模式下的生成随机性。

        思考模式下不传递该参数。

    max_tokens:
        最大输出 Token 数量。

    timeout:
        单次 API 请求超时时间，单位为秒。

    max_retries:
        请求失败后的最大重试次数。

    api_key_env:
        保存 DeepSeek API Key 的环境变量名称。

    base_url:
        DeepSeek OpenAI 兼容接口地址。
    """

    try:
        from langchain_openai import (
            ChatOpenAI,
        )

    except ImportError as error:
        raise RuntimeError(
            "当前环境没有安装 "
            "langchain-openai。\n"
            "请执行：\n"
            "pip install -U langchain-openai"
        ) from error

    normalized_model_name = (
        str(model_name).strip()
    )

    supported_models = {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    }

    if normalized_model_name not in (
        supported_models
    ):
        raise ValueError(
            "DeepSeek V4 模型名称无效："
            f"{normalized_model_name!r}。\n"
            "请使用：\n"
            "deepseek-v4-flash\n"
            "或\n"
            "deepseek-v4-pro"
        )

    normalized_base_url = (
        str(base_url).strip()
    )

    if not normalized_base_url:
        raise ValueError(
            "DeepSeek base_url 不能为空。"
        )

    normalized_effort = (
        str(reasoning_effort)
        .strip()
        .lower()
    )

    if normalized_effort not in {
        "high",
        "max",
    }:
        raise ValueError(
            "reasoning_effort 只能是 "
            "'high' 或 'max'。"
        )

    api_key = read_api_key(
        api_key_env
    )

    kwargs: dict[str, Any] = {
        "model": normalized_model_name,
        "api_key": api_key,
        "base_url": normalized_base_url,
        "timeout": timeout,
        "max_retries": int(
            max_retries
        ),
        "extra_body": {
            "thinking": {
                "type": (
                    "enabled"
                    if thinking_enabled
                    else "disabled"
                )
            }
        },
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = int(
            max_tokens
        )

    if thinking_enabled:
        kwargs[
            "reasoning_effort"
        ] = normalized_effort

    else:
        kwargs["temperature"] = float(
            temperature
        )

    return ChatOpenAI(
        **kwargs
    )


def create_openai_chat_model(
    *,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float | None = 60.0,
    max_retries: int = 2,
    api_key_env: str = (
        "OPENAI_API_KEY"
    ),
    base_url: str | None = None,
) -> Any:
    """创建 OpenAI Chat Model。"""

    try:
        from langchain_openai import (
            ChatOpenAI,
        )

    except ImportError as error:
        raise RuntimeError(
            "当前环境没有安装 "
            "langchain-openai。\n"
            "请执行：\n"
            "pip install -U langchain-openai"
        ) from error

    api_key = read_api_key(
        api_key_env
    )

    kwargs: dict[str, Any] = {
        "model": str(
            model_name
        ).strip(),
        "temperature": float(
            temperature
        ),
        "timeout": timeout,
        "max_retries": int(
            max_retries
        ),
        "api_key": api_key,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = int(
            max_tokens
        )

    if base_url is not None:
        normalized_base_url = (
            str(base_url).strip()
        )

        if normalized_base_url:
            kwargs["base_url"] = (
                normalized_base_url
            )

    return ChatOpenAI(
        **kwargs
    )


def create_ollama_chat_model(
    *,
    model_name: str,
    temperature: float = 0.0,
    num_predict: int | None = None,
    base_url: str | None = None,
) -> Any:
    """创建 Ollama 本地 Chat Model。"""

    try:
        from langchain_ollama import (
            ChatOllama,
        )

    except ImportError as error:
        raise RuntimeError(
            "当前环境没有安装 "
            "langchain-ollama。\n"
            "请执行：\n"
            "pip install -U langchain-ollama"
        ) from error

    kwargs: dict[str, Any] = {
        "model": str(
            model_name
        ).strip(),
        "temperature": float(
            temperature
        ),
    }

    if num_predict is not None:
        kwargs["num_predict"] = int(
            num_predict
        )

    if base_url is not None:
        normalized_base_url = (
            str(base_url).strip()
        )

        if normalized_base_url:
            kwargs["base_url"] = (
                normalized_base_url
            )

    return ChatOllama(
        **kwargs
    )


def create_chat_model(
    *,
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_output_tokens: int | None = (
        1024
    ),
    timeout: float | None = 120.0,
    max_retries: int = 2,
    api_key_env: str | None = None,
    base_url: str | None = None,
    thinking_enabled: bool = False,
    reasoning_effort: str = "high",
) -> Any:
    """根据 provider 创建聊天模型。

    Parameters
    ----------
    provider:
        当前支持：

        deepseek
        openai
        ollama

    model_name:
        具体模型名称。

    temperature:
        非思考模式下的生成随机性。

    max_output_tokens:
        最大输出 Token 数量。

    timeout:
        API 请求超时时间。

    max_retries:
        API 请求失败重试次数。

    api_key_env:
        API Key 环境变量名称。

        DeepSeek 默认：
        DEEPSEEK_API_KEY

        OpenAI 默认：
        OPENAI_API_KEY

    base_url:
        API 服务地址。

    thinking_enabled:
        是否开启 DeepSeek V4 思考模式。

    reasoning_effort:
        DeepSeek V4 思考强度。
    """

    normalized_provider = (
        normalize_provider(provider)
    )

    normalized_model_name = (
        str(model_name).strip()
    )

    if not normalized_model_name:
        raise ValueError(
            "model_name 不能为空。"
        )

    if normalized_provider == (
        "deepseek"
    ):
        return create_deepseek_chat_model(
            model_name=(
                normalized_model_name
            ),
            thinking_enabled=(
                thinking_enabled
            ),
            reasoning_effort=(
                reasoning_effort
            ),
            temperature=temperature,
            max_tokens=(
                max_output_tokens
            ),
            timeout=timeout,
            max_retries=max_retries,
            api_key_env=(
                api_key_env
                or "DEEPSEEK_API_KEY"
            ),
            base_url=(
                base_url
                or "https://api.deepseek.com"
            ),
        )

    if normalized_provider == (
        "openai"
    ):
        return create_openai_chat_model(
            model_name=(
                normalized_model_name
            ),
            temperature=temperature,
            max_tokens=(
                max_output_tokens
            ),
            timeout=timeout,
            max_retries=max_retries,
            api_key_env=(
                api_key_env
                or "OPENAI_API_KEY"
            ),
            base_url=base_url,
        )

    if normalized_provider == (
        "ollama"
    ):
        return create_ollama_chat_model(
            model_name=(
                normalized_model_name
            ),
            temperature=temperature,
            num_predict=(
                max_output_tokens
            ),
            base_url=base_url,
        )

    raise RuntimeError(
        "无法创建聊天模型："
        f"provider={normalized_provider}"
    )