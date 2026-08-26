"""RAG Prompt 构造模块。

主要职责：

用户问题
+
检索得到的受控上下文
+
回答约束
↓
生成可以直接交给聊天模型的消息列表

本模块不负责：

1. 文档检索；
2. 上下文截断；
3. 模型调用；
4. 答案评测。
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from raglab.generation.context_builder import (
    BuiltContext,
)


DEFAULT_SYSTEM_PROMPT = """你是一个基于检索资料回答问题的助手。

请严格遵守以下规则：

1. 只能依据用户提供的“检索资料”回答问题。
2. 不得将模型自身记忆或外部知识冒充为检索资料中的内容。
3. 回答中的事实性结论应尽量标注对应资料编号，例如 [资料1]。
4. 当一句话同时依据多条资料时，可以标注多个编号，例如 [资料1][资料2]。
5. 如果检索资料不足以回答问题，应明确说明“现有检索资料不足以回答该问题”，不要猜测。
6. 如果资料之间存在冲突，应指出冲突，不要自行选择其中一个结论作为确定事实。
7. 不要编造资料中没有出现的名称、数字、步骤、接口、参数或结论。
8. 回答应直接、清晰，并优先解决用户实际提出的问题。
9. 不要输出检索资料中不存在的网页链接或参考文献。
10. 引用编号必须使用检索资料中已经提供的编号，不得编造新的资料编号。"""


@dataclass(frozen=True)
class RAGPrompt:
    """保存一次完整的 RAG Prompt。"""

    question: str
    context: str
    system_prompt: str
    user_prompt: str
    messages: list[BaseMessage]
    context_available: bool


def normalize_question(
    question: str,
) -> str:
    """清理并检查用户问题。"""

    if not isinstance(
        question,
        str,
    ):
        raise TypeError(
            "question 必须是字符串，"
            f"实际类型：{type(question)!r}"
        )

    normalized = question.strip()

    if not normalized:
        raise ValueError(
            "question 不能为空。"
        )

    return normalized


def build_user_prompt(
    question: str,
    context: str,
) -> str:
    """构造发送给模型的用户消息。

    Parameters
    ----------
    question:
        用户提出的问题。

    context:
        context_builder 生成的检索上下文。

    Returns
    -------
    str
        完整用户消息。
    """

    normalized_question = (
        normalize_question(question)
    )

    normalized_context = context.strip()

    if normalized_context:
        context_section = (
            "以下是本次检索得到的资料：\n\n"
            f"{normalized_context}"
        )
    else:
        context_section = (
            "本次没有检索到可用资料。"
        )

    return (
        f"{context_section}\n\n"
        "请根据以上检索资料回答下面的问题。\n\n"
        f"用户问题：\n{normalized_question}\n\n"
        "回答要求：\n"
        "1. 直接回答问题，不要复述完整检索资料。\n"
        "2. 关键事实后标注对应资料编号。\n"
        "3. 资料不足时明确说明，不要猜测。\n"
        "4. 不要使用检索资料之外的信息补全答案。"
    )


def build_rag_prompt(
    question: str,
    built_context: BuiltContext,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> RAGPrompt:
    """构造完整的 RAG Prompt。

    Parameters
    ----------
    question:
        用户问题。

    built_context:
        build_context() 返回的 BuiltContext。

    system_prompt:
        系统级回答规则。

    Returns
    -------
    RAGPrompt
        包含系统消息、用户消息和必要的调试信息。
    """

    normalized_question = (
        normalize_question(question)
    )

    if not isinstance(
        built_context,
        BuiltContext,
    ):
        raise TypeError(
            "built_context 必须是 BuiltContext，"
            f"实际类型：{type(built_context)!r}"
        )

    if not isinstance(
        system_prompt,
        str,
    ):
        raise TypeError(
            "system_prompt 必须是字符串。"
        )

    normalized_system_prompt = (
        system_prompt.strip()
    )

    if not normalized_system_prompt:
        raise ValueError(
            "system_prompt 不能为空。"
        )

    context_text = (
        built_context.text.strip()
    )

    user_prompt = build_user_prompt(
        question=normalized_question,
        context=context_text,
    )

    messages: list[BaseMessage] = [
        SystemMessage(
            content=normalized_system_prompt
        ),
        HumanMessage(
            content=user_prompt
        ),
    ]

    return RAGPrompt(
        question=normalized_question,
        context=context_text,
        system_prompt=(
            normalized_system_prompt
        ),
        user_prompt=user_prompt,
        messages=messages,
        context_available=bool(
            context_text
        ),
    )


def messages_to_text(
    messages: list[BaseMessage],
) -> str:
    """将消息列表转换成便于调试的纯文本。

    注意：
    该函数只用于控制台查看，不用于正式模型调用。
    """

    blocks: list[str] = []

    for message in messages:
        if isinstance(
            message,
            SystemMessage,
        ):
            role = "SYSTEM"

        elif isinstance(
            message,
            HumanMessage,
        ):
            role = "USER"

        else:
            role = (
                message.__class__.__name__
                .upper()
            )

        blocks.append(
            f"[{role}]\n"
            f"{message.content}"
        )

    return "\n\n".join(blocks)