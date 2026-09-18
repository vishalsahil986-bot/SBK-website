"""
LLM Service

Primary:
    Alibaba / Qwen model rotation

Fallback:
    Gemini API key rotation

Flow:
Alibaba models
    ↓
Gemini key 1
    ↓
Gemini key 2
    ↓
Gemini key 3
    ↓
Gemini key 4
"""

import asyncio
from typing import Any

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from config.settings import settings
from prompts.system_prompt import SYSTEM_PROMPT

from services.alibaba_service import (
    AlibabaApplicationError,
    AlibabaNotConfiguredError,
    AlibabaPartialStreamError,
    AlibabaProviderUnavailableError,
    AllAlibabaModelsFailed,
    alibaba_service,
)

from utils.logger import logger


# ─────────────────────────────────────
# Exceptions
# ─────────────────────────────────────

class AllLLMProvidersFailed(Exception):
    pass


class LLMApplicationError(Exception):
    pass


# ─────────────────────────────────────
# Gemini API Keys
# ─────────────────────────────────────

API_KEYS = [
    settings.google_api_key_1,
    settings.google_api_key_2,
    settings.google_api_key_3,
    settings.google_api_key_4,
]

# Remove empty keys
API_KEYS = [
    key.strip()
    for key in API_KEYS
    if key and key.strip()
]

_current_key_index = 0


# ─────────────────────────────────────
# Gemini Helpers
# ─────────────────────────────────────

def _build_gemini_llm(
    api_key: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_budget=0,
        google_api_key=api_key,
    )


def _gemini_key_order() -> list[int]:
    """
    Start from currently preferred key,
    then try every other configured key.
    """

    if not API_KEYS:
        return []

    return [
        (
            _current_key_index + offset
        )
        % len(API_KEYS)
        for offset in range(
            len(API_KEYS)
        )
    ]


def _set_current_key(
    index: int,
) -> None:
    global _current_key_index

    _current_key_index = index


def _extract_response_text(
    response: Any,
) -> str:

    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:

            if isinstance(block, str):
                parts.append(block)
                continue

            if isinstance(block, dict):
                text = block.get(
                    "text",
                    "",
                )

                if isinstance(
                    text,
                    str,
                ):
                    parts.append(text)

                continue

            text = getattr(
                block,
                "text",
                "",
            )

            if isinstance(text, str):
                parts.append(text)

        return "".join(
            parts
        ).strip()

    return str(content).strip()


def _is_non_retryable_gemini_error(
    exc: Exception,
) -> bool:
    """
    Errors where changing API keys is
    unlikely to help.
    """

    message = str(exc).lower()

    non_retryable = (
        "400",
        "422",
        "invalid argument",
        "invalid_argument",
        "bad request",
        "malformed",
    )

    return any(
        keyword in message
        for keyword in non_retryable
    )


def _is_retryable_gemini_error(
    exc: Exception,
) -> bool:

    message = str(exc).lower()

    retryable = (
        "429",
        "quota",
        "resource exhausted",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "too many requests",
        "limit exceeded",

        "401",
        "403",

        "408",
        "500",
        "502",
        "503",
        "504",

        "timeout",
        "timed out",
        "deadline exceeded",

        "service unavailable",
        "temporarily unavailable",
        "internal error",

        "connection",
        "network",
        "connection reset",
    )

    return any(
        keyword in message
        for keyword in retryable
    )


# ─────────────────────────────────────
# Gemini Invocation
# ─────────────────────────────────────

async def _invoke_gemini_raw(
    input_data,
    *,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """
    Try every Gemini API key.

    Each key receives a strict timeout.
    """

    if not API_KEYS:
        raise AllLLMProvidersFailed(
            "No Gemini API keys configured"
        )

    last_error = None

    for index in _gemini_key_order():

        key = API_KEYS[index]

        logger.info(
            f"LLM provider=Gemini "
            f"key_index={index} attempting"
        )

        try:
            llm = _build_gemini_llm(
                api_key=key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response = await asyncio.wait_for(
                llm.ainvoke(
                    input_data
                ),
                timeout=float(
                    settings.llm_timeout_seconds
                ),
            )

            _set_current_key(index)

            logger.info(
                f"LLM provider=Gemini "
                f"key_index={index} success"
            )

            return response

        except asyncio.TimeoutError as exc:

            last_error = exc

            logger.warning(
                f"Gemini key index {index} "
                f"timed out after "
                f"{settings.llm_timeout_seconds}s "
                "— rotating"
            )

            continue

        except Exception as exc:

            last_error = exc

            if _is_non_retryable_gemini_error(
                exc
            ):
                logger.error(
                    "Gemini rejected the "
                    f"request: {exc}"
                )

                raise LLMApplicationError(
                    str(exc)
                ) from exc

            if _is_retryable_gemini_error(
                exc
            ):
                logger.warning(
                    f"Gemini key index {index} "
                    f"failed: {exc} "
                    "— rotating"
                )

                continue

            # Unknown provider error:
            # try the next key instead of
            # immediately killing the chat.
            logger.warning(
                f"Gemini key index {index} "
                f"unexpected error: {exc} "
                "— trying next key"
            )

            continue

    raise AllLLMProvidersFailed(
        "All Gemini API keys failed"
    ) from last_error


async def _invoke_gemini(
    user_context: str,
) -> str:

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=user_context
        ),
    ]

    response = await _invoke_gemini_raw(
        messages,
        temperature=0.3,
        max_tokens=1024,
    )

    reply = _extract_response_text(
        response
    )

    if not reply:
        raise AllLLMProvidersFailed(
            "Gemini returned empty response"
        )

    return reply


# ─────────────────────────────────────
# Main LLM Invocation
# ─────────────────────────────────────

async def invoke_llm(
    user_context: str,
) -> str:
    """
    Primary:
        Alibaba / Qwen

    Fallback:
        Gemini key rotation
    """

    # ── 1. Alibaba first ──────────────

    try:
        logger.info(
            "LLM primary provider: Alibaba"
        )

        reply = await (
            alibaba_service.generate(
                user_context
            )
        )

        if reply:
            logger.info(
                "LLM response served "
                "by Alibaba"
            )

            return reply

        logger.warning(
            "Alibaba returned an empty "
            "response — switching to Gemini"
        )

    except AlibabaNotConfiguredError:

        logger.warning(
            "Alibaba not configured "
            "— switching to Gemini"
        )

    except AlibabaProviderUnavailableError as exc:

        logger.warning(
            f"Alibaba unavailable: {exc} "
            "— switching to Gemini"
        )

    except AllAlibabaModelsFailed as exc:

        logger.warning(
            f"{exc} "
            "— switching to Gemini"
        )

    except AlibabaPartialStreamError as exc:
        # Since this chatbot returns the final
        # complete answer rather than streaming
        # chunks directly to the browser,
        # Gemini can safely retry the request.
        logger.warning(
            f"Alibaba partial response "
            f"failed: {exc} "
            "— switching to Gemini"
        )

    except AlibabaApplicationError as exc:

        # Bad request / application error.
        # Do not blindly send the same invalid
        # request through every provider.
        logger.error(
            f"Alibaba application error: "
            f"{exc}"
        )

        raise LLMApplicationError(
            str(exc)
        ) from exc

    except Exception as exc:

        # Unexpected Alibaba provider problem.
        # Gemini still gets a chance.
        logger.warning(
            f"Unexpected Alibaba error: "
            f"{exc} "
            "— switching to Gemini"
        )

    # ── 2. Gemini fallback ────────────

    logger.info(
        "LLM fallback: Alibaba → Gemini"
    )

    return await _invoke_gemini(
        user_context
    )


# ─────────────────────────────────────
# Public Safe Wrapper
# ─────────────────────────────────────

async def safe_invoke_llm(
    user_context: str,
) -> str:

    try:
        return await invoke_llm(
            user_context
        )

    except LLMApplicationError:
        raise

    except AllLLMProvidersFailed:
        raise

    except Exception as exc:

        logger.exception(
            f"safe_invoke_llm failed: "
            f"{exc}"
        )

        raise AllLLMProvidersFailed(
            "All LLM providers failed"
        ) from exc


# ─────────────────────────────────────
# Summarizer Adapter
# ─────────────────────────────────────

class SummarizerLLM:
    """
    Uses the same provider flow as the main chatbot:

    Alibaba first
        ↓
    Gemini fallback only if Alibaba fails
    """

    async def ainvoke(
        self,
        input_data,
    ):
        # Convert input to plain text
        if isinstance(input_data, str):
            prompt = input_data
        else:
            prompt = str(input_data)

        # Uses:
        # Alibaba → Gemini fallback
        return await invoke_llm(prompt)


_summarizer_llm = SummarizerLLM()


def get_summarizer_llm():
    return _summarizer_llm


# ─────────────────────────────────────
# Compatibility
# ─────────────────────────────────────

def get_llm():
    """
    Kept for compatibility with any older
    code that imports get_llm().

    Returns the currently preferred
    Gemini model instance.
    """

    if not API_KEYS:
        raise RuntimeError(
            "No Gemini API keys configured"
        )

    return _build_gemini_llm(
        API_KEYS[_current_key_index]
    )