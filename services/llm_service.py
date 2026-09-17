"""
LLM Service with API Key Rotation for Gemini
Automatically rotates to next key when quota is exceeded.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from prompts.system_prompt import SYSTEM_PROMPT
from utils.logger import logger


# ── API Key Pool ──────────────────────────────────────────────
API_KEYS = [
    settings.google_api_key_1,
    settings.google_api_key_2,
    settings.google_api_key_3,
    settings.google_api_key_4,
]

# Filter out empty keys
API_KEYS = [k for k in API_KEYS if k and k != ""]

# Current key index tracker
_current_key_index = 0


def _get_current_key() -> str:
    return API_KEYS[_current_key_index]


def _rotate_key():
    """Move to next available API key."""
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(API_KEYS)
    logger.warning(f"Rotated to API key index: {_current_key_index}")


def _build_llm(
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


async def invoke_llm(user_context: str) -> str:
    """
    Invoke LLM with automatic API key rotation on quota errors.
    Tries all available keys before giving up.
    """
    if not API_KEYS:
        raise RuntimeError("No API keys configured.")

    last_error = None

    for attempt in range(len(API_KEYS)):
        current_key = _get_current_key()
        try:
            llm = _build_llm(current_key)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_context),
            ]
            response = await llm.ainvoke(messages)

            # Handle Gemini response content type
            if isinstance(response.content, list):
                reply = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in response.content
                ).strip()
            else:
                reply = response.content.strip()

            logger.debug(f"Response from key index {_current_key_index}")
            return reply

        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a quota/rate limit error
            if any(keyword in error_msg for keyword in [
                "quota", "rate limit", "resource exhausted",
                "429", "limit exceeded", "too many requests"
            ]):
                logger.warning(f"Key index {_current_key_index} quota exceeded. Rotating...")
                _rotate_key()
                last_error = e
            else:
                # Non-quota error — don't rotate, just raise
                logger.error(f"LLM error (non-quota): {e}")
                raise

    # All keys exhausted
    logger.error("All API keys exhausted!")
    raise RuntimeError("All Gemini API keys have exceeded their quota.") from last_error


async def safe_invoke_llm(user_context: str) -> str:
    """Wrapper with top-level error handling."""
    try:
        return await invoke_llm(user_context)
    except Exception as e:
        logger.error(f"safe_invoke_llm failed: {e}")
        raise


def get_summarizer_llm():
    """Returns summarizer LLM with current active key."""
    return _build_llm(
        _get_current_key(),
        temperature=0.1,
        max_tokens=300,
    )


def get_llm():
    """Returns main LLM with current active key."""
    return _build_llm(_get_current_key())