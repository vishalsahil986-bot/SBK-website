"""
Sadabahar Restaurant Chatbot — Conversation Summarizer

Calls the LLM to generate a structured summary of a single user↔bot exchange.
Each summary captures:
  - user_intent  : what the user was trying to accomplish
  - bot_response : what the bot said (condensed)
  - context      : any important context for future turns
"""

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from utils.logger import logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
async def summarize_conversation(
    user_message: str,
    bot_response: str,
    llm_chain,
) -> dict:
    """
    Summarizes a single conversation turn into a structured dict.

    Args:
        user_message : The user's message in this turn.
        bot_response : The bot's response in this turn.
        llm_chain    : The LangChain LLM chain instance.

    Returns:
        {
            "user_intent": str,
            "bot_response": str,
            "context": str
        }

    Falls back to a basic summary if LLM call or JSON parsing fails.
    """
    summarization_prompt = f"""
You are a conversation summarizer for a restaurant chatbot system.

Summarize the following exchange in strict JSON format with exactly these three keys:
- "user_intent"  : A brief description of what the user wanted or asked (1 sentence).
- "bot_response" : A brief summary of what the bot replied (1-2 sentences).
- "context"      : Any useful context that should be remembered for future turns
                   (e.g., user preferences, items they liked, delivery area, etc.).

Exchange to summarize:
USER: {user_message}
BOT: {bot_response}

Respond ONLY with valid JSON. No preamble, no markdown, no explanation.
Example format:
{{
  "user_intent": "User asked about available pizzas.",
  "bot_response": "Bot listed Margherita and Pepperoni as available options.",
  "context": "User is interested in pizza. May want to place an order."
}}
""".strip()

    try:
        raw = await llm_chain.ainvoke({"input": summarization_prompt})
        raw_text = raw.content if hasattr(raw, "content") else str(raw)

        # Strip markdown fences if present
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        summary = json.loads(clean)

        # Validate expected keys
        required_keys = {"user_intent", "bot_response", "context"}
        if not required_keys.issubset(summary.keys()):
            raise ValueError(f"Summary missing keys: {required_keys - summary.keys()}")

        logger.debug(f"Summary generated: {summary}")
        return summary

    except Exception as e:
        logger.warning(f"Summarization failed ({e}), using fallback summary.")
        return _fallback_summary(user_message, bot_response)


def _fallback_summary(user_message: str, bot_response: str) -> dict:
    """
    Basic fallback summary used when the LLM summarizer fails.
    Truncates long strings to keep context lean.
    """
    return {
        "user_intent": user_message[:150],
        "bot_response": bot_response[:150],
        "context": "Summary auto-generated due to LLM failure.",
    }