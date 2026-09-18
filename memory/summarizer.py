import json

from utils.logger import logger


async def summarize_conversation(
    user_message: str,
    bot_response: str,
    llm_chain,
) -> dict:
    """
    Create a short memory summary of one chat exchange.
    """

    prompt = f"""
Summarize this SADA BAHAR KOHISTAN chatbot exchange.

Return ONLY valid JSON:

{{
  "user_intent": "short intent",
  "bot_response": "short response summary",
  "context": "important context for future follow-ups"
}}

USER:
{user_message}

ASSISTANT:
{bot_response}
""".strip()

    try:
        # Send STRING directly, not {"input": prompt}
        response = await llm_chain.ainvoke(prompt)

        raw_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        clean = raw_text.strip()

        # Remove ```json fences if present
        if clean.startswith("```"):
            clean = clean.split("```", 2)[1]

            if clean.lower().startswith("json"):
                clean = clean[4:]

            clean = clean.strip()

        summary = json.loads(clean)

        return {
            "user_intent": str(
                summary.get(
                    "user_intent",
                    "",
                )
            )[:150],

            "bot_response": str(
                summary.get(
                    "bot_response",
                    "",
                )
            )[:200],

            "context": str(
                summary.get(
                    "context",
                    "",
                )
            )[:200],
        }

    except Exception as exc:
        logger.warning(
            f"Summarization failed ({exc}), "
            "using fallback summary."
        )

        return {
            "user_intent": (
                user_message[:150]
            ),
            "bot_response": (
                bot_response[:200]
            ),
            "context": (
                "Previous conversation "
                "retained using fallback memory."
            ),
        }