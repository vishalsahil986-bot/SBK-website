from utils.logger import logger


def get_context_for_llm(
    session: dict,
    current_message: str,
    retrieved_context: str = "",
) -> str:
    """
    Build the complete context sent to the LLM.

    Combines:
    1. Relevant RAG portfolio knowledge
    2. Conversation memory
    3. Current user message
    """

    msg_count = session.get("message_count", 0)
    summaries = session.get("summaries", [])
    last_messages = session.get("last_messages", [])

    # RAG knowledge
    rag_section = ""

    if retrieved_context and retrieved_context.strip():
        rag_section = (
            "[Relevant Portfolio Knowledge]\n"
            f"{retrieved_context.strip()}\n\n"
        )

    # First message
    if msg_count == 0:
        logger.debug(
            "Context phase: 1 (first message + RAG)"
        )

        return (
            f"{rag_section}"
            "[Current Message]\n"
            f"{current_message}"
        )

    # Second message
    if msg_count == 1 and last_messages:
        logger.debug(
            "Context phase: 2 (previous exchange + RAG)"
        )

        prev_user = (
            last_messages[0].get("content", "")
            if len(last_messages) > 0
            else ""
        )

        prev_bot = (
            last_messages[1].get("content", "")
            if len(last_messages) > 1
            else ""
        )

        return (
            f"{rag_section}"
            "[Previous Conversation]\n"
            f"User: {prev_user}\n"
            f"Assistant: {prev_bot}\n\n"
            "[Current Message]\n"
            f"{current_message}"
        )

    # Third message and beyond
    logger.debug(
        f"Context phase: 3+ "
        f"(msg #{msg_count + 1}, "
        f"{len(summaries)} summaries + RAG)"
    )

    context_parts = []

    if rag_section:
        context_parts.append(
            rag_section.strip()
        )

    if summaries:
        context_parts.append(
            "[Conversation Summary]"
        )

        for i, summary in enumerate(
            summaries,
            start=1,
        ):
            context_parts.append(
                f"[Exchange {i}]\n"
                f"User Intent: "
                f"{summary.get('user_intent', '')}\n"
                f"Bot Response: "
                f"{summary.get('bot_response', '')}\n"
                f"Context: "
                f"{summary.get('context', '')}"
            )

    context_parts.append(
        f"[Current Message]\n"
        f"{current_message}"
    )

    return "\n\n".join(context_parts)