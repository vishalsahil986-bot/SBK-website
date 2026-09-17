from fastapi import HTTPException
import asyncio

from memory.memory_manager import (
    get_session,
    save_session,
    increment_message_count,
    append_summary,
    set_last_messages,
)
from memory.context_builder import get_context_for_llm
from memory.summarizer import summarize_conversation
from services.llm_service import safe_invoke_llm, get_summarizer_llm
from rag.retriever import retriever
from utils.logger import logger


async def process_chat(session_id: str, user_message: str) -> dict:
    """
    Main chat handler.

    Flow:
    1. Load conversation memory
    2. Retrieve relevant portfolio knowledge from Pinecone
    3. Combine RAG knowledge + conversation memory
    4. Send combined context to Gemini
    5. Save conversation memory
    """

    # Load session
    session = get_session(session_id)
    msg_count = session["message_count"]

    logger.info(
        f"[{session_id}] Processing message "
        f"#{msg_count + 1}: '{user_message[:60]}...'"
    )

    # ─────────────────────────────────────
    # Retrieve relevant knowledge from RAG
    # ─────────────────────────────────────

    try:
        retrieved_context = await asyncio.to_thread(
            retriever.retrieve,
            user_message,
        )

        logger.debug(
            f"[{session_id}] RAG context retrieved "
            f"({len(retrieved_context)} chars)"
        )

    except Exception as e:
        # RAG failure should not completely break the chatbot
        logger.warning(
            f"[{session_id}] RAG retrieval failed: {e}"
        )
        retrieved_context = ""

    # ─────────────────────────────────────
    # Build final LLM context
    # ─────────────────────────────────────

    context = get_context_for_llm(
        session=session,
        current_message=user_message,
        retrieved_context=retrieved_context,
    )

    logger.debug(
        f"[{session_id}] Context built "
        f"({len(context)} chars)"
    )

    # ─────────────────────────────────────
    # Invoke LLM
    # ─────────────────────────────────────

    try:
        bot_response = await safe_invoke_llm(context)

    except Exception as e:
        logger.error(
            f"[{session_id}] LLM failure: {e}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "LLM service is temporarily unavailable. "
                "Please try again in a moment."
            ),
        )

    # ─────────────────────────────────────
    # Update conversation memory
    # ─────────────────────────────────────

    session = increment_message_count(session)

    if session["message_count"] >= 2:
        try:
            summary = await summarize_conversation(
                user_message=user_message,
                bot_response=bot_response,
                llm_chain=get_summarizer_llm(),
            )

            session = append_summary(
                session,
                summary,
            )

            logger.debug(
                f"[{session_id}] Summary appended. "
                f"Total summaries: "
                f"{len(session['summaries'])}"
            )

        except Exception as e:
            logger.warning(
                f"[{session_id}] "
                f"Summarization skipped: {e}"
            )

    # Store last user + assistant messages
    session = set_last_messages(
        session,
        user_message,
        bot_response,
    )

    # Save session
    save_session(
        session_id,
        session,
    )

    logger.info(
        f"[{session_id}] Response sent "
        f"({len(bot_response)} chars)"
    )

    return {
        "session_id": session_id,
        "response": bot_response,
        "message_count": session["message_count"],
    }