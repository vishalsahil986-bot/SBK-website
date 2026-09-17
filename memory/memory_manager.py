import json

from config.settings import settings
from utils.logger import logger


# In-memory session storage
_IN_MEMORY_STORE: dict = {}

# Redis client
_redis_client = None


def _get_redis():
    global _redis_client

    if _redis_client is None:
        import redis

        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
        )

        logger.info(
            f"Redis connected: "
            f"{settings.redis_host}:{settings.redis_port}"
        )

    return _redis_client


def _empty_session() -> dict:
    return {
        "summaries": [],
        "last_messages": [],
        "message_count": 0,
    }


def _read_session(session_id: str) -> dict:
    if settings.memory_backend == "redis":
        raw = _get_redis().get(
            f"session:{session_id}"
        )

        if raw:
            return json.loads(raw)

        return _empty_session()

    return _IN_MEMORY_STORE.get(
        session_id,
        _empty_session(),
    )


def _write_session(
    session_id: str,
    data: dict,
) -> None:

    if settings.memory_backend == "redis":
        _get_redis().set(
            f"session:{session_id}",
            json.dumps(data),
            ex=86400,
        )
        return

    _IN_MEMORY_STORE[session_id] = data


# ─────────────────────────────────────
# Public API
# ─────────────────────────────────────

def get_session(session_id: str) -> dict:
    session = _read_session(session_id)

    logger.debug(
        f"[{session_id}] Session loaded — "
        f"msg_count={session['message_count']}"
    )

    return session


def save_session(
    session_id: str,
    session: dict,
) -> None:

    _write_session(
        session_id,
        session,
    )

    logger.debug(
        f"[{session_id}] Session saved — "
        f"msg_count={session['message_count']}"
    )


def increment_message_count(
    session: dict,
) -> dict:

    session["message_count"] += 1

    return session


def append_summary(
    session: dict,
    summary: dict,
) -> dict:

    session["summaries"].append(
        summary
    )

    if (
        len(session["summaries"])
        > settings.max_summaries
    ):
        session["summaries"].pop(0)

    return session


def set_last_messages(
    session: dict,
    user_msg: str,
    bot_response: str,
) -> dict:

    session["last_messages"] = [
        {
            "role": "user",
            "content": user_msg,
        },
        {
            "role": "assistant",
            "content": bot_response,
        },
    ]

    return session


def delete_session(
    session_id: str,
) -> None:

    if settings.memory_backend == "redis":
        _get_redis().delete(
            f"session:{session_id}"
        )
    else:
        _IN_MEMORY_STORE.pop(
            session_id,
            None,
        )

    logger.info(
        f"[{session_id}] Session deleted."
    )