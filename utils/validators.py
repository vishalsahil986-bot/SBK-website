from fastapi import HTTPException

MAX_MESSAGE_LENGTH = 2000

def validate_message(message: str) -> str:
    """
    Validates and sanitizes the incoming user message.
 
    Raises:
        HTTPException 400 — if message is empty or too long.
 
    Returns:
        Stripped, validated message string.
    """
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
 
    message = message.strip()
 
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed.",
        )
 
    return message

def validate_session_id(session_id: str) -> str:
    """
    Validates the session_id field.
 
    Raises:
        HTTPException 400 — if session_id is empty or malformed.
 
    Returns:
        Stripped session_id string.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")
 
    session_id = session_id.strip()
 
    if len(session_id) > 128:
        raise HTTPException(status_code=400, detail="session_id is too long.")
 
    return session_id