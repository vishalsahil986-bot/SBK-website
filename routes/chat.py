from fastapi import APIRouter
from pydantic import BaseModel
 
from services.chat_service import process_chat
from utils.validators import validate_message, validate_session_id
from utils.logger import logger
 
router = APIRouter()

# Request Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
 
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user_abc123",
                "message": "What pizzas do you have?",
            }
        }
    }


# Response Models
class ChatResponse(BaseModel):
    session_id: str
    response: str
    message_count: int

# Endpoint
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the restaurant chatbot",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message and receive an AI response from Sada,
    the Sadabahar Restaurant chatbot.
 
    - **session_id**: Unique identifier for the user's session.
    - **message**: The user's message (max 2000 characters).
    """

    # Validate inputs
    session_id = validate_session_id(request.session_id)
    message = validate_message(request.message)
 
    logger.info(f"POST /chat — session={session_id}")

    # Delegate to chat service
    result = await process_chat(session_id=session_id, user_message=message)
 
    return ChatResponse(**result)