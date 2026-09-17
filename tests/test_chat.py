import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
 
from main import app
 
client = TestClient(app)

# Health Endpoint
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Sadabahar Restaurant Chatbot"
    assert "timestamp" in data

# Chat Endpoint — Input Validation
def test_chat_empty_message_returns_400():
    response = client.post("/chat", json={"session_id": "test1", "message": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_chat_whitespace_message_returns_400():
    response = client.post("/chat", json={"session_id": "test1", "message": "   "})
    assert response.status_code == 400

def test_chat_empty_session_id_returns_400():
    response = client.post("/chat", json={"session_id": "", "message": "Hello"})
    assert response.status_code == 400

def test_chat_message_too_long_returns_400():
    long_msg = "a" * 2001
    response = client.post("/chat", json={"session_id": "test1", "message": long_msg})
    assert response.status_code == 400
    assert "too long" in response.json()["detail"].lower() 

def test_chat_missing_fields_returns_422():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 422

# Chat Endpoint — Successful Response
@patch("services.chat_service.safe_invoke_llm", new_callable=AsyncMock)
def test_chat_returns_valid_response(mock_llm):
    mock_llm.return_value = "We have Margherita and Pepperoni pizzas!"
 
    response = client.post("/chat", json={
        "session_id": "integration_test_1",
        "message": "What pizzas do you have?"
    })
 
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "integration_test_1"
    assert data["response"] == "We have Margherita and Pepperoni pizzas!"
    assert data["message_count"] == 1

 
@patch("services.chat_service.safe_invoke_llm", new_callable=AsyncMock)
def test_chat_increments_message_count(mock_llm):
    mock_llm.return_value = "Bot reply"
    sid = "count_test_session"
 
    for i in range(3):
        response = client.post("/chat", json={"session_id": sid, "message": f"Message {i}"})
        assert response.status_code == 200
        assert response.json()["message_count"] == i + 1

# Chat Endpoint — LLM Failure Handling
@patch("services.chat_service.safe_invoke_llm", new_callable=AsyncMock)
def test_chat_llm_failure_returns_503(mock_llm):
    mock_llm.side_effect = Exception("LLM connection failed")
 
    response = client.post("/chat", json={
        "session_id": "llm_fail_test",
        "message": "Hello"
    })
 
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()

# Session Isolation Test
@patch("services.chat_service.safe_invoke_llm", new_callable=AsyncMock)
def test_sessions_are_isolated(mock_llm):
    mock_llm.return_value = "Reply"
 
    client.post("/chat", json={"session_id": "session_A", "message": "Hi"})
    client.post("/chat", json={"session_id": "session_A", "message": "Again"})
 
    # Session B should start fresh
    response = client.post("/chat", json={"session_id": "session_B", "message": "First"})
    assert response.json()["message_count"] == 1
 