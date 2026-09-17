import pytest
from memory.memory_manager import (
    get_session,
    save_session,
    increment_message_count,
    append_summary,
    set_last_messages,
    delete_session,
    _empty_session,
)
from memory.context_builder import get_context_for_llm

# Session Management Tests
def test_get_session_creates_empty_session():
    session = get_session("test_new_session_xyz")
    assert session["summaries"] == []
    assert session["last_messages"] == []
    assert session["message_count"] == 0

 
def test_save_and_retrieve_session():
    sid = "test_save_retrieve"
    session = _empty_session()
    session["message_count"] = 5
    save_session(sid, session)
 
    loaded = get_session(sid)
    assert loaded["message_count"] == 5
    delete_session(sid)

def test_delete_session():
    sid = "test_delete"
    session = _empty_session()
    save_session(sid, session)
    delete_session(sid)
 
    fresh = get_session(sid)
    assert fresh["message_count"] == 0

# Summary Rolling Window Tests
def test_append_summary_within_limit():
    session = _empty_session()
    for i in range(3):
        session = append_summary(session, {
            "user_intent": f"intent {i}",
            "bot_response": f"response {i}",
            "context": f"context {i}",
        })
    assert len(session["summaries"]) == 3

def test_append_summary_enforces_max_5():
    session = _empty_session()
    for i in range(7):
        session = append_summary(session, {
            "user_intent": f"intent {i}",
            "bot_response": f"response {i}",
            "context": f"context {i}",
        })
    assert len(session["summaries"]) == 5
    # Oldest should be dropped — first summary should now be index 2
    assert session["summaries"][0]["user_intent"] == "intent 2"

 
def test_set_last_messages():
    session = _empty_session()
    session = set_last_messages(session, "hello", "hi there!")
    assert session["last_messages"][0]["role"] == "user"
    assert session["last_messages"][0]["content"] == "hello"
    assert session["last_messages"][1]["role"] == "assistant"
    assert session["last_messages"][1]["content"] == "hi there!"

# Context Phase Tests
def test_phase_1_returns_message_only():
    session = _empty_session()
    context = get_context_for_llm(session, "What burgers do you have?")
    assert context == "What burgers do you have?"
    assert "[Previous Conversation]" not in context
    assert "[Conversation Summary]" not in context

def test_phase_2_includes_previous_exchange():
    session = _empty_session()
    session["message_count"] = 1
    session = set_last_messages(session, "Hello", "Hi! Welcome to Sadabahar!")
 
    context = get_context_for_llm(session, "What pizzas do you have?")
    assert "[Previous Conversation]" in context
    assert "Hello" in context
    assert "Hi! Welcome to Sadabahar!" in context
    assert "What pizzas do you have?" in context

def test_phase_3_uses_summaries():
    session = _empty_session()
    session["message_count"] = 3
    session = append_summary(session, {
        "user_intent": "User asked about pizzas",
        "bot_response": "Recommended Margherita and Pepperoni",
        "context": "User likes pizza",
    })
    session = append_summary(session, {
        "user_intent": "User asked about delivery",
        "bot_response": "Delivery within 10 km",
        "context": "User is within delivery range",
    })
 
    context = get_context_for_llm(session, "Can I order a Margherita?")
    assert "[Conversation Summary]" in context
    assert "User asked about pizzas" in context
    assert "Can I order a Margherita?" in context
    assert "[Previous Conversation]" not in context

 