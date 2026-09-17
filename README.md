# 🤖 Portfolio AI Assistant

> A production-ready AI-powered personal assistant for my portfolio website, built with FastAPI, LangChain, and Gemini.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [File Structure](#file-structure)
5. [Setup & Installation](#setup--installation)
6. [Environment Variables](#environment-variables)
7. [API Reference](#api-reference)
8. [Memory System](#memory-system)
9. [Development Phases](#development-phases)
10. [Error Handling](#error-handling)
11. [Production Deployment](#production-deployment)

---

## Overview

**Portfolio AI Assistant** is a conversational AI chatbot that lives on my portfolio website. Visitors can ask it about my skills, experience, projects, and how to get in touch.

- Answers questions about Vishal Sahil's skills and experience
- Session-isolated conversations (no cross-user data leakage)
- Intelligent hybrid memory with progressive summarization
- Embeddable in any React / Next.js website

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                     │
│              Portfolio Website — Floating Chat Widget       │
└────────────────────────┬────────────────────────────────────┘
                         │  HTTP POST /chat
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Routes     │  │  Middleware  │  │  Error Handlers  │  │
│  │  /chat       │  │  CORS        │  │  Validation      │  │
│  │  /health     │  │  Logging     │  │  LLM Fallback    │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Chat Service                           │   │
│  │  ┌─────────────────┐   ┌──────────────────────────┐ │   │
│  │  │  Memory Manager │   │    LangChain LLM Chain   │ │   │
│  │  │  - summarize()  │   │    - System Prompt       │ │   │
│  │  │  - update()     │   │    - Context Builder     │ │   │
│  │  │  - get_context()│   │    - Response Generator  │ │   │
│  │  └────────┬────────┘   └──────────────────────────┘ │   │
│  └───────────┼──────────────────────────────────────────┘   │
└──────────────┼──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Memory Storage Layer                      │
│  In-Memory Dict (Demo) / Redis (Production)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### Core Features
- **Session Management** — Each visitor gets a unique `session_id`; conversations are fully isolated
- **Hybrid Summarization Memory** — Smart 3-phase memory that keeps context without bloating the LLM prompt
- **Portfolio-Aware Responses** — Only answers about Vishal's skills, projects, and experience
- **No Hallucination** — Strictly uses provided profile data only

### Technical Features
- REST API with FastAPI
- CORS-enabled for Next.js integration
- Structured logging with Loguru
- Graceful error handling (empty input, LLM failure)
- Modular codebase (routes, services, memory, utils)

---

## File Structure

```
portfolio-chatbot/
│
├── README.md
├── requirements.txt
├── .env                             ← never commit
├── .env.example
├── .gitignore
├── render.yaml                      ← Render deployment config
│
├── main.py                          ← FastAPI entry point
│
├── routes/
│   ├── __init__.py
│   ├── chat.py                      ← POST /chat
│   └── health.py                    ← GET /health
│
├── services/
│   ├── __init__.py
│   ├── chat_service.py              ← Chat pipeline orchestration
│   └── llm_service.py              ← Gemini LLM setup & invocation
│
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py            ← Session store (dict/Redis)
│   ├── summarizer.py                ← Summarization logic
│   └── context_builder.py          ← 3-phase context builder
│
├── utils/
│   ├── __init__.py
│   ├── validators.py                ← Input validation
│   └── logger.py                    ← Structured logging
│
├── config/
│   ├── __init__.py
│   └── settings.py                  ← Pydantic settings / env loader
│
└── prompts/
    └── system_prompt.py             ← Portfolio info + AI rules
```

---

## Setup & Installation

This is a **private repository** for personal use only.

Want to build something similar for yourself? Use the public open-source version as a base:

```bash
git clone https://github.com/vishalsahilai/chatbot-fastapi.git
```

Then:
1. Replace `prompts/system_prompt.py` with your own personal info
2. Remove `utils/menu.py` (not needed for portfolio)
3. Follow the setup steps from that repo's README
4. Deploy to Render for free

---

## Environment Variables

```env
# ─────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────
GOOGLE_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512

# ─────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false

# ─────────────────────────────────────────
# Memory Configuration
# ─────────────────────────────────────────
MEMORY_BACKEND=dict
MAX_SUMMARIES=5

# ─────────────────────────────────────────
# Redis (optional — for production)
# ─────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# ─────────────────────────────────────────
# CORS
# ─────────────────────────────────────────
CORS_ORIGINS=["https://vishalsahilai.vercel.app"]
```

---

## API Reference

### `POST /chat`

Send a message to the portfolio assistant.

**Request:**
```json
{
  "session_id": "user_abc123",
  "message": "What are Vishal's skills?"
}
```

**Success Response `200 OK`:**
```json
{
  "session_id": "user_abc123",
  "response": "Vishal specializes in AI Automation, LangChain, FastAPI, and Prompt Engineering...",
  "message_count": 1
}
```

**Error Response `400 Bad Request`:**
```json
{
  "detail": "Message cannot be empty."
}
```

**Error Response `503 Service Unavailable`:**
```json
{
  "detail": "LLM service is temporarily unavailable. Please try again."
}
```

---

### `GET /health`

```json
{
  "status": "healthy",
  "service": "Portfolio AI Assistant",
  "version": "1.0.0",
  "environment": "production",
  "timestamp": "2026-07-30T10:30:00Z"
}
```

---

## Memory System

3-phase hybrid memory — keeps context lean without bloating the LLM prompt.

```
Message 1  →  Send directly to LLM (no prior context)
Message 2  →  Send full previous exchange + current message
Message 3+ →  Send summaries only + current message
             (NO full history — prevents context bloat)
```

Each summary stored in memory:
```json
{
  "user_intent": "Asked about Vishal's AI skills",
  "bot_response": "Listed LangChain, Gemini, OpenAI experience",
  "context": "Visitor may be evaluating Vishal for a project or hire"
}
```

Max 5 summaries stored per session — oldest dropped automatically.

---

## Development Phases

- **Phase 1** — Project setup + virtual environment
- **Phase 2** — FastAPI backend + CORS middleware
- **Phase 3** — Gemini LLM integration + portfolio system prompt
- **Phase 4** — Hybrid summarization memory system
- **Phase 5** — API endpoints + input validation
- **Phase 6** — Next.js floating chat widget integration
- **Phase 7** — Deploy to Render (free tier)

---

## Error Handling

| Scenario | HTTP Status | Response |
|---|---|---|
| Empty message | `400` | `"Message cannot be empty."` |
| Message too long (>2000 chars) | `400` | `"Message too long."` |
| LLM timeout or failure | `503` | `"LLM service unavailable."` |
| Session not found | `200` | New session auto-created |
| Invalid JSON body | `422` | FastAPI validation error |

---

## Production Deployment

Deployed on **Render** (free tier) — connected to portfolio at [vishalsahilai.vercel.app](https://vishalsahilai.vercel.app).

```yaml
services:
  - type: web
    name: portfolio-chatbot
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python -m uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.10+) |
| LLM Framework | LangChain |
| LLM Provider | Google Gemini (gemini-1.5-flash) |
| Memory (dev) | In-memory Python dict |
| Memory (prod) | Redis |
| Frontend | Next.js — [vishalsahilai.vercel.app](https://vishalsahilai.vercel.app) |
| Deployment | Render (free tier) |

---

## License

Private — Personal use only. Not open for public contributions.

---

> Built by [Vishal Sahil](https://vishalsahilai.vercel.app) · AI Automation Engineer · Karachi, Pakistan