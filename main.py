from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from routes.chat import router as chat_router
from routes.health import router as health_router
from utils.logger import logger


# ─────────────────────────────────────
# App Initialization
# ─────────────────────────────────────

app = FastAPI(
    title="SADA BAHAR KOHISTAN AI Assistant",
    description=(
        "RAG-powered AI assistant for the "
        "SADA BAHAR KOHISTAN website."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────
# CORS
# ─────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────
# Routers
# ─────────────────────────────────────

app.include_router(chat_router)
app.include_router(health_router)


# ─────────────────────────────────────
# Root Route
# ─────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "SADA BAHAR KOHISTAN AI Assistant",
        "status": "running",
        "health": "/health",
        "docs": "/docs",
    }


# ─────────────────────────────────────
# Global Exception Handler
# ─────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc,
):
    logger.exception(
        f"Unhandled exception: {exc}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "An unexpected error occurred. "
                "Please try again."
            )
        },
    )


# ─────────────────────────────────────
# Startup
# ─────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    logger.info(
        "SADA BAHAR KOHISTAN AI Assistant "
        "API is starting up..."
    )

    logger.info(
        f"Environment: {settings.app_env}"
    )

    logger.info(
        f"Pinecone index: "
        f"{settings.pinecone_index_name}"
    )

    logger.info(
        f"Pinecone namespace: "
        f"{settings.pinecone_namespace}"
    )


# ─────────────────────────────────────
# Shutdown
# ─────────────────────────────────────

@app.on_event("shutdown")
async def on_shutdown():
    logger.info(
        "SADA BAHAR KOHISTAN AI Assistant "
        "API is shutting down..."
    )