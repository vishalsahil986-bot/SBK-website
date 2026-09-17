from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from routes.chat import router as chat_router
from routes.health import router as health_router
from utils.logger import logger 


# App Initialization
app = FastAPI(
    title="Portfolio AI Assistant",
    description="AI-powered personal assistant for Vishal Sahil's portfolio",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"], #Accept any HTTP method — GET, POST, PUT, DELETE, PATCH, etc.
    allow_headers=["*"] #Accept any request header — Authorization, Content-Type, custom headers, etc.
)

# Routers
app.include_router(chat_router)
app.include_router(health_router)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

#startup Event
#Exception is the Base Class of ALL errors in Python
@app.on_event("startup")
async def on_startup():
    logger.info("Sadabahar Restaurant Chatbot API is starting up...")
    logger.info(f"Environment: {settings.app_env}")

#shutdown Event
@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Sadabahar Restaurant Chatbot API is shutting down...")
