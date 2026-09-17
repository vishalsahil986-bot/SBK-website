import sys 
from loguru import logger

#Remove default handler
logger.remove()

# Console handler — colorized, human-readable
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# File handler — structured for production log parsing
logger.add(
    "logs/sadabahar.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} — {message}",
    level="INFO",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

__all__ = ["logger"]