import logging
from pathlib import Path

def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)