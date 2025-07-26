# refactored_fyers_swing/utils/logger.py

import logging
from config import LOG_FILE

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
