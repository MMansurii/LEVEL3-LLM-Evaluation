import logging
import sys
from pathlib import Path

from config.settings import Settings


def get_logger(name: str) -> logging.Logger:
    """Get configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        formatter = logging.Formatter(Settings.LOG_FORMAT)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        log_file = Settings.OUTPUT_DIR / "evaluation.log"
        Settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.setLevel(getattr(logging, Settings.LOG_LEVEL))
    
    return logger
