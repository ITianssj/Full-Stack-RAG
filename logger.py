"""
Logging Configuration Module

This module sets up centralized logging for the RAG Search Engine application using Loguru.
It provides both console and file logging with appropriate formatting and rotation policies.

Features:
- Colored console output for development
- File logging with automatic rotation and retention
- Timestamped log files for better organization
"""

import os
from loguru import logger
import sys
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate timestamped log file name
log_file = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

# Remove default logger configuration
logger.remove()

# Add console handler with colored output for development
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
)

# Add file handler with rotation and retention for production logging
logger.add(
    log_file,
    level="DEBUG",
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="7 days"  # Keep logs for 7 days
)

# Log initialization confirmation
logger.info("Logger ready — All actions will be tracked")
