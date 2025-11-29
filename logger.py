import os
from loguru import logger
import sys
from datetime import datetime


os.makedirs("logs", exist_ok=True)
log_file = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(log_file, level="DEBUG", rotation="10 MB", retention="7 days")


logger.info("Logger ready — All actions will be tracked")