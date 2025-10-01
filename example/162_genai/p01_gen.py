import numpy as np
import pandas as pd
import sys
import os
from sqlite_minutils import *
from loguru import logger
from google import genai
from google.genai import types

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
