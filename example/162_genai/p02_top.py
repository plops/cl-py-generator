# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_top.py
import numpy as np
import pandas as pd
import sys
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
from p02_impl import GenerationConfig, GenAIJob

cfg = GenerationConfig(
    prompt_text="make a summary of the imminent government shutdown in the US. show historical parallels.",
    model="gemini-flash-latest",
    output_yaml_path="out.yaml",
    use_search=True,
    think_budget=-1,
    include_thoughts=True,
)
job = GenAIJob(cfg)
result = job.run()
logger.info(f"thoughts: {result.thoughts}")
logger.info(f"answer: {result.answer}")
logger.info(f"usage: {result.usage_summary}")
