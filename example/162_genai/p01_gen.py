import numpy as np
import pandas as pd
import sys
import os
import yaml
import time
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

# NEW: class-based generation
from g01_gen import GenerationConfig, GenAIJob  # noqa: E402


def main():
    cfg = GenerationConfig(
        prompt_text="make a summary about the most recent news about any vaccine developments against herpes viruses (mrna based or any other).",
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
    logger.info(f"{result.usage_summary}")


if __name__ == "__main__":
    main()
