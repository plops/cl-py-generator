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
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model = "gemini-flash-latest"
contents = [
    types.Content(
        role="user", parts=[types.Part.from_text(text=r"""tell me a jokes""")]
    )
]
tools = [types.Tool(googleSearch=types.GoogleSearch())]
generate_content_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinkingBudget=24576), tools=tools
)
for chunk in client.models.generate_content_stream(
    model=model, contents=contents, config=generate_content_config
):
    print(chunk.text, end="")
