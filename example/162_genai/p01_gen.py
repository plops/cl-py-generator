import numpy as np
import pandas as pd
import sys
import os
import yaml
import pydantic_core
from sqlite_minutils import *
from loguru import logger
from google import genai
from pydantic import BaseModel
from google.genai import types

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")


class Recipe(BaseModel):
    title: str
    summary: list[str]


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model = "gemini-flash-latest"
contents = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=r"""make a summary about the most recent news about bill gates"""
            )
        ],
    )
]
tools = [types.Tool(googleSearch=types.GoogleSearch())]
think_max_budget_flash = 24576
think_auto_budget = -1
think_off = 0
generate_content_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinkingBudget=think_auto_budget, include_thoughts=True
    ),
    tools=tools,
    response_mime_type="text/plain",
    response_schema=list[Recipe],
)
thoughts = ""
answer = ""
responses = []
for chunk in client.models.generate_content_stream(
    model=model, contents=contents, config=generate_content_config
):
    for part in chunk.candidates[0].content.parts:
        responses.append(chunk)
        print(chunk)
        if not (part.text):
            continue
        elif part.thought:
            print(part.text)
            thoughts += part.text
        else:
            print(part.text)
            answer += part.text
with open("out.yaml", "w", encoding="utf-8") as f:
    yaml.dump(responses, f, allow_unicode=True, indent=2)
print(thoughts)
print(answer)
