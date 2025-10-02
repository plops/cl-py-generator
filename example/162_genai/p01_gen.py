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
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model = "gemini-flash-latest"
contents = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=r"""make a summary about the most recent news about ethris stock"""
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
)
thoughts = ""
answer = ""
responses = []
t_submit = time.monotonic()
first_thought_time = None
last_thought_time = None
first_answer_time = None
final_answer_time = None
for chunk in client.models.generate_content_stream(
    model=model, contents=contents, config=generate_content_config
):
    for part in chunk.candidates[0].content.parts:
        responses.append(chunk)
        print(chunk)
        if not (part.text):
            continue
        elif part.thought:
            now = time.monotonic()
            if first_thought_time is None:
                logger.info("first thought")
                first_thought_time = now
            last_thought_time = now
            print(part.text)
            thoughts += part.text
        else:
            now = time.monotonic()
            if first_answer_time is None:
                logger.info("first answer")
                first_answer_time = now
            final_answer_time = now
            print(part.text)
            answer += part.text
# persist raw responses
with open("out.yaml", "w", encoding="utf-8") as f:
    yaml.dump(responses, f, allow_unicode=True, indent=2)


# helper to find the first non-null value across all responses using a provided extractor
def find_first(responses, extractor):
    for r in responses:
        try:
            v = extractor(r)
        except Exception:
            v = None
        if not (v is None):
            return v
    return None


def find_last(responses, extractor):
    for r in reversed(responses):
        try:
            v = extractor(r)
        except Exception:
            v = None
        if not (v is None):
            return v
    return None


# find the last response that contains usage metadata (for the aggregated token counts)
last_with_usage = None
for resp in reversed(responses):
    if getattr(resp, "usage_metadata", None) is not None:
        last_with_usage = resp
        break
if last_with_usage is not None:
    um = last_with_usage.usage_metadata
    d = {}
    d["candidates_token_count"] = getattr(um, "candidates_token_count", None)
    d["prompt_token_count"] = getattr(um, "prompt_token_count", None)
    d["thoughts_token_count"] = getattr(um, "thoughts_token_count", None)
    d["response_id"] = find_first(responses, lambda r: getattr(r, "response_id", None))
    d["model_version"] = find_first(
        responses, lambda r: getattr(r, "model_version", None)
    )
    totals = [
        getattr(getattr(r, "usage_metadata", None), "total_token_count", None)
        for r in responses
    ]
    valid_totals = [
        (tot)
        if (
            isinstance(
                tot,
                (
                    int,
                    float,
                ),
            )
        )
        else (None)
        for tot in totals
    ]
    d["total_token_count"] = (max(valid_totals)) if (valid_totals) else (None)
    d["finish_reason"] = find_last(
        responses,
        lambda r: (
            (getattr(r, "finish_reason", None))
            or (
                (getattr(getattr(r, "candidates", [None])[0], "finish_reason", None))
                if (getattr(r, "candidates", None))
                else (None)
            )
        ),
    )
    d["sdk_date"] = find_first(
        responses,
        lambda r: (
            (getattr(r, "sdk_http_response", None))
            and (getattr(r.sdk_http_response, "headers", {}).get("date"))
        ),
    )
    d["prompt_parsing_time"] = (first_thought_time) - (t_submit)
    d["thinking_time"] = (last_thought_time) - (first_thought_time)
    d["answer_time"] = (final_answer_time) - (last_thought_time)
logger.info(f"thoughts: {thoughts}")
logger.info(f"answer: {answer}")
logger.info(f"{d}")
