import numpy as np
import pandas as pd
import sys
import os
import yaml
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
                text=r"""make a summary about the most recent news about eli lily stock"""
            )
        ],
    )
]
tools = [types.Tool(googleSearch=types.GoogleSearch())]
think_max_budget_flash = 24576
think_auto_budget = -1
think_off = 0
generate_content_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinkingBudget=think_auto_budget),
    tools=tools,
    response_mime_type="text/plain",
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
    d["create_time"] = find_first(responses, lambda r: getattr(r, "create_time", None))
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
    d["finish_reason"] = find_first(
        responses,
        lambda r: (
            (
                (getattr(r, "finish_reason", None))
                and (getattr(r.candidates[0], "finish_reason", None))
            )
            or (None)
        ),
    )
    d["sdk_date"] = find_first(
        responses,
        lambda r: (
            (getattr(r, "sdk_http_response", None))
            and (getattr(r.sdk_http_response, "headers", {}).get("date"))
        ),
    )
logger.info(f"thoughts: {thoughts}")
logger.info(f"answer: {answer}")
logger.info(f"{d}")
