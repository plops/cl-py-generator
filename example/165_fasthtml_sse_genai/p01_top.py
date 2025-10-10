# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_top.py
from __future__ import annotations
import datetime
import argparse
from fasthtml.common import (
    Script,
    fast_app,
    Titled,
    Form,
    Fieldset,
    Legend,
    Div,
    Label,
    Textarea,
    Button,
    Request,
    signal_shutdown,
    sse_message,
    Article,
    EventStream,
    serve,
)
import os
import sys
import asyncio
from dataclasses import dataclass, field
from typing import List, Any, Dict
from loguru import logger
from google import genai
from google.genai import types

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the SSE AI Responder website")
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase verbosity: -v for DEBUG, -vv for TRACE",
)
args = parser.parse_args()
# Determine log level based on verbosity
if (args.verbose) == (1):
    log_level = "DEBUG"
elif (args.verbose) >= (2):
    log_level = "TRACE"
else:
    log_level = "INFO"
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level=log_level,
    enqueue=True,
)
logger.info("Logger configured")
# import after logger exists
from p02_impl import GenerationConfig, GenAIJob

hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs, live=True)


@rt
def index():
    return Titled(
        "SSE AI Responder",
        Form(
            Fieldset(
                Legend("Submit a prompt for the AI to respond to"),
                Div(
                    Label(
                        "Enter your prompt here (e.g. Make a list of european companies like Bosch, Siemens, group by topic, innovation and moat.)",
                        _for="prompt_text",
                    ),
                    Textarea(
                        placeholder="Enter prompt text here",
                        style="height: 300px; width: 60%;",
                        id="prompt_text",
                        name="prompt_text",
                    ),
                    Button("Submit"),
                ),
            ),
            data_hx_post="/process_transcript",
            data_hx_swap="afterbegin",
            data_hx_target="#response-list",
        ),
        Div(
            data_hx_ext="sse",
            data_sse_connect="/time-sender",
            data_hx_swap="innerHTML",
            data_sse_swap="message",
            data_sse_close="close",
        ),
        Div(id="response-list"),
    )


@app.post("/process_transcript")
def process_transcript(prompt_text: str, request: Request):
    # Return a new SSE Div with the prompt in the connect URL
    logger.trace(
        f"POST process_transcript client={request.client.host} prompt='{prompt_text}'"
    )
    uid = f"id-{datetime.datetime.now().timestamp()}"
    return Div(
        Article(f"Prompt: {prompt_text}"),
        Div("Thoughts:", Div(id=f"{uid}-thoughts")),
        Div("Answer:", Div(id=f"{uid}-answer")),
        Div(id=f"{uid}-error"),
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?prompt_text={prompt_text}&uid={uid}",
        data_sse_swap="thought,answer,final_answer,error",
        data_hx_swap_oob="true",
        data_sse_close="close",
    )


event = signal_shutdown()


async def time_generator():
    logger.trace("time_generator init")
    count = 0
    while not ((event.is_set()) or ((7) < (count))):
        count += 1
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        logger.trace(f"time_generator sends {time_str}")
        yield (sse_message(Article(time_str), event="message"))
        await asyncio.sleep(1)
    yield (sse_message(Article(time_str), event="close"))
    logger.trace("time_generator shutdown")


@app.get("/time-sender")
async def time_sender():
    logger.trace(f"GET time-sender")
    return EventStream(time_generator())


@app.get("/response-stream")
async def response_stream(prompt_text: str, uid: str):
    async def gen():
        logger.trace(f"GET response-stream prompt_text={prompt_text}")
        config = GenerationConfig(
            prompt_text=prompt_text,
            model="gemini-flash-latest",
            use_search=False,
            think_budget=0,
            include_thoughts=True,
        )
        logger.trace("created a genai configuration")
        job = GenAIJob(config)
        logger.trace("configured genai job")
        async for msg in job.run():
            logger.trace(f"genai.job async for {msg}")
            if (msg["type"]) == ("thought"):
                yield (
                    sse_message(
                        Div(
                            f"{msg['text']}",
                            id=f"{uid}-thoughts",
                            hx_swap_oob="beforeend",
                        ),
                        event="thought",
                    )
                )
            elif (msg["type"]) == ("answer"):
                yield (
                    sse_message(
                        Div(
                            f"{msg['text']}", id=f"{uid}-answer", hx_swap_oob="beforeend"
                        ),
                        event="answer",
                    )
                )
            elif (msg["type"]) == ("complete"):
                yield (
                    sse_message(
                        Div(
                            f"Final Answer: {msg['answer']}",
                            id=f"{uid}-answer",
                            hx_swap_oob="innerHTML",
                        ),
                        event="final_answer",
                    )
                )
                yield (sse_message(" ", event="close"))
                break
            elif (msg["type"]) == ("error"):
                yield (
                    sse_message(
                        Div(
                            f"Error: {msg['message']}",
                            id=f"{uid}-error",
                            hx_swap_oob="innerHTML",
                        ),
                        event="error",
                    )
                )
                yield (sse_message(" ", event="close"))
                break

    return EventStream(gen())


serve()
