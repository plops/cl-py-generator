# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_top.py
import random
import time
import asyncio
from urllib.parse import quote
from fasthtml.common import *
from loguru import logger


logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger.info("Logger configured")
# import after logger exists
from p02_impl import GenerationConfig, GenAIJob

cfg = GenerationConfig(
    prompt_text=r"""Make a list of european companies like Bosch, Siemens, group by topic, innovation and moat""",
    model="gemini-flash-latest",
    use_search=True,
    think_budget=-1,
    include_thoughts=True,
)
hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs, live=True)


@rt
def index():
    return Titled(
        "SSE Random Number Generator",
        "GenAI Prompt",
        Form(
            Input(type="text", name="prompt", placeholder="Enter your prompt"),
            Button("Submit", type="submit"),
            hx_post="/generate",
            hx_target="#output",
            hx_swap="innerHTML"
        ),
        Div(id="output")
    )


@rt
def generate(prompt: str):
    return Div(
        hx_ext="sse",
        sse_connect=f"/stream?prompt={quote(prompt)}",
        hx_swap="beforeend",
        sse_swap="message",
        id="output")


@rt("/stream")
async def stream(prompt: str):
    config = GenerationConfig(
        prompt_text=prompt,
        model="gemini-flash-latest",
        use_search=True,
        think_budget=-1,
        include_thoughts=True,
    )
    job = GenAIJob(config)
    async for msg in job.run():
        if msg['type'] == 'thought':
            yield sse_message(Div(f"Thought: {msg['text']}"))
        elif msg['type'] == 'answer':
            yield sse_message(Div(f"Answer: {msg['text']}"))
        elif msg['type'] == 'complete':
            yield sse_message(Div(f"Final Answer: {msg['answer']}"))
            break
        elif msg['type'] == 'error':
            yield sse_message(Div(f"Error: {msg['message']}"))
            break

serve()