# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_top.py
import asyncio
from loguru import logger
from fasthtml.common import *

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

hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs, live=True)


@rt
def index():
    return Titled(
        "SSE AI Responder",
        P("See the response to the prompt"),
        Div(
            hx_ext="sse",
            sse_connect="/response-stream",
            hx_swap="beforeend show:bottom",
            sse_swap="message",
        ),
    )


@rt("/response-stream")
async def get():
    config = GenerationConfig(
        prompt_text=r"""Make a list of european companies like Bosch, Siemens, group by topic, innovation and moat""",
        model="gemini-flash-latest",
        use_search=False,
        think_budget=0,
        include_thoughts=False,
    )
    job = GenAIJob(config)
    async for msg in job.run():
        if (msg["type"]) == ("thought"):
            yield (sse_message(Div(f"Thought: {msg['text']}")))
        elif (msg["type"]) == ("answer"):
            yield (sse_message(Div(f"Answer: {msg['text']}")))
        elif (msg["type"]) == ("complete"):
            yield (sse_message(Div(f"Final Answer: {msg['answer']}")))
            break
        elif (msg["type"]) == ("error"):
            yield (sse_message(Div(f"Error: {msg['message']}")))
            break


serve()
