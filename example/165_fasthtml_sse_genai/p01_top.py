# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_top.py
import asyncio
import datetime
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
        Form(
            Fieldset(
                Legend("Submit a prompt for the AI to respond to"),
                Div(
                    Label("Write or paste your prompt here", _for="prompt-text"),
                    Textarea(
                        placeholder="Make a list of european companies like Bosch, Siemens, group by topic, innovation and moat",
                        style="height: 300px; width: 60%;",
                        id="prompt-text",
                        name="prompt-text",
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
        ),
        Div(
            data_hx_ext="sse",
            data_sse_connect="/response-stream",
            data_hx_swap="beforeend show:bottom",
            data_sse_swap="message",
        ),
        Div(id="summary-list"),
    )


@rt("/process_transcript")
def post(prompt_text: str, request: Request):
    return prompt_text


event = signal_shutdown()


async def time_generator():
    while not (event.is_set()):
        yield (
            sse_message(
                Article(datetime.datetime.now().strftime("%H:%M:%S")), event="message"
            )
        )
        await asyncio.sleep(1)


@rt("/time-sender")
async def get():
    return EventStream(time_generator())


@rt("/response-stream")
async def get(prompt: str):
    config = GenerationConfig(
        prompt_text=prompt,
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
