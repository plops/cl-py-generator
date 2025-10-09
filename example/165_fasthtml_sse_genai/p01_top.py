# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_top.py
import argparse
import asyncio
import datetime
import sys
from loguru import logger

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the SSE AI Responder")
parser.add_argument('-v', '--verbose', action='count', default=0, help="Increase verbosity: -v for DEBUG, -vv for TRACE")
args = parser.parse_args()

# Determine log level based on verbosity
if args.verbose == 1:
    log_level = "DEBUG"
elif args.verbose >= 2:
    log_level = "TRACE"
else:
    log_level = "INFO"

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS} UTC</green> <level>{level}</level> <cyan>{name}</cyan>: <level>{message}</level>",
    colorize=True,
    level=log_level,
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
        ),
        Div(id="response-list"),
    )


@rt("/process_transcript")
def post(prompt_text: str, request: Request):
    # Return a new SSE Div with the prompt in the connect URL
    return Div(
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?prompt_text={prompt_text}",
        data_hx_swap="beforeend show:bottom",
        data_sse_swap="message",
    )


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
async def get(prompt_text: str):
    config = GenerationConfig(
        prompt_text=prompt_text,
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