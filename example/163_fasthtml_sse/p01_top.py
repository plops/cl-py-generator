import random
import time
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
# https://www.npmjs.com/package/htmx-ext-sse https://x.com/jeremyphoward/status/1829897065440428144/photo/1
hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs)


@rt
def index():
    return Titled(
        "SSE Random Number Generator",
        P("Generate pairs of random numbers, as the list grows scroll downwards."),
        Div(
            hx_ext="sse",
            sse_connect="/number-stream",
            hx_swap="beforeend show:bottom",
            sse_swap="message",
        ),
    )


shutdown_event = signal_shutdown()


async def number_generator():
    while not (shutdown_event.is_set()):
        data = Div(Article(random.randint(1, 100)), Article(random.randint(1, 100)))
        yield (sse_message(data))
        await time.sleep(1)


@rt("/number-stream")
async def get():
    return EventStream(number_generator())


serve()
