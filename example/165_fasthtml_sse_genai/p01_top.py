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
@dataclass
class GenerationConfig:
    prompt_text: str
    model: str = "gemini-flash-latest"
    output_yaml_path: str = "out.yaml"
    use_search: bool = True
    think_budget: int = -1
    include_thoughts: bool = True
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class StreamResult:
    thought: str = ""
    answer: str = ""
    responses: List[Any] = field(default_factory=list)


class GenAIJob:
    def __init__(self, config: GenerationConfig):
        logger.trace("GenAIJob.__init__")
        self.config = config
        self.client = genai.Client(api_key=os.environ.get(config.api_key_env))

    def _build_request(self) -> Dict[str, Any]:
        logger.trace("GenAIJob._build_request")
        tools = (
            ([types.Tool(googleSearch=types.GoogleSearch())])
            if (self.config.use_search)
            else ([])
        )
        safety = [
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinkingBudget=self.config.think_budget,
                include_thoughts=self.config.include_thoughts,
            ),
            safety_settings=safety,
            tools=tools,
        )
        contents = [
            types.Content(
                role="user", parts=[types.Part.from_text(text=self.config.prompt_text)]
            )
        ]
        logger.debug(f"_build_request {self.config.prompt_text}")
        return dict(
            model=self.config.model, contents=contents, config=generate_content_config
        )

    async def run(self):
        req = self._build_request()
        result = StreamResult()
        logger.debug("Starting streaming generation")
        error_in_parts = False
        try:
            async for chunk in self.client.models.generate_content_stream(**req):
                logger.debug("received chunk")
                try:
                    parts = chunk.candidates[0].content.parts
                except Exception as e:
                    logger.debug(f"exception when accessing chunk: {e}")
                    continue
                try:
                    for part in parts:
                        if getattr(part, "text", None):
                            logger.trace(f"{part}")
                            if getattr(part, "thought", False):
                                result.thought += part.text
                                yield (dict(type="thought", text=part.text))
                            else:
                                result.answer += part.text
                                yield (dict(type="answer", text=part.text))
                except Exception as e:
                    error_in_parts = True
                    logger.warning(f"genai {e}")
        except Exception as e:
            logger.error(f"genai {e}")
            yield (dict(type="error", message=str(e)))
            return
        logger.debug(f"Thought: {result.thought}")
        logger.debug(f"Answer: {result.answer}")
        yield (
            dict(
                type="complete",
                thought=result.thought,
                answer=result.answer,
                error=error_in_parts,
            )
        )


__all__ = ["GenerationConfig", "StreamResult", "GenAIJob"]
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


@app.post("/process_transcript")
def process_transcript(prompt_text: str, request: Request):
    # Return a new SSE Div with the prompt in the connect URL
    logger.trace(
        f"POST process_transcript client={request.client.host} prompt='{prompt_text}'"
    )
    return Div(
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?prompt_text={prompt_text}",
        data_hx_swap="beforeend show:bottom",
        data_sse_swap="message",
    )


event = signal_shutdown()


async def time_generator():
    logger.trace("time_generator init")
    while not (event.is_set()):
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        logger.trace(f"time_generator sends {time_str}")
        yield (sse_message(Article(time_str), event="message"))
        await asyncio.sleep(1)
    logger.trace("time_generator shutdown")


@app.get("/time-sender")
async def time_sender():
    time_gen = time_generator()
    logger.trace(f"GET time-sender {time_gen}")
    return EventStream(time_gen)


@app.get("/response-stream")
async def response_stream(prompt_text: str):
    logger.trace(f"GET response-stream prompt_text={prompt_text}")
    config = GenerationConfig(
        prompt_text=prompt_text,
        model="gemini-flash-latest",
        use_search=False,
        think_budget=0,
        include_thoughts=False,
    )
    logger.trace("created a genai configuration")
    job = GenAIJob(config)
    logger.trace("configured genai job")
    async for msg in job.run():
        logger.trace(f"genai.job async for {msg}")
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
