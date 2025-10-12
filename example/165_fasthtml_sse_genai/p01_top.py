# export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p01_top.py
from __future__ import annotations
import time
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
    sse_message,
    EventStream,
    serve,
)
import os
import sys
import asyncio
from dataclasses import dataclass
from typing import Any, Dict
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


class GenAIJob:
    def __init__(self, config: GenerationConfig):
        # GenAIJob.__init__
        self.config = config
        self.client = genai.Client(api_key=os.environ.get(config.api_key_env))

    def _build_request(self) -> Dict[str, Any]:
        # GenAIJob._build_request
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
        # _build_request
        return dict(
            model=self.config.model, contents=contents, config=generate_content_config
        )

    async def run(self):
        req = self._build_request()
        result = StreamResult()
        # Starting streaming generation
        error_in_parts = False
        try:
            for chunk in self.client.models.generate_content_stream(**req):
                # received chunk
                try:
                    parts = chunk.candidates[0].content.parts
                except Exception as e:
                    # exception when accessing chunk:
                    continue
                try:
                    for part in parts:
                        if getattr(part, "text", None):
                            #
                            if getattr(part, "thought", False):
                                result.thought += part.text
                                yield (dict(type="thought", text=part.text))
                            else:
                                result.answer += part.text
                                yield (dict(type="answer", text=part.text))
                except Exception as e:
                    error_in_parts = True
                    # genai
        except Exception as e:
            # genai
            yield (dict(type="error", message=str(e)))
            return
        # Thought:
        # Answer:
        yield (
            dict(
                type="complete",
                thought=result.thought,
                answer=result.answer,
                error=error_in_parts,
            )
        )


hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs)


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
        Div(id="response-list"),
    )


_job_store: Dict[str, Dict[str, Any]] = {}
_job_store_lock = asyncio.Lock()


@app.post("/process_transcript")
async def process_transcript(prompt_text: str, request: Request):
    # Return a new SSE Div with the uid (prompt is stored server side)
    id_str = int(((1000) * (datetime.datetime.now().timestamp())))
    uid = f"id-{id_str}"
    # POST process_transcript
    async with _job_store_lock:
        _job_store[uid] = dict(
            prompt_text=prompt_text,
            model="gemini-flash-latest",
            use_search=False,
            include_thoughts=False,
            think_budget=0,
            created_at=time.time(),
        )
    return Div(
        Div("Thoughts:", Div(id=f"{uid}-thoughts")),
        Div("Answer:", Div(id=f"{uid}-answer")),
        Div(id=f"{uid}-final_answer"),
        Div(id=f"{uid}-error"),
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?uid={uid}",
        data_sse_swap="thought,answer,final_answer,error",
        data_hx_target="#response-list",
        data_sse_close="close",
    )


@app.get("/response-stream")
async def response_stream(uid: str):
    async def gen():
        # GET response-stream
        async with _job_store_lock:
            job_info = _job_store.get(uid)
        if not (job_info):
            yield (
                sse_message(
                    Div(
                        f"Error: unkown or expired uid {uid}",
                        id=f"{uid}-error",
                        data_hx_swap_oob="innerHTML",
                    ),
                    event="error",
                )
            )
            yield (sse_message("", event="close"))
            return
        prompt_text = job_info["prompt_text"]
        include_thought = False
        config = GenerationConfig(
            prompt_text=prompt_text,
            model=job_info.get("model", "gemini-flash-latest"),
            use_search=job_info.get("use_search", False),
            think_budget=job_info.get(
                "think_budget", (-1) if (include_thought) else (0)
            ),
            include_thoughts=include_thought,
        )
        # created a genai configuration
        job = GenAIJob(config)
        # configured genai job
        try:
            async for msg in job.run():
                # genai.job async for
                if (include_thought) and ((msg["type"]) == ("thought")):
                    yield (
                        sse_message(
                            Div(
                                f"{msg['text']}",
                                id=f"{uid}-thoughts",
                                data_hx_swap_oob="beforeend",
                            ),
                            event="thought",
                        )
                    )
                elif (msg["type"]) == ("answer"):
                    yield (
                        sse_message(
                            Div(
                                f"{msg['text']}",
                                id=f"{uid}-answer",
                                data_hx_swap_oob="beforeend",
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
                                data_hx_swap_oob="innerHTML",
                            ),
                            event="final_answer",
                        )
                    )
                    break
                elif (msg["type"]) == ("error"):
                    err_text = msg.get("message", msg.get("text", "Unknown error"))
                    yield (
                        sse_message(
                            Div(
                                f"Error: {err_text}",
                                id=f"{uid}-error",
                                data_hx_swap_oob="innerHTML",
                            ),
                            event="error",
                        )
                    )
                    break
        finally:
            async with _job_store_lock:
                if uid in _job_store:
                    del _job_store[uid]
                    # cleaned up job store for
        yield (sse_message("", event="close"))

    return EventStream(gen())


serve()
