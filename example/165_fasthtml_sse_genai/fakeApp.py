# Save as fakeApp.py and run with: uvicorn fakeApp:app --reload
from __future__ import annotations
import time
import asyncio
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
    sse_message,
    EventStream,
    H4,
    P,
    Strong,
)
from typing import Any, Dict


# -----------------------------------------------------------------------------
# 1. FAKE GenAI Implementation to emulate streaming
# -----------------------------------------------------------------------------
class FakeGenAIJob:
    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text
        self.fake_response = "This is a simulated streaming response from the AI. It will arrive in several chunks, just like a real one would."
        self.final_answer = f"Based on your prompt about '{prompt_text[:30]}...', the final conclusion is that this simulation works."

    async def run(self):
        """An async generator that yields data chunks to simulate a real API."""
        await asyncio.sleep(0.5)
        yield dict(
            type="thought",
            text="Thinking about the prompt and preparing the simulated response...",
        )

        for word in self.fake_response.split():
            await asyncio.sleep(0.2)
            yield dict(type="answer_chunk", text=f"{word} ")

        await asyncio.sleep(0.5)

        yield dict(
            type="final_answer",
            answer=self.final_answer,
            error=False,
        )


# -----------------------------------------------------------------------------
# 2. FastHTML App Setup
# -----------------------------------------------------------------------------
hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),
        Script(src="https://unpkg.com/htmx.org@1.9.12/dist/ext/debug.js"))
app, rt = fast_app(hdrs=hdrs)

_job_store: Dict[str, Dict[str, Any]] = {}
_job_store_lock = asyncio.Lock()


@rt
def index():
    return Titled(
        "SSE AI Responder (Corrected)",
        Form(
            Fieldset(
                Legend("Submit a prompt to the simulated AI"),
                Div(
                    Label("Enter your prompt here:", _for="prompt_text"),
                    Textarea(
                        "Create a summary of industries in Europe, grouped by topic, innovation, and moat. Also indicate the impact that increasing US Import Tariffs will have on these industries.",
                        style="height: 150px; width: 99%;",
                        id="prompt_text",
                        name="prompt_text",
                    ),
                    Button("Submit"),
                ),
            ),
            data_hx_post="/process_prompt",
            data_hx_target="#response-container",
            data_hx_swap="innerHTML",
            data_hx_ext="debug",
        ),
        Div(id="response-container"),
    )


@app.post("/process_prompt")
async def process_prompt(prompt_text: str):
    uid = f"id-{int(time.time() * 1000)}"
    async with _job_store_lock:
        _job_store[uid] = dict(prompt_text=prompt_text)

    return Div(
        H4("Response Stream:"),
        Div(
            "Waiting for AI to start thinking...",
            data_sse_swap="thought",
            data_hx_swap="innerHTML",
            style="color: grey; font-style: italic;",
        ),
        Div(
            id=f"{uid}-final-container",
            children=[
                Div(
                    data_sse_swap="answer_chunk",
                    data_hx_swap="beforeend",
                    style="border: 1px solid #ccc; padding: 10px; min-height: 50px; margin-top: 10px;",
                )
            ],
        ),
        Div(
            data_sse_swap="final_answer",
            data_hx_target=f"#{uid}-final-container",
            data_hx_swap="outerHTML",
        ),
        Div(
            data_sse_swap="error",
            data_hx_swap="innerHTML",
            style="color: red; margin-top: 10px;",
        ),
        # --- HTML Attributes (Keyword Arguments) ---
        id=f"{uid}-container",
        data_hx_ext="sse,debug",
        data_sse_connect=f"/response-stream?uid={uid}",
        data_sse_close="close",
    )


@app.get("/response-stream")
async def response_stream(uid: str):
    async def gen():
        try:
            async with _job_store_lock:
                job_info = _job_store.get(uid)

            if not job_info:
                yield sse_message("Error: unknown UID", event="error")
                return

            job = FakeGenAIJob(prompt_text=job_info["prompt_text"])
            async for msg in job.run():
                if msg["type"] == "thought":
                    yield sse_message(msg["text"], event="thought")
                elif msg["type"] == "answer_chunk":
                    yield sse_message(msg["text"], event="answer_chunk")
                elif msg["type"] == "final_answer":
                    final_html_content = Div(
                        id=f"{uid}-final-container",
                        children=[
                            Div(
                                Strong("Final Answer:"),
                                P(msg["answer"]),
                                style="border: 1px solid #ccc; padding: 10px; min-height: 50px; margin-top: 10px;",
                            )
                        ],
                    )
                    yield sse_message(str(final_html_content), event="final_answer")
                    break
        finally:
            async with _job_store_lock:
                if uid in _job_store:
                    del _job_store[uid]
            yield sse_message("", event="close")

    return EventStream(gen())


# -----------------------------------------------------------------------------
# 3. Run the App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
