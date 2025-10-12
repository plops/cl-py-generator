# Save as mini_htmx_sse.py and run with: uv run mini_htmx_sse.py
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
    serve,
    Span,
    H4,  # MODIFICATION: Import H4
    P,  # MODIFICATION: Import P
    Strong,  # MODIFICATION: Import Strong
)
from typing import Any, Dict


# -----------------------------------------------------------------------------
# 1. FAKE GenAI Implementation to emulate streaming
#    This replaces the entire `google.genai` part of your code.
# -----------------------------------------------------------------------------
class FakeGenAIJob:
    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text
        self.fake_response = "This is a simulated streaming response from the AI. It will arrive in several chunks, just like a real one would."
        self.final_answer = f"Based on your prompt about '{prompt_text[:30]}...', the final conclusion is that this simulation works."

    async def run(self):
        """An async generator that yields data chunks to simulate a real API."""
        # MODIFICATION: Yield a "thought" message first.
        await asyncio.sleep(0.5)
        yield dict(
            type="thought",
            text="Thinking about the prompt and preparing the simulated response...",
        )

        for word in self.fake_response.split():
            await asyncio.sleep(0.2)  # Simulate network latency/computation
            # MODIFICATION: Change event type to 'answer_chunk'
            yield dict(type="answer_chunk", text=f"{word} ")

        await asyncio.sleep(0.5)

        # MODIFICATION: Change event type to 'final_answer'
        yield dict(
            type="final_answer",
            answer=self.final_answer,
            error=False,
        )


# -----------------------------------------------------------------------------
# 2. FastHTML App Setup
#    This is almost identical to your original code.
# -----------------------------------------------------------------------------
hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs)

_job_store: Dict[str, Dict[str, Any]] = {}
_job_store_lock = asyncio.Lock()


@rt
def index():
    return Titled(
        "SSE AI Responder (Simulation)",
        Form(
            Fieldset(
                Legend("Submit a prompt to the simulated AI"),
                Div(
                    Label("Enter your prompt here:", _for="prompt_text"),
                    Textarea(
                        placeholder="Your prompt doesn't matter, the response is fake.",
                        style="height: 150px; width: 60%;",
                        id="prompt_text",
                        name="prompt_text",
                    ),
                    Button("Submit"),
                ),
            ),
            # HTMX attributes for the form submission
            data_hx_post="/process_prompt",
            # MODIFICATION: Target #response-container and use innerHTML swap
            data_hx_target="#response-container",
            data_hx_swap="innerHTML",
        ),
        # MODIFICATION: Change id to response-container
        Div(id="response-container"),
    )


@app.post("/process_prompt")
async def process_prompt(prompt_text: str):
    uid = f"id-{int(time.time() * 1000)}"
    async with _job_store_lock:
        _job_store[uid] = dict(prompt_text=prompt_text)

    # MODIFICATION: This container now matches the structure from the README.
    return Div(
        H4("Response Stream:"),
        Div(
            "Waiting for AI to start thinking...",
            id=f"{uid}-thoughts",
            style="color: grey; font-style: italic;",
        ),
        Div(
            id=f"{uid}-answer-stream",
            style="border: 1px solid #ccc; padding: 10px; min-height: 50px; margin-top: 10px;",
        ),
        Div(id=f"{uid}-error", style="color: red; margin-top: 10px;"),
        id=f"{uid}-container",
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?uid={uid}",
        # MODIFICATION: Listen for the new event names
        data_sse_swap="thought,answer_chunk,final_answer,error",
        data_sse_close="close",
    )


@app.get("/response-stream")
async def response_stream(uid: str):
    """
    The streaming endpoint, now corrected to match the README example.
    """

    async def gen():
        try:
            async with _job_store_lock:
                job_info = _job_store.get(uid)

            if not job_info:
                # MODIFICATION: Target the new error div
                yield sse_message(
                    Div(
                        f"Error: unknown UID {uid}",
                        id=f"{uid}-error",
                        hx_swap_oob="innerHTML",
                    ),
                    event="error",
                )
                return

            job = FakeGenAIJob(prompt_text=job_info["prompt_text"])
            async for msg in job.run():
                # MODIFICATION: Handle 'thought' event
                if msg["type"] == "thought":
                    yield sse_message(
                        Div(msg["text"], id=f"{uid}-thoughts", hx_swap_oob="innerHTML"),
                        event="thought",
                    )
                # MODIFICATION: Handle 'answer_chunk' event
                elif msg["type"] == "answer_chunk":
                    yield sse_message(
                        Span(
                            msg["text"],
                            id=f"{uid}-answer-stream",
                            hx_swap_oob="beforeend",
                        ),
                        event="answer_chunk",
                    )
                # MODIFICATION: Handle 'final_answer' event
                elif msg["type"] == "final_answer":
                    yield sse_message(
                        Div(
                            Strong("Final Answer:"),
                            P(msg["answer"]),
                            id=f"{uid}-answer-stream",
                            hx_swap_oob="innerHTML",
                        ),
                        event="final_answer",
                    )
                    break
        finally:
            # Clean up the job from the server-side store
            async with _job_store_lock:
                if uid in _job_store:
                    del _job_store[uid]

            # Tell the client to close the SSE connection.
            yield sse_message("", event="close")

    return EventStream(gen())


# -----------------------------------------------------------------------------
# 3. Run the App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    serve()
