# Save as mini_htmx_sse.py and run with: uvicorn mini_htmx_sse:app --reload
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
#    (No changes needed in this section)
# -----------------------------------------------------------------------------
class FakeGenAIJob:
    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text
        self.fake_response = "This is a simulated streaming response from the AI. It will arrive in several chunks, just like a real one would."
        self.final_answer = f"Based on your prompt about '{prompt_text[:30]}...', the final conclusion is that this simulation works."

    async def run(self):
        """An async generator that yields data chunks to simulate a real API."""
        # Yield a "thought" message first.
        await asyncio.sleep(0.5)
        yield dict(
            type="thought",
            text="Thinking about the prompt and preparing the simulated response...",
        )

        # Yield response chunks.
        for word in self.fake_response.split():
            await asyncio.sleep(0.2)
            yield dict(type="answer_chunk", text=f"{word} ")

        await asyncio.sleep(0.5)

        # Yield the final answer.
        yield dict(
            type="final_answer",
            answer=self.final_answer,
            error=False,
        )


# -----------------------------------------------------------------------------
# 2. FastHTML App Setup
# -----------------------------------------------------------------------------
# Use the latest version of the SSE extension
hdrs = (Script(src="https://unpkg.com/htmx.org@1.9.12"), Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs)

_job_store: Dict[str, Dict[str, Any]] = {}
_job_store_lock = asyncio.Lock()


@rt
def index():
    # The initial page remains the same. The form correctly targets the response container.
    return Titled(
        "SSE AI Responder (Corrected)",
        Form(
            Fieldset(
                Legend("Submit a prompt to the simulated AI"),
                Div(
                    Label("Enter your prompt here:", _for="prompt_text"),
                    Textarea(
                        "Write a short story about a robot who discovers music.",
                        style="height: 150px; width: 60%;",
                        id="prompt_text",
                        name="prompt_text",
                    ),
                    Button("Submit"),
                ),
            ),
            data_hx_post="/process_prompt",
            data_hx_target="#response-container",
            data_hx_swap="innerHTML",
        ),
        Div(id="response-container"),
    )


@app.post("/process_prompt")
async def process_prompt(prompt_text: str):
    """
    CORRECTED: This endpoint now returns the proper client-side setup with dedicated
    listeners for each SSE event, as described in the revised document.
    """
    uid = f"id-{int(time.time() * 1000)}"
    async with _job_store_lock:
        _job_store[uid] = dict(prompt_text=prompt_text)

    return Div(
        # The main container establishes the connection. Note attributes start with "data_".
        id=f"{uid}-container",
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?uid={uid}",
        data_sse_close="close",

        # Child elements act as LISTENERS for specific events from the server.
        H4("Response Stream:"),

        # 1. This div listens for the 'thought' event and swaps its innerHTML.
        Div(
            "Waiting for AI to start thinking...",
            data_sse_swap="thought",
            data_hx_swap="innerHTML",
            style="color: grey; font-style: italic;",
        ),

        # 2. This outer container will be completely replaced by the 'final_answer' event.
        Div(
            id=f"{uid}-final-container",
            # 2a. The inner div listens for 'answer_chunk' events and appends them.
            children=[Div(
                data_sse_swap="answer_chunk",
                data_hx_swap="beforeend",
                style="border: 1px solid #ccc; padding: 10px; min-height: 50px; margin-top: 10px;",
            )]
        ),

        # 3. This empty div listens for 'final_answer'. When it receives the event,
        #    it targets the container above (#uid-final-container) and replaces it.
        Div(
            data_sse_swap="final_answer",
            data_hx_target=f"#{uid}-final-container",
            data_hx_swap="outerHTML"
        ),

        # 4. This div listens for the 'error' event and displays the message.
        Div(
            data_sse_swap="error",
            data_hx_swap="innerHTML",
            style="color: red; margin-top: 10px;",
        ),
    )


@app.get("/response-stream")
async def response_stream(uid: str):
    """
    CORRECTED: The streaming endpoint now sends raw content for each event,
    removing all `hx-swap-oob` logic. The client-side listeners handle the swaps.
    """

    async def gen():
        try:
            async with _job_store_lock:
                job_info = _job_store.get(uid)

            if not job_info:
                # Data is just the error message. The 'error' listener on the client will render it.
                yield sse_message("Error: unknown UID", event="error")
                return

            job = FakeGenAIJob(prompt_text=job_info["prompt_text"])
            async for msg in job.run():
                if msg["type"] == "thought":
                    # Data is the thought text. The 'thought' listener swaps it.
                    yield sse_message(msg["text"], event="thought")

                elif msg["type"] == "answer_chunk":
                    # Data is the next text chunk. The 'answer_chunk' listener appends it.
                    yield sse_message(msg["text"], event="answer_chunk")

                elif msg["type"] == "final_answer":
                    # Data is a complete HTML block to replace the streaming container.
                    # The 'final_answer' listener on the client performs the outerHTML swap.
                    final_html_content = Div(
                        id=f"{uid}-final-container", # The ID must match the target for replacement
                        children=[Div(
                            Strong("Final Answer:"),
                            P(msg["answer"]),
                            style="border: 1px solid #ccc; padding: 10px; min-height: 50px; margin-top: 10px;",
                        )]
                    )
                    yield sse_message(str(final_html_content), event="final_answer")
                    break
        finally:
            # Clean up and tell the client to close the connection.
            async with _job_store_lock:
                if uid in _job_store:
                    del _job_store[uid]
            yield sse_message("", event="close")

    return EventStream(gen())


# -----------------------------------------------------------------------------
# 3. Run the App
# -----------------------------------------------------------------------------
# This part is commented out to prevent execution in this environment.
# To run locally:
# 1. `pip install uvicorn fasthtml`
# 2. `uvicorn mini_htmx_sse:app --reload`
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)