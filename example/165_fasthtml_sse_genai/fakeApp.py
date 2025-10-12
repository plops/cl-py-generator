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
        # Yield a few "answer" chunks
        for word in self.fake_response.split():
            await asyncio.sleep(0.1)  # Simulate network latency/computation
            yield dict(type="answer", text=f"{word} ")

        await asyncio.sleep(0.5)

        # Yield the "complete" message
        yield dict(
            type="complete",
            answer=self.final_answer,
            thought="",  # No thoughts in this simple simulation
            error=False,
        )


# -----------------------------------------------------------------------------
# 2. FastHTML App Setup
#    This is almost identical to your original code.
# -----------------------------------------------------------------------------
hdrs = (Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),)
app, rt = fast_app(hdrs=hdrs)

# Server-side store to hold prompts between the POST and GET requests
_job_store: Dict[str, Dict[str, Any]] = {}
_job_store_lock = asyncio.Lock()


@rt
def index():
    """The main page with the form."""
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
            data_hx_swap="afterbegin",
            data_hx_target="#response-list",
        ),
        Div(id="response-list"),
    )


@app.post("/process_prompt")
async def process_prompt(prompt_text: str):
    """
    Step 2: This endpoint now returns an HTML snippet with a unique ID
    on the container element that initiates the SSE connection.
    """
    uid = f"id-{int(time.time() * 1000)}"
    async with _job_store_lock:
        _job_store[uid] = dict(prompt_text=prompt_text)

    # The key change is adding `id=f"{uid}-container"` to this Div.
    # This allows us to target this specific element for removal later.
    return Div(
        Div(Div(id=f"{uid}-answer", cl="answer-box")),
        Div(id=f"{uid}-error"),
        id=f"{uid}-container",  # MODIFICATION: Added a unique ID to the container
        data_hx_ext="sse",
        data_sse_connect=f"/response-stream?uid={uid}",
        data_sse_swap="answer,error",
        data_sse_close="close",
    )


@app.get("/response-stream")
async def response_stream(uid: str):
    """
    Step 3: The streaming endpoint. It now sends a final OOB swap
    message to remove the SSE-initiating element from the DOM.
    """

    async def gen():
        job_info = None
        # MODIFICATION: Encapsulate the entire logic in a try/finally block
        # to guarantee cleanup and connection termination.
        try:
            async with _job_store_lock:
                job_info = _job_store.get(uid)

            if not job_info:
                # Send a visible error message to the user
                yield sse_message(
                    Div(f"Error: unknown UID {uid}", id=f"{uid}-error", hx_swap_oob="innerHTML"),
                    event="error"
                )
                return  # Exit the generator

            job = FakeGenAIJob(prompt_text=job_info["prompt_text"])
            async for msg in job.run():
                if msg["type"] == "answer":
                    yield sse_message(
                        Div(msg["text"], id=f"{uid}-answer", hx_swap_oob="beforeend"),
                        event="answer",
                    )
                elif msg["type"] == "complete":
                    yield sse_message(
                        Div(
                            f"Final Answer: {msg['answer']}",
                            id=f"{uid}-answer",
                            hx_swap_oob="innerHTML",
                        ),
                        event="answer",
                    )
                    break
        finally:
            # MODIFICATION: This block now executes after the stream is finished or fails.
            # Its primary job is to remove the element that started the SSE connection.
            yield sse_message(
                # This is an empty Div targeting the container by its ID.
                # hx-swap-oob="outerHTML" will replace the entire element with
                # this Div's content (which is nothing), effectively deleting it.
                Div("", id=f"{uid}-container", hx_swap_oob="outerHTML"),
                event="answer",  # Use an event the client is already listening to.
            )

            # Clean up the job from the server-side store
            async with _job_store_lock:
                if uid in _job_store:
                    del _job_store[uid]

            # Formally tell the client to close the connection. While removing the
            # element usually handles this, sending the close event is good practice.
            yield sse_message("", event="close")

    return EventStream(gen())


# -----------------------------------------------------------------------------
# 3. Run the App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    serve()
#
#
#
#
# ### How the HTMX and SSE Flow Works (The Important Part)
#
# This is a step-by-step breakdown of what happens when you click "Submit".
#
# **Step 1: The Form Submission**
# - You fill out the `Textarea` and click the "Submit" button.
# - The `Form` has `data_hx_post="/process_prompt"`. HTMX sends an AJAX POST request to that URL.
# - The `data_hx_target="#response-list"` and `data_hx_swap="afterbegin"` attributes tell HTMX: "When you get a response from the POST request, take the HTML content and put it at the very beginning of the `<div id="response-list">`."
#
# **Step 2: The "Bridge" Endpoint (`/process_prompt`)**
# - This is the crucial part. This endpoint does **not** do the long-running AI work.
# - It quickly does two things:
# 1.  It generates a unique ID (`uid`) and stores the user's prompt on the server in the `_job_store` dictionary. This is necessary because the next request (the SSE connection) is a separate HTTP GET request and needs a way to retrieve the prompt.
# 2.  It returns a small snippet of HTML:
# ```html
# <div>
# <div><div id="id-12345-answer" class="answer-box"></div></div>
# <div id="id-12345-error"></div>
# </div>
# ```
# - **This snippet contains instructions for the HTMX SSE extension:**
# - `data_hx_ext="sse"`: "Hey HTMX, enable the SSE extension for this element."
# - `data_sse_connect="/response-stream?uid=..."`: "Now, immediately open a Server-Sent Events connection to this URL."
# - `data_sse_swap="answer,error"`: "Listen for messages from this stream. If you get a message with `event: answer`, swap its content. If you get `event: error`, swap its content."
# - `data_sse_close="close"`: "If you get a message with `event: close`, terminate the SSE connection."
#
# **Step 3: The SSE Stream (`/response-stream`)**
# - As soon as the browser receives the HTML snippet from Step 2, the HTMX SSE extension opens a GET request to `/response-stream?uid=...`. This connection is kept open.
# - The server-side `response_stream` function is an async generator (`async def` with `yield`).
# - It looks up the `uid` to get the original prompt.
#                                             - It starts the `FakeGenAIJob`.
#                                                             - As the `FakeGenAIJob` `yield`s data chunks, the `response_stream` function wraps them in an SSE message format using `sse_message()`:
# - `yield sse_message(..., event="answer", id=f"{uid}-answer", data_hx_swap_oob="beforeend")`
# - This sends a message to the browser that looks something like this:
# ```
# event: answer
# data: <div id="id-12345-answer" hx-swap-oob="beforeend">This </div>
#                                                                ```
#                                                                  - **HTMX on the client sees this message:**
# - It sees `event: answer`, which matches one of the events in `data_sse_swap`.
#                                                     - It looks at the `data` payload.
#                                                                              - It sees the `hx-swap-oob="beforeend"` ("Out-Of-Band" swap). This is a powerful feature that says: "Don't swap this content into the element that initiated the SSE connection. Instead, find the element on the page with `id="id-12345-answer"` and append this content to it."
#                                                                                                                                                                                                                                                                                                                              - This process repeats for every word, creating the streaming effect in the browser.
#
# **Step 4: Closing the Connection**
#                       - Once the `FakeGenAIJob` is done, the `finally` block in `response_stream` runs.
# - It sends one final message: `yield sse_message("", event="close")`.
# - HTMX on the client sees `event: close`, matches it to `data_sse_close="close"`, and closes the connection. The process is complete.
