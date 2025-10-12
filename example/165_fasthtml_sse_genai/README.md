https://github.com/google/langextract this seems useful 


example error message when quota runs out:

```
INFO:     127.0.0.1:43460 - "GET /response-stream?prompt_text=tell%20me%20a%20joke HTTP/1.1" 200 OK
2025-10-09 22:55:42.640 UTC TRACE p01_top: GET response-stream prompt_text=tell me a joke
2025-10-09 22:55:42.640 UTC TRACE p01_top: created a genai configuration
2025-10-09 22:55:42.640 UTC TRACE p01_top: GenAIJob.__init__
2025-10-09 22:55:42.651 UTC TRACE p01_top: configured genai job
2025-10-09 22:55:42.651 UTC TRACE p01_top: GenAIJob._build_request
2025-10-09 22:55:42.651 UTC DEBUG p01_top: _build_request tell me a joke
2025-10-09 22:55:42.651 UTC DEBUG p01_top: Starting streaming generation
2025-10-09 22:55:42.900 UTC ERROR p01_top: genai 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 10\nPlease retry in 17.156209832s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '10'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '17s'}]}}
2025-10-09 22:55:42.901 UTC TRACE p01_top: genai.job async for {'type': 'error', 'message': "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.\\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 10\\nPlease retry in 17.156209832s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '10'}]}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '17s'}]}}"}

```

Here is a detailed description of the use case, perfect for a prompt or documentation.

***

### Use Case: Building a Real-Time Generative AI Streaming Interface with HTMX and Server-Sent Events (SSE)

This use case describes a web application that allows a user to submit a text prompt to a generative AI. The application provides a real-time, streaming response back to the user, showing the AI's "thoughts" and partial answers as they are generated. Once the final answer is ready, it replaces the intermediate content. The entire process is managed with minimal JavaScript by leveraging HTMX and its Server-Sent Events (SSE) extension.

#### The Workflow Step-by-Step:

1.  **User Interaction (The Initial Request):**
    *   The user is presented with a simple HTML form containing a textarea to enter their prompt and a "Submit" button.
    *   Using HTMX attributes (`hx-post`, `hx-target`), the form is configured to send an AJAX POST request to a server endpoint (e.g., `/process_prompt`) without requiring a full page reload.

2.  **Server-Side Preparation (The "Bridge" Response):**
    *   The `/process_prompt` endpoint does **not** run the time-consuming AI job directly. Instead, it acts as a bridge.
    *   It immediately generates a **Unique ID (UID)** for the user's request (e.g., `id-1672531200000`).
    *   It stores the user's prompt and the UID in a temporary, server-side store (like a dictionary or a Redis cache). This is crucial because the subsequent SSE connection is a separate HTTP request and needs a way to retrieve the original prompt.
    *   The endpoint instantly returns a small snippet of HTML to be injected into the user's page. This snippet is the key to initiating the real-time updates.

3.  **Client-Side Setup (The SSE Initiator Snippet):**
    *   The returned HTML snippet is a container `<div>` that includes several specifically prepared, empty child elements, each with an ID derived from the UID. For example:
        *   `<div id="{uid}-container">...</div>`: The main container that holds everything for this specific request.
        *   `<div id="{uid}-thoughts"></div>`: A placeholder to show the AI's reasoning or thought process.
        *   `<div id="{uid}-answer-stream"></div>`: A placeholder where partial, streaming answer chunks will be appended.
        *   `<div id="{uid}-error"></div>`: A placeholder to display any error messages.
    *   Crucially, the main container is decorated with HTMX SSE attributes to automatically establish and manage a connection.
        *   `hx-ext="sse"`: Enables the HTMX Server-Sent Events extension.
        *   `sse-connect="/response-stream?uid={uid}"`: Tells HTMX to immediately open an SSE connection to this URL, passing the unique ID.
        *   `sse-swap="thought,answer_chunk,final_answer,error"`: Instructs HTMX to listen for specific named events from the stream and swap the content into the page.
        *   `sse-close="close"`: Defines a special event name (`close`) that, when received from the server, will gracefully terminate the SSE connection.

4.  **Real-Time Asynchronous Updates (The SSE Stream):**
    *   Once the snippet is on the page, the browser opens a persistent GET request to `/response-stream?uid={uid}`.
    *   The server-side logic for this endpoint retrieves the user's prompt from the store using the UID.
    *   It then starts the actual AI generation process. As the AI produces output, the server sends a **stream of messages** back to the client over the open connection. Each message is formatted with an `event` name and `data`.
    *   **Out-of-Band (OOB) Swaps:** The `data` payload for each message is a small piece of HTML containing an `hx-swap-oob` attribute. This powerful HTMX feature tells the browser to update a *specific element on the page* (identified by its ID) rather than swapping the content into the main container.
        *   **Showing Thoughts:** `event: thought\ndata: <div id="{uid}-thoughts" hx-swap-oob="innerHTML">Thinking about topic...</div>`
        *   **Streaming the Answer:** `event: answer_chunk\ndata: <span id="{uid}-answer-stream" hx-swap-oob="beforeend">...next few words...</span>` (using `beforeend` to append content).

5.  **Finalization and Cleanup:**
    *   **Displaying the Final Message:** When the AI process is complete, the server sends a final message. This could be the complete answer or an error.
        *   **Success:** `event: final_answer\ndata: <div id="{uid}-answer-stream" hx-swap-oob="innerHTML">**Final Answer:** ...full text...</div>` (using `innerHTML` to replace all the intermediate chunks).
        *   **Error:** `event: error\ndata: <div id="{uid}-error" hx-swap-oob="innerHTML">An error occurred: ...details...</div>`
    *   **Terminating the Connection:** After sending the final message, the server's `finally` block executes.
        *   It sends the special `close` event: `event: close\ndata: `
        *   On the client, HTMX sees this event, matches it to the `sse-close="close"` attribute, and shuts down the SSE connection, preventing any further requests or reconnection attempts.
    *   **Server-Side Cleanup:** The server removes the UID and its associated data from the temporary store to free up memory. This ensures the system remains clean and scalable.


### 1. Initial Prompt Submission

This diagram shows the sequence when a user submits the prompt form. The server responds by creating a new container in the `#response-list` div, which then initiates an SSE connection.

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant Server

    User->>Browser: Fills out and submits form
    Browser->>Server: POST /process_transcript with prompt_text
    Server-->>Browser: Returns HTML Div with hx-sse attributes
    Note right of Browser: The response includes placeholder Divs for thoughts, answer, and error.
    Browser->>Browser: Inserts the new Div into element with id=response-list
    Browser->>Server: Opens SSE connection to /response-stream
```

### 2. SSE Streaming and UI Updates

This diagram illustrates how the browser receives Server-Sent Events (SSE) and uses them to update different parts of the page in real-time using "Out of Band" (OOB) swaps.

```mermaid
sequenceDiagram
    participant Browser
    participant Server

    Browser->>Server: GET /response-stream (SSE connection established)
    loop Streaming Events
        Server-->>Browser: SSE event: 'thought' with HTML content
        Browser->>Browser: Appends content to `uid-thoughts` div (hx-swap-oob="beforeend")

        Server-->>Browser: SSE event: 'answer' with HTML content
        Browser->>Browser: Appends content to `uid-answer` div (hx-swap-oob="beforeend")
    end

    alt Final State
        Server-->>Browser: SSE event: 'complete' with final answer
        Browser->>Browser: Replaces content of `uid-answer` div (hx-swap-oob="innerHTML")
    else Error State
        Server-->>Browser: SSE event: 'error' with error message
        Browser->>Browser: Replaces content of `uid-error` div (hx-swap-oob="innerHTML")
    end

    Server-->>Browser: SSE event: 'close'
    Browser->>Browser: Closes the SSE connection
```