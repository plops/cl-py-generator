Here is the Architecture Design Document for the refactoring and enhancement of the Video Transcript Summarizer application.

# Architecture Design Document: RocketRecap v2

## 1. Introduction

### 1.1 Purpose
This document outlines the architecture for the next iteration of the "RocketRecap" Video Transcript Summarizer. The goal is to refactor the existing monolithic FastHTML prototype (`p04_host.py`) into a robust, scalable, and maintainable application.

### 1.2 Scope of Changes
The refactoring encompasses the following key technical and functional requirements:
1.  **GenAI API Migration:** Upgrade from the deprecated synchronous API to the new asynchronous `google.genai` API (v2), supporting streaming and "thinking" models.
2.  **Database Migration:** Move from SQLite (`sqlite_minutils`) to asynchronous PostgreSQL.
3.  **Real-time Updates:** Replace HTMX polling with Server-Sent Events (SSE) for live generation feedback.
4.  **Structured Data:** Utilize Python `dataclasses` for form handling and internal data modeling.
5.  **Localization:** Implement multi-language support for the UI, outputs, and prompts.
6.  **Authentication:** Implement User Authentication via OAuth (initially Google) with session management.

## 2. Architectural Overview

The application will move from a single-file script to a **Layered Architecture**. This ensures separation of concerns, improves testability, and makes the codebase easier to navigate.

### 2.1 High-Level Layers

1.  **Presentation Layer (FastHTML):** Handles HTTP requests, renders HTML using FastTags, manages HTMX/SSE interactions, and defines routes.
2.  **Service Layer (Business Logic):** Orchestrates workflows (authentication, summarization), manages job states, and acts as the intermediary between the presentation and infrastructure.
3.  **Infrastructure Layer:** Deals with external systems. Contains adapters for the Database, GenAI API, and YouTube downloading/parsing.
4.  **Domain/Models:** Contains `dataclasses` representing the core entities and data transfer objects (DTOs) shared across layers.

## 3. Detailed Component Design

### 3.1 Domain/Models (Dataclasses)

These classes define the schema of data moving through the application.

*   **`User`**: Represents an authenticated user.
    *   Fields: `id` (UUID/int), `email`, `provider` (e.g., 'google'), `preferred_language`, `created_at`.
*   **`SummaryRequestForm`**: Binds to the UI submission form.
    *   Fields: `source_url` (Optional[str]), `manual_transcript` (Optional[str]), `model_id` (str), `output_language` (str).
*   **`SummaryJob`**: Represents the state of a summarization attempt. Mirror of the database table.
    *   Fields: `id` (PK), `user_id` (FK), `status` (Enum: PENDING, DOWNLOADING, GENERATING, COMPLETED, FAILED), `source_url`, `transcript_raw`, `summary_markdown`, `thinking_trace`, `error_message`, `cost_usd`, `token_usage` (dict), timestamps (start/end).
*   **`JobUpdate`**: A transient DTO pushed via SSE to the client.
    *   Fields: `job_id`, `status`, `partial_text` (for streaming), `is_thinking` (bool).

### 3.2 Infrastructure Layer

This layer isolates external dependencies. The existing helper scripts (`s01` to `s04`) will be utilized here or in the presentation layer as utilities.

*   **`DatabaseAdapter` (Async PostgreSQL)**
    *   **Responsibilities:** Manages the connection pool to PostgreSQL (using `asyncpg` or an async wrapper compatible with FastHTML practices). Executes queries.
    *   **Key Components:**
        *   `UserRepository`: Methods for `get_by_email`, `create_user`, `update_preferences`.
        *   `SummaryRepository`: Methods for `create_job`, `update_job_state`, `append_to_summary`, `get_job_by_id`, `get_recent_jobs_for_user`.
*   **`GenAIAdapter`**
    *   **Responsibilities:** Wraps the new `google.genai` Async Client (incorporating logic from `p02_impl.py`).
    *   **Key Methods:**
        *   `generate_summary_stream(transcript, prompt, model_config) -> AsyncIterator[StreamResult]`
    *   **Implementation Details:** Integrates the `UsageAggregator` and `PricingEstimator` logic from the reference `p02_impl.py` to calculate costs and separate "thinking" parts from the final response.
*   **`TranscriptProvider`**
    *   **Responsibilities:** Fetches transcripts from external sources.
    *   **Key Methods:**
        *   `get_from_youtube(url) -> str`: Wraps `yt-dlp` subprocess calls and utilizes `s02_parse_vtt_file.py`. Uses `s01_validate_youtube_url.py` for validation before execution. Handles timeouts and errors.

### 3.3 Service Layer

Contains the core application logic.

*   **`AuthService`**
    *   **Responsibilities:** Handles OAuth callbacks, retrieves user info from providers, creates/retrieves `User` entities via `UserRepository`, and manages FastHTML session data.
*   **`LocalizationService`**
    *   **Responsibilities:** Loads translation files. Provides methods to get UI strings based on language codes. Generates the appropriate prompt for the GenAI model, instructing it to output in the target language.
*   **`JobStateManager`**
    *   **Responsibilities:** Manages in-memory `asyncio.Queues` (or similar broadcast mechanism) for active jobs.
    *   **Key Methods:**
        *   `subscribe(job_id) -> AsyncIterator[ServerSentEvent]`: Used by the SSE route.
        *   `publish_update(job_id, update_data)`: Used by the `SummarizationWorkflow`.
*   **`SummarizationWorkflow` (Background Task)**
    *   **Responsibilities:** Orchestrates the summarization process asynchronously.
    *   **Process Flow:**
        1.  Receives `SummaryRequestForm` and `user_id`.
        2.  Creates initial `SummaryJob` in DB (status: PENDING).
        3.  If YouTube URL, calls `TranscriptProvider` (status: DOWNLOADING). Update DB & `JobStateManager`.
        4.  Validates transcript length.
        5.  Calls `LocalizationService` to build the prompt.
        6.  Calls `GenAIAdapter.generate_summary_stream` (status: GENERATING).
        7.  Iterates over the stream. If "thinking", publish thinking events. If "content", append to DB and publish content events via `JobStateManager`.
        8.  On completion, calculate final costs, generate embeddings (if required), update DB to COMPLETED.

### 3.4 Presentation Layer (FastHTML)

Handles the UI and HTTP interaction.

*   **Middleware:**
    *   `AuthMiddleware`: Verifies session existence for protected routes. Redirects to login if necessary. Adds the current `User` object to the `request.scope`.
*   **Routes (`main.py` / `routes.py`):**
    *   `GET /`: Landing page. If logged in, shows the `SubmitForm` and recent history.
    *   `GET /login`: Renders login options (OAuth buttons).
    *   `GET /auth/{provider}/callback`: OAuth redirect handler. Calls `AuthService`.
    *   `POST /submit`: Accepts `SubmitForm`. Validates dataclass. Initiates `SummarizationWorkflow` as a background task. Returns an HTML placeholder containing the HTMX SSE connection attributes pointing to `/sse/{new_job_id}`.
    *   `GET /sse/{job_id}`: Establishes the EventStream. Calls `JobStateManager.subscribe()`. Yields SSE messages formatted for HTMX to update the UI (e.g., swapping "thinking" indicators or appending text).
    *   `GET /job/{job_id}`: Renders a completed job. Uses `s03_convert_markdown...` and `s04_convert_html_timestamps...` to format the final output for display.
*   **UI Components (`components/`):**
    *   **`forms.py`**: Renders the `SummaryRequestForm` dataclass into FastTags, including the language selection dropdown.
    *   **`job_display.py`**: Components for the "Thinking..." state, the live streaming output area, and the final formatted summary card. Utilizes localization for labels.

## 4. Key Technical Decisions

### 4.1 PostgreSQL vs. SQLite
While SQLite is excellent for prototypes, PostgreSQL provides necessary features for a multi-user production environment, including better concurrency handling (essential for multiple long-running async GenAI jobs) and robust data integrity types. Using an async driver is crucial to not block the FastHTML event loop during DB operations.

### 4.2 Server-Sent Events (SSE) vs. Polling
The GenAI models, especially with "thinking" enabled, have variable response times. Polling (`every 1s`) is inefficient, creating unnecessary server load and latency in UI updates. SSE provides a persistent, unidirectional channel perfectly explicitly supported by FastHTML (`EventStream`) and HTMX (`hx-ext="sse"`), allowing the backend to push "thinking" states and partial summaries the instant they are available.

### 4.3 Dataclasses for Forms
Moving from unstructured dictionaries to `dataclasses` provides type safety, automatic validation hooks, and clearer documentation of what data the application expects from the user. FastHTML supports binding form data directly to dataclasses.

### 4.4 Async GenAI with "Thinking"
Adopting the `p02_impl.py` approach allows the UI to show the model's "thought process" separate from the final answer. This greatly improves perceived performance and user engagement compared to staring at a loading spinner for 30+ seconds.

## 5. Testability Strategy

The layered architecture significantly improves testability compared to the original script.

1.  **Unit Testing Domain/Utils:** `dataclasses` and existing utility functions (`s01`-`s04`) can be tested in isolation (already partially done in `t01`-`t04`).
2.  **Mocking Infrastructure:**
    *   We can create a `MockGenAIAdapter` that yields predefined streams of thinking/content without hitting the Google API.
    *   We can create a `MockTranscriptProvider` that returns fixed strings instead of calling `yt-dlp`.
    *   We can use an in-memory database or test-scoped PostgreSQL container for DB tests.
3.  **Testing Service Logic:** By injecting mocks into the `SummarizationWorkflow`, we can test state transitions, DB updates, and SSE publishing without external side effects.
4.  **FastHTML Route Testing:** FastHTML's test client can be used to verify that routes return the correct HTML structure and HTMX attributes, given specific states in the mocked service layer.

## 6. Migration Plan Summary

1.  **Setup:** Initialize new project structure, setup PostgreSQL, install dependencies (`asyncpg`, `google-genai`, etc.).
2.  **Infra Core:** Implement `DatabaseAdapter` and repositories. Port `s0*` utilities.
3.  **GenAI Core:** Implement `GenAIAdapter` using `p02_impl.py` as reference.
4.  **Auth:** Implement `AuthService` and basic Google OAuth routes.
5.  **Service/Workflow:** Implement `SummarizationWorkflow` and `JobStateManager` (dummy `TranscriptProvider` initially).
6.  **UI & SSE:** Create dataclass forms, implement `/submit` and `/sse/{id}` routes, and connect HTMX frontend.
7.  **YouTube Integration:** Implement the real `TranscriptProvider`.
8.  **Localization:** Extract hardcoded strings and implement `LocalizationService`.
9.  **Final Polish:** Styling, error handling, and comprehensive testing.