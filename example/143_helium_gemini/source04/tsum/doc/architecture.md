### 1. High-Level Architecture (Component Diagram)

This diagram illustrates the high-level architecture of the simplified RocketRecap application, highlighting its main components, external dependencies, and how they interact. The FastHTML framework serves as the presentation layer, handling user requests and rendering the UI. The core application logic in `p04_host.py` orchestrates the summarization process, utilizing external APIs for generative AI and transcript downloading, along with several utility scripts and a local SQLite database for persistence.

The user interacts with the FastHTML application, which initiates a background thread for summarization. This thread may download transcripts from YouTube, process them, call the Google Generative AI API for summary generation and embeddings, and store the results in the SQLite database. The UI is updated asynchronously via HTMX polling.

```mermaid
graph TD
    subgraph External Systems
        user[<i class='fa fa-user'></i> User Browser]
        youtube[<i class='fa fa-youtube'></i> YouTube]
        genai_api[<i class='fa fa-robot'></i> Google GenAI API]
    end

    subgraph "RocketRecap Simplified App"
        subgraph "Presentation Layer [Presentation Layer - FastHTML]"
            direction TB
            routes["Routes & UI Rendering"]
        end

        subgraph "Application Logic [Application Logic - p04_host.py]"
            direction TB
            summarization_process["Summarization Process (Threaded)"]
            transcript_downloader["Transcript Downloader (yt-dlp)"]
            genai_integrator["GenAI Integrator"]
            db_manager["Database Manager"]
            utils["Utility Functions"]
        end

        subgraph "Data Storage"
            sqlite_db[<i class='fa fa-database'></i> SQLite DB]
        end

        subgraph "Utility Scripts"
            s01["s01_validate_youtube_url.py"]
            s02["s02_parse_vtt_file.py"]
            s03["s03_convert_markdown_to_youtube_format.py"]
            s04["s04_convert_html_timestamps_to_youtube_links.py"]
        end
    end

    %% --- Interactions ---
    user --> routes
    routes --> db_manager
    routes --> summarization_process
    summarization_process --> transcript_downloader
    transcript_downloader --> youtube
    transcript_downloader --> s01
    transcript_downloader --> s02
    summarization_process --> genai_integrator
    genai_integrator --> genai_api
    summarization_process --> db_manager
    genai_integrator --> s03
    routes --> user
    user -- HTMX polling --> routes
    routes --> s04
    db_manager --> sqlite_db
    s01 -.-> transcript_downloader
    s02 -.-> transcript_downloader
    s03 -.-> genai_integrator
    s04 -.-> routes

    classDef external fill:#D1E8E2,stroke:#333,stroke-width:2px;
    classDef app fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class user,youtube,genai_api external;
    class routes,summarization_process,transcript_downloader,genai_integrator,db_manager,utils,sqlite_db,s01,s02,s03,s04 app;
```

### 2. Asynchronous Summarization Flow (Sequence Diagram)

This sequence diagram details the asynchronous workflow for generating a video summary within the `p04_host.py` application. The user initiates the process via a web request, which immediately triggers a non-blocking background task. The client-side UI uses HTMX to poll the server for updates, receiving partial summaries and status changes as the background task progresses through transcript downloading, AI generation, and post-processing.

```mermaid
sequenceDiagram
    actor User
    participant Browser
    participant FastHTML Server (p04_host.py)
    participant Background Thread (download_and_generate)
    participant yt-dlp
    participant GenAI API
    participant SQLite DB

    User->>Browser: Enters YouTube URL/Transcript, Clicks 'Summarize'
    Browser->>FastHTML Server: POST /process_transcript (url, transcript, model)
    activate FastHTML Server
    FastHTML Server->>SQLite DB: INSERT new job (status: PENDING)
    SQLite DB-->>FastHTML Server: Returns new identifier
    FastHTML Server->>Background Thread: thread.start(download_and_generate(identifier))
    FastHTML Server-->>Browser: Returns initial HTML with HTMX polling (generation_preview)
    deactivate FastHTML Server

    Browser->>FastHTML Server: GET /generations/{identifier} (HTMX polling)
    activate FastHTML Server

    alt If transcript not provided
        Background Thread->>yt-dlp: Execute yt-dlp to download subtitles
        yt-dlp-->>Background Thread: Returns VTT file content
        Background Thread->>SQLite DB: UPDATE job (transcript, status: DOWNLOADING_COMPLETE)
        Note over FastHTML Server, Browser: HTMX updates UI with 'Downloading transcript...' status
    end

    Background Thread->>GenAI API: Request summary (prompt, transcript, model)
    loop For each summary chunk
        GenAI API-->>Background Thread: Yields content chunk
        Background Thread->>SQLite DB: UPDATE job (summary = summary + chunk)
        Note over FastHTML Server, Browser: HTMX polls /generations, refreshes UI with partial summary
    end
    GenAI API-->>Background Thread: Final response (usage metadata)
    Background Thread->>SQLite DB: UPDATE job (summary_done=True, input/output tokens, timestamp)
    
    opt Generate Embeddings
        Background Thread->>GenAI API: Embed transcript
        GenAI API-->>Background Thread: Returns embedding
        Background Thread->>SQLite DB: UPDATE job (full_embedding)

        Background Thread->>GenAI API: Embed summary
        GenAI API-->>Background Thread: Returns embedding
        Background Thread->>SQLite DB: UPDATE job (embedding)
    end

    Background Thread->>SQLite DB: UPDATE job (timestamps_done=True, timestamped_summary_in_youtube_format, timestamps_timestamp_end)
    Background Thread-->>FastHTML Server: (Implicit completion)

    FastHTML Server->>Browser: Renders final summary with formatted timestamps (via HTMX poll)
    deactivate FastHTML Server
    Note over Browser: HTMX polling stops (data_hx_trigger="")
```

---

### 3. Database Schema (Entity Relationship Diagram)

This ERD visualizes the logical structure of the SQLite database used by `p04_host.py`. The application uses a single table, `summaries`, to store all information related to a summarization job. This table includes fields for the original source, the AI model used, the generated summary (both raw and YouTube-formatted), cost details, token usage, and vector embeddings for both the transcript and the summary. The `identifier` serves as the primary key.

```mermaid
erDiagram
    summaries {
        INTEGER identifier PK "Auto-incrementing primary key"
        TEXT model "AI model used for summarization"
        TEXT transcript "Full video transcript"
        TEXT host "Client host IP address"
        TEXT original_source_link "Original YouTube URL or source"
        BOOLEAN include_comments "Flag for including comments (currently unused)"
        BOOLEAN include_timestamps "Flag for including timestamps (currently unused)"
        BOOLEAN include_glossary "Flag for including glossary (currently unused)"
        TEXT output_language "Desired output language (currently unused)"
        TEXT summary "Partial or complete AI-generated summary"
        BOOLEAN summary_done "True if summary generation is complete"
        INTEGER summary_input_tokens "Tokens used for summary input"
        INTEGER summary_output_tokens "Tokens generated for summary output"
        TEXT summary_timestamp_start "Timestamp when summary generation started"
        TEXT summary_timestamp_end "Timestamp when summary generation ended"
        TEXT timestamps "Intermediate timestamp data (currently unused in final output)"
        BOOLEAN timestamps_done "True if timestamp formatting is complete"
        INTEGER timestamps_input_tokens "Tokens used for timestamp conversion input (currently 0)"
        INTEGER timestamps_output_tokens "Tokens generated for timestamp conversion output (currently 0)"
        TEXT timestamps_timestamp_end "Timestamp when timestamp formatting ended"
        TEXT timestamped_summary_in_youtube_format "Final summary formatted for YouTube with clickable links"
        REAL cost "Estimated cost of AI generation in USD"
        BLOB embedding "Vector embedding of the final summary (for similarity search)"
        BLOB full_embedding "Vector embedding of the full transcript (for similarity search)"
    }
```

---

### 4. Job Processing Lifecycle (State Diagram)

This state diagram models the lifecycle of a `summary` job as it progresses through various stages in the `p04_host.py` application. A job begins in the `INITIAL` state upon user submission and transitions through stages of transcript acquisition, AI summarization, and post-processing, including embedding generation and YouTube formatting. The process concludes in either a `COMPLETED` state upon successful execution or a `FAILED` state if an error occurs at any point.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> INITIAL: User submits request

    state "Processing Transcript" as Processing {
        INITIAL --> DOWNLOADING_TRANSCRIPT: Transcript not provided
        INITIAL --> VALIDATING_TRANSCRIPT: Transcript provided

        DOWNLOADING_TRANSCRIPT --> VALIDATING_TRANSCRIPT: Transcript downloaded
        DOWNLOADING_TRANSCRIPT --> FAILED: Download error

        VALIDATING_TRANSCRIPT --> GENERATING_SUMMARY: Transcript valid
        VALIDATING_TRANSCRIPT --> FAILED: Validation error (length, format)

        GENERATING_SUMMARY --> EMBEDDING_TRANSCRIPT: Summary stream complete
        GENERATING_SUMMARY --> FAILED: GenAI summary error

        EMBEDDING_TRANSCRIPT --> FORMATTING_SUMMARY_FOR_YOUTUBE: Transcript embedding done (or skipped)
        EMBEDDING_TRANSCRIPT --> FAILED: Embedding error (transcript)

        FORMATTING_SUMMARY_FOR_YOUTUBE --> EMBEDDING_SUMMARY: Summary formatted
        FORMATTING_SUMMARY_FOR_YOUTUBE --> FAILED: Formatting error

        EMBEDDING_SUMMARY --> COMPLETED: Summary embedding done (or skipped)
        EMBEDDING_SUMMARY --> FAILED: Embedding error (summary)
    }

    COMPLETED --> [*]
    FAILED --> [*]
```

### 5. API Routes and Endpoints

This section provides a detailed overview of all the HTTP routes supported by the `p04_host.py` application. These endpoints are designed to be consumed by an HTMX-powered front-end, handling everything from serving the main user interface to processing form submissions and providing real-time updates via polling.

| Endpoint | Method | Description | Request Data | Success Response |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | Renders the main application page. This includes the submission form and a list of recently completed summaries fetched from the SQLite database. | - | Full HTML page (`text/html`). |
| `/process_transcript` | `POST` | The primary action endpoint that initiates a new summarization job. It creates a record in the `summaries` table, starts the `download_and_generate` function in a background thread, and returns an initial placeholder UI. | Form Data (`source_url`, `manual_transcript`, `model_id`). | **HTML Partial.** Returns a `<div>` containing the HTMX attributes needed to start polling the `/generations/{identifier}` endpoint for the newly created job. |
| `/generations/{id}` | `GET` | **HTMX Polling Endpoint.** Provides real-time progress updates for a specific job. It is polled every second by the client. The server checks the job's status in the database (`summary_done`, `timestamps_done`) and returns the appropriate HTML partial. | `id` (Path Parameter). | **HTML Partial.** Returns an "in-progress" view while the job is running. When complete, it returns the final formatted summary and includes an `HX-Trigger` header to stop the polling. |

