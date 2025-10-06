Of course. Here is a proposed overview of the source and test files for the implementation, grouped by their semantic function. This structure promotes separation of concerns and high testability, directly reflecting the architectural design.

---

### **Source and Test File Overview: RocketRecap v2**

This file structure organizes the project into logical layers: configuration, data models, infrastructure (external communication), services (business logic), and presentation (web routes and UI).

#### 1. Core Application & Configuration

This group contains the application's entrypoint, configuration management, and top-level definitions.

| File Path | Purpose | Testing Strategy |
| :--- | :--- | :--- |
| `rocket_recap/main.py` | The main application entrypoint. Initializes the FastHTML app, mounts middleware, includes the route modules, and defines the `serve()` command. | **Integration Test:** `tests/test_main.py` will use a test client to ensure the app starts and basic routes like `/` and `/login` return a `200 OK` status. |
| `rocket_recap/config.py` | Loads and validates all configuration from environment variables (e.g., Database URL, API keys, OAuth secrets) into a strongly-typed settings object. | **Unit Test:** `tests/test_config.py` will test loading settings from mock environment variables and ensure defaults are applied correctly. |
| `rocket_recap/models.py` | Defines all core `dataclasses` for the application, such as `User`, `SummaryJob`, and `SummaryRequestForm`. This is the single source of truth for data schemas. | **Unit Test:** `tests/test_models.py` will verify dataclass defaults and type correctness. |

---

#### 2. Infrastructure Layer (External Services)

These modules are wrappers around external systems, isolating the core business logic from the details of external communication.

| File Path | Purpose | Testing Strategy |
| :--- | :--- | :--- |
| `rocket_recap/infra/database.py` | Manages the asynchronous PostgreSQL connection pool and provides a dependency-injectable way to get a database session. | **Integration Test:** `tests/infra/test_database.py` will test the ability to connect to a test database and perform a basic query. |
| `rocket_recap/infra/repositories.py` | Contains repository classes (`UserRepository`, `SummaryRepository`) that encapsulate all SQL queries. The service layer will call these repositories instead of writing SQL directly. | **Integration Test:** `tests/infra/test_repositories.py` will test each repository method against a real test database to ensure queries are correct and return the expected data. |
| `rocket_recap/infra/genai_adapter.py` | **Replaces `p04_host.py`'s direct API calls.** Implements the modern, asynchronous Google GenAI client, incorporating logic from `p02_impl.py` for streaming, "thinking" detection, and cost calculation. | **Unit Test:** `tests/infra/test_genai_adapter.py` will use `unittest.mock` to patch the `genai` client and test the adapter's ability to handle mock API responses, correctly parse streams, and calculate costs without making real API calls. |
| `rocket_recap/infra/transcript_provider.py` | **Incorporates logic from `s01` and `s02`.** Handles downloading and parsing video transcripts. Contains the logic for calling `yt-dlp` and processing VTT files. | **Unit Test:** `tests/infra/test_transcript_provider.py` will test the VTT parsing logic with local file fixtures. The `yt-dlp` subprocess call will be mocked to test error handling and command construction. |

---

#### 3. Service Layer (Business Logic)

This is the heart of the application, orchestrating the workflows and containing the core business rules.

| File Path | Purpose | Testing Strategy |
| :--- | :--- | :--- |
| `rocket_recap/services/summarization_service.py` | Contains the primary `SummarizationWorkflow`. This background task orchestrates the entire process: calling the transcript provider, updating job status in the DB, invoking the GenAI adapter, and publishing updates to the SSE service. | **Unit Test:** `tests/services/test_summarization_service.py` will be a key test suite. It will test the workflow's state transitions by injecting mocked repositories, adapters, and SSE service, ensuring the correct methods are called in the correct order. |
| `rocket_recap/services/sse_service.py` | Manages the in-memory Pub/Sub channels for Server-Sent Events. Provides methods for the summarization service to `publish` updates and for the SSE web route to `subscribe` to them. | **Unit Test:** `tests/services/test_sse_service.py` will test the publish/subscribe mechanism in-memory to ensure messages are correctly broadcast to subscribers for a given job ID. |
| `rocket_recap/services/auth_service.py` | Handles the business logic for OAuth callbacks: exchanging codes for tokens, fetching user profiles, and creating or updating user records in the database via the `UserRepository`. | **Unit Test:** `tests/services/test_auth_service.py` will test the logic using a mocked OAuth client and a mocked `UserRepository`, verifying that it correctly processes user data and creates the right session information. |
| `rocket_recap/services/localization_service.py`| Manages loading and providing translated strings for the UI and for constructing language-specific AI prompts. | **Unit Test:** `tests/services/test_localization_service.py` will test the service's ability to load language files and return the correct strings for given keys and locales. |

---

#### 4. Presentation Layer (Web Routes & UI)

This layer is responsible for handling HTTP requests and rendering the user interface.

| File Path | Purpose | Testing Strategy |
| :--- | :--- | :--- |
| `rocket_recap/routes/ui.py` | Contains standard page routes like `/`, `/history`, and `/settings`. These routes primarily fetch data and render full HTML pages. | **Integration Test:** `tests/routes/test_ui.py` will use the test client to make `GET` requests to these endpoints, asserting that they return a `200 OK` and that the HTML contains expected elements (e.g., a title, a settings form). |
| `rocket_recap/routes/auth.py` | Contains all authentication-related endpoints: `/login`, `/logout`, and the OAuth `/callback` handler. | **Integration Test:** `tests/routes/test_auth.py` will simulate the OAuth flow by mocking the external provider, ensuring the routes correctly set session cookies and perform redirects. |
| `rocket_recap/routes/jobs.py` | Contains the core workflow endpoints: `POST /summarize` (to start a job) and `GET /sse/{job_id}` (to stream results). | **Integration Test:** `tests/routes/test_jobs.py` will be critical. It will test the `POST /summarize` endpoint to ensure a background task is started and the correct HTMX partial is returned. It will also test the SSE endpoint with a mock stream. |
| `rocket_recap/components.py` | Contains reusable UI components written as FastTag functions (e.g., `render_job_card`, `render_summary_form`). | **Unit Test:** `tests/test_components.py` will test these functions to ensure they generate the correct HTML structure and attributes when called with different data. |
| `rocket_recap/middleware.py` | Defines custom middleware, primarily the authentication middleware that checks for a valid session and attaches the `User` object to the request. | **Integration Test:** `tests/test_middleware.py` will test that protected routes correctly redirect unauthenticated users and allow authenticated users to pass. |

---

#### 5. Utilities & Supporting Files

| File Path | Purpose | Testing Strategy |
| :--- | :--- | :--- |
| `rocket_recap/utils/formatters.py` | **Incorporates logic from `s03` and `s04`.** Contains pure utility functions for text manipulation, like converting markdown for YouTube and replacing timestamps in HTML with clickable links. | **Unit Test:** `tests/utils/test_formatters.py` will contain parameterized tests for these pure functions, using the existing test cases from `t03` and `t04` as a starting point. |
| `alembic/` (directory) | Contains database migration scripts, managed by the Alembic tool, to handle schema changes over time. | N/A (Alembic has its own versioning system). |
| `static/` (directory) | Contains all static assets like CSS stylesheets, client-side JavaScript, and images. | N/A (Tested via end-to-end or visual regression tests, outside the scope of unit/integration tests). |
| `tests/conftest.py` | Contains shared Pytest fixtures, such as fixtures to provide a test client instance, a fresh test database for each test run, and authenticated user sessions. | N/A (Provides setup for other tests). |