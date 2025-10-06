Of course. Here is a comprehensive overview of the necessary API endpoints to implement the proposed architecture for RocketRecap v2.

This design is centered around a clear separation of concerns: user authentication, the core real-time summarization workflow, and historical data retrieval. The endpoints are designed to be consumed by an HTMX-powered front-end.

---

### **API Endpoint Overview: RocketRecap v2**

The endpoints are grouped into logical categories based on their function.

#### 1. User Authentication & Session Management

These endpoints handle the entire user lifecycle, from logging in via OAuth to logging out.

| Endpoint | Method | Authentication | Description | Request Data | Success Response |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET /` | `GET` | Optional | The main application page. If the user is authenticated, it renders the dashboard with the submission form and recent jobs. If not, it renders a public landing page with a login prompt. | - | Full HTML page. |
| `GET /login` | `GET` | Not Required | Displays the login page where the user can choose an OAuth provider (e.g., "Sign in with Google"). | - | Full HTML page with login buttons. |
| `GET /auth/google/login` | `GET` | Not Required | Initiates the OAuth flow. This endpoint does not return content; it redirects the user to the Google consent screen. | - | HTTP 302 Redirect to Google's OAuth endpoint. |
| `GET /auth/google/callback` | `GET` | Not Required | The callback URL that Google redirects to after user consent. It handles the code exchange, fetches user info, creates/updates the user in the DB, sets the session cookie, and redirects to the dashboard. | `code`, `state` (Query Params from Google) | HTTP 303 Redirect to `/`. |
| `POST /logout` | `POST` | **Required** | Logs the user out by clearing their session cookie. Using `POST` prevents CSRF logout attacks. | - | HTTP 303 Redirect to `/login`. |

---

#### 2. Core Summarization Workflow

This is the primary user journey. It involves submitting a new job and receiving real-time updates via Server-Sent Events (SSE).

| Endpoint | Method | Authentication | Description | Request Data | Success Response |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `POST /summarize` | `POST` | **Required** | The main action endpoint. It accepts the user's request, creates a new `summary_jobs` record in the database with `PENDING` status, and starts the `SummarizationWorkflow` as a background task. | Form Data bound to the `SummaryRequestForm` dataclass (`source_url`, `model_id`, etc.). | **HTML Partial.** Returns an initial placeholder `<div>` containing the HTMX attributes needed to connect to the SSE stream for the newly created job. e.g., `<div hx-ext="sse" sse-connect="/sse/{new_job_id}">Starting...</div>` |
| `GET /sse/{job_id}` | `GET` | **Required** | The Server-Sent Events (SSE) endpoint. A client connects here to receive real-time progress updates for a specific job. The connection is long-lived. The backend must verify that the authenticated user owns this `job_id`. | `job_id` (Path Parameter). | **`text/event-stream`**. Pushes a stream of events like `event: thinking`, `event: content`, `event: status_update`, and `event: complete`. Each event carries a `data:` payload with HTML content or JSON status information for the client to swap into the DOM. |

---

#### 3. User Dashboard & History

These endpoints allow users to view their completed summaries and manage their account.

| Endpoint | Method | Authentication | Description | Request Data | Success Response |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET /history` | `GET` | **Required** | Displays a paginated list of all the user's past summarization jobs. | Optional query params for pagination (e.g., `?page=2`). | Full HTML page containing a table or list of completed jobs. |
| `GET /job/{job_id}` | `GET` | **Required** | Displays the full details of a single, completed summarization job. The backend must verify that the authenticated user owns this `job_id`. | `job_id` (Path Parameter). | Full HTML page showing the final formatted summary, metadata (cost, tokens), and a link to the original source. |
| `GET /settings` | `GET` | **Required** | Renders a page where the user can view their profile and update settings, such as their preferred language. | - | Full HTML page with a settings form. |
| `POST /settings` | `POST` | **Required** | Processes updates to user settings (e.g., changing `preferred_language`). | Form Data with user preferences. | HTTP 303 Redirect back to `/settings` with a success toast/message. |

---

#### 4. Static Assets

| Endpoint | Method | Authentication | Description | Request Data | Success Response |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET /static/{filepath:path}` | `GET` | Not Required | Serves static files like CSS, JavaScript, images, and fonts. This is typically handled by FastHTML's/Starlette's built-in static file serving middleware. | `filepath` (Path Parameter). | The requested file with the appropriate `Content-Type` header. |

This set of endpoints provides a complete and logical structure for the application, clearly separating the real-time, event-driven core from the standard request-response patterns of authentication and data retrieval.