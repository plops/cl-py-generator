Of course. Here is the implementation for Part 4: Presentation Layer.

This layer connects all the underlying services and infrastructure to the web. It defines the HTTP routes, handles user input, and renders the UI using FastHTML's FastTags. It's the final piece that brings the entire application to life.

### Directory Structure Update

The new files will be placed inside new `routes` and `middleware` directories, and a `components.py` file will be added.

```
.
├── rocket_recap/
│   ├── __init__.py
│   ├── main.py                     # <-- Will be updated
│   ├── ... (config, models)
│   ├── infra/
│   │   └── ...
│   ├── services/
│   │   └── ...
│   ├── components.py               # <-- Implementation Below
│   ├── middleware.py               # <-- Implementation Below
│   └── routes/
│       ├── __init__.py
│       ├── ui.py                   # <-- Implementation Below
│       ├── auth.py                 # <-- Implementation Below
│       └── jobs.py                 # <-- Implementation Below
└── ...
```

---
### 1. `rocket_recap/components.py`

This file contains reusable UI components. Defining them as functions keeps the route logic clean and promotes consistency across the application.

```python
# rocket_recap/components.py

from fasthtml.common import *
from .models import SummaryRequestForm

def Page(title: str, *content):
    """A standard page layout component."""
    return Titled(f"RocketRecap - {title}",
        # Basic navigation header
        Header(
            Nav(
                Ul(Li(H1("🚀 RocketRecap"))),
                Ul(
                    Li(A("History", href="/history")),
                    Li(A("Settings", href="/settings")),
                    Li(Button("Logout", hx_post="/logout", cls="secondary"))
                )
            )
        ),
        Main(*content, cls="container")
    )

def SummaryForm():
    """Renders the main form for submitting a new summarization job."""
    return Form(
        Fieldset(
            Label("YouTube URL", fr="source_url"),
            Input(id="source_url", name="source_url", placeholder="https://www.youtube.com/watch?v=..."),

            Label("Or Paste Transcript", fr="manual_transcript"),
            Textarea(id="manual_transcript", name="manual_transcript", rows=5),

            # In a real app, model and language options would be dynamic
            Label("Model", fr="model_id"),
            Select(
                Option("gemini-1.5-flash-latest", value="gemini-1.5-flash-latest", selected=True),
                Option("gemini-1.5-pro-latest", value="gemini-1.5-pro-latest"),
                name="model_id", id="model_id"
            ),

            Button("Generate Summary", type="submit")
        ),
        hx_post="/summarize",
        hx_target="#summary-results",
        hx_swap="innerHTML"
    )

def JobInProgress(job_id: int):
    """
    Renders the initial placeholder for a new job, which connects to the SSE stream.
    """
    return Div(
        H4(f"Processing Job #{job_id}..."),
        Article(
            Header(P("Status: PENDING", id=f"job-status-{job_id}")),
            Pre(
                Code("Waiting for updates...", id=f"job-content-{job_id}"),
                style="white-space: pre-wrap; word-wrap: break-word;"
            ),
            aria_busy="true",
        ),
        # HTMX attributes to connect to the SSE stream
        hx_ext="sse",
        sse_connect=f"/sse/{job_id}",
        # Swap specific events from the stream into the DOM
        sse_swap="message",
        id=f"job-container-{job_id}"
    )

```

---

### 2. `rocket_recap/middleware.py`

This file defines the authentication middleware. It checks the session for a user ID on protected routes and can redirect to login if the user is not authenticated.

```python
# rocket_recap/middleware.py

from starlette.responses import RedirectResponse
from fasthtml.core import Beforeware

# For now, we'll just check for a 'user_id' in the session.
# A full implementation would fetch the User object from the DB.
def auth_check(req, sess):
    """
    Beforeware function that checks for a user_id in the session.
    If not found, it redirects to the login page.
    """
    if not sess.get("user_id"):
        return RedirectResponse("/login", status_code=303)

# Define the routes to skip authentication for.
# This must include login, auth callbacks, and static assets.
auth_middleware = Beforeware(
    auth_check,
    skip=[
        r'/login',
        r'/auth/.*',
        r'/static/.*',
        r'/health',
        r'favicon.ico'
    ]
)
```

---

### 3. `rocket_recap/routes/auth.py`

This router handles all user authentication and session management endpoints.

```python
# rocket_recap/routes/auth.py

from fasthtml.common import *
from ..services.auth_service import auth_service

# All routes in this file are managed by this APIRouter instance.
router = APIRouter()
rt = router.route

@rt("/login")
def login_page():
    """Displays the login page."""
    return Titled("Login",
        Main(
            H2("Sign In"),
            # This link will trigger the OAuth flow
            A("Sign in with Google", href="/auth/google/login", role="button"),
            cls="container"
        )
    )

@rt("/auth/google/login")
def google_login():
    """Redirects the user to Google's OAuth consent screen."""
    # In a real app, you would use an OAuth library to generate this URL
    # with the correct client_id, redirect_uri, scope, and state.
    google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?..." # Placeholder
    print(f"Redirecting to: {google_auth_url}")
    return RedirectResponse(google_auth_url, status_code=303)

@rt("/auth/google/callback")
async def google_callback(req, sess, code: str = None):
    """
    Handles the callback from Google. Exchanges the code for user info,
    logs the user in, and redirects to the home page.
    """
    if not code:
        return "Error: No code provided.", 400

    # Mock user profile, as we don't have a real OAuth client setup
    mock_user_profile = {"email": "testuser@example.com", "sub": "12345"}
    
    # Use the auth service to handle the business logic
    user = await auth_service.handle_oauth_callback("google", mock_user_profile)
    
    # Store user ID in the session to log them in
    sess["user_id"] = user.id
    sess["user_email"] = user.email

    return RedirectResponse("/", status_code=303)

@rt("/logout", methods=["POST"])
def logout(sess):
    """Clears the session to log the user out."""
    sess.clear()
    return RedirectResponse("/login", status_code=303)
```

---
### 4. `rocket_recap/routes/jobs.py`

This router manages the core application workflow: submitting new jobs and streaming their results.

```python
# rocket_recap/routes/jobs.py

from fasthtml.common import *
from ..models import SummaryRequestForm
from ..services.summarization_service import start_summarization_workflow
from ..services.sse_service import sse_service
from ..infra.repositories import SummaryRepository
from ..components import JobInProgress

router = APIRouter()
rt = router.route

@rt("/summarize", methods=["POST"])
async def create_summary_job(sess, summary_form: SummaryRequestForm):
    """
    Endpoint to create a new summarization job.
    Starts a background task and returns an initial SSE container.
    """
    user_id = sess.get("user_id")
    if not user_id:
        return "Unauthorized", 401

    # 1. Create the job record in the database
    repo = SummaryRepository()
    job = await repo.create_job(user_id, summary_form)
    
    # 2. Start the background workflow
    start_summarization_workflow(job.id, summary_form)

    # 3. Return the initial component that connects to the SSE stream
    return JobInProgress(job.id)


@rt("/sse/{job_id:int}")
async def stream_job_updates(req, sess, job_id: int):
    """Endpoint for Server-Sent Events (SSE)."""
    user_id = sess.get("user_id")
    # TODO: Add logic to verify that this user_id owns the job_id

    async def event_generator():
        # Subscribe to the SSE service for this job's channel
        async for message in sse_service.subscribe(job_id):
            # Format the message into an SSE event
            event = message.get("type", "message")
            if event == "content":
                # For content, we create an OOB swap to append to the code block
                data = to_xml(Pre(Code(message['text']), hx_swap_oob="beforeend", id=f"job-content-{job_id}"))
            elif event == "status_update":
                data = to_xml(P(f"Status: {message['status']}", hx_swap_oob="innerHTML", id=f"job-status-{job_id}"))
            else:
                data = str(message) # Fallback for other message types
            
            yield sse_message(data, event=event)

    return EventStream(event_generator())
```

---

### 5. `rocket_recap/routes/ui.py`

This router handles the main user interface pages.

```python
# rocket_recap/routes/ui.py

from fasthtml.common import *
from ..components import Page, SummaryForm

router = APIRouter()
rt = router.route

@rt("/")
def home(sess):
    """The main dashboard page."""
    user_email = sess.get("user_email", "Guest")
    return Page(
        title="Dashboard",
        H2(f"Welcome, {user_email}"),
        P("Submit a YouTube URL below to get started."),
        SummaryForm(),
        # This div is the target for new job results
        Div(id="summary-results")
    )

# Placeholders for future pages
@rt("/history")
def history_page():
    return Page("History", P("Your past jobs will appear here."))

@rt("/settings")
def settings_page():
    return Page("Settings", P("User settings will be available here."))

```

---

### 6. `rocket_recap/main.py` (Updated)

Finally, update the main entrypoint to import and use the new middleware and routers.

```python
# rocket_recap/main.py (Updated)

from fasthtml.common import *
from fasthtml.fastapp import fast_app

from .config import settings
# Import the new modules
from .routes import ui, auth, jobs
from .middleware import auth_middleware

def create_app():
    """
    Creates and configures the main FastHTML application instance.
    """
    app, rt = fast_app(
        secret_key=settings.SESSION_SECRET_KEY,
        # Apply the authentication middleware
        before=auth_middleware
    )

    # --- Mount Routers ---
    # Register the routes from the separate files with the main app.
    ui.router.to_app(app)
    auth.router.to_app(app)
    jobs.router.to_app(app)

    # Health check endpoint remains useful for deployment monitoring
    @rt("/health")
    def health_check():
        return {"status": "ok"}

    return app

app = create_app()

if __name__ == "__main__":
    serve()
```