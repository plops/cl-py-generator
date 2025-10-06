show the implementations for part 1 core application & configuration



Of course. Here are the implementations for Part 1: Core Application & Configuration.

This code establishes the foundational structure of the application, including the main entrypoint, configuration management, and the core data models that will be used across all layers.

### Directory Structure

For context, the files will be organized as follows:

```
.
├── rocket_recap/
│   ├── __init__.py
│   ├── main.py         # <-- Implementation Below
│   ├── config.py       # <-- Implementation Below
│   └── models.py       # <-- Implementation Below
├── tests/
│   └── ... (to be created later)
└── .env                # File for local environment variables
```

---

### 1. `rocket_recap/config.py`

This file uses `pydantic-settings` to load configuration from environment variables (and a `.env` file for local development). This approach provides type safety, validation, and a single source of truth for all settings.

```python
# rocket_recap/config.py

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Loads and validates application settings from environment variables.
    For local development, create a .env file in the project root.
    """
    # Use model_config to specify the .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # --- Database Configuration ---
    # The DSN for connecting to the asynchronous PostgreSQL database.
    # Format: postgresql+asyncpg://user:password@host:port/dbname
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/rocket_recap_db"

    # --- Google GenAI API ---
    # The API key for accessing Google's Generative AI services.
    # This is a critical secret and must be set in the environment.
    GEMINI_API_KEY: str

    # --- Session and Authentication ---
    # A secret key for signing session cookies. This should be a long, random
    # string. You can generate one with: openssl rand -hex 32
    SESSION_SECRET_KEY: str

    # --- Google OAuth 2.0 Credentials ---
    # These are obtained from the Google Cloud Console for your OAuth client.
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str

    # --- Application Settings ---
    LOG_LEVEL: str = "INFO"


# Create a single, globally accessible instance of the settings.
# This instance will be imported by other parts of the application.
settings = Settings()

```

---

### 2. `rocket_recap/models.py`

This file defines the core data structures of the application using Python's `enum` and `dataclasses`. These models serve as the schema for our application's logic, form handling, and database interactions.

```python
# rocket_recap/models.py

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

class JobStatus(str, Enum):
    """Enumeration for the lifecycle status of a summarization job."""
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    GENERATING = "GENERATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class User:
    """Represents an authenticated user in the system."""
    id: Optional[int] = None
    email: str = ""
    oauth_provider: str = ""
    oauth_provider_id: str = ""
    preferred_language: str = "en-US"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


@dataclass
class SummaryRequestForm:
    """Binds to the data submitted from the main summarization form."""
    source_url: Optional[str] = None
    manual_transcript: Optional[str] = None
    model_id: str = "gemini-1.5-flash-latest"
    output_language: str = "en-US"


@dataclass
class SummaryJob:
    """Represents a single summarization job and its complete state."""
    id: Optional[int] = None
    user_id: Optional[int] = None
    status: JobStatus = JobStatus.PENDING
    source_url: Optional[str] = None
    model_id: str = "gemini-1.5-flash-latest"
    transcript_raw: Optional[str] = None
    summary_markdown: Optional[str] = None
    summary_youtube_format: Optional[str] = None
    error_message: Optional[str] = None
    output_language: str = "en-US"
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    grounding_used: bool = False
    cost_input_usd: Optional[Decimal] = None
    cost_output_usd: Optional[Decimal] = None
    cost_grounding_usd: Optional[Decimal] = None
    cost_total_usd: Optional[Decimal] = None
    # Omitting embedding fields for now, can be added later
    client_host: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

```

---

### 3. `rocket_recap/main.py`

This is the application's main entrypoint. It initializes the FastHTML application, integrates the configuration, and prepares to mount the various route modules from the `routes` directory (which will be created in Part 4).

```python
# rocket_recap/main.py

from fasthtml.common import *
from fasthtml.fastapp import fast_app

# Import the centralized settings object
from .config import settings

# --- Placeholder for future imports ---
# As we build the application, we will import and mount route modules here.
# from .routes import ui, auth, jobs
# from .middleware import AuthMiddleware

def create_app():
    """
    Creates and configures the main FastHTML application instance.
    """
    # Define middleware that will be applied to the application.
    # The AuthMiddleware will be implemented later.
    middleware = [
        # Middleware(AuthMiddleware)
    ]

    # Create the FastHTML application instance.
    # The secret_key is essential for secure session management.
    app, rt = fast_app(
        secret_key=settings.SESSION_SECRET_KEY,
        # middleware=middleware # To be uncommented later
    )

    # --- Mount Routers ---
    # In a larger application, we organize routes into separate files
    # and "include" them here. These will be implemented in Part 4.
    # ui.router.to_app(app)
    # auth.router.to_app(app)
    # jobs.router.to_app(app)


    # --- Example Top-Level Route ---
    # A simple health check endpoint to confirm the server is running.
    @rt("/health")
    def health_check():
        return {"status": "ok"}

    # Placeholder for the main route, will be moved to routes/ui.py
    @rt("/")
    def home():
        return Titled("Welcome to RocketRecap v2", P("Coming soon..."))


    return app

# Create the application instance
app = create_app()

# The serve() function from FastHTML handles running the Uvicorn server.
# This should be the final call in the entrypoint script.
# `serve()` is designed to be called only when the script is executed directly.
if __name__ == "__main__":
    serve()

```

To make this runnable, you would also need to:
1.  Create an empty `rocket_recap/__init__.py` file.
2.  Install the necessary dependencies: `pip install python-fasthtml uvicorn pydantic-settings python-dotenv`.
3.  Create a `.env` file in the root directory with the required secrets.