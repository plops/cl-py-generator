Of course. Here is the implementation for Part 3: Service Layer.

This layer contains the core business logic of the application. It acts as an orchestrator, using the infrastructure components from Part 2 to perform complex workflows. These services are designed to be stateless and easily testable by injecting their dependencies.

### Directory Structure Update

The new files will be placed inside a new `services` directory:

```
.
├── rocket_recap/
│   ├── __init__.py
│   ├── ... (config, models)
│   ├── infra/
│   │   └── ... (database, repositories, etc.)
│   └── services/
│       ├── __init__.py
│       ├── sse_service.py            # <-- Implementation Below
│       ├── localization_service.py   # <-- Implementation Below
│       ├── auth_service.py           # <-- Implementation Below
│       └── summarization_service.py  # <-- Implementation Below
└── ...
```
---

### 1. `rocket_recap/services/sse_service.py`

This is the in-memory message bus for Server-Sent Events. It uses `asyncio.Queue` to manage real-time communication between the background summarization task (the publisher) and the web request handler (the subscriber) without touching the database for every update.

```python
# rocket_recap/services/sse_service.py

import asyncio
from typing import Dict, AsyncIterator, Any

class SSEService:
    """
    Manages in-memory channels for Server-Sent Events (SSE).
    This allows for a pub/sub model where a background task can publish
    updates that are streamed to a client via a web request handler.
    """

    def __init__(self):
        # A dictionary to hold asyncio Queues, keyed by a channel identifier (e.g., job_id).
        self._channels: Dict[int, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, channel_id: int) -> AsyncIterator[Any]:
        """
        Subscribes to a channel to receive messages. Creates the channel if it doesn't exist.
        Yields messages as they are published.
        """
        async with self._lock:
            if channel_id not in self._channels:
                self._channels[channel_id] = asyncio.Queue()
        
        queue = self._channels[channel_id]

        try:
            while True:
                message = await queue.get()
                # A special sentinel value to signal the end of the stream.
                if message.get("type") == "EOF":
                    break
                yield message
        finally:
            # Clean up the queue once the subscriber disconnects or the stream ends.
            async with self._lock:
                if channel_id in self._channels:
                    # Check if queue is empty before deleting
                    if self._channels[channel_id].empty():
                         del self._channels[channel_id]

    async def publish(self, channel_id: int, message: Any):
        """Publishes a message to all subscribers of a channel."""
        # It's possible to publish before a client has subscribed.
        # Ensure the queue exists.
        async with self._lock:
            if channel_id not in self._channels:
                self._channels[channel_id] = asyncio.Queue()

        await self._channels[channel_id].put(message)

    async def close_channel(self, channel_id: int):
        """Publishes the end-of-stream signal to a channel."""
        await self.publish(channel_id, {"type": "EOF"})


# Create a single, shared instance to be used across the application.
sse_service = SSEService()
```

---

### 2. `rocket_recap/services/localization_service.py`

This service manages translated text for both the UI and for constructing prompts for the AI, ensuring the model receives instructions in the desired language.

```python
# rocket_recap/services/localization_service.py

class LocalizationService:
    """Manages translated strings and language-specific AI prompts."""

    def __init__(self):
        # In a real application, this would be loaded from JSON or YAML files.
        self._translations = {
            "en-US": {
                "summary_instructions": "Please give an abstract of the transcript and then summarize it in a self-contained bullet list format.",
                "title": "Welcome to RocketRecap"
            },
            "de": {
                "summary_instructions": "Bitte erstellen Sie eine Zusammenfassung des Transkripts und fassen Sie es dann in einer eigenständigen Aufzählungsliste zusammen.",
                "title": "Willkommen bei RocketRecap"
            },
            "es": {
                "summary_instructions": "Por favor, dé un resumen de la transcripción y luego resúmala en un formato de lista de viñetas autocontenida.",
                "title": "Bienvenido a RocketRecap"
            }
        }
        self._fallback_lang = "en-US"

    def get_string(self, key: str, language: str) -> str:
        """Gets a translated string for a given key and language."""
        lang_map = self._translations.get(language, self._translations[self._fallback_lang])
        return lang_map.get(key, f"<{key}>")

    def get_prompt(self, transcript: str, language: str) -> str:
        """Constructs the full, localized prompt for the summarization task."""
        instructions = self.get_string("summary_instructions", language)
        # Using the prompt structure from p04_host.py
        return f"""
        {instructions}
        Include starting timestamps, important details, and key takeaways.

        Here is the transcript. Please summarize it in {language}:
        {transcript}
        """

# Create a single, shared instance.
localization_service = LocalizationService()
```
---

### 3. `rocket_recap/services/auth_service.py`

This service contains the logic to handle the OAuth callback. It's responsible for interacting with the `UserRepository` to find or create a user after a successful external login.

```python
# rocket_recap/services/auth_service.py

from ..infra.repositories import UserRepository
from ..models import User

class AuthService:
    """Handles business logic related to user authentication."""

    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    async def handle_oauth_callback(self, provider: str, user_profile: dict) -> User:
        """
        Processes the user profile received from an OAuth provider.
        Gets or creates a user in the database and returns the User object.
        
        Args:
            provider: The name of the OAuth provider (e.g., 'google').
            user_profile: A dictionary containing user details from the provider.
                          Expected to have at least an 'email' key.
        """
        email = user_profile.get("email")
        if not email:
            raise ValueError("User profile from OAuth provider is missing an email.")

        provider_details = {
            "provider": provider,
            "provider_id": user_profile.get("sub") # 'sub' is a standard OAuth ID claim
        }

        # Use the repository to abstract away the database interaction.
        user = await self.user_repository.get_or_create_user(
            email=email,
            provider_details=provider_details
        )
        return user

# Create an instance, injecting its repository dependency.
# In a larger app, this would be handled by a dependency injection container.
auth_service = AuthService(UserRepository())
```
---

### 4. `rocket_recap/services/summarization_service.py`

This is the core orchestrator. A `SummarizationWorkflow` instance is created for each job. Its `run` method is executed as a background task. It uses all the other services and infrastructure adapters to perform its job.

```python
# rocket_recap/services/summarization_service.py

import asyncio
from typing import Dict

from ..models import SummaryRequestForm, JobStatus
from ..infra.repositories import SummaryRepository
from ..infra.transcript_provider import TranscriptProvider
from ..infra.genai_adapter import GenAIAdapter
from .sse_service import sse_service
from .localization_service import localization_service

class SummarizationWorkflow:
    """Orchestrates the entire summarization process for a single job."""

    def __init__(
        self,
        summary_repo: SummaryRepository,
        transcript_provider: TranscriptProvider,
        genai_adapter: GenAIAdapter
    ):
        # Dependencies are injected for testability
        self.summary_repo = summary_repo
        self.transcript_provider = transcript_provider
        self.genai_adapter = genai_adapter3