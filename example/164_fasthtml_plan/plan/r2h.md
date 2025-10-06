show implementation for part 2

Of course. Here are the implementations for Part 2: Infrastructure Layer.

This layer is crucial for isolating the application's core logic from the details of external systems. We will implement wrappers for the PostgreSQL database, the Google GenAI API, and the YouTube transcript downloader.

### Directory Structure Update

The new files will be placed inside a new `infra` directory:

```
.
├── rocket_recap/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   └── infra/
│       ├── __init__.py
│       ├── database.py             # <-- Implementation Below
│       ├── repositories.py         # <-- Implementation Below
│       ├── genai_adapter.py        # <-- Implementation Below
│       └── transcript_provider.py    # <-- Implementation Below
└── ...
```

---

### 1. `rocket_recap/infra/database.py`

This module manages the connection to the PostgreSQL database using `SQLAlchemy`'s asynchronous engine. It provides a simple, dependency-injectable way to get a database session for use in repositories and services.

```python
# rocket_recap/infra/database.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from ..config import settings

# Create an asynchronous engine instance.
# The engine manages a pool of connections to the database.
engine = create_async_engine(settings.DATABASE_URL, echo=False)

# Create a sessionmaker factory. This factory will be used to create
# new asynchronous sessions (database connections).
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Important for async usage
)

@asynccontextmanager
async def get_db_session() -> AsyncSession:
    """
    Provides an asynchronous database session within a context manager.
    This ensures that the session is properly closed after use.

    Usage:
    async with get_db_session() as session:
        # do database work...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

```

---

### 2. `rocket_recap/infra/repositories.py`

This file will contain the data access logic. For now, it's a placeholder because we first need to define the SQLAlchemy models that map our dataclasses to database tables. This would typically be done in a separate `db_models.py` or within the repositories file itself for smaller projects.

*Note: For a full implementation, you would use SQLAlchemy's Declarative Base to define table mappings. For brevity, this part is described in comments and will be assumed to exist for the service layer logic.*

```python
# rocket_recap/infra/repositories.py

# This file will contain the data access logic.
# A complete implementation requires SQLAlchemy table models.
# For example:
#
# from sqlalchemy.orm import declarative_base, Mapped, mapped_column
# from sqlalchemy import BigInteger, Text, Enum as SQLEnum
# from ..models import JobStatus
#
# Base = declarative_base()
#
# class SummaryJobDB(Base):
#     __tablename__ = 'summary_jobs'
#     id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
#     status: Mapped[JobStatus] = mapped_column(SQLEnum(JobStatus))
#     # ... other columns
#
# We will proceed assuming such models exist.

from ..models import SummaryJob, User
from .database import get_db_session
# from .db_models import SummaryJobDB, UserDB # Assuming these exist

class SummaryRepository:
    """Handles all database operations for SummaryJob objects."""

    async def create_job(self, user_id: int, form_data) -> SummaryJob:
        """Creates a new summary job record in the database."""
        async with get_db_session() as session:
            # In a real implementation, you would create a SummaryJobDB instance
            # and add it to the session.
            print(f"DATABASE: Creating job for user {user_id} with model {form_data.model_id}")
            # This is a mock implementation for now.
            new_job = SummaryJob(
                user_id=user_id,
                model_id=form_data.model_id,
                source_url=form_data.source_url,
                output_language=form_data.output_language
            )
            # In real code: session.add(db_job); await session.flush(); return SummaryJob.from_orm(db_job)
            new_job.id = 123 # Mock ID
            return new_job

    async def update_job_status(self, job_id: int, status, error_message: str = None):
        """Updates the status of a job."""
        async with get_db_session() as session:
            print(f"DATABASE: Updating job {job_id} to status {status}")
            # Real implementation:
            # await session.execute(
            #     update(SummaryJobDB).where(SummaryJobDB.id == job_id).values(status=status, ...)
            # )
            pass

    async def append_summary_chunk(self, job_id: int, chunk: str):
        """Appends a chunk of text to the summary_markdown field."""
        async with get_db_session() as session:
            print(f"DATABASE: Appending chunk to job {job_id}")
            # Real implementation using string concatenation operator:
            # await session.execute(
            #     update(SummaryJobDB).where(SummaryJobDB.id == job_id).values(
            #         summary_markdown=SummaryJobDB.summary_markdown + chunk
            #     )
            # )
            pass


class UserRepository:
    """Handles all database operations for User objects."""

    async def get_or_create_user(self, email: str, provider_details: dict) -> User:
        """Finds a user by email or creates a new one if they don't exist."""
        async with get_db_session() as session:
            print(f"DATABASE: Get or create user for email {email}")
            # Real implementation would query for the user and create if not found.
            return User(id=1, email=email, oauth_provider="google")

```
---

### 3. `rocket_recap/infra/genai_adapter.py`

This module is a crucial refactoring. It encapsulates all interactions with the Google GenAI API, using the new asynchronous client. It directly incorporates the patterns from `p02_impl.py`.

```python
# rocket_recap/infra/genai_adapter.py

import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..config import settings
from ..models import SummaryJob

# Configure the genai client with the API key from settings
genai.configure(api_key=settings.GEMINI_API_KEY)

# Pre-defined prompts and examples can be stored here or loaded from files
# For now, using a simplified version of the prompt from p04_host.py
EXAMPLE_INPUT = "..." # Load from file or define here
EXAMPLE_OUTPUT = "..." # Load from file or define here

class GenAIAdapter:
    """Adapter for interacting with the Google Generative AI API."""

    def _get_prompt(self, transcript: str) -> str:
        """Constructs the full prompt for the summarization task."""
        return f"""
        Below, I will provide an example transcript and the corresponding summary I expect.
        Afterward, I will provide a new transcript that I want you to summarize in the same format.
        Please give an abstract of the transcript and then summarize it in a self-contained bullet list format.
        Include starting timestamps, important details, and key takeaways.

        Example Input:
        {EXAMPLE_INPUT}

        Example Output:
        {EXAMPLE_OUTPUT}

        Here is the real transcript. Please summarize it:
        {transcript}
        """

    async def generate_summary_stream(self, job: SummaryJob, transcript: str):
        """
        Generates a summary asynchronously and streams the results.
        Yields chunks of text or thinking metadata.
        """
        model = genai.GenerativeModel(job.model_id)

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = self._get_prompt(transcript)

        try:
            # Use the asynchronous streaming method
            response_stream = await model.generate_content_async(
                prompt,
                stream=True,
                safety_settings=safety_settings
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield {"type": "content", "text": chunk.text}

            # After the stream, we can get the full response for metadata
            full_response = await response_stream.response
            usage_metadata = full_response.usage_metadata

            yield {
                "type": "complete",
                "prompt_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count,
                # Note: 'thinking_tokens' might not be available on all models/versions
                "thinking_tokens": getattr(usage_metadata, 'thinking_token_count', 0)
            }

        except Exception as e:
            print(f"GENAI ERROR: {e}")
            yield {"type": "error", "message": str(e)}

```
---

### 4. `rocket_recap/infra/transcript_provider.py`

This module isolates the logic for fetching and parsing video transcripts. It refactors the functionality from the original `s01_...` and `s02_...` scripts into a reusable class.

```python
# rocket_recap/infra/transcript_provider.py

import asyncio
import glob
import os
import re
import tempfile
import webvtt
from typing import Optional

class TranscriptProvider:
    """Provides methods to fetch and parse video transcripts."""

    def _validate_youtube_url(self, url: str) -> Optional[str]:
        """
        Validates various YouTube URL formats and returns the video ID.
        Returns None if the URL is not a valid YouTube link.
        """
        patterns = [
            r"https://(www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
            r"https://(www\.)?youtube\.com/live/([A-Za-z0-9_-]{11})",
            r"https://(www\.)?youtu\.be/([A-Za-z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                return match.group(2)
        return None

    def _parse_vtt(self, vtt_content: str) -> str:
        """
        Parses VTT content from a string, deduplicates, and formats it.
        """
        captions = webvtt.read_buffer(vtt_content.splitlines())
        
        output_lines = []
        last_text = None

        for caption in captions:
            # Simple deduplication
            if caption.text != last_text:
                timestamp = caption.start.split('.')[0] # Remove milliseconds
                text = caption.text.replace('\n', ' ').strip()
                output_lines.append(f"{timestamp} {text}")
                last_text = caption.text
        
        return "\n".join(output_lines)

    async def get_from_youtube(self, url: str) -> str:
        """
        Downloads a YouTube transcript using yt-dlp as a subprocess.
        Returns the parsed transcript as a string.
        Raises ValueError for invalid URLs and RuntimeError for download failures.
        """
        video_id = self._validate_youtube_url(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL provided.")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, f"{video_id}.%(ext)s")
            
            # Command to download English auto-captions
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-subs",
                "--sub-lang", "en",
                "-o", output_template,
                "--", video_id,
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_message = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"yt-dlp failed: {error_message}")

            # Find the downloaded .vtt file
            vtt_files = glob.glob(os.path.join(temp_dir, "*.vtt"))
            if not vtt_files:
                raise RuntimeError("Failed to download VTT subtitles for the video.")

            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                vtt_content = f.read()

            return self._parse_vtt(vtt_content)

```