#!/usr/bin/env python3
# Alternative 1: running with uv: GEMINI_API_KEY=`cat api_key.txt` uv run uvicorn p04_host:app --port 5001
# Alternative 2: install dependencies with pip: pip install -U google-generativeai python-fasthtml markdown
# Alternative 3: install dependencies with micromamba: micromamba install python-fasthtml markdown yt-dlp uvicorn numpy; pip install  webvtt-py
import google.generativeai as genai
import google.api_core.exceptions
import markdown
import sqlite_minutils.db
import datetime
import subprocess
import time
import glob
import numpy as np
import os
import logging
from google.generativeai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fasthtml.common import *
from s01_validate_youtube_url import *
from s02_parse_vtt_file import *
from s03_convert_markdown_to_youtube_format import *


# Configure logging with UTC timestamps and file output
class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        return dt.isoformat()


# Create formatter with UTC timestamps
formatter = UTCFormatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Clear any existing handlers
logger.handlers.clear()
# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# File handler
file_handler = logging.FileHandler("transcript_summarizer.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Get logger for this module
logger = logging.getLogger(__name__)
# Read the demonstration transcript and corresponding summary from disk
try:
    with open("example_input.txt") as f:
        g_example_input = f.read()
    with open("example_output.txt") as f:
        g_example_output = f.read()
    with open("example_output_abstract.txt") as f:
        g_example_output_abstract = f.read()
except FileNotFoundError as e:
    logger.error(f"Required example file not found: {e}")
    raise

# Use environment variable for API key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

MODEL_OPTIONS = [
    "gemini-2.5-flash| input-price: 0.3 output-price: 2.5 max-context-length: 128_000",
    "gemini-2.5-flash-lite| input-price: 0.1 output-price: 0.4 max-context-length: 128_000",
    "gemini-2.5-pro| input-price: 1.25 output-price: 10 max-context-length: 200_000",
]


def validate_transcript_length(transcript: str, max_words: int = 280_000) -> bool:
    """Validate transcript length to prevent processing overly large inputs."""
    if (not (transcript)) or (not (transcript.strip())):
        raise (ValueError("Transcript cannot be empty"))
    words = transcript.split()
    if (max_words) < (len(words)):
        raise (
            ValueError(f"Transcript too long: {len(words)} words (max: {max_words})")
        )
    return True


def validate_youtube_id(youtube_id: str) -> bool:
    if (not (youtube_id)) or ((len(youtube_id)) != (11)):
        return False
    # YouTube IDs are alphanumeric with _ and -
    return all(((c.isalnum()) or (c in "_-")) for c in youtube_id)


def render(summary: Summary):
    identifier = summary.identifier
    sid = f"gen-{identifier}"
    if summary.timestamps_done:
        return generation_preview(identifier)
    elif summary.summary_done:
        return Div(
            NotStr(markdown.markdown(summary.summary)),
            id=sid,
            hx_post=f"/generations/{identifier}",
            hx_trigger=("") if (summary.timestamps_done) else ("every 1s"),
            hx_swap="outerHTML",
        )
    else:
        return Div(
            NotStr(markdown.markdown(summary.summary)),
            id=sid,
            hx_post=f"/generations/{identifier}",
            hx_trigger="every 1s",
            hx_swap="outerHTML",
        )


# open website
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt, summaries, Summary = fast_app(
    db_file="data/summaries.db",
    live=False,
    render=render,
    identifier=int,
    model=str,
    transcript=str,
    host=str,
    original_source_link=str,
    include_comments=bool,
    include_timestamps=bool,
    include_glossary=bool,
    output_language=str,
    summary=str,
    summary_done=bool,
    summary_input_tokens=int,
    summary_output_tokens=int,
    summary_timestamp_start=str,
    summary_timestamp_end=str,
    timestamps=str,
    timestamps_done=bool,
    timestamps_input_tokens=int,
    timestamps_output_tokens=int,
    timestamps_timestamp_start=str,
    timestamps_timestamp_end=str,
    timestamped_summary_in_youtube_format=str,
    cost=float,
    embedding=bytes,
    full_embedding=bytes,
    pk="identifier",
)
documentation = (
    """**Get Your YouTube Summary:**

1.  **Copy** the video link.
2.  **Paste** it into the input field.
3.  **Click** 'Summarize' to get your summary with timestamps.

**Important Note on Subtitles:**

*   Automatic summary generation requires **English subtitles** on the video.
*   **If the video has no English subtitles, the automatic download of the transcript using the link will fail.**
*   **Manual Alternative:** You can still get a summary!
    1.  Find the transcript on YouTube (usually below the video description when viewed on a desktop browser).
    2.  **Copy** the entire transcript text manually. (Need help finding/copying? Watch the 'Demo Video' linked at the top right of this page).
    3.  **(Optional)** Add any additional instructions *after* the transcript (e.g., 'Translate the summary to German.', 'Add a glossary of medical terms and jargon to the summary.').

**For videos longer than 50 minutes:**

*   Select a **Pro model** for automatic summarization. Note that Google seems to not allow free use of Pro model anymore.
"""
) + (
    """*   If the Pro limit is reached (or if you prefer using your own tool), use the **Copy Prompt** button, paste the prompt into your AI tool, and run it there.
"""
)


def get_transcript(url, identifier):
    # Call yt-dlp to download the subtitles. Modifies the timestamp to have second granularity. Returns a single string
    try:
        youtube_id = validate_youtube_url(url)
        if not (youtube_id):
            logger.warning(f"Invalid YouTube URL: {url}")
            return "URL couldn't be validated"
        if not (validate_youtube_id(youtube_id)):
            logger.warning(f"Invalid YouTube ID format: {youtube_id}")
            return "Invalid YouTube ID format"
        sub_file = f"/dev/shm/o_{identifier}"
        sub_file_en = f"/dev/shm/o_{identifier}.en.vtt"
        # First, try to get English subtitles
        cmds_en = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--cookies-from-browser",
            "firefox",
            "--sub-lang",
            "en",
            "-o",
            sub_file,
            "--",
            youtube_id,
        ]
        logger.info(f"Downloading English subtitles: {' '.join(cmds_en)}")
        result = subprocess.run(cmds_en, capture_output=True, text=True, timeout=60)
        if (result.returncode) != (0):
            logger.warning(
                f"yt-dlp failed with return code {result.returncode}: {result.stderr}"
            )
        sub_file_to_parse = None
        if os.path.exists(sub_file_en):
            sub_file_to_parse = sub_file_en
        else:
            # If English subtitles are not found, try to download any available subtitle
            logger.info(
                "English subtitles not found. Trying to download original language subtitles."
            )
            cmds_any = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-subs",
                "--write-subs",
                "--cookies-from-browser",
                "firefox",
                "-o",
                sub_file,
                "--",
                youtube_id,
            ]
            logger.info(f"Downloading any subtitles: {' '.join(cmds_any)}")
            result = subprocess.run(
                cmds_any, capture_output=True, text=True, timeout=60
            )
            # Find the downloaded subtitle file
            subtitle_files = glob.glob(f"{sub_file}.*.vtt")
            if subtitle_files:
                sub_file_to_parse = subtitle_files[0]
                logger.info(
                    f"Parse transcript from {sub_file_to_parse} out of the subtitle files: {subtitle_files}"
                )
        ostr = "Problem getting subscript."
        if (sub_file_to_parse) and (os.path.exists(sub_file_to_parse)):
            try:
                ostr = parse_vtt_file(sub_file_to_parse)
                logger.info(f"Successfully parsed subtitle file: {sub_file_to_parse}")
                os.remove(sub_file_to_parse)
            except FileNotFoundError:
                logger.error(f"Subtitle file not found: {sub_file_to_parse}")
                ostr = "Error: Subtitle file disappeared"
            except PermissionError:
                logger.error(f"Permission denied removing file: {sub_file_to_parse}")
                ostr = "Error: Permission denied cleaning up subtitle file"
            except Exception as e:
                logger.error(f"Error processing subtitle file: {e}")
                ostr = f"Error: problem when processing subtitle file {e}"
        else:
            logger.error("No subtitle file found")
            ostr = "Error: No subtitles found for this video. Please provide the transcript manually."
        # Cleanup any other subtitle files that might have been downloaded
        other_subs = glob.glob(f"{sub_file}.*.vtt")
        for sub in other_subs:
            try:
                os.remove(sub)
            except OSError as e:
                logger.warning(f"Error removing file {sub}: {e}")
        return ostr
    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp timeout for identifier {identifier}")
        return "Error: Download timeout"
    except Exception as e:
        logger.error(f"Unexpected error in get_transcript: {e}")
        return f"Error712: {str(e)}"


documentation_html = markdown.markdown(documentation)


@rt("/")
def get(request: Request):
    logger.info(f"Request from host: {request.client.host}")
    nav = Nav(
        Ul(Li(Strong("Transcript Summarizer"))),
        Ul(
            Li(A("Map", href="https://rocketrecap.com/exports/index.html")),
            Li(A("Demo Video", href="https://www.youtube.com/watch?v=ttuDW1YrkpU")),
            Li(
                A(
                    "Documentation",
                    href="https://github.com/plops/gemini-competition/blob/main/README.md",
                )
            ),
        ),
    )
    transcript = Textarea(
        placeholder="(Optional) Paste YouTube transcript here",
        style="height: 300px; width=60%;",
        name="transcript",
    )
    selector = [Option(opt, value=opt) for opt in MODEL_OPTIONS]
    model = Div(
        Select(*selector, style="width: 100%;", name="model"),
        style="display: flex; align-items: center; width: 100%;",
    )
    form = Form(
        Group(
            Div(
                Textarea(
                    placeholder="Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)",
                    name="original_source_link",
                ),
                transcript,
                model,
                Div(
                    Label("Output Language", _for="output_language"),
                    Select(
                        Option("en"),
                        style="width: 100%;",
                        name="output_language",
                        id="output_language",
                    ),
                    style="display: none; align-items: center; width: 100%;",
                ),
                Div(
                    Input(
                        type="checkbox",
                        id="include_comments",
                        name="include_comments",
                        checked=False,
                    ),
                    Label("Include User Comments", _for="include_comments"),
                    style="display: none; align-items: center; width: 100%;",
                ),
                Div(
                    Input(
                        type="checkbox",
                        id="include_timestamps",
                        name="include_timestamps",
                        checked=True,
                    ),
                    Label("Include Timestamps", _for="include_timestamps"),
                    style="display: none; align-items: center; width: 100%;",
                ),
                Div(
                    Input(
                        type="checkbox",
                        id="include_glossary",
                        name="include_glossary",
                        checked=False,
                    ),
                    Label("Include Glossary", _for="include_glossary"),
                    style="display: none; align-items: center; width: 100%;",
                ),
                Button("Summarize Transcript"),
                style="display: flex; flex-direction:column;",
            )
        ),
        hx_post="/process_transcript",
        hx_swap="afterbegin",
        target_id="gen-list",
    )
    gen_list = Div(id="gen-list")
    summaries_to_show = summaries(order_by="identifier DESC", limit=3)
    summary_list = Ul(*summaries_to_show, id="summaries")
    return Title("Video Transcript Summarizer"), Main(
        nav,
        NotStr(documentation_html),
        form,
        gen_list,
        summary_list,
        Script("""function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}"""),
        cls="container",
    )


# A pending preview keeps polling this route until we return the summary
def generation_preview(identifier):
    sid = f"gen-{identifier}"
    text = "Generating ..."
    trigger = "every 1s"
    price_input = {
        ("gemini-2.5-flash"): (0.30),
        ("gemini-2.5-flash-lite"): (0.10),
        ("gemini-2.5-pro"): (1.250),
    }
    price_output = {
        ("gemini-2.5-flash"): (2.50),
        ("gemini-2.5-flash-lite"): (0.40),
        ("gemini-2.5-pro"): (10),
    }
    try:
        s = summaries[identifier]
        if s.timestamps_done:
            # this is for <= 128k tokens
            real_model = s.model.split("|")[0]
            price_input_token_usd_per_mio = -1
            price_output_token_usd_per_mio = -1
            try:
                price_input_token_usd_per_mio = price_input[real_model]
                price_output_token_usd_per_mio = price_output[real_model]
            except Exception as e:
                pass
            input_tokens = (s.summary_input_tokens) + (s.timestamps_input_tokens)
            output_tokens = (s.summary_output_tokens) + (s.timestamps_output_tokens)
            cost = (
                ((input_tokens) / (1_000_000)) * (price_input_token_usd_per_mio)
            ) + (((output_tokens) / (1_000_000)) * (price_output_token_usd_per_mio))
            summaries.update(pk_values=identifier, cost=cost)
            if (cost) < (2.00e-2):
                cost_str = f"${cost:.4f}"
            else:
                cost_str = f"${cost:.2f}"
            text = f"""{s.timestamped_summary_in_youtube_format}

I used {s.model} on rocketrecap dot com to summarize the transcript.
Cost (if I didn't use the free tier): {cost_str}
Input tokens: {input_tokens}
Output tokens: {output_tokens}"""
            trigger = ""
        elif s.summary_done:
            text = s.summary
        elif (s.summary is not None) and ((0) < (len(s.summary))):
            text = s.summary
        elif len(s.transcript):
            text = f"Generating from transcript: {s.transcript[0 : min(100, len(s.transcript))]}"
        summary_details = Div(
            P(B("identifier:"), Span(f"{s.identifier}")),
            P(B("model:"), Span(f"{s.model}")),
            P(B("host:"), Span(f"{s.host}")),
            A(
                f"{s.original_source_link}",
                target="_blank",
                href=f"{s.original_source_link}",
                id="source-link",
            ),
            P(B("include_comments:"), Span(f"{s.include_comments}")),
            P(B("include_timestamps:"), Span(f"{s.include_timestamps}")),
            P(B("include_glossary:"), Span(f"{s.include_glossary}")),
            P(B("output_language:"), Span(f"{s.output_language}")),
            P(B("cost:"), Span(f"{s.cost}")),
            cls="summary-details",
        )
        summary_container = Div(summary_details, cls="summary-container")
        title = summary_container
        html = markdown.markdown(s.summary)
        pre = Div(
            Div(
                Pre(text, id=f"pre-{identifier}"),
                id="hidden-markdown",
                style="display: none;",
            ),
            Div(NotStr(html)),
        )
        button = Button("Copy Summary", onclick=f"copyPreContent('pre-{identifier}')")
        prompt_text = get_prompt(s)
        prompt_pre = Pre(
            prompt_text, id=f"prompt-pre-{identifier}", style="display: none;"
        )
        prompt_button = Button(
            "Copy Prompt", onclick=f"copyPreContent('prompt-pre-{identifier}')"
        )
        if (trigger) == (""):
            return Div(title, pre, prompt_pre, button, prompt_button, id=sid)
        else:
            return Div(
                title,
                pre,
                prompt_pre,
                button,
                prompt_button,
                id=sid,
                hx_post=f"/generations/{identifier}",
                hx_trigger=trigger,
                hx_swap="outerHTML",
            )
    except Exception as e:
        return Div(
            f"line 1897 id: {identifier} e: {e}",
            Pre(text),
            id=sid,
            hx_post=f"/generations/{identifier}",
            hx_trigger=trigger,
            hx_swap="outerHTML",
        )


@app.post("/generations/{identifier}")
def get(identifier: int):
    return generation_preview(identifier)


@rt("/process_transcript")
def post(summary: Summary, request: Request):
    summary.host = request.client.host
    summary.summary_timestamp_start = datetime.datetime.now().isoformat()
    summary.summary = ""
    if (0) == (len(summary.transcript)):
        summary.summary = "Downloading transcript..."
    s2 = summaries.insert(summary)
    download_and_generate(s2.identifier)
    return generation_preview(s2.identifier)


@threaded
def download_and_generate(identifier: int):
    try:
        s = wait_until_row_exists(identifier)
        if (s) == (-1):
            logger.error(f"Row {identifier} never appeared in database")
            return
        if (s.transcript is None) or ((0) == (len(s.transcript))):
            # No transcript given, try to download from URL
            transcript = get_transcript(s.original_source_link, identifier)
            summaries.update(pk_values=identifier, transcript=transcript)
        # re-fetch summary with transcript
        s = summaries[identifier]
        # Validate transcript length
        try:
            validate_transcript_length(s.transcript)
        except ValueError as e:
            logger.error(f"Transcript validation failed for {identifier}: {e}")
            summaries.update(
                pk_values=identifier, summary=f"Error1031: {str(e)}", summary_done=True
            )
            return
        words = s.transcript.split()
        if (len(words)) < (30):
            summaries.update(
                pk_values=identifier,
                summary="Error: Transcript is too short. No summary necessary",
                summary_done=True,
            )
            return
        if (280_000) < (len(words)):
            if "-pro" in s.model:
                summaries.update(
                    pk_values=identifier,
                    summary="Error: Transcript exceeds 280,000 words. Please shorten it or don't use the pro model.",
                    summary_done=True,
                )
                return
        logger.info(f"Processing link: {s.original_source_link}")
        summaries.update(pk_values=identifier, summary="")
        generate_and_save(identifier)
    except Exception as e:
        logger.error(f"Error in download_and_generate for {identifier}: {e}")
        try:
            summaries.update(
                pk_values=identifier, summary=f"Error1055: {str(e)}", summary_done=True
            )
        except Exception as update_error:
            logger.error(
                f"Failed to update database with error for {identifier}: {update_error}"
            )


def wait_until_row_exists(identifier):
    for i in range(400):
        try:
            s = summaries[identifier]
            return s
        except sqlite_minutils.db.NotFoundError:
            logger.debug(f"Entry {identifier} not found, attempt {i + 1}")
        except Exception as e:
            logger.error(f"Unknown exception waiting for row {identifier}: {e}")
        time.sleep((0.10))
    logger.error(f"Row {identifier} did not appear after 400 attempts")
    return -1


def get_prompt(summary: Summary) -> str:
    r"""Generate prompt from a given Summary object. It will use the contained transcript."""
    prompt = f"""Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding abstract and summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please give an abstract of the transcript and then summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
{g_example_input}
Example Output:
{g_example_output_abstract}
{g_example_output}
Here is the real transcript. Please summarize it: 
{(summary.transcript)}"""
    return prompt


def generate_and_save(identifier: int):
    """
    Generates a summary for the given identifier, stores it in the database, and computes embeddings for both
    the transcript and the summary. Handles errors and updates the database accordingly.

    Args:
        identifier (int): The unique identifier for the summary entry in the database."""
    logger.info(f"generate_and_save id={identifier}")
    try:
        s = wait_until_row_exists(identifier)
        if (s) == (-1):
            logger.error(f"Could not find summary with id {identifier}")
            return
        logger.info(f"generate_and_save model={s.model}")
        m = genai.GenerativeModel(s.model.split("|")[0])
        safety = {
            (HarmCategory.HARM_CATEGORY_HATE_SPEECH): (HarmBlockThreshold.BLOCK_NONE),
            (HarmCategory.HARM_CATEGORY_HARASSMENT): (HarmBlockThreshold.BLOCK_NONE),
            (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT): (
                HarmBlockThreshold.BLOCK_NONE
            ),
            (HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT): (
                HarmBlockThreshold.BLOCK_NONE
            ),
        }
        prompt = get_prompt(s)
        response = m.generate_content(prompt, safety_settings=safety, stream=True)
        for chunk in response:
            try:
                logger.debug(f"Adding text chunk to id={identifier}")
                summaries.update(
                    pk_values=identifier,
                    summary=((summaries[identifier].summary) + (chunk.text)),
                )
            except ValueError as e:
                logger.warning(f"ValueError processing chunk for {identifier}: {e}")
                summaries.update(
                    pk_values=identifier,
                    summary=(
                        (summaries[identifier].summary)
                        + (f"\nError: value error {str(e)}")
                    ),
                )
            except Exception as e:
                logger.error(f"Error processing chunk for {identifier}: {e}")
                summaries.update(
                    pk_values=identifier,
                    summary=(
                        (summaries[identifier].summary) + (f"\nError1189: {str(e)}")
                    ),
                )
        prompt_token_count = response.usage_metadata.prompt_token_count
        candidates_token_count = response.usage_metadata.candidates_token_count
        try:
            logger.info(f"Usage metadata: {response.usage_metadata}")
            thinking_token_count = response.usage_metadata.thinking_token_count
        except AttributeError:
            logger.info("No thinking token count available")
            thinking_token_count = 0
        summaries.update(
            pk_values=identifier,
            summary_done=True,
            summary_input_tokens=prompt_token_count,
            summary_output_tokens=((candidates_token_count) + (thinking_token_count)),
            summary_timestamp_end=datetime.datetime.now().isoformat(),
            timestamps="",
            timestamps_timestamp_start=datetime.datetime.now().isoformat(),
        )
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Resource exhausted for {identifier}: {e}")
        summaries.update(
            pk_values=identifier,
            summary_done=False,
            summary=(
                (summaries[identifier].summary) + ("\nError1234: resource exhausted")
            ),
            summary_timestamp_end=datetime.datetime.now().isoformat(),
            timestamps="",
            timestamps_timestamp_start=datetime.datetime.now().isoformat(),
        )
        return
    except Exception as e:
        logger.error(f"Unexpected error in generate_and_save for {identifier}: {e}")
        try:
            summaries.update(
                pk_values=identifier,
                summary_done=False,
                summary=((summaries[identifier].summary) + (f"Error1254: {str(e)}")),
                summary_timestamp_end=datetime.datetime.now().isoformat(),
            )
        except Exception as update_error:
            logger.error(
                f"Failed to update database with error for {identifier}: {update_error}"
            )
        return
    try:
        # Generate and store the embedding of the transcript
        transcript_text = summaries[identifier].transcript
        if transcript_text:
            logger.info(f"Generating embedded transcript: {identifier}...")
            embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=transcript_text,
                task_type="clustering",
            )
            vector_blob = np.array(
                embedding_result["embedding"], dtype=np.float32
            ).tobytes()
            summaries.update(pk_values=identifier, full_embedding=vector_blob)
            logger.info(f"Embedding stored for identifier {identifier}.")
    except google.api_core.exceptions.ResourceExhausted:
        logger.warning("Resource exhausted when embedding full transcript")
    except Exception as e:
        logger.error(f"Error during full embedding for identifier {identifier}: {e}")
    try:
        text = summaries[identifier].summary
        text = convert_markdown_to_youtube_format(text)
        summaries.update(
            pk_values=identifier,
            timestamps_done=True,
            timestamped_summary_in_youtube_format=text,
            timestamps_input_tokens=0,
            timestamps_output_tokens=0,
            timestamps_timestamp_end=datetime.datetime.now().isoformat(),
        )
    except google.api_core.exceptions.ResourceExhausted:
        logger.warning("Resource exhausted during summary update")
        summaries.update(
            pk_values=identifier,
            timestamps_done=False,
            timestamped_summary_in_youtube_format=f"resource exhausted",
            timestamps_timestamp_end=datetime.datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error during summary update for identifier {identifier}: {e}")
    try:
        # Generate and store the embedding of the summary
        summary_text = summaries[identifier].summary
        if summary_text:
            logger.info(f"Generating summary embedding for identifier {identifier}...")
            embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=summary_text,
                task_type="clustering",
            )
            vector_blob = np.array(
                embedding_result["embedding"], dtype=np.float32
            ).tobytes()
            summaries.update(pk_values=identifier, embedding=vector_blob)
            logger.info(f"Embedding stored for identifier {identifier}.")
    except google.api_core.exceptions.ResourceExhausted:
        logger.warning("Resource exhausted during embedding of summary")
    except Exception as e:
        logger.error(f"Error during embedding for identifier {identifier}: {e}")


# in production run this script with: GEMINI_API_KEY=`cat api_key.txt` uvicorn p04_host:app --port 5001
