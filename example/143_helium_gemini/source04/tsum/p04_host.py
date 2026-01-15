#!/usr/bin/env python3
# Alternative 1: running with uv: GEMINI_API_KEY=`cat api_key.txt` uv run uvicorn p04_host:app --port 5001
# Alternative 2: install dependencies with pip: pip install -U google-generativeai python-fasthtml markdown; apt-get install nodejs; wget https://github.com/denoland/deno/releases/download/v2.6.2/deno-x86_64-unknown-linux-gnu.zip; unzip deno-x86*.zip; sudo mv deno /usr/bin
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
import re
from zoneinfo import ZoneInfo
from google.generativeai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fasthtml.common import *
from s01_validate_youtube_url import *
from s02_parse_vtt_file import *
from s03_convert_markdown_to_youtube_format import *
from s04_convert_html_timestamps_to_youtube_links import *


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
logger.info("Logger initialized")
logger.info("Read the demonstration transcript and corresponding summary from disk")
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

logger.info("Use environment variable for API key")
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

MODEL_OPTIONS = [
    "gemini-2.5-flash-lite-preview-09-2025| input-price: 0.1 output-price: 0.4 max-context-length: 128_000",
    "gemini-2.5-flash-preview-09-2025| input-price: 0.3 output-price: 2.5 max-context-length: 128_000",
    "gemini-3-flash-preview| input-price: 0.5 output-price: 3 max-context-length: 128_000",
]

# Counters for tracking daily usage
model_counts = {opt.split("|")[0]: 0 for opt in MODEL_OPTIONS}
last_reset_day = None


def check_reset_counters():
    global last_reset_day
    try:
        la_tz = ZoneInfo("America/Los_Angeles")
        now = datetime.datetime.now(la_tz)
        today = now.date()
        if last_reset_day != today:
            logger.info(f"Resetting quota counters. New day: {today}")
            for k in model_counts:
                model_counts[k] = 0
            last_reset_day = today
    except Exception as e:
        logger.warning(f"Timezone reset check failed: {e}")


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
            data_hx_post=f"/generations/{identifier}",
            data_hx_trigger=("") if (summary.timestamps_done) else ("every 1s"),
            data_hx_swap="outerHTML",
        )
    else:
        return Div(
            NotStr(markdown.markdown(summary.summary)),
            id=sid,
            data_hx_post=f"/generations/{identifier}",
            data_hx_trigger="every 1s",
            data_hx_swap="outerHTML",
        )


logger.info("Create website app")
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt, summaries, Summary = fast_app(
    db_file="data/summaries.db",
    live=False,
    render=render,
    htmlkw=dict(lang="en-US"),
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
    embedding_model=str,
    full_embedding=bytes,
    pk="identifier",
)
documentation = (
    """**Get Your Summary**

1.  **For YouTube videos:** Paste the link into the input field for automatic transcript download.
2.  **For other text:** Paste articles, meeting notes, or manually copied transcripts directly into the text area below.
3.  **Click 'Summarize':** The tool will process your request using the selected model.

### Browser Extension Available
To make this process faster, you can use the **new browser addon** for Chrome and Firefox. This extension simplifies the workflow and also enables usage on **iPhone**.

### Available Models
You can choose between three models with different capabilities. While these models have commercial costs, we utilize **Google's Free Tier**, so you are not charged on this website.
*   **Gemini 3 Flash** (~$0.50/1M tokens): Highest capability, great for long or complex videos.
*   **Gemini 2.5 Flash** (~$0.30/1M tokens): Balanced performance.
*   **Gemini 2.5 Flash-Lite** (~$0.10/1M tokens): Fastest and lightweight.
*(Note: The free tier allows approximately 20 requests per day for each model. This is for the entire website, so don't tell anyone it exists ;-) )*

### Important Notes & Troubleshooting

**YouTube Captions & Languages**
*   **Automatic Download:** The software now automatically downloads captions corresponding to the **original audio language** of the video.
*   **Missing/Wrong Captions:** Some videos may have incorrect language settings or no captions at all. If the automatic download fails:
    1.  Open the video on YouTube (this usually requires a **desktop browser**).
    2.  Open the transcript tab on YouTube.
    3.  Copy the entire transcript.
    4.  Paste it manually into the text area below.

**Tips for Pasting Text**
*   **Timestamps:** The summarizer is optimized for content that includes timestamps (e.g., `00:15:23 Key point is made`).
*   **Best Results:** While the tool works with any block of text (articles/notes), providing timestamped transcripts generally produces the most detailed and well-structured summaries.
"""
) + (
    """* If the daily request limit is reached, use the **Copy Prompt** button, paste the prompt into your AI tool, and run it there.
"""
)
PREFERRED_BASE = [
    "en",
    "de",
    "fr",
    "pl",
    "ar",
    "bn",
    "bg",
    "zh-Hans",
    "zh-Hant",
    "hr",
    "cs",
    "da",
    "nl",
    "et",
    "fi",
    "el",
    "iw",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lv",
    "lt",
    "no",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "th",
    "tr",
    "uk",
    "vi",
]


def pick_best_language(list_output: str) -> str | None:
    # Collect available language codes from yt-dlp --list-subs output
    langs = set()
    for line in list_output.splitlines():
        m = re.match(r"""^\s*([A-Za-z0-9\-]+)\s+""", line)
        if m:
            langs.add(m.group(1))
    if not (langs):
        return None

    def base(code: str) -> str:
        # Group langs by base code (strip trailing -orig if present)
        return (code[:-5]) if (code.endswith("-orig")) else (code)

    orig_langs = [l for l in langs if (l.endswith("-orig"))]
    # 1) Prefer any -orig. If multiple, choose by PREFERRED_BASE order using base code
    if orig_langs:
        # Map base -> full code with orig
        base_to_orig = {base(l): l for l in orig_langs}
        for pref in PREFERRED_BASE:
            if pref in base_to_orig:
                return base_to_orig[pref]
        # If none of the bases are in the list, pick the first deterministically
        return sorted(orig_langs)[0]
    # 2) No -orig, choose by PREFERRED_BASE
    available_bases = {l for l in langs}
    for pref in PREFERRED_BASE:
        if pref in available_bases:
            return pref
    # 3) Fallbacks
    for l in sorted(langs):
        if l.startswith("en"):
            return l
    return sorted(langs)[0]


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
        list_cmd = [
            "uvx",
            "yt-dlp",
            "--list-subs",
            "--cookies-from-browser",
            "firefox",
            "--js-runtimes",
            "deno",
            "--remote-components",
            "ejs:npm",
            "--",
            youtube_id,
        ]
        logger.info(f"Listing subtitles: {' '.join(list_cmd)}")
        list_res = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
        if (0) != (list_res.returncode):
            logger.warning(f"yt-dlp --list-subs failed: {list_res.stderr}")
            return "Error: Could not list subtitles"
        chosen_lang = pick_best_language(list_res.stdout)
        if not (chosen_lang):
            logger.error("No subtitles listed by yt-dlp")
            return "Error: No subtitles found for this video. Please provide the transcript manually."
        sub_file_prefix = f"/dev/shm/o_{identifier}"
        dl_cmd = [
            "uvx",
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--cookies-from-browser",
            "firefox",
            "--js-runtimes",
            "deno",
            "--remote-components",
            "ejs:npm",
            "--sub-langs",
            chosen_lang,
            "-o",
            sub_file_prefix,
            "--",
            youtube_id,
        ]
        logger.info(f"Downloading subtitles ({chosen_lang}): {' '.join(dl_cmd)}")
        dl_res = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60)
        if (dl_res.returncode) != (0):
            logger.warning(f"yt-dlp download failed: {dl_res.stderr}")
        vtt_files = glob.glob(f"{sub_file_prefix}.*.vtt")
        if not (vtt_files):
            logger.error("No subtitle file downloaded")
            return "Error: No subtitles found for this video. Please provide the transcript manually."
        sub_file_to_parse = vtt_files[0]
        try:
            ostr = parse_vtt_file(sub_file_to_parse)
            logger.info(f"Successfully parsed subtitle file: {sub_file_to_parse}")
        except FileNotFoundError:
            logger.error(f"Subtitle file not found: {sub_file_to_parse}")
            ostr = "Error: Subtitle file disappeared or was not present"
        except PermissionError:
            logger.error(f"Permission denied removing file: {sub_file_to_parse}")
            ostr = "Error: Permission denied cleaning up subtitle file"
        except Exception as e:
            logger.error(f"Error processing subtitle file: {e}")
            ostr = f"Error: problem when processing subtitle file {e}"
        for sub in glob.glob(f"{sub_file_prefix}.*.vtt"):
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
    check_reset_counters()
    nav = Nav(
        Ul(Li(H1("RocketRecap Content Summarizer"))),
        Ul(
            Li(A("Map", href="https://rocketrecap.com/exports/index.html")),
            Li(A("FAQ", href="https://rocketrecap.com/exports/faq.html")),
            Li(A("Extension", href="https://rocketrecap.com/exports/extension.html")),
            Li(
                A("Privacy Policy", href="https://rocketrecap.com/exports/privacy.html")
            ),
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
        style="height: 300px; width: 60%;",
        name="transcript",
        id="transcript-paste",
    )
    selector = []
    for opt in MODEL_OPTIONS:
        model_name = opt.split("|")[0]
        used = model_counts.get(model_name, 0)
        remaining = max(0, 20 - used)
        label = f"{model_name} | {remaining} requests left"
        selector.append(Option(opt, value=opt, label=label))

    model = Div(
        Label("Select Model", _for="model-select", cls="visually-hidden"),
        Select(*selector, id="model-select", style="width: 100%;", name="model"),
        style="display: flex; align-items: center; width: 100%;",
    )
    form = Form(
        Fieldset(
            Legend("Submit Text for Summarization"),
            Div(
                Label(
                    "Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)",
                    _for="youtube-link",
                ),
                Textarea(
                    placeholder="Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)",
                    id="youtube-link",
                    name="original_source_link",
                ),
                Label(
                    "(Optional) Paste YouTube transcript here", _for="transcript-paste"
                ),
                transcript,
                model,
                Button("Summarize Transcript"),
                style="display: flex; flex-direction:column;",
            ),
        ),
        data_hx_post="/process_transcript",
        data_hx_swap="afterbegin",
        data_hx_target="#summary-list",
    )
    summaries_to_show = summaries(order_by="identifier DESC", limit=3)
    summary_list_container = Div(*summaries_to_show, id="summary-list")
    return (
        Title("Video Transcript Summarizer"),
        Meta(
            name="description",
            content="Get AI-powered summaries of YouTube videos and websites. Paste a link or transcript to receive a concise summary with timestamps.",
        ),
        Main(
            nav,
            NotStr(documentation_html),
            form,
            summary_list_container,
            Script("""function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}"""),
            cls="container",
        ),
        Style(""".visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}"""),
    )


# A pending preview keeps polling this route until we return the summary
def generation_preview(identifier):
    sid = f"gen-{identifier}"
    text = "Generating ..."
    trigger = "every 1s"
    price_input = {
        ("gemini-2.5-flash-lite-preview-09-2025"): (0.10),
        ("gemini-2.5-flash-preview-09-2025"): (0.30),
        ("gemini-3-flash-preview"): (0.50),
    }
    price_output = {
        ("gemini-2.5-flash-lite-preview-09-2025"): (0.40),
        ("gemini-2.5-flash-preview-09-2025"): (2.50),
        ("gemini-3-flash-preview"): (3),
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
            text = f"""*AI Summary*

{s.timestamped_summary_in_youtube_format}

AI-generated summary created with {s.model.split("|")[0]} for free via RocketRecap-dot-com. (Input: {input_tokens:,} tokens, Output: {output_tokens:,} tokens, Est. cost: {cost_str})."""
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
            A(
                f"{s.original_source_link}",
                target="_blank",
                href=f"{s.original_source_link}",
                id=f"source-link-{identifier}",
            ),
            P(B("embedding_model:"), Span(f"{s.embedding_model}")),
            cls="summary-details",
        )
        summary_container = Div(summary_details, cls="summary-container")
        title = summary_container
        html0 = markdown.markdown(s.summary)
        if ("") == (html0):
            real_model = s.model.split("|")[0]
            html = f"Waiting for {real_model} to respond to request..."
        else:
            html = replace_timestamps_in_html(html0, s.original_source_link)
        prompt_id = f"pompt-pre-{identifier}"
        hidden_pre_for_prompt = Pre(get_prompt(s), id=prompt_id)
        prompt_button = Button("Copy Prompt", onclick=f"copyPreContent('{prompt_id}')")
        hidden_pre_for_copy = Div(
            Pre(text, id=f"pre-{identifier}"),
            hidden_pre_for_prompt,
            id=f"hidden-markdown-{identifier}",
            style="display: none;",
        )
        card_content = [
            Header(
                H4(
                    A(
                        f"{s.original_source_link}",
                        target="_blank",
                        href=f"{s.original_source_link}",
                    )
                ),
                P(
                    f"ID: {s.identifier} | Model: {s.model.split('|')[0]}",
                    style="font-size: 0.9em; color: var(--pico-secondary-foreground); margin-bottom: 0;",
                ),
            ),
            Div(NotStr(html), style="white-space: normal;"),
            Footer(
                hidden_pre_for_copy,
                Button(
                    "Copy Summary",
                    onclick=f"copyPreContent('pre-{identifier}')",
                    cls="outline",
                ),
                prompt_button,
            ),
        ]
        if (trigger) == (""):
            return Article(*card_content, id=sid)
        else:
            attrs = dict(
                id=sid,
                data_hx_post=f"/generations/{identifier}",
                data_hx_trigger=trigger,
                data_hx_swap="outerHTML",
            )
            if not ((s.summary_timestamp_end) or (s.summary_done)):
                attrs["aria-busy"] = "true"
                attrs["aria-live"] = "polite"
            return Article(*card_content, **attrs)
    except Exception as e:
        return Article(
            Header(H4(f"Error processing Summary ID: {identifier}")),
            Div(
                P(
                    "An error occurred while trying to render the summary. The page will continue to refresh automatically."
                ),
                P(B("Details:"), Code(f"{e}")),
                Pre(text),
            ),
            id=sid,
            data_hx_post=f"/generations/{identifier}",
            data_hx_trigger=trigger,
            data_hx_swap="outerHTML",
            style="border-color: var(--pico-del-color);",
        )


@app.post("/generations/{identifier}")
def get(identifier: int):
    return generation_preview(identifier)


@rt("/process_transcript")
def post(summary: Summary, request: Request):
    summary.host = request.client.host
    summary.summary_timestamp_start = datetime.datetime.now().isoformat()
    summary.summary = ""
    if summary.transcript is not None:
        if (0 == len(summary.transcript)):
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
                summary="Error: Transcript is too short. Probably I couldn't download it. You can provide it manually.",
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
    prompt = f"""Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding abstract and summary I expect. Afterward, I will provide a new transcript that I want a summarization in the same format. 

**Please give an abstract of the transcript and then summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
{g_example_input}
Example Output:
{g_example_output_abstract}
{g_example_output}
Here is the real transcript. What would be a good group of people to review this topic? Please summarize provide a summary like they would: 
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

        # Update usage counter
        check_reset_counters()
        real_model = s.model.split("|")[0]
        if real_model in model_counts:
            model_counts[real_model] += 1

        logger.info(f"generate_and_save model={s.model}")
        m = genai.GenerativeModel(
            s.model.split("|")[0],
            system_instruction=r"""### CORE INSTRUCTION
You are an advanced, adaptive knowledge synthesis engine. Your goal is to provide high-fidelity summaries of input material. You possess the capability to analyze text, determine the specific domain of expertise required to understand it, and fully adopt the persona of a senior expert in that field to perform the summary.

### PROCESS PROTOCOL
For every input provided, follow this strict three-step process:

1.  **Analyze and Adopt:** 
    - Scan the input material to determine its domain (e.g., Legal, Astrophysics, Culinary, Software Engineering).
    - Adopt the persona of a Top-Tier Senior Analyst or Expert in that specific domain.
    - Calibrate your vocabulary, tone, and focus to match that expert persona.

2.  **Summarize (Strict Objectivity):**
    - Generate a summary of the input material *as that expert*.
    - **Constraint:** Your summary must reflect *only* the information contained in the source text. Do not offer agreement, disagreement, or external opinions within the summary.
    - **Style:** Use American English. Be direct, efficient, and dense. Avoid """,
        )
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
                        (summaries[identifier].summary) + (f"\n[Error1189: {str(e)}]")
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
                (summaries[identifier].summary)
                + ("\nError1234: resource exhausted. Try again with a different model.")
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
            embedding_model = "gemini-embedding-001"
            embedding_result = genai.embed_content(
                model=f"models/{embedding_model}",
                content=summary_text,
                task_type="clustering",
                output_dimensionality=3072,
            )
            vector_array = np.array(embedding_result["embedding"], dtype=np.float32)
            vector_blob = vector_array.tobytes()
            summaries.update(
                pk_values=identifier,
                embedding=vector_blob,
                embedding_model=embedding_model,
            )
            logger.info(
                f"Embedding stored for identifier {identifier} shape={vector_array.shape}."
            )
    except google.api_core.exceptions.ResourceExhausted:
        logger.warning("Resource exhausted during embedding of summary")
    except Exception as e:
        logger.error(f"Error during embedding for identifier {identifier}: {e}")


# in production run this script with: GEMINI_API_KEY=`cat api_key.txt` uvicorn p04_host:app --port 5001 --log-config=uvicorn_log_conf.yaml
