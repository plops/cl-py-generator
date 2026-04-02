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
import numpy as np
import os
import logging
import hashlib
import re
import tempfile
from zoneinfo import ZoneInfo
from google.generativeai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fasthtml.common import *
from fastlite import database
from s01_validate_youtube_url import *
from s02_parse_vtt_file import *
from s03_convert_markdown_to_youtube_format import *
from s04_convert_html_timestamps_to_youtube_links import *
# Configure logging with UTC timestamps and file output
class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt = None):
        dt=datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        return dt.isoformat()
# Create formatter with UTC timestamps
formatter=UTCFormatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Configure root logger
logger=logging.getLogger()
logger.setLevel(logging.INFO)
# Clear any existing handlers
logger.handlers.clear()
# Console handler
console_handler=logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# File handler
file_handler=logging.FileHandler("transcript_summarizer.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Get logger for this module
logger=logging.getLogger(__name__)
logger.info("Logger initialized")
logger.info("Read the demonstration transcript and corresponding summary from disk")
try:
    with open("example_input.txt") as f:
        g_example_input=f.read()
    with open("example_output.txt") as f:
        g_example_output=f.read()
    with open("example_output_abstract.txt") as f:
        g_example_output_abstract=f.read()
except FileNotFoundError as e:
    logger.error(f"Required example file not found: {e}")
    raise
 
logger.info("Use environment variable for API key")
api_key=os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
 
MODEL_OPTIONS=["gemini-3-flash-preview| input: $0.5 | output: $3.0 | context: 1_000_000 | rpm: 5 | rpd: 20", "gemini-3.1-flash-lite-preview| input: $0.25 | output: $1.5 | context: 1_000_000 | rpm: 15 | rpd: 500", "gemini-2.5-flash| input: $0.3 | output: $2.5 | context: 1_000_000 | rpm: 5 | rpd: 20", "gemini-2.5-flash-lite| input: $0.1 | output: $0.4 | context: 1_000_000 | rpm: 10 | rpd: 20", "gemini-robotics-er-1.5-preview| input: $0.3 | output: $2.5 | context: 1_000_000 | rpm: 10 | rpd: 20"]
 
# Counters for tracking daily usage
model_counts={opt.split("|")[0].strip():0 for opt in MODEL_OPTIONS}
last_reset_day=None
 
def check_reset_counters():
    global last_reset_day
    try:
        la_tz=ZoneInfo("America/Los_Angeles")
        now=datetime.datetime.now(la_tz)
        today=now.date()
        if ( ((last_reset_day)!=(today)) ):
            logger.info(f"Resetting quota counters. New day: {today}")
            for k in model_counts:
                model_counts[k]=0
            last_reset_day=today
    except Exception as e:
        logger.warning(f"Timezone reset check failed: {e}")
 
def validate_transcript_length(transcript: str, max_words: int = 280_000)->bool:
    """Validate transcript length to prevent processing overly large inputs."""
    if ( ((not(transcript)) or (not(transcript.strip()))) ):
        raise(ValueError("Transcript cannot be empty"))
    words=transcript.split()
    if ( ((max_words)<(len(words))) ):
        raise(ValueError(f"Transcript too long: {len(words)} words (max: {max_words})"))
    return True
 
def validate_youtube_id(youtube_id: str)->bool:
    if ( ((not(youtube_id)) or (((len(youtube_id))!=(11)))) ):
        return False
    # YouTube IDs are alphanumeric with _ and -
    return all(((c.isalnum()) or ((c in "_-"))) for c in youtube_id)
 
def render(summary):
    identifier=summary.identifier
    sid=f"gen-{identifier}"
    if ( summary.timestamps_done ):
        return generation_preview(identifier)
    elif ( summary.summary_done ):
        return Div(NotStr(markdown.markdown(summary.summary)), id=sid, data_hx_post=f"/generations/{identifier}", data_hx_trigger=("") if (summary.timestamps_done) else ("every 1s"), data_hx_swap="outerHTML")
    else:
        return Div(NotStr(markdown.markdown(summary.summary)), id=sid, data_hx_post=f"/generations/{identifier}", data_hx_trigger="every 1s", data_hx_swap="outerHTML")
 
logger.info("Initialize database manually")
db = database("data/summaries.db")
summaries = db.t.items
if not summaries.exists():
    summaries.create(identifier=int, model=str, transcript=str, transcript_hash=str, host=str, original_source_link=str, include_comments=bool, include_timestamps=bool, include_glossary=bool, output_language=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float, embedding=bytes, embedding_model=str, full_embedding=bytes, pk="identifier")
existing_columns={column.name for column in summaries.columns}
if ( "transcript_hash" not in existing_columns ):
    logger.info("Adding transcript_hash column for fast transcript deduplication")
    db.execute("alter table [items] add column [transcript_hash] text")

logger.info("Create website app without automatic db loading")
app, rt = fast_app(live=False, render=render, htmlkw=dict(lang="en-US"))

SUMMARY_LIST_SELECT="identifier, summary, summary_done, timestamps_done"
SUMMARY_PREVIEW_SELECT="""identifier, model, original_source_link, embedding_model, summary, summary_done,
summary_timestamp_end, timestamps_done, summary_input_tokens, timestamps_input_tokens,
summary_output_tokens, timestamps_output_tokens, timestamped_summary_in_youtube_format, cost,
substr(coalesce(transcript, ''), 1, 100) as transcript_preview"""
SUMMARY_STREAM_FLUSH_SECONDS=(0.50    )
SUMMARY_STREAM_FLUSH_CHARS=512

def fetch_summary_row(where: str, where_args, select: str = "*"):
    try:
        rows=list(summaries.rows_where(where=where, where_args=where_args, select=select, limit=1))
    except Exception as e:
        logger.error(f"Error fetching summary row ({where}): {e}")
        return None
    if ( not(rows) ):
        return None
    return AttrDict(rows[0])

def get_summaries(limit=3, order_by="-identifier"):
    """Get summaries with proper error handling."""
    try:
        return [AttrDict(row) for row in summaries.rows_where(order_by=order_by, limit=limit, select=SUMMARY_LIST_SELECT)]
    except Exception as e:
        logger.error(f"Error fetching summaries: {e}")
        return []

def get_summary(identifier: int, select: str = "*"):
    """Get a single summary by identifier."""
    return fetch_summary_row("identifier = ?", [identifier], select=select)

def get_summary_preview(identifier: int):
    return get_summary(identifier, select=SUMMARY_PREVIEW_SELECT)
# Optimization: Ensure indexes exist for fast deduplication lookups.
try:
    summaries.create_index(["original_source_link", "model", "summary_timestamp_start"], if_not_exists=True)
    summaries.create_index(["model", "summary_timestamp_start"], if_not_exists=True)
    summaries.create_index(["transcript_hash", "model", "summary_timestamp_start"], if_not_exists=True)
except Exception as e:
    logger.warning(f"Index creation failed (this is harmless if they exist): {e}")
documentation=((""""""))
def compute_transcript_hash(transcript: str | None)->str | None:
    if ( ((transcript is None)) or (not(transcript.strip())) ):
        return None
    normalized=transcript.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def extract_yt_dlp_error(output: str)->str | None:
    for line in reversed(output.splitlines()):
        stripped=line.strip()
        if ( stripped.startswith("ERROR:") ):
            return stripped[6:].strip()
    return None

def extract_subtitle_filename(output: str)->str | None:
    for line in output.splitlines():
        m=re.search(r"""(?:Writing video subtitles to:|Destination:)\s+(.+\.vtt)\s*$""", line)
        if ( m ):
            return os.path.basename(m.group(1).strip())
    return None

def parse_subtitle_metadata(filename: str, youtube_id: str)->tuple[str, str] | None:
    m=re.match(rf"""^(?P<title>.+)\s\[{re.escape(youtube_id)}\]\.(?P<language>[^.]+)\.vtt$""", filename)
    if ( not(m) ):
        return None
    return (m.group("title").strip(), m.group("language"))

def get_transcript(url, identifier):
    # Call yt-dlp once, let yt-dlp choose the subtitle filename, and prepend title/language metadata to the parsed transcript.
    try:
        youtube_id=validate_youtube_url(url)
        if ( not(youtube_id) ):
            logger.warning(f"Invalid YouTube URL: {url}")
            return "URL couldn't be validated"
        if ( not(validate_youtube_id(youtube_id)) ):
            logger.warning(f"Invalid YouTube ID format: {youtube_id}")
            return "Invalid YouTube ID format"
        temp_dir_base="/dev/shm" if ( os.path.isdir("/dev/shm") ) else None
        with tempfile.TemporaryDirectory(prefix=f"yt-dlp-{identifier}-", dir=temp_dir_base) as temp_dir:
            dl_cmd=["uvx", "yt-dlp", "--cookies-from-browser", "firefox", "--write-subs", "--write-auto-subs", "--skip-download", "--sub-langs", ".*-orig", "--", youtube_id]
            logger.info(f"Downloading subtitles: {' '.join(dl_cmd)} (cwd={temp_dir})")
            dl_res=subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60, cwd=temp_dir)
            combined_output="\n".join([part for part in [dl_res.stdout, dl_res.stderr] if part])
            if ( ((dl_res.returncode)!=(0)) ):
                error_message=extract_yt_dlp_error(combined_output)
                logger.warning(f"yt-dlp download failed: {combined_output}")
                if ( error_message ):
                    return f"Error: {error_message}"
                return "Error: Could not download subtitles"
            subtitle_filename=extract_subtitle_filename(combined_output)
            if ( not(subtitle_filename) ):
                logger.error(f"Could not determine subtitle filename from yt-dlp output: {combined_output}")
                return "Error: No subtitles found for this video. Please provide the transcript manually."
            metadata=parse_subtitle_metadata(subtitle_filename, youtube_id)
            if ( not(metadata) ):
                logger.error(f"Could not parse subtitle metadata from filename: {subtitle_filename}")
                return "Error: Could not determine subtitle metadata"
            title, language=metadata
            sub_file_to_parse=os.path.join(temp_dir, subtitle_filename)
            if ( not(os.path.exists(sub_file_to_parse)) ):
                logger.error(f"Subtitle file missing after download: {sub_file_to_parse}")
                return "Error: Subtitle file disappeared or was not present"
            try:
                transcript_body=parse_vtt_file(sub_file_to_parse)
                logger.info(f"Successfully parsed subtitle file: {sub_file_to_parse}")
            except FileNotFoundError:
                logger.error(f"Subtitle file not found: {sub_file_to_parse}")
                return "Error: Subtitle file disappeared or was not present"
            except PermissionError:
                logger.error(f"Permission denied reading subtitle file: {sub_file_to_parse}")
                return "Error: Permission denied processing subtitle file"
            except Exception as e:
                logger.error(f"Error processing subtitle file: {e}")
                return f"Error: problem when processing subtitle file {e}"
            return f"Title: {title}\nLanguage: {language}\n\n{transcript_body}"
    except subprocess.TimeoutExpired:
        logger.error(f"yt-dlp timeout for identifier {identifier}")
        return "Error: Download timeout"
    except Exception as e:
        logger.error(f"Unexpected error in get_transcript: {e}")
        return f"Error712: {str(e)}"
 
documentation_html=markdown.markdown(documentation)
@rt("/")
def get(request: Request):
    country=request.headers.get("x-country-code", "XX")
    FORBIDDEN_COUNTRIES=set(["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "CH", "GB", "LI", "IS", "NO"])
    is_forbidden=(country in FORBIDDEN_COUNTRIES)
    logger.info(f"Request from: {request.client.host} (Country: {country})")
    logger.info(f"Request from host: {request.client.host}")
    check_reset_counters()
    nav=Nav(Ul(Li(H1("RocketRecap Content Summarizer"))), Ul(Li(A("Map", href="https://rocketrecap.com/exports/index.html")), Li(A("FAQ", href="https://rocketrecap.com/exports/faq.html")), Li(A("Extension", href="https://rocketrecap.com/exports/extension.html")), Li(A("Privacy Policy", href="https://rocketrecap.com/exports/privacy.html")), Li(A("Demo Video", href="https://www.youtube.com/watch?v=ttuDW1YrkpU")), Li(A("Documentation", href="https://github.com/plops/gemini-competition/blob/main/README.md"))))
    error_notice=""
    if ( is_forbidden ):
        error_notice=Div(P(B("Notice: "), "Due to Google's Terms of Service for Gemini in the EU/UK/CH, manual transcript submission is disabled in your region. YouTube link processing is still available."), style="color: #d9534f; background: #f9f2f2; padding: 10px; border-radius: 5px; margin-bottom: 20px;")
    summaries_to_show=[render(s) for s in get_summaries(limit=3, order_by="-identifier")]
    selector=[]
    for opt in MODEL_OPTIONS:
        parts=opt.split("|")
        model_name=parts[0].strip()
        rpd_limit=int(parts[-1].split(":")[1].strip())
        used=model_counts.get(model_name, 0)
        remaining=max(0, ((rpd_limit)-(used)))
        label=f"{model_name} | {remaining} / {rpd_limit} RPD left"
        selector.append(Option(opt, value=opt, label=label))
    model=Div(Label("Select Model", _for="model-select", cls="visually-hidden"), Select(*selector, id="model-select", style="width: 100%;", name="model"), style="display: flex; align-items: center; width: 100%;")
    transcript=Textarea(placeholder="(Optional) Paste YouTube transcript here", style="height: 300px; width: 60%;", name="transcript", id="transcript-paste", disabled=is_forbidden)
    form=Form(Fieldset(Legend("Submit Text for Summarization"), Div(Label("Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)", _for="youtube-link"), Textarea(placeholder="Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)", id="youtube-link", name="original_source_link", style="width: 60%;"), Label("Paste YouTube transcript here", _for="transcript-paste"), transcript, model, Button("Summarize Transcript"), style="display: flex; flex-direction:column;")), data_hx_post="/process_transcript", data_hx_swap="afterbegin", data_hx_target="#summary-list-container")
    
    summary_cards_div=Div(*summaries_to_show, cls="summary-cards", id="summary-list-container")
    
    main_body = Main(
        nav, 
        NotStr(documentation_html),
        H1("RocketRecap Content Summarizer"),
        P(Em("Summarize YouTube videos and transcripts with AI-powered analysis.")),
        Section(
            H3("Submit New Transcript"),
            P("Paste a YouTube URL or transcript to generate an AI-powered summary with timestamps."),
            form
        ),
        Div(
            H2("Recent Summaries", cls="summary-list"),
            error_notice,
            summary_cards_div
        )
    )
    return Title("Video Transcript Summarizer"), Meta(name="description", content="Get AI-powered summaries of YouTube videos and websites. Paste a link or transcript to receive a concise summary with timestamps."), main_body, Script("""function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}


document.getElementById('transcript-paste').addEventListener('paste', (e) => {
    const html = e.clipboardData.getData('text/html');
    
    if (html) {
        e.preventDefault();
        const container = document.createElement('div');
        container.innerHTML = html;
        
        // 1. Handle Links: Convert to "Text (URL) "
        container.querySelectorAll('a').forEach(link => {
            link.innerText = `${link.innerText} (${link.href}) `;
        });

        // 2. Prevent Merging: Add a space to every child element 
        // This ensures that when <div>Text</div><div>37:28</div> 
        // is flattened, it becomes "Text 37:28 "
        const allElements = container.querySelectorAll('*');
        allElements.forEach(el => {
            if (el.innerText && el.children.length === 0) {
                el.innerText = el.innerText + ' ';
            }
        });
        
        // 3. Clean up double spaces and insert
        const cleanText = container.innerText.replace(/[ ]+/g, ' ').trim();
        document.execCommand("insertText", false, cleanText);
    }
})
""") , Style(""".visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}""")
 
# A pending preview keeps polling this route until we return the summary
def generation_preview(identifier):
    sid=f"gen-{identifier}"
    text="Generating ..."
    trigger="every 1s"
    price_input={("gemini-3-flash-preview"):((0.50    )), ("gemini-3.1-flash-lite-preview"):((0.250    )), ("gemini-2.5-flash"):((0.30    )), ("gemini-2.5-flash-lite"):((0.10    )), ("gemini-robotics-er-1.5-preview"):((0.30    ))}
    price_output={("gemini-3-flash-preview"):((3.0    )), ("gemini-3.1-flash-lite-preview"):((1.50    )), ("gemini-2.5-flash"):((2.50    )), ("gemini-2.5-flash-lite"):((0.40    )), ("gemini-robotics-er-1.5-preview"):((2.50    ))}
    try:
        s=get_summary_preview(identifier)
        if not s:
            return Div(P("Summary not found"))
        if ( s.timestamps_done ):
            # this is for <= 128k tokens
            real_model=s.model.split("|")[0]
            price_input_token_usd_per_mio=-1
            price_output_token_usd_per_mio=-1
            try:
                price_input_token_usd_per_mio=price_input[real_model]
                price_output_token_usd_per_mio=price_output[real_model]
            except Exception as e:
                pass
            input_tokens=((s.summary_input_tokens)+(s.timestamps_input_tokens))
            output_tokens=((s.summary_output_tokens)+(s.timestamps_output_tokens))
            cost=((((((input_tokens)/(1_000_000)))*(price_input_token_usd_per_mio)))+(((((output_tokens)/(1_000_000)))*(price_output_token_usd_per_mio))))
            if ( ((s.cost is None)) or (((1.00e-12)<(abs(((s.cost)-(cost)))))) ):
                summaries.update(pk_values=identifier, cost=cost)
            if ( ((cost)<((2.00e-2))) ):
                cost_str=f"${cost:.4f}"
            else:
                cost_str=f"${cost:.2f}"
            text=f"""*AI Summary*

{s.timestamped_summary_in_youtube_format}

AI-generated summary created with {s.model.split('|')[0]} for free via RocketRecap-dot-com. (Input: {input_tokens:,} tokens, Output: {output_tokens:,} tokens, Est. cost: {cost_str})."""
            trigger=""
        elif ( s.summary_done ):
            text=s.summary
        elif ( ((s.summary) and (((0)<(len(s.summary))))) ):
            text=s.summary
        elif ( ((s.transcript_preview) and (((0)<(len(s.transcript_preview))))) ):
            text=f"Generating from transcript: {s.transcript_preview}"
        summary_details=Div(P(B("identifier:"), Span(f"{s.identifier}")), P(B("model:"), Span(f"{s.model}")), A(f"{s.original_source_link}", target="_blank", href=f"{s.original_source_link}", id=f"source-link-{identifier}"), P(B("embedding_model:"), Span(f"{s.embedding_model}")), cls="summary-details")
        summary_container=Div(summary_details, cls="summary-container")
        title=summary_container
        html0=markdown.markdown(text)
        if ( (("")==(html0)) ):
            real_model=s.model.split("|")[0]
            html=f"Waiting for {real_model} to respond to request..."
        else:
            html=replace_timestamps_in_html(html0, s.original_source_link)
        hidden_pre_for_copy=Div(Pre(text, id=f"pre-{identifier}"), id=f"hidden-markdown-{identifier}", style="display: none;")
        card_content=[
            Header(
                H4(
                    A(f"{s.original_source_link}", 
                      target="_blank", 
                      href=f"{s.original_source_link}", 
                      id=f"source-link-{identifier}",
                      style="text-decoration: none; color: var(--pico-primary);")
                ),
                P(f"ID: {s.identifier} | Model: {s.model.split('|')[0]}", 
                  style="font-size: 0.9em; color: var(--pico-secondary-foreground); margin-bottom: 0.5em;"),
                Div(NotStr(html), 
                   style="margin: 1em 0; padding: 1em; border: 1px solid var(--pico-muted); border-radius: 0.5em; background: var(--pico-background-color);")
            ),
            Footer(
                hidden_pre_for_copy, 
                Button("Copy Summary", 
                        onclick=f"copyPreContent('pre-{identifier}')", 
                        cls="outline secondary")
            )
        ]
        if ( ((trigger)==("")) ):
            return Article(*card_content, id=sid)
        else:
            attrs=dict(id=sid, data_hx_post=f"/generations/{identifier}", data_hx_trigger=trigger, data_hx_swap="outerHTML")
            if ( not(((s.summary_timestamp_end) or (s.summary_done))) ):
                attrs["aria-busy"]="true"
                attrs["aria-live"]="polite"
            return Article(*card_content, **attrs)
    except Exception as e:
        return Article(Header(H4(f"Error processing Summary ID: {identifier}")), Div(P("An error occurred while trying to render the summary. The page will continue to refresh automatically."), P(B("Details:"), Code(f"{e}")), Pre(text)), id=sid, data_hx_post=f"/generations/{identifier}", data_hx_trigger=trigger, data_hx_swap="outerHTML", style="border-color: var(--pico-del-color);")
 
@app.post("/generations/{identifier}")
def get(identifier: int):
    return generation_preview(identifier)
 
@rt("/process_transcript")
def post(request: Request, model: str = "", transcript: str = "", original_source_link: str = ""):
    summary = AttrDict(
        model=model,
        transcript=transcript,
        transcript_hash=compute_transcript_hash(transcript),
        original_source_link=original_source_link,
        include_comments=False,
        include_timestamps=True,
        include_glossary=False,
        output_language="en",
        summary_done=False,
        timestamps_done=False
    )
    summary.host=request.client.host if request.client else "unknown"
    summary.summary_timestamp_start=datetime.datetime.now().isoformat()
    summary.summary=""
    t_start=time.perf_counter()
    # Define a lookback window (e.g., 5 minutes) to catch double-clicks or re-submissions.
    lookback_limit=((datetime.datetime.now())-(datetime.timedelta(minutes=5)))
    existing_entry=None
    if ( ((summary.original_source_link) and (((0)<(len(summary.original_source_link.strip()))))) ):
        # Criteria 1: Check by YouTube Link + Model
        matches=list(summaries.rows_where(where="original_source_link = ? AND model = ? AND summary_timestamp_start > ?", where_args=[summary.original_source_link.strip(), summary.model, lookback_limit.isoformat()], order_by="-identifier", select="identifier", limit=1))
        if ( ((0)<(len(matches))) ):
            existing_entry=AttrDict(matches[0])
    elif ( summary.transcript_hash ):
        # Criteria 2: Check by transcript hash + Model (if no link provided)
        matches=list(summaries.rows_where(where="transcript_hash = ? AND model = ? AND summary_timestamp_start > ?", where_args=[summary.transcript_hash, summary.model, lookback_limit.isoformat()], order_by="-identifier", select="identifier", limit=1))
        if ( ((0)<(len(matches))) ):
            existing_entry=AttrDict(matches[0])
    t_end=time.perf_counter()
    duration=((t_end)-(t_start))
    if ( ((duration)>((0.50    ))) ):
        logger.warning(f"Slow deduplication lookup: {duration:.4f}s")
    else:
        logger.info(f"Deduplication lookup took: {duration:.4f}s")
    if ( existing_entry ):
        #  If a duplicate is found, log it and return the PREVIEW of the existing entry instead of starting a new generation job.
        logger.info(f"Duplicate request detected (ID: {existing_entry.identifier}). Skipping new generation.")
        return generation_preview(existing_entry.identifier)
    if ( (summary.transcript is not None) ):
        if ( ((0)==(len(summary.transcript))) ):
            summary.summary="Downloading transcript..."
    s2=AttrDict(summaries.insert(summary))
    download_and_generate(s2.identifier)
    return generation_preview(s2.identifier)
 
@threaded
def download_and_generate(identifier: int):
    try:
        s=wait_until_row_exists(identifier)
        if ( ((s)==(-1)) ):
            logger.error(f"Row {identifier} never appeared in database")
            return 
        if ( (((s["transcript"] is None)) or (((0)==(len(s["transcript"]))))) ):
            # No transcript given, try to download from URL
            transcript=get_transcript(s["original_source_link"], identifier)
            summaries.update(pk_values=identifier, transcript=transcript, transcript_hash=compute_transcript_hash(transcript))
        # re-fetch summary with transcript
        s=get_summary(identifier)
        # Validate transcript length
        try:
            validate_transcript_length(s["transcript"])
        except ValueError as e:
            logger.error(f"Transcript validation failed for {identifier}: {e}")
            summaries.update(pk_values=identifier, summary=f"Error1031: {str(e)}", summary_done=True)
            return 
        words=s["transcript"].split()
        if ( ((len(words))<(30)) ):
            summaries.update(pk_values=identifier, summary="Error: Transcript is too short. Probably I couldn't download it. You can provide it manually.", summary_done=True)
            return 
        if ( ((280_000)<(len(words))) ):
            if ( ("-pro" in s["model"]) ):
                summaries.update(pk_values=identifier, summary="Error: Transcript exceeds 280,000 words. Please shorten it or don't use the pro model.", summary_done=True)
                return 
        logger.info(f"Processing link: {s.original_source_link}")
        summaries.update(pk_values=identifier, summary="")
        generate_and_save(identifier)
    except Exception as e:
        logger.error(f"Error in download_and_generate for {identifier}: {e}")
        try:
            summaries.update(pk_values=identifier, summary=f"Error1055: {str(e)}", summary_done=True)
        except Exception as update_error:
            logger.error(f"Failed to update database with error for {identifier}: {update_error}")
 
def wait_until_row_exists(identifier):
    for i in range(400):
        s=get_summary(identifier)
        if s:
            return s
        time.sleep((0.10    ))
    logger.error(f"Row {identifier} did not appear after 400 attempts")
    return -1
 
def get_prompt(summary)->str:
    r"""Generate prompt from a given Summary object. It will use the contained transcript."""
    prompt=f"""Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding abstract and summary I expect. Afterward, I will provide a new transcript that I want a summarization in the same format. 

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
    summary_text=""
    persisted_summary=""
    last_summary_flush=time.perf_counter()
    try:
        s=wait_until_row_exists(identifier)
        if ( ((s)==(-1)) ):
            logger.error(f"Could not find summary with id {identifier}")
            return 
        # Update usage counter
        check_reset_counters()
        real_model=s["model"].split("|")[0]
        if ( (real_model in model_counts) ):
            model_counts[real_model] += 1
        summary_text=(s["summary"] or "")
        persisted_summary=summary_text
        logger.info(f"generate_and_save model={s['model']}")
        m=genai.GenerativeModel(s["model"].split("|")[0], system_instruction=r"""### CORE INSTRUCTION
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
    - **Style:** Use American English. Be direct, efficient, and dense. Avoid """)
        safety={(HarmCategory.HARM_CATEGORY_HATE_SPEECH):(HarmBlockThreshold.BLOCK_NONE), (HarmCategory.HARM_CATEGORY_HARASSMENT):(HarmBlockThreshold.BLOCK_NONE), (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT):(HarmBlockThreshold.BLOCK_NONE), (HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT):(HarmBlockThreshold.BLOCK_NONE)}
        prompt=get_prompt(s)
        response=m.generate_content(prompt, safety_settings=safety, stream=True)
        for chunk in response:
            try:
                logger.debug(f"Adding text chunk to id={identifier}")
                chunk_text=(chunk.text or "")
            except ValueError as e:
                logger.warning(f"ValueError processing chunk for {identifier}: {e}")
                summary_text += f"\nError: value error {str(e)}"
                summaries.update(pk_values=identifier, summary=summary_text)
                persisted_summary=summary_text
                last_summary_flush=time.perf_counter()
            except Exception as e:
                logger.error(f"Error processing chunk for {identifier}: {e}")
                summary_text += f"\n[Error1189: {str(e)}]"
                summaries.update(pk_values=identifier, summary=summary_text)
                persisted_summary=summary_text
                last_summary_flush=time.perf_counter()
            else:
                if ( not(chunk_text) ):
                    continue
                summary_text += chunk_text
                now=time.perf_counter()
                if ( (((SUMMARY_STREAM_FLUSH_CHARS)<=(((len(summary_text))-(len(persisted_summary)))))) or (((SUMMARY_STREAM_FLUSH_SECONDS)<=((now)-(last_summary_flush)))) ):
                    summaries.update(pk_values=identifier, summary=summary_text)
                    persisted_summary=summary_text
                    last_summary_flush=now
        if ( summary_text != persisted_summary ):
            summaries.update(pk_values=identifier, summary=summary_text)
        prompt_token_count=response.usage_metadata.prompt_token_count
        candidates_token_count=response.usage_metadata.candidates_token_count
        try:
            logger.info(f"Usage metadata: {response.usage_metadata}")
            thinking_token_count=response.usage_metadata.thinking_token_count
        except AttributeError:
            logger.info("No thinking token count available")
            thinking_token_count=0
        summaries.update(pk_values=identifier, summary=summary_text, summary_done=True, summary_input_tokens=prompt_token_count, summary_output_tokens=((candidates_token_count)+(thinking_token_count)), summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Resource exhausted for {identifier}: {e}")
        summary_text += "\nError1234: resource exhausted. Try again with a different model."
        summaries.update(pk_values=identifier, summary_done=False, summary=summary_text, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
        return 
    except Exception as e:
        logger.error(f"Unexpected error in generate_and_save for {identifier}: {e}")
        try:
            summary_text += f"Error1254: {str(e)}"
            summaries.update(pk_values=identifier, summary_done=False, summary=summary_text, summary_timestamp_end=datetime.datetime.now().isoformat())
        except Exception as update_error:
            logger.error(f"Failed to update database with error for {identifier}: {update_error}")
        return 
    try:
        text=summary_text
        text=convert_markdown_to_youtube_format(text)
        summaries.update(pk_values=identifier, timestamps_done=True, timestamped_summary_in_youtube_format=text, timestamps_input_tokens=0, timestamps_output_tokens=0, timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        logger.warning("Resource exhausted during summary update")
        summaries.update(pk_values=identifier, timestamps_done=False, timestamped_summary_in_youtube_format=f"resource exhausted", timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Error during summary update for identifier {identifier}: {e}")
    try:
        # Generate and store the embedding of the summary
        if ( summary_text ):
            logger.info(f"Generating summary embedding for identifier {identifier}...")
            embedding_model="gemini-embedding-001"
            embedding_result=genai.embed_content(model=f"models/{embedding_model}", content=summary_text, task_type="clustering", output_dimensionality=3072)
            vector_array=np.array(embedding_result["embedding"], dtype=np.float32)
            vector_blob=vector_array.tobytes()
            summaries.update(pk_values=identifier, embedding=vector_blob, embedding_model=embedding_model)
            logger.info(f"Embedding stored for identifier {identifier} shape={vector_array.shape}.")
    except google.api_core.exceptions.ResourceExhausted:
        logger.warning("Resource exhausted during embedding of summary")
    except Exception as e:
        logger.error(f"Error during embedding for identifier {identifier}: {e}")
     
# in production run this script with: GEMINI_API_KEY=`cat api_key.txt` uvicorn p04_host:app --port 5001 --log-config=uvicorn_log_conf.yaml
