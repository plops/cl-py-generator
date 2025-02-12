#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml markdown
# micromamba install python-fasthtml markdown yt-dlp; pip install  webvtt-py
import google.generativeai as genai
import os
import google.api_core.exceptions
import re
import markdown
import sqlite_minutils.db
import datetime
import subprocess
import webvtt
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fasthtml.common import *
 
# Read the gemini api key from disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
genai.configure(api_key=api_key)
 
def render(summary: Summary):
    identifier=summary.identifier
    sid=f"gen-{identifier}"
    if ( summary.timestamps_done ):
        return generation_preview(identifier)
    elif ( summary.summary_done ):
        return Div(NotStr(markdown.markdown(summary.summary)), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=("") if (summary.timestamps_done) else ("every 1s"), hx_swap="outerHTML")
    else:
        return Div(NotStr(markdown.markdown(summary.summary)), id=sid, hx_post=f"/generations/{identifier}", hx_trigger="every 1s", hx_swap="outerHTML")
 
# open website
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt, summaries, Summary=fast_app(db_file="data/summaries.db", live=False, render=render, identifier=int, model=str, transcript=str, host=str, original_source_link=str, include_comments=bool, include_timestamps=bool, include_glossary=bool, output_language=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float, pk="identifier")
 
documentation="""###### To use the YouTube summarizer:

1. **Copy the YouTube video link.**
2. **Paste the link into the provided input field.**
3. **Click the 'Summarize' button.** The summary with timestamps will be generated.

"""
 
def validate_youtube_url(url):
    """Validates various YouTube URL formats."""
    patterns=[r"""^https://(www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]{11}.*""", r"""^https://(www\.)?youtube\.com/live/[A-Za-z0-9_-]{11}.*""", r"""^https://(www\.)?youtu\.be/[A-Za-z0-9_-]{11}.*"""]
    for pattern in patterns:
        if ( re.match(pattern, url) ):
            return True
    print("Error: Invalid YouTube URL")
    return False
def get_transcript(url):
    # Call yt-dlp to download the subtitles
    if ( not(validate_youtube_url(url)) ):
        return ""
    sub_file="/dev/shm/o"
    sub_file_="/dev/shm/o.en.vtt"
    cmds=["yt-dlp", "--skip-download", "--write-auto-subs", "--write-subs", "--cookies-from-browser", "firefox", "--sub-lang", "en", "-o", sub_file, url]
    print(" ".join(cmds))
    subprocess.run(cmds)
    ostr=""
    try:
        for c in webvtt.read(sub_file_):
            # we don't need sub-second time resolution. trim it away
            start=c.start.split(".")[0]
            # remove internal newlines within a caption
            cap=c.text.strip().replace("\n", " ")
            # write <start> <c.text> into each line of ostr
            ostr += f"{start} {cap}\n"
        os.remove(sub_file_)
    except FileNotFoundError:
        print("Error: Subtitle file not found")
    except Exception as e:
        print("Error: problem when processing subtitle file")
    return ostr
 
documentation_html=markdown.markdown(documentation)
@rt("/")
def get(request: Request):
    print(request.client.host)
    nav=Nav(Ul(Li(Strong("Transcript Summarizer"))), Ul(Li(A("Demo Video", href="https://www.youtube.com/watch?v=ttuDW1YrkpU")), Li(A("Documentation", href="https://github.com/plops/gemini-competition/blob/main/README.md"))))
    transcript=Textarea(placeholder="(Optional) Paste YouTube transcript here", style="height: 300px; width=60%;", name="transcript")
    model=Div(Select(Option("gemini-2.0-flash-exp"), Option("gemini-2.0-pro-exp-02-05"), Option("gemini-1.5-pro-exp-0827"), Option("gemini-2.0-flash-lite-preview-02-05"), Option("gemini-2.0-flash-thinking-exp-01-21"), Option("gemini-2.0-flash-001"), Option("gemini-exp-1206"), Option("gemini-exp-1121"), Option("gemini-exp-1114"), Option("learnlm-1.5-pro-experimental"), Option("gemini-1.5-flash-002"), Option("gemini-1.5-pro-002"), Option("gemini-1.5-pro-exp-0801"), Option("gemini-1.5-flash-exp-0827"), Option("gemini-1.5-flash-8b-exp-0924"), Option("gemini-1.5-flash-latest"), Option("gemma-2-2b-it"), Option("gemma-2-9b-it"), Option("gemma-2-27b-it"), Option("gemini-1.5-flash"), Option("gemini-1.5-pro"), Option("gemini-1.0-pro"), style="width: 100%;", name="model"), style="display: flex; align-items: center; width: 100%;")
    form=Form(Group(Div(Textarea(placeholder="Link to youtube video (e.g. https://youtube.com/watch?v=j9fzsGuTTJA)", name="original_source_link"), transcript, model, Div(Label("Output Language", _for="output_language"), Select(Option("en"), Option("de"), Option("fr"), Option("ch"), Option("nl"), Option("pt"), Option("cz"), Option("it"), Option("jp"), Option("ar"), style="width: 100%;", name="output_language", id="output_language"), style="display: none; align-items: center; width: 100%;"), Div(Input(type="checkbox", id="include_comments", name="include_comments", checked=False), Label("Include User Comments", _for="include_comments"), style="display: none; align-items: center; width: 100%;"), Div(Input(type="checkbox", id="include_timestamps", name="include_timestamps", checked=True), Label("Include Timestamps", _for="include_timestamps"), style="display: none; align-items: center; width: 100%;"), Div(Input(type="checkbox", id="include_glossary", name="include_glossary", checked=False), Label("Include Glossary", _for="include_glossary"), style="display: none; align-items: center; width: 100%;"), Button("Summarize Transcript"), style="display: flex; flex-direction:column;")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    gen_list=Div(id="gen-list")
    summaries_to_show=summaries(order_by="identifier DESC")
    summaries_to_show=summaries_to_show[0:min(3, len(summaries_to_show))]
    summary_list=Ul(*summaries_to_show, id="summaries")
    return Title("Video Transcript Summarizer"), Main(nav, NotStr(documentation_html), form, gen_list, summary_list, Script("""function copyPreContent(elementId) {
  var preElement = document.getElementById(elementId);
  var textToCopy = preElement.textContent;

  navigator.clipboard.writeText(textToCopy);
}"""), cls="container")
 
# A pending preview keeps polling this route until we return the summary
def generation_preview(identifier):
    sid=f"gen-{identifier}"
    text="Generating ..."
    trigger="every 1s"
    try:
        s=summaries[identifier]
        if ( s.timestamps_done ):
            # this is for <= 128k tokens
            if ( ((s.model.startswith("gemini-1.5-pro")) or (s.model.startswith("gemini-2.0-pro"))) ):
                price_input_token_usd_per_mio=(1.250    )
                price_output_token_usd_per_mio=(5.0    )
            else:
                if ( ("flash" in s.model) ):
                    price_input_token_usd_per_mio=(0.10    )
                    price_output_token_usd_per_mio=(0.40    )
                else:
                    if ( s.model.startswith("gemini-1.0-pro") ):
                        price_input_token_usd_per_mio=(0.50    )
                        price_output_token_usd_per_mio=(1.50    )
                    else:
                        price_input_token_usd_per_mio=-1
                        price_output_token_usd_per_mio=-1
            input_tokens=((s.summary_input_tokens)+(s.timestamps_input_tokens))
            output_tokens=((s.summary_output_tokens)+(s.timestamps_output_tokens))
            if ( ("flash" in s.model) ):
                cost=((price_input_token_usd_per_mio)+(price_output_token_usd_per_mio))
            else:
                cost=((((((input_tokens)/(1_000_000)))*(price_input_token_usd_per_mio)))+(((((output_tokens)/(1_000_000)))*(price_output_token_usd_per_mio))))
            summaries.update(pk_values=identifier, cost=cost)
            if ( ((cost)<((2.00e-2))) ):
                cost_str=f"${cost:.4f}"
            else:
                cost_str=f"${cost:.2f}"
            text=f"""{s.timestamped_summary_in_youtube_format}

I used {s.model} on rocketrecap dot com to summarize the transcript.
Cost (if I didn't use the free tier): {cost_str}
Input tokens: {input_tokens}
Output tokens: {output_tokens}"""
            trigger=""
        elif ( s.summary_done ):
            text=s.summary
        elif ( ((0)<(len(s.summary))) ):
            text=s.summary
        elif ( ((len(s.transcript))) ):
            text=f"Generating from transcript: {s.transcript[0:min(100,len(s.transcript))]}"
        summary_details=Div(P(B("identifier:"), Span(f"{s.identifier}")), P(B("model:"), Span(f"{s.model}")), P(B("host:"), Span(f"{s.host}")), A(f"{s.original_source_link}", target="_blank", href=f"{s.original_source_link}", id="source-link"), P(B("include_comments:"), Span(f"{s.include_comments}")), P(B("include_timestamps:"), Span(f"{s.include_timestamps}")), P(B("include_glossary:"), Span(f"{s.include_glossary}")), P(B("output_language:"), Span(f"{s.output_language}")), P(B("cost:"), Span(f"{s.cost}")), cls="summary-details")
        summary_container=Div(summary_details, cls="summary-container")
        title=summary_container
        html=markdown.markdown(s.summary)
        pre=Div(Div(Pre(text, id=f"pre-{identifier}"), id="hidden-markdown", style="display: none;"), Div(NotStr(html)))
        button=Button("Copy", onclick=f"copyPreContent('pre-{identifier}')")
        prompt_text=get_prompt(s)
        prompt_pre=Pre(prompt_text, id=f"prompt-pre-{identifier}", style="display: none;")
        prompt_button=Button("Copy Prompt", onclick=f"copyPreContent('prompt-pre-{identifier}')")
        if ( ((trigger)==("")) ):
            return Div(title, pre, prompt_pre, prompt_button, id=sid)
        else:
            return Div(title, pre, prompt_pre, button, prompt_button, id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
    except Exception as e:
        return Div(f"id: {identifier} e: {e}", Pre(text), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
 
@app.post("/generations/{identifier}")
def get(identifier: int):
    return generation_preview(identifier)
 
@rt("/process_transcript")
def post(summary: Summary, request: Request):
    if ( ((0)==(len(summary.transcript))) ):
        # No transcript given, try to download from URL
        summary.transcript=get_transcript(summary.original_source_link)
    words=summary.transcript.split()
    if ( ((len(words))<(30)) ):
        return Div("Error: Transcript is too short. No summary necessary", id="summary")
    if ( ((280_000)<(len(words))) ):
        if ( ("-pro" in summary.model) ):
            return Div("Error: Transcript exceeds 280,000 words. Please shorten it or don't use the pro model.", id="summary")
    summary.host=request.client.host
    summary.summary_timestamp_start=datetime.datetime.now().isoformat()
    print(f"link: {summary.original_source_link}")
    summary.summary=""
    s2=summaries.insert(summary)
    # first identifier is 1
    generate_and_save(s2.identifier)
    return generation_preview(s2.identifier)
 
def wait_until_row_exists(identifier):
    for i in range(10):
        try:
            s=summaries[identifier]
            return s
        except sqlite_minutils.db.NotFoundError:
            print("entry not found")
        except Exception as e:
            print(f"unknown exception {e}")
        time.sleep((0.10    ))
    print("row did not appear")
    return -1
 
def get_prompt(summary: Summary)->str:
    r"""Generate prompt from a given Summary object. It will use the contained transcript."""
    prompt=f"""Below, I will provide input for an example video (comprising of title, description, and transcript, in this order) and the corresponding summary I expect. Afterward, I will provide a new transcript that I want you to summarize in the same format. 

**Please summarize the transcript in a self-contained bullet list format.** Include starting timestamps, important details and key takeaways. 

Example Input: 
input
Example Output:
output
Here is the real transcript. Please summarize it: 
{s.transcript}"""
    return prompt
 
@threaded
def generate_and_save(identifier: int):
    print(f"generate_and_save id={identifier}")
    s=wait_until_row_exists(identifier)
    print(f"generate_and_save model={s.model}")
    m=genai.GenerativeModel(s.model)
    safety={(HarmCategory.HARM_CATEGORY_HATE_SPEECH):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_HARASSMENT):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT):(HarmBlockThreshold.BLOCK_NONE),(HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT):(HarmBlockThreshold.BLOCK_NONE)}
    try:
        prompt=get_prompt(s)
        response=m.generate_content(prompt, safety_settings=safety, stream=True)
        for chunk in response:
            try:
                print(f"add text to id={identifier}: {chunk.text}")
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(chunk.text)))
            except ValueError:
                
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+("\nError: value error")))
                print("Value Error ")
            except Exception as e:
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(f"\nError: {str(e)}")))
                print("Error")
        summaries.update(pk_values=identifier, summary_done=True, summary_input_tokens=response.usage_metadata.prompt_token_count, summary_output_tokens=response.usage_metadata.candidates_token_count, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, summary_done=False, summary=((summaries[identifier].summary)+("\nError: resource exhausted")), summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
        return
    try:
        text=summaries[identifier].summary
        # adapt the markdown to YouTube formatting
        text=text.replace("**:", ":**")
        text=text.replace("**,", ",**")
        text=text.replace("**.", ".**")
        text=text.replace("**", "*")
        # markdown title starting with ## with fat text
        text=re.sub(r"""^##\s*(.*)""", r"""*\1*""", text)
        # find any text that looks like an url and replace the . with -dot-
        text=re.sub(r"""((?:https?://)?(?:www\.)?\S+)\.(com|org|de|us|gov|net|edu|info|io|co\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr)""", r"""\1-dot-\2""", text)
        summaries.update(pk_values=identifier, timestamps_done=True, timestamped_summary_in_youtube_format=text, timestamps_input_tokens=0, timestamps_output_tokens=0, timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, timestamps_done=False, timestamped_summary_in_youtube_format=f"resource exhausted", timestamps_timestamp_end=datetime.datetime.now().isoformat())
        return
 
serve(host="0.0.0.0", port=5001)