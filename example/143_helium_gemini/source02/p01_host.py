#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml
import google.generativeai as genai
import re
import sqlite_minutils.db
import datetime
import time
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
        return Div(Pre(summary.summary), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=("") if (summary.timestamps_done) else ("every 1s"), hx_swap="outerHTML")
    else:
        return Div(Pre(summary.summary), id=sid, hx_post=f"/generations/{identifier}", hx_trigger="every 1s", hx_swap="outerHTML")
 
# open website
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt, summaries, Summary=fast_app(db_file="data/summaries.db", live=False, render=render, identifier=int, model=str, transcript=str, host=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float, pk="identifier")
 
 
@rt("/")
def get(request: Request):
    print(request.client.host)
    nav=Nav(Ul(Li(Strong("Transcript Summarizer"))), Ul(Li(A("Demo Video", href="https://www.youtube.com/watch?v=ttuDW1YrkpU")), Li(A("Documentation", href="https://github.com/plops/gemini-competition/blob/main/README.md"))))
    transcript=Textarea(placeholder="Paste YouTube transcript here", name="transcript")
    model=Select(Option("gemini-1.5-flash-latest"), Option("gemini-1.5-pro-exp-0801"), name="model")
    form=Form(Group(transcript, model, Button("Summarize Transcript")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    gen_list=Div(id="gen-list")
    summaries_to_show=summaries(order_by="identifier DESC")
    summaries_to_show=summaries_to_show[0:min(3, len(summaries_to_show))]
    summary_list=Ul(*summaries_to_show, id="summaries")
    return Title("Video Transcript Summarizer"), Main(nav, form, gen_list, summary_list, Script("""function copyPreContent(elementId) {
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
            if ( ((s.model)==("gemini-1.5-pro-exp-0801")) ):
                price_input_token_usd_per_mio=(3.50    )
                price_output_token_usd_per_mio=(10.50    )
            else:
                price_input_token_usd_per_mio=(7.50e-2)
                price_output_token_usd_per_mio=(0.30    )
            input_tokens=((s.summary_input_tokens)+(s.timestamps_input_tokens))
            output_tokens=((s.summary_output_tokens)+(s.timestamps_output_tokens))
            cost=((((((input_tokens)/(1_000_000)))*(price_input_token_usd_per_mio)))+(((((output_tokens)/(1_000_000)))*(price_output_token_usd_per_mio))))
            if ( ((cost)<((2.00e-2))) ):
                cost_str=f"${cost:.4f}"
            else:
                cost_str=f"${cost:.2f}"
            text=f"""{s.timestamped_summary_in_youtube_format}

I used {s.model} to summarize the transcript.
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
        title=f"{s.summary_timestamp_start} id: {identifier} summary: {s.summary_done} timestamps: {s.timestamps_done}"
        pre=Pre(text, id=f"pre-{identifier}")
        button=Button("Copy", onclick=f"copyPreContent('pre-{identifier}')")
        if ( ((trigger)==("")) ):
            return Div(title, pre, button, id=sid)
        else:
            return Div(title, pre, button, id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
    except Exception as e:
        return Div(f"id: {identifier}", Pre(text), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=trigger, hx_swap="outerHTML")
 
@app.post("/generations/{identifier}")
def get(identifier: int):
    return generation_preview(identifier)
 
@rt("/process_transcript")
def post(summary: Summary, request: Request):
    words=summary.transcript.split()
    if ( ((20_000)<(len(words))) ):
        return Div("Error: Transcript exceeds 20,000 words. Please shorten it.", id="summary")
    summary.host=request.client.host
    summary.summary_timestamp_start=datetime.datetime.now().isoformat()
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
            print(f"entry not found")
        time.sleep((0.10    ))
    print("row did not appear")
    return -1
 
@threaded
def generate_and_save(identifier: int):
    print(f"generate_and_save id={identifier}")
    s=wait_until_row_exists(identifier)
    print(f"generate_and_save model={s.model}")
    m=genai.GenerativeModel(s.model)
    try:
        response=m.generate_content(f"I don't want to watch the video. Create a self-contained bullet list summary: {s.transcript}", stream=True)
        for chunk in response:
            try:
                print(f"add text to id={identifier}: {chunk.text}")
                summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(chunk.text)))
            except ValueError:
                
                print("Value Error ")
        summaries.update(pk_values=identifier, summary_done=True, summary_input_tokens=response.usage_metadata.prompt_token_count, summary_output_tokens=response.usage_metadata.candidates_token_count, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, summary_done=False, summary_timestamp_end=datetime.datetime.now().isoformat(), timestamps="", timestamps_timestamp_start=datetime.datetime.now().isoformat())
        return
    try:
        print("generate timestamps")
        s=summaries[identifier]
        response2=m.generate_content(f"Add a title to the summary and add a starting (not stopping) timestamp to each bullet point in the following summary: {s.summary}nThe full transcript is: {s.transcript}", stream=True)
        for chunk in response2:
            try:
                print(f"add timestamped text to id={identifier}: {chunk.text}")
                summaries.update(pk_values=identifier, timestamps=((summaries[identifier].timestamps)+(chunk.text)))
            except ValueError:
                
                print("Value Error")
        text=summaries[identifier].timestamps
        # adapt the markdown to youtube formatting
        text=text.replace("**:", ":**")
        text=text.replace("**,", ",**")
        text=text.replace("**.", ".**")
        text=text.replace("**", "*")
        # markdown title starting with ## with fat text
        text=re.sub(r"""^##\s*(.*)""", r"""*\1*""", text)
        summaries.update(pk_values=identifier, timestamps_done=True, timestamped_summary_in_youtube_format=text, timestamps_input_tokens=response2.usage_metadata.prompt_token_count, timestamps_output_tokens=response2.usage_metadata.candidates_token_count, timestamps_timestamp_end=datetime.datetime.now().isoformat())
    except google.api_core.exceptions.ResourceExhausted:
        summaries.update(pk_values=identifier, timestamps_done=False, timestamped_summary_in_youtube_format=text, timestamps_timestamp_end=datetime.datetime.now().isoformat())
        return
 
serve(port=5001)