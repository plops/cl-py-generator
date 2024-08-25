#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml
import google.generativeai as genai
import google.generativeai.types.answer_types
import datetime
from fasthtml.common import *
 
# Read the gemini api key from disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
genai.configure(api_key=api_key)
 
def render(summary: Summary):
    identifier=summary.identifier
    sid=f"gen-{identifier}"
    if ( summary.summary_done ):
        return Div(Pre(summary.timestamps), id=sid, hx_post=f"/generations/{identifier}", hx_trigger=("") if (summary.timestamps_done) else ("every 1s"), hx_swap="outerHTML")
    else:
        return Div(Pre(summary.summary), id=sid, hx_post=f"/generations/{identifier}", hx_trigger="every 1s", hx_swap="outerHTML")
 
# open website
# summaries is of class 'sqlite_minutils.db.Table, see https://github.com/AnswerDotAI/sqlite-minutils. Reference documentation: https://sqlite-utils.datasette.io/en/stable/reference.html#sqlite-utils-db-table
app, rt, summaries, Summary=fast_app(db_file="data/summaries.db", live=True, render=render, identifier=int, model=str, transcript=str, host=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, cost=float, pk="identifier")
 
def render(summary: Summary):
    return Li(A(summary.summary_timestamp_start, href=f"/summaries/{summary.identifier}"))
 
@rt("/")
def get(request: Request):
    print(request.client.host)
    nav=Nav(Ul(Li(Strong("Transcript Summarizer"))), Ul(Li(A("About", href="#")), Li(A("Documentation", href="#"))))
    transcript=Textarea(placeholder="Paste YouTube transcript here", name="transcript")
    model=Select(Option("gemini-1.5-flash-latest"), Option("gemini-1.5-pro-exp-0801"), name="model")
    form=Form(Group(transcript, model, Button("Send Transcript")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    summary_list=Ul(*summaries(order_by="identifier DESC"), id="summaries")
    return Title("Video Transcript Summarizer"), Main(nav, H1("Summarizer Demo"), form, summary_list, cls="container")
 
# A pending preview keeps polling this route until we return the summary
def generation_preview(identifier):
    sid=f"gen-{identifier}"
    try:
        summary=summaries[identifier].summary
        return Div(Pre(summary), id=sid)
    except NotFoundError:
        
        return Div("Generating ...", id=sid, hx_post=f"/generations/{identifier}", hx_trigger="every 1s", hx_swap="outerHTML")
 
@app.post("/generations/{id}")
def get(identifier: int):
    return generation_preview(identifier)
 
@rt("/process_transcript")
def post(summary: Summary, request: Request):
    words=summary.transcript.split()
    if ( ((20_000)<(len(words))) ):
        return Div("Error: Transcript exceeds 20,000 words. Please shorten it.", id="summary")
    summary.host=request.client.host
    summary.timestamp_summary_start=datetime.datetime.now().isoformat()
    summary.summary=""
    s2=summaries.insert(summary)
    # first identifier is 1
    generate_and_save(s2.identifier)
    return generation_preview(s2.identifier)
 
@threaded
def generate_and_save(identifier: int):
    print(f"generate_and_save id={identifier}")
    s=summaries[identifier]
    print(f"generate_and_save model={s.model}")
    m=genai.GenerativeModel(s.model)
    response=m.generate_content(f"I don't want to watch the video. Create a self-contained bullet list summary: {s.transcript}", stream=True)
    if ( ((google.generativeai.types.answer_types.FinishReason.SAFETY)==(response.candidates[0].safety_ratings)) ):
        print("stopped because of safety")
    for chunk in response:
        try:
            print(f"add text to id={identifier}: {chunk.text}")
            summaries.update(pk_values=identifier, summary=((summaries[identifier].summary)+(chunk.text)))
        except ValueError:
            
            print("Value Error")
    if ( response._done ):
        summaries.update(pk_values=identifier, summary_done=True, summary_input_tokens=response.usage_metadata.prompt_token_count, summary_output_tokens=response.usage_metadata.candidates_token_count, summary_timestamp_end=datetime.datetime.now().isoformat())
    else:
        print("Warning: Did not finish!")
 
serve(port=5002)