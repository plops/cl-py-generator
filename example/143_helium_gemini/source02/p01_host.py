#!/usr/bin/env python3
# pip install -U google-generativeai python-fasthtml
import google.generativeai as genai
import os
import datetime
from fasthtml.common import *
 
# Read the gemini api key from disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
genai.configure(api_key=api_key)
 
def render(summary):
    if ( summary.summary_done ):
        return Div(Pre(summary.timestamps), id=sid, hx_post=f"/generations/{id}", hx_trigger=("") if (summary.timestamps_done) else ("every 1s"), hx_swap="outerHTML")
    else:
        return Div(Pre(summary.summary), id=sid, hx_post=f"/generations/{id}", hx_trigger="every 1s", hx_swap="outerHTML")
 
# open website
app, rt, summaries, summary=fast_app(db_file="data/summaries.db", live=True, render=render, id=int, model=str, transcript=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, cost=float, pk="id")
 
def render(summary):
    return Li(A(summary.summary_timestamp_start, href=f"/summaries/{summary.id}"))
 
@rt("/")
def get():
    nav=Nav(Ul(Li(Strong("Transcript Summarizer"))), Ul(Li(A("About", href="#")), Li(A("Documentation", href="#"))))
    transcript=Textarea(placeholder="Paste YouTube transcript here", name="transcript")
    model=Select(Option("gemini-1.5-flash-latest"), Option("gemini-1.5-pro-exp-0801"), name="model")
    form=Form(Group(transcript, model, Button("Send Transcript")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    gen_list=Div(id="gen-list")
    return Title("Video Transcript Summarizer"), Main(nav, H1("Summarizer Demo"), form, gen_list, cls="container")
 
# A pending preview keeps polling this route until we return the summary
def generation_preview(id):
    sid=f"gen-{id}"
    if ( summaries[id] ):
        summary=summaries[id].summary
        return Div(Pre(summary), id=sid)
    else:
        return Div("Generating ...", id=sid, hx_post=f"/generations/{id}", hx_trigger="every 1s", hx_swap="outerHTML")
 
@app.post("/generations/{id}")
def get(id: int):
    return generation_preview(id)
 
@rt("/process_transcript")
def post(transcript: str, model: str):
    words=transcript.split()
    if ( ((20_000)<(len(words))) ):
        return Div("Error: Transcript exceeds 20,000 words. Please shorten it.", id="summary")
    generate_and_save(transcript, id, model)
    return generation_preview(id)
 
@threaded
def generate_and_save(prompt, id, model):
    m=genai.GenerativeModel(model)
    response=m.generate_content(f"I don't want to watch the video. Create a self-contained bullet list summary: {prompt}", stream=True)
    s=summaries()[id]
    with open(pre_file, "w") as f:
        s["summary"]=""
        for chunk in response:
            print(chunk)
            s["summary"]=((s["summary"])+(chunk.text))
            f.write(chunk.text)
        if ( response._done ):
            s["summary_done"]=True
            f.write(f"\nSummarized with {model}")
            f.write(f"\nInput tokens: {response.usage_metadata.prompt_token_count}")
            f.write(f"\nOutput tokens: {response.usage_metadata.candidates_token_count}")
        else:
            f.write("Warning: Did not finish!")
 
serve(port=5002)