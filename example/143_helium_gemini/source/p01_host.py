#!/usr/bin/env python3
# pip install -U google-generativeai
# https://docs.fastht.ml/tutorials/by_example.html#full-example-2---image-generation-app
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
app, rt, summaries, summary=fast_app("summaries.db", live=True, render=render, id=int, model=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, cost=float, pk="id")
# create a folder with current datetime gens_<datetime>/
generations=[]
dt_now=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder=f"gens_{dt_now}/"
os.makedirs(folder, exist_ok=True)
 
@rt("/")
def get():
    transcript=Textarea(placeholder="Paste YouTube transcript here", name="transcript")
    model=Select(Option("gemini-1.5-pro-exp-0801"), Option("gemini-1.5-flash-latest"), name="model")
    form=Form(Group(transcript, model, Button("Send Transcript")), hx_post="/process_transcript", hx_swap="afterbegin", target_id="gen-list")
    gen_list=Div(id="gen-list")
    return Title("Video Transcript Summarizer"), Main(H1("Summarizer Demo"), form, gen_list, cls="container")
 
# A pending preview keeps polling this route until we return the summary
def generation_preview(id):
    sid=f"gen-{id}"
    pre_filename=f"{folder}/{id}.pre"
    filename=f"{folder}/{id}.md"
    if ( os.path.exists(filename) ):
        # Load potentially partial response from the file
        with open(filename) as f:
            summary=f.read()
        return Div(Pre(summary), id=sid)
    else:
        if ( os.path.exists(pre_filename) ):
            with open(pre_filename) as f:
                summary_pre=f.read()
            return Div(Pre(summary_pre), id=sid, hx_post=f"/generations/{id}", hx_trigger="every 1s", hx_swap="outerHTML")
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
    id=len(generations)
    generate_and_save(transcript, id, model)
    generations.append(transcript)
    return generation_preview(id)
 
@threaded
def generate_and_save(prompt, id, model):
    m=genai.GenerativeModel(model)
    response=m.generate_content(f"I don't want to watch the video. Create a self-contained bullet list summary: {prompt}", stream=True)
    # consecutively append output to {folder}/{id}.pre and finally move from .pre to .md file
    pre_file=f"{folder}/{id}.pre"
    md_file=f"{folder}/{id}.md"
    with open(pre_file, "w") as f:
        for chunk in response:
            print(chunk)
            f.write(chunk.text)
        if ( response._done ):
            f.write(f"\nSummarized with {model}")
            f.write(f"\nInput tokens: {response.usage_metadata.prompt_token_count}")
            f.write(f"\nOutput tokens: {response.usage_metadata.candidates_token_count}")
        else:
            f.write("Warning: Did not finish!")
    os.rename(pre_file, md_file)
 
serve(port=5002)