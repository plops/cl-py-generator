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
 
# open website
app, rt=fast_app(live=True)
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
    filename=f"{folder}/{id}.md"
    if ( os.path.exists(filename) ):
        # Load potentially partial response from the file
        with open(filename) as f:
            summary_pre=f.read()
        return Div(Pre(summary_pre, id=sid))
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
    # consecutively append output to {folder}/{id}.md
    with open(f"{folder}/{id}.md", "w") as f:
        for chunk in response:
            f.write(chunk.text)
            print(f"{model} {id} {chunk.text}")
serve(port=5002)