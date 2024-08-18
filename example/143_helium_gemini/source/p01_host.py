#!/usr/bin/env python3
# pip install -U google-generativeai
import google.generativeai as genai
from fasthtml.common import *
with open("api_key.txt") as f:
    api_key=f.read().strip()
genai.configure(api_key=api_key)
@rt("/")
def get():
    frm=Form(Group(Textarea(placeholder="Paste YouTube transcript here"), Select(Option("gemini-1.5-pro-exp-0801"), Option("gemini-1.5-flash-latest"), name="model"), Button("Send Transcript", hx_post="/process_transcript")), hx_post="/process_transcript", hx_target="#summary")
    return (Title("Video Transcript Summarizer"),Main(H1("Summarizer Demo"), Card(Div(id="summary"), header=frm)),)
@rt("/process_transcript")
async def post(transcript: str, model: str):
    words=transcript.split()
    if ( ((20_000)<(len(words))) ):
        return Div("Error: Transcript exceeds 20,000 words. Please shorten it.", id="summary")