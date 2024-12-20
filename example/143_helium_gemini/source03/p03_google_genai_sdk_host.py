#!/usr/bin/env python3
# micromamba install python-fasthtml markdown; pip install google-genai
import markdown
import sqlite_minutils.db
import datetime
import time
from google import genai
from google.genai import types
# genai manual: https://googleapis.github.io/python-genai/
 
# Read the gemini api key from  disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
client=genai.Client(api_key=api_key)
prompt="Tell me a joke about rockets!"
model="gemini-2.0-flash-exp"
safeties=[]
for harm in types.HarmCategory.__args__[1:]:
    # skip unspecified
    safeties.append(types.SafetySetting(category=harm, threshold="BLOCK_NONE"))
config=types.GenerateContentConfig(temperature=(2.0    ), safety_settings=safeties)
for chunk in client.models.generate_content_stream(model=model, contents=prompt, config=config):
    print(chunk.text)