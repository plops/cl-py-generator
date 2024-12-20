#!/usr/bin/env python3
# micromamba install python-fasthtml markdown; pip install google-genai
import markdown
import sqlite_minutils.db
import datetime
import time
from fasthtml.common import *
from google import genai
from google.genai import types
# genai manual: https://googleapis.github.io/python-genai/
 
# Read the gemini api key from  disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
client=genai.Client(api_key=api_key)