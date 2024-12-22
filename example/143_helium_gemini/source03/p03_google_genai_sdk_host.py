#!/usr/bin/env python3
# micromamba install python-fasthtml markdown; pip install google-genai webvtt-py
import sqlite_minutils.db
import datetime
import time
import subprocess
import webvtt
from google import genai
from google.genai import types
start_time=time.time()
# Call yt-dlp to download the subtitles
url="https://www.youtube.com/watch?v=ttuDW1YrkpU"
sub_file="/dev/shm/o"
sub_file_="/dev/shm/o.en.vtt"
subprocess.run(["yt-dlp", "--skip-download", "--write-auto-subs", "--write-subs", "--sub-lang", "en", "-o", sub_file, url])
for c in webvtt.read(sub_file_):
    start=c.start.split(".")[0]
    print("{} nil start={} c.text={}".format(((time.time())-(start_time)), start, c.text))