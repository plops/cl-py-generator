#!/bin/bash

export GEMINI_API_KEY=`cat ~/api_key.txt`
export PATH="/home/kiel/.local/bin:$PATH
uv run python -i p01_top.py -vv
 
