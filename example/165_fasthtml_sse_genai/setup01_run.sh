#!/bin/bash

export GEMINI_API_KEY=`cat ~/api_key.txt`
export PATH="/home/kiel/.local/bin:$PATH"
# enable jit
export PYTHON_JIT=1 
uv run python -i p01_top.py -vv
 
