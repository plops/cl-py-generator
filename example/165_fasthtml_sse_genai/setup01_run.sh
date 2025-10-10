#!/bin/bash

export GEMINI_API_KEY=`cat ~/api_key.txt`
export PATH="/home/kiel/.local/bin:$PATH"

# enable jit
 export PYTHON_JIT=1 ;uv run python p01_top.py -vv

# alternative: try pypy (fasthtml doesn't work with that)
#uv run --python pypy3 p01_top.py -vv

