#!/bin/bash

rm -rf .venv uv.lock 
export PATH="/home/kiel/.local/bin:$PATH
# uv python install pypy@3.11
uv python install 3.14
uv sync
