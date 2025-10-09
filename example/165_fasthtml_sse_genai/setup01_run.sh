#!/bin/bash

export GEMINI_API_KEY=`cat ~/api_key.txt`
uv run python -i p01_top.py
