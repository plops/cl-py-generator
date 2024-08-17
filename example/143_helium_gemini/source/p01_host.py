#!/usr/bin/env python3
# pip install -U google-generativeai
import os
import google.generativeai as genai
genai.configure(api_key=os.environ("API_KEY"))