#!/usr/bin/env python
# micromamba activate
# pip install google-generativeai PySocks

import google.generativeai as genai
import socket
import socks

socks.set_default_proxy(socks.SOCKS5, "localhost", 8080)
socket.socket = socks.socksocket


# read api key from file
with open("/home/martin/api_key.txt") as f:
    key = f.read().strip()

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-1.5-flash')
response =model.generate_content("Give me python code to sort a list")
print(response.text)