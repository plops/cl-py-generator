#!/usr/bin/env python
# micromamba activate
# pip install google-generativeai PySocks

# export http_proxy="socks4://localhost:8080"; export https_proxy="socks4://localhost:8080"

import google.generativeai as genai
#import socket
#import socks

#socks.set_default_proxy(socks.SOCKS5, "localhost", 8080)
#socket.socket = socks.socksocket


# read api key from file
with open("/home/martin/api_key.txt") as f:
    key = f.read().strip()

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-1.5-flash')\
prompt = "Tell me a joke"
response =model.generate_content(prompt,
                                 generation_config={"temperature": 2.0},
                                 safety_settings={"threshold": BLOCK_NONE})
print(response.text)