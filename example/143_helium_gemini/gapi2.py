#!/usr/bin/env python3
import requests
import argparse
import zlib
import json
import sys

# Set up the proxy settings
proxies = {
    "http": "socks5://localhost:8080",
    "https": "socks5://localhost:8080",
}

# load api key from file
with open("/home/martin/api_key.txt") as f:
    api_key = f.read().strip()


# command line arguments:
# -f <file> read file into prompt2 variable
# if no -f argument is given read prompt2 from stdin
# if -C is given, compress the post request
# if -P is given, use the pro model

argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--file", help="file to read")
argparser.add_argument("-C", "--compress", help="compress the post request", action="store_true")
argparser.add_argument("-P", "--pro", help="use the pro model", action="store_true")
args = argparser.parse_args()
if args.file:
    with open(args.file) as f:
        prompt2 = f.read()
else:
    prompt2 = input("Enter prompt2: ")

prompt = "I don't want to watch the video. Create a self-contained bullet list summary from the following transcript that I can understand without watching the video. "

if args.pro:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key=" + api_key
else:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + api_key

data = {"contents": [{"parts": [{"text": prompt + prompt2}]}]}
data = json.dumps(data)
# compress the post request
if args.compress:
    headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip", "Content-Encoding": "gzip"}
    data = zlib.compress(data.encode('utf-8'))
else:
    headers = {"Content-Type": "application/json"}


response = requests.post(url, headers=headers,
                         data=data, proxies=proxies)

if response.status_code == 200:
    #print(response.json())
    summary = response.json()['candidates'][0]['content']['parts'][0]['text']
    print(summary)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
    sys.exit(1)

prompt3 = f"Add starting (not stopping) timestamp to each bullet point in the following summary: {summary}\nThe full transcript is: {prompt2}"

response2 = requests.post(url, headers=headers,
                            data=json.dumps({"contents": [{"parts": [{"text": prompt3}]}]}),
                            proxies=proxies)

if response2.status_code == 200:
    summary2 = response2.json()['candidates'][0]['content']['parts'][0]['text']
    print(summary2)
else:
    print(f"Error: {response2.status_code}")
    print(response2.text)
    sys.exit(1)