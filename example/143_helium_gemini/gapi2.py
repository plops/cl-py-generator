#!/usr/bin/env python3
import requests
import argparse
import zlib
import json
import sys
import time

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
# if -v is given, print verbose output

argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--file", help="file to read")
argparser.add_argument("-C", "--compress", help="compress the post request", action="store_true")
argparser.add_argument("-P", "--pro", help="use the pro model", action="store_true")
argparser.add_argument("-v", "--verbose", help="print verbose output", action="store_true")
args = argparser.parse_args()
if args.file:
    with open(args.file) as f:
        prompt2 = f.read()
else:
    prompt2 = input("Enter prompt2: ")

start = time.time()

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
    summary = response.json()['candidates'][0]['content']['parts'][0]['text']
    input_tokens = response.json()['usageMetadata']['promptTokenCount']
    output_tokens = response.json()['usageMetadata']['candidatesTokenCount']
    if args.verbose:
        print(response)
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
    input_tokens2 = response2.json()['usageMetadata']['promptTokenCount']
    output_tokens2 = response2.json()['usageMetadata']['candidatesTokenCount']
    if args.verbose:
        print(response2)
        print(summary2)
else:
    print(f"Error: {response2.status_code}")
    print(response2.text)
    sys.exit(1)

# replace '**' with '*' in summary2

summary2 = summary2.replace("**", "*")

print("*Summary*")
print(summary2)
if args.pro:
    print("I used Google Gemini 1.5 Pro to summarize the transcript.")
else:
    print("I used Google Gemini 1.5 Flash to summarize the transcript.")

# this is for <= 128k tokens
if args.pro:
    price_input_token_usd_per_mio = 3.5
    price_output_token_usd_per_mio = 10.5
else:
    price_input_token_usd_per_mio = 0.075
    price_output_token_usd_per_mio = 0.3

end = time.time()

cost_input = (input_tokens+input_tokens2) / 1_000_000 * price_input_token_usd_per_mio
cost_output = (output_tokens+output_tokens2) / 1_000_000 * price_output_token_usd_per_mio
print(f"Cost (if I didn't use the free tier): ${cost_input+cost_output:.4f}")
print(f"Time: {end-start:.2f} seconds")
print(f"Input tokens: {input_tokens+input_tokens2}")
print(f"Output tokens: {output_tokens+output_tokens2}")

# googles free tier quota was exhausted after 5 summaries