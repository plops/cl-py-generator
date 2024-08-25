#!/usr/bin/env python3
"""
This script uses the Google Gemini API to generate a summarized bullet list from a YouTube video transcript. 
It takes the transcript as input, either from a file or by downloading it directly from YouTube using the video ID. 
The script then sends the transcript to the Gemini API, which generates a summary. Finally, it adds timestamps to 
each bullet point and saves the summary to a file.

Requirements:

* Google Gemini Key: You will need a Google Gemini API key to use this script. You can obtain one from the Google Cloud Console. 
Make sure to enable the Gemini API for your project.
* VM in US (for free tier): The free tier of the Gemini API is only available in the US. You will need to run this script on 
a VM located in the US to take advantage of the free tier.
* yt-dlp: This script uses `yt-dlp` to download YouTube transcripts. You can install it using `pip install yt-dlp`.

Usage:

```
python gapi2.py [-f <file>] [-C] [-P] [-v] [-y <youtube_id>]
```

Arguments:

* `-f <file>`: Read the transcript from the specified file.
* `-C`: Compress the POST request to the Gemini API. (Not working yet)
* `-P`: Use the Gemini Pro model (slower and less quota but only model that creates reasonable timestamps).
* `-v`: Print verbose output.
* `-y <youtube_id>`: Download the transcript for the YouTube video with the specified ID.

Examples:

* Summarize a transcript from a file:

```
python your_script_name.py -f transcript.txt
```

* Summarize a transcript from a YouTube video:

```
python your_script_name.py -y DmgqT9CQSf8
```

* Use the Gemini Pro model:

```
python your_script_name.py -y DmgqT9CQSf8 -P
```

Important Notes:

* Free Tier: The free tier of the Gemini API has limitations on the number of tokens you can use per day. 
If you exceed this limit, you will be charged.
* Rate Limits: The free tier also has rate limits. The script includes a mechanism to mitigate these limits 
by pausing for 61 seconds if the expected token usage exceeds the limit.
* API Key Security: Do not share your API key publicly. Store it securely, ideally in an environment variable or 
a secure configuration file.

This script provides a basic framework for using the Google Gemini API to summarize YouTube transcripts. 
You can modify and extend it to suit your specific needs.
"""



import requests
import argparse
import zlib
import json
import sys
import time
import datetime
import os


LOG_FILE_PATH = "logfile.log"

def log_request():
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{datetime.datetime.now()}\n")

def count_requests_today():
    """
    Counts the number of requests logged today.

    Returns:
        int: The number of requests logged today.
    """
    today = datetime.date.today()
    count = 0
    try:
        with open(LOG_FILE_PATH, "r") as log_file:
            for line in log_file:
                timestamp = datetime.datetime.fromisoformat(line.strip())
                if timestamp.date() == today:
                    count += 1
    except FileNotFoundError:
        print("Log file does not exist. No requests have been logged.")
    return count

# Example usage
print(f"Requests today: {count_requests_today()}")

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
# alternatively you can use -y DmgqT9CQSf8 to download the youtube video transcript with that id
# if -C is given, compress the post request
# if -P is given, use the pro model
# if -v is given, print verbose output

argparser = argparse.ArgumentParser()
argparser.add_argument("-f", "--file", help="file to read")
argparser.add_argument("-C", "--compress", help="compress the post request", action="store_true")
argparser.add_argument("-P", "--pro", help="use the pro model", action="store_true")
argparser.add_argument("-v", "--verbose", help="print verbose output", action="store_true")
argparser.add_argument("-y", "--youtube", help="download youtube video transcript with that id")
args = argparser.parse_args()


start = time.time()

if args.file:
    # If you don't have yt-dlp, then you can click on the description of many
    # youtube videos to get access to the transcript. Scroll to the bottom of the transcript.
    # Select all the text upwards. For best results include the description and title of the video.
    # Store the text in a file and use the -f argument to read the file.
    with open(args.file) as f:
        prompt2 = f.read()
elif args.youtube:
    # don't run the following two commands if {args.youtube}.txt already exists
    if not os.path.exists(f"{args.youtube}.txt"):
        # download the youtube video transcript with that id
        # yt-dlp --write-auto-sub --convert-subs=srt -k --skip-download -o <id> <id>
        # yt-dlp might be violating youtube's terms of service (I'm not sure if it does for downloading subtitles)
        # There is no official youtube API for downloading subtitles for third party transcripts.
        os.system(f"yt-dlp --write-auto-sub --convert-subs=srt -k --skip-download -o {args.youtube} {args.youtube}")
        # now use awk to clean up the srt file (make timestamps less verbose)
        # awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' <id>.en.srt > <id>.txt
        os.system("""awk -F ' --> ' '/^[0-9]+$/{next} NF==2{gsub(",[0-9]+", "", $1); print $1} NF==1' """ + args.youtube + ".en.srt > " + args.youtube + ".txt")

    with open(f"{args.youtube}.txt") as f:
        prompt2 = f.read()
    
else:
    prompt2 = input("Enter prompt2: ")


prompt = "I don't want to watch the video. Create a self-contained bullet list summary from the following transcript that I can understand without watching the video. "

# Documentation of the REST API:
# https://ai.google.dev/api/generate-content#text_gen_text_only_prompt-SHELL

if args.pro:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-exp-0801:generateContent?key=" + api_key
else:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + api_key

if args.verbose:
    # write prompt+prompt2 to a file
    with open("prompt.txt", "w") as f:
        f.write(prompt + prompt2)

# The following categories are available in the safety settings.
# By default all categories are set to BLOCK_SOME.

# HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_SEXUALLY_EXPLICIT, 
# HARM_CATEGORY_DANGEROUS_CONTENT, HARM_CATEGORY_HARASSMENT

safety = [{"category": "HARM_CATEGORY_HATE_SPEECH","threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold": "BLOCK_NONE"},]
data = {"contents": [{"parts": [{"text": prompt + prompt2}]}],
        "generationConfig": {"temperature": 2.0},
        "safetySettings": safety}
data = json.dumps(data)
# compress the post request
if args.compress:
    headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip", "Content-Encoding": "gzip"}
    data = zlib.compress(data.encode('utf-8'))
else:
    headers = {"Content-Type": "application/json"}

if args.pro:
    log_request()


response = requests.post(url, headers=headers,
                         data=data, proxies=proxies)

if response.status_code == 200:
    if args.verbose:
        print("Response received")
        print(response.text)
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

rate_limit_mitigation = 0
expected_tokens = 2*(input_tokens+output_tokens)
if args.pro:
    if expected_tokens > 32_000:
        print(f"Warning: expected tokens {expected_tokens} exceeds 32k limit")
        print("Wait 1 min to prevent rate limit")
        rate_limit_mitigation = 61
        time.sleep(rate_limit_mitigation)
else:
    if expected_tokens > 1_000_000:
        print(f"Warning: expected tokens {expected_tokens} exceeds 1e6 limit")
        print("Wait 1 min to prevent rate limit")
        rate_limit_mitigation = 61
        time.sleep(rate_limit_mitigation)


prompt3 = f"Add starting (not stopping) timestamp to each bullet point in the following summary: {summary}\nThe full transcript is: {prompt2}"

if args.pro:
    log_request()

response2 = requests.post(url, headers=headers,
                            data=json.dumps({"contents": [{"parts": [{"text": prompt3}]}],
                                             "safetySettings": safety}),
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

# Modify the markdown so that it works in a Youtube comment

# replace '**' with '*' in summary2

summary2 = summary2.replace("**", "*")

# replace '*:' with ':*' in summary2

summary2 = summary2.replace("*:", ":*")

# Create the filename with datetime and ID
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"/home/martin/gemini/{timestamp}_{args.youtube}.md"

# Function to write to both stdout and file
def print_and_write(text, file_handle):
    print(text)
    file_handle.write(text + "\n")

# Open the file for writing
with open(filename, "w") as f:
    print_and_write("*Summary*", f)
    print_and_write(summary2, f)  # Assuming 'summary2' holds the summary text
    if args.pro:
        print_and_write("I used Google Gemini 1.5 Pro exp 0801 to summarize the transcript.", f)
    else:
        print_and_write("I used Google Gemini 1.5 Flash to summarize the transcript.", f)

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

    print_and_write(f"Cost (if I didn't use the free tier): ${cost_input+cost_output:.4f}", f)
    print_and_write(f"Time: {end-start:.2f} seconds", f)
    if rate_limit_mitigation > 0:
        print_and_write("I added a 61 second delay to prevent a rate limit of the free tier.", f)
    print_and_write(f"Input tokens: {input_tokens+input_tokens2}", f)
    print_and_write(f"Output tokens: {output_tokens+output_tokens2}", f)
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

    print_and_write(f"Cost (if I didn't use the free tier): ${cost_input+cost_output:.4f}", f)
    print_and_write(f"Time: {end-start:.2f} seconds", f)
    if rate_limit_mitigation > 0:
        print_and_write("I added a 61 second delay to prevent a rate limit of the free tier.", f)
    print_and_write(f"Input tokens: {input_tokens+input_tokens2}", f)
    print_and_write(f"Output tokens: {output_tokens+output_tokens2}", f)
