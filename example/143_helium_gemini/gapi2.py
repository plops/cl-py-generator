#!/usr/bin/env python3
import requests

# Set up the proxy settings
proxies = {
    "http": "socks5://localhost:8080",
    "https": "socks5://localhost:8080",
}

# load api key from file
with open("/home/martin/api_key.txt") as f:
    api_key = f.read().strip()

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + api_key
headers = {"Content-Type": "application/json"}
data = {"contents": [{"parts": [{"text": "Explain how AI works"}]}]}

response = requests.post(url, headers=headers, json=data, proxies=proxies)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)