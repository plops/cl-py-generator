# the following url is a search request for recent phones:
# https://www.tutti.ch/de/q/handys/Ak8CqY2VsbFBob25lc5TAwMDA?sorting=newest&page=1
# download the first two pages

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

num_pages=2
"""Download phone listings from tutti.ch"""
base_url = "https://www.tutti.ch/de/q/handys/Ak8CqY2VsbFBob25lc5TAwMDA"

all_listings = []

for page in range(1, num_pages + 1):
    params = {
        'sorting': 'newest',
        'page': page
    }

    try:
        print(f"Downloading page {page}...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Store the HTML content or parse specific elements
        all_listings.append({
            'page': page,
            'url': response.url,
            'content': soup
        })

        print(f"Successfully downloaded page {page}")

        # Be polite to the server
        if page < num_pages:
            time.sleep(1)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading page {page}: {e}")

df = pd.DataFrame(all_listings)
df.to_csv("e01_phones.csv", index=False)





# create a folder with datetime and store the html files there
import os
from datetime import datetime
folder_name = datetime.now().strftime("phones_%Y%m%d_%H%M%S")
os.makedirs(folder_name, exist_ok=True)
for listing in all_listings:
    page = listing['page']
    content = listing['content']
    with open(os.path.join(folder_name, f"page_{page}.html"), "w", encoding="utf-8") as f:
        f.write(str(content))
print(f"HTML files saved in folder: {folder_name}")
print("All done.")