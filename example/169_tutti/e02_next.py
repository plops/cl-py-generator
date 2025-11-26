import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_build_id(session, base_url):
    """Fetch the current Next.js build ID from the homepage."""
    print("Fetching build ID...")
    response = session.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    script = soup.find('script', id='__NEXT_DATA__')

    if not script:
        raise ValueError("Could not find __NEXT_DATA__ tag. Website structure may have changed.")

    data = json.loads(script.string)
    build_id = data.get('buildId')
    print(f"Found Build ID: {build_id}")
    return build_id

def download_tutti_json(pages=2):
    session = requests.Session()
    # Headers are critical to avoid being blocked
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'x-nextjs-data': '1' # Hints to the server we want the Next.js data payload
    })

    # 1. Get the dynamic build ID
    # The category path we want to scrape
    category_slug = "handys/Ak8CqY2VsbFBob25lc5TAwMDA"
    base_url = f"https://www.tutti.ch/de/q/{category_slug}"

    try:
        build_id = get_build_id(session, base_url)
    except Exception as e:
        print(f"Failed to init: {e}")
        return

    all_devices = []

    # 2. Iterate through pages using the internal JSON endpoint
    for page in range(1, pages + 1):
        # Construct the internal Next.js data URL
        # Format: /_next/data/<BUILD_ID>/<LANG>/q/<SLUG>.json
        json_url = f"https://www.tutti.ch/_next/data/{build_id}/de/q/{category_slug}.json"

        params = {
            'sorting': 'newest',
            'page': page,
            # The slug parameters are sometimes required in the query string by Next.js routers
            'slug': ['handys', 'Ak8CqY2VsbFBob25lc5TAwMDA']
        }

        print(f"Downloading JSON for page {page}...")

        try:
            resp = session.get(json_url, params=params)

            if resp.status_code != 200:
                print(f"Failed to fetch page {page} (Status: {resp.status_code})")
                continue

            data = resp.json()

            # Navigate the JSON path to find listings
            # Path: pageProps -> dehydratedState -> queries -> [0] -> state -> data -> listings -> edges
            try:
                queries = data['pageProps']['dehydratedState']['queries']
                # Find the query containing listing data (usually the first one)
                listing_query = next((q for q in queries if 'listings' in q['state']['data']), None)

                if not listing_query:
                    print("No listing data found in JSON.")
                    break

                edges = listing_query['state']['data']['listings']['edges']

                for edge in edges:
                    node = edge['node']
                    all_devices.append({
                        'id': node.get('listingID'),
                        'title': node.get('title'),
                        'price': node.get('formattedPrice'),
                        'location': node.get('postcodeInformation', {}).get('locationName'),
                        'zip': node.get('postcodeInformation', {}).get('postcode'),
                        'date': node.get('timestamp'),
                        'description': node.get('body'),
                        'link': f"https://www.tutti.ch/de/vi/{node.get('listingID')}"
                    })

                print(f"Found {len(edges)} items.")

            except KeyError as e:
                print(f"JSON structure changed, key not found: {e}")

            time.sleep(1) # Be polite

        except Exception as e:
            print(f"Error on page {page}: {e}")

    return pd.DataFrame(all_devices)

if __name__ == "__main__":
    df = download_tutti_json(2)
    if df is not None and not df.empty:
        print(f"\nTotal items extracted: {len(df)}")
        print(df[['title', 'price', 'date']].head())
        # df.to_csv('tutti_data_clean.csv', index=False)
    else:
        print("No data found.")


# >>> df
#           id                                              title    price                location   zip                       date                                        description                                 link
# 0   72023485                     Schöne Hülle für iPhone 16 Pro     10.-                    Lyss  3250  2025-11-25T07:50:43+01:00                       Inkl. Porto, TWINT moeglich.  https://www.tutti.ch/de/vi/72023485
# 1   76543915                               Iphone 14 pro 256 gb    499.-                 Locarno  6600  2025-11-25T07:44:19+01:00  Da tip top natel locarno Telefono in buono sta...  https://www.tutti.ch/de/vi/76543915
# 2   77691845  Samsung Galaxy S20 Ultra 5G Neuwertig mit Sams...    200.-           Bremgarten AG  5620  2025-11-25T07:38:56+01:00  SAMSUNG Galaxy S20 Ultra 5G 128GB, Pearl White...  https://www.tutti.ch/de/vi/77691845
