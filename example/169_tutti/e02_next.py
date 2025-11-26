import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime
import os
from fastcore.utils import BasicRepr, store_attr

# Function provided in your prompt context implies these imports exist in the library
try:
    from gaspard import Client
except ImportError:
    # Fallback/Mock for dry-run if gaspard isn't actually installed
    print("Gaspard library not found. Ensure it is installed via pip.")
    exit(1)


def get_build_id(session, base_url):
    """Fetch the current Next.js build ID from the homepage."""
    # The Build ID is a unique hash (a string of random characters) generated every time the developers at Tutti release a new version of their website.
    print("Fetching build ID...")
    response = session.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")

    if not script:
        raise ValueError(
            "Could not find __NEXT_DATA__ tag. Website structure may have changed."
        )

    data = json.loads(script.string)
    build_id = data.get("buildId")
    print(f"Found Build ID: {build_id}")
    return build_id


def download_tutti_json(pages=2):
    session = requests.Session()
    # Headers are critical to avoid being blocked
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "*/*",
            "x-nextjs-data": "1",  # Hints to the server we want the Next.js data payload
        }
    )
    # 1. Get the dynamic build ID
    # The category path we want to scrape
    slug = ["handys", "Ak6xtb2JpbGUgcGhvbmWqY2VsbFBob25lc5TAwMDA"]
    # In Next.js, a "slug" is the part of the URL that identifies the specific page or category you are looking at.
    # that long string is the Category ID. It describes what you are looking for, not which version of the site you are viewing.

    # It will only change in two specific scenarios:
    # You want to scrape a different category:
    # If you decide you want to search for "Laptops" or "Furniture" instead of phones, you will need to find the specific long string for those categories.
    # Tutti restructures their entire database (Very Rare):
    # This happens maybe once every few years. If they completely rebuild how they categorize items, the IDs might change. But for daily or weekly use, this string is permanent.

    category_slug = f"{slug[0]}/{slug[1]}"
    base_url = f"https://www.tutti.ch/de/q/{category_slug}"

    try:
        build_id = get_build_id(session, base_url)
    except Exception as e:
        print(f"Failed to init: {e}")
        return pd.DataFrame()

    all_devices = []

    # 2. Iterate through pages using the internal JSON endpoint
    for page in range(1, pages + 1):
        # Construct the internal Next.js data URL
        # Format: /_next/data/<BUILD_ID>/<LANG>/q/<SLUG>.json
        json_url = (
            f"https://www.tutti.ch/_next/data/{build_id}/de/q/{category_slug}.json"
        )
        params = {"sorting": "newest", "page": page, "slug": slug}

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
                queries = data["pageProps"]["dehydratedState"]["queries"]
                # Find the query containing listing data (usually the first one)
                listing_query = next(
                    (q for q in queries if "listings" in q["state"]["data"]), None
                )

                if not listing_query:
                    print("No listing data found in JSON.")
                    break

                edges = listing_query["state"]["data"]["listings"]["edges"]

                for edge in edges:
                    node = edge["node"]
                    all_devices.append(
                        {
                            "id": node.get("listingID"),
                            "title": node.get("title"),
                            "price": node.get("formattedPrice"),
                            "location": node.get("postcodeInformation", {}).get(
                                "locationName"
                            ),
                            "zip": node.get("postcodeInformation", {}).get("postcode"),
                            "date": node.get("timestamp"),
                            "description": node.get("body"),
                            "link": f"https://www.tutti.ch/de/vi/{node.get('listingID')}",
                        }
                    )
                print(f"Found {len(edges)} items.")

            except KeyError as e:
                print(f"JSON structure changed, key not found: {e}")

            time.sleep(1)  # Be polite

        except Exception as e:
            print(f"Error on page {page}: {e}")

    return pd.DataFrame(all_devices)


def evaluate_phones(df, max_price=None):
    """
    Filters phones by price and uses Gaspard (Gemini) to rate them as 5G routers.
    """
    if df.empty:
        return df

    # 1. Parse and filter price
    def clean_price(p):
        if not p or not isinstance(p, str):
            return 99999.0
        # Remove '.-' and separators like "'"
        p_clean = p.replace(".-", "").replace("'", "").replace("CHF", "").strip()
        if p_clean.lower() in ["gratis", "free"]:
            return 0.0
        if p_clean.lower() == "auf anfrage":
            return 99999.0
        try:
            return float(p_clean)
        except ValueError:
            return 99999.0

    df["price_numeric"] = df["price"].apply(clean_price)

    if max_price is not None:
        initial_count = len(df)
        df = df[df["price_numeric"] <= max_price].copy()
        print(
            f"Filtered {initial_count} items down to {len(df)} based on price limit ({max_price})."
        )

    if df.empty:
        print("No items left after price filtering.")
        return df

    # 2. Setup Gaspard
    if "GEMINI_API_KEY" not in os.environ:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    # Using the flash-exp model as suggested in docs for speed/free tier
    model_name = "gemini-flash-latest"
    print(f"Initializing Gaspard with model: {model_name}")
    cli = Client(model_name)

    # Structured Output Class matching Gaspard/fastcore syntax
    class PhoneRating(BasicRepr):
        "Evaluation of a phone's suitability as a static 5G router."

        def __init__(
            self,
            score: int,  # Rating 0-100 (100 is perfect match)
            reasoning: str,  # Brief explanation of the rating
            has_5g: bool,  # Does it clearly support 5G?
            screen_condition: str,  # Brief note on screen (e.g. "Cracked", "OK", "Unknown")
        ):
            store_attr()

    results = []
    print("Scoring listings with AI...")

    for idx, row in df.iterrows():
        # Context for the LLM
        item_text = (
            f"Title: {row['title']}\n"
            f"Price: {row['price']}\n"
            f"Description: {row['description']}"
        )

        prompt = (
            "I need a phone to use as a permanently plugged-in 5G router. "
            "Requirements:\n"
            "- MUST have 5G (fast preferred).\n"
            "- Display/Touch must work (scratches fine, cracks bad/risk of ghost touch).\n"
            "- Camera status irrelevant.\n"
            "- Battery health irrelevant (plugged in).\n\n"
            "Assess this listing:"
            f"\n---\n{item_text}\n---"
        )

        try:
            # Call Gaspard structured output
            # Returns a list of objects, we take the first
            res_objs = cli.structured(prompt, PhoneRating)
            if res_objs:
                rating = res_objs[0]
                results.append(
                    {
                        "score": rating.score,
                        "ai_reasoning": rating.reasoning,
                        "ai_has_5g": rating.has_5g,
                        "ai_screen": rating.screen_condition,
                    }
                )
            else:
                results.append(
                    {
                        "score": 0,
                        "ai_reasoning": "No response",
                        "ai_has_5g": False,
                        "ai_screen": "?",
                    }
                )

        except Exception as e:
            print(f"AI Error on row {idx}: {e}")
            results.append(
                {
                    "score": -1,
                    "ai_reasoning": f"Error: {e}",
                    "ai_has_5g": False,
                    "ai_screen": "?",
                }
            )
            time.sleep(2)  # Backoff on error

    # 3. Merge and Sort
    metrics = pd.DataFrame(results)
    # Align indices just in case
    df = df.reset_index(drop=True)
    final_df = pd.concat([df, metrics], axis=1)

    # Sort by score descending
    final_df.sort_values(by="score", ascending=False, inplace=True)
    return final_df


if __name__ == "__main__":
    # 1. Scrape
    df = download_tutti_json(pages=22)  # Adjust pages as needed

    if df is not None and not df.empty:
        print(f"\nExtracted {len(df)} items. Starting AI evaluation...")

        # 2. Filter and Rate (Max price, e.g. 120.-, for a router seems reasonable)
        scored_df = evaluate_phones(df, max_price=120.0)

        # 3. Display Top Results
        print("\n=== Top 5 Recommendations ===")
        cols = [
            "score",
            "title",
            "price",
            "ai_has_5g",
            "ai_screen",
            "ai_reasoning",
            "link",
        ]
        # Handle case where AI failed for all
        if "score" in scored_df.columns:
            pd.set_option("display.max_colwidth", 100)
            print(scored_df[cols].head(5).to_string(index=False))

            # 4. Save
            dt = datetime.datetime.now().isoformat().replace(":", "-")
            fn = f"tutti_router_candidates_{dt}.csv"
            scored_df.to_csv(fn, index=False)
            print(f"\nFull results saved to {fn}")
        else:
            print("Scoring failed or returned no data.")
    else:
        print("No listings found to evaluate.")

# >>> df
#           id                                              title    price                location   zip                       date                                        description                                 link
# 0   72023485                     Schöne Hülle für iPhone 16 Pro     10.-                    Lyss  3250  2025-11-25T07:50:43+01:00                       Inkl. Porto, TWINT moeglich.  https://www.tutti.ch/de/vi/72023485
# 1   76543915                               Iphone 14 pro 256 gb    499.-                 Locarno  6600  2025-11-25T07:44:19+01:00  Da tip top natel locarno Telefono in buono sta...  https://www.tutti.ch/de/vi/76543915
# 2   77691845  Samsung Galaxy S20 Ultra 5G Neuwertig mit Sams...    200.-           Bremgarten AG  5620  2025-11-25T07:38:56+01:00  SAMSUNG Galaxy S20 Ultra 5G 128GB, Pearl White...  https://www.tutti.ch/de/vi/77691845

# From the dataframe a prompt for an LLM containing listings like this for each phone
# Use a function that allows setting and upper limit on the price
# Idx=1 Titel="Iphone 14 pro 256 gb" Price=499 Description="Da tip top natel ..."


# export GEMINI_API_KEY=`cat ~/api_key.txt` ;  uv run python -i e02_next.py
