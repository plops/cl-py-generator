import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import time
import datetime
import os
import math
from fastcore.utils import BasicRepr, store_attr
import argparse
from glob import glob
from loguru import logger

# Function provided in your prompt context implies these imports exist in the library
try:
    from gaspard import Client
except ImportError:
    # Fallback/Mock for dry-run if gaspard isn't actually installed
    # replace print with logger after logger config in main
    # but keep early exit here
    print("Gaspard library not found. Ensure it is installed via pip.")
    exit(1)


def get_build_id(session, base_url):
    """Fetch the current Next.js build ID from the homepage."""
    # The Build ID is a unique hash (a string of random characters) generated every time the developers at Tutti release a new version of their website.
    logger.info("Fetching build ID...")
    response = session.get(base_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")

    if not script:
        raise ValueError(
            "Could not find __NEXT_DATA__ tag. Website structure may have changed."
        )

    data = json.loads(script.string)
    build_id = data.get("buildId")
    logger.info(f"Found Build ID: {build_id}")
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
    slug = ["handys", "Ak8CqY2VsbFBob25lc5TAwMDA"]
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
        logger.error(f"Failed to init: {e}")
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

        logger.info(f"Downloading JSON for page {page}...")
        try:
            resp = session.get(json_url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch page {page} (Status: {resp.status_code})"
                )
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
                    logger.warning("No listing data found in JSON.")
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
                logger.info(f"Found {len(edges)} items.")
            except KeyError as e:
                logger.error(f"JSON structure changed, key not found: {e}")
            time.sleep(1)  # Be polite
        except Exception as e:
            logger.error(f"Error on page {page}: {e}")
    return pd.DataFrame(all_devices)


def find_latest_candidates_csv(pattern="tutti_router_candidates_*.csv"):
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found.")
    latest = max(files, key=os.path.getmtime)
    logger.info(f"Using cached CSV: {latest}")
    return latest


# python
def evaluate_phones(df, min_price=None, max_price=None, skip_scored=True):
    """
    Filters phones by price and uses Gaspard (Gemini) to rate them as 5G routers.
    If skip_scored is True (default), entries with existing score > 0 are not re-submitted.
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

    df = df.copy()
    df["price_numeric"] = df["price"].apply(clean_price)

    # Apply min_price and/or max_price filters if provided
    if min_price is not None or max_price is not None:
        initial_count = len(df)
        if min_price is not None:
            df = df[df["price_numeric"] >= min_price].copy()
        if max_price is not None:
            df = df[df["price_numeric"] <= max_price].copy()
        logger.info(
            f"Filtered {initial_count} items down to {len(df)} based on price limits (min={min_price}, max={max_price})."
        )

    if df.empty:
        logger.info("No items left after price filtering.")
        return df

    # Determine which rows to evaluate
    if "score" in df.columns and skip_scored:
        to_eval_mask = df["score"].fillna(-1) <= 0
    else:
        to_eval_mask = pd.Series([True] * len(df), index=df.index)

    to_eval_df = df[to_eval_mask].copy()
    if to_eval_df.empty:
        logger.info(
            "No items to evaluate (all already scored). Returning original dataframe."
        )
        return df

    # --- 2. Setup Gaspard ---
    if "GEMINI_API_KEY" not in os.environ:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    model_name = "gemini-flash-latest"
    logger.info(f"Initializing Gaspard with model: {model_name}")
    cli = Client(model_name)

    # Structured Output Class matching Gaspard/fastcore syntax
    class PhoneRating(BasicRepr):
        "Evaluation of a phone's suitability as a static 5G router."

        def __init__(
            self,
            listing_id: str,  # The ID of the listing being evaluated
            score: int,  # Rating 0-100 (100 is perfect match)
            reasoning: str,  # Brief explanation
            has_5g: bool,  # Does it clearly support 5G?
            screen_condition: str,  # e.g. "Cracked", "OK"
        ):
            store_attr()

    # --- 3. Batch Processing ---
    results_map = {}  # Store results by listing ID

    # Reduce batch size to avoid token/function-call issues
    BATCH_SIZE = 1500
    # Delay: 10 seconds between requests ensures we stay under 10 RPM (60s / 10 = 6s minimum)
    DELAY_SECONDS = 10

    listings_data = to_eval_df.to_dict("records")
    total_batches = math.ceil(len(listings_data) / BATCH_SIZE)
    logger.info(f"Starting AI evaluation in {total_batches} batches...")

    for i in range(0, len(listings_data), BATCH_SIZE):
        batch = listings_data[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        # Construct a combined prompt for the batch
        batch_text = ""
        for item in batch:
            batch_text += (
                f"--- ITEM START ---\n"
                # f"ID: {item['id']}\n"
                f"Title: {item['title']}\n"
                f"Price: {item['price']}\n"
                f"Description: {item['description']}\n"
                f"--- ITEM END ---\n\n"
            )

        prompt = (
            "I need phones to use as permanently plugged-in 5G routers. "
            "Requirements:\n"
            "- MUST have 5G (fast preferred).\n"
            "- Display/Touch must work (scratches fine, cracks bad/risk of ghost touch).\n"
            "- Camera/Battery health irrelevant.\n\n"
            "- Prefer Pixel over iPhone, then any phone that is supported by lineageos then samsung, then others."
            "Evaluate ALL of the following listings evaluating each one individually. "
            "Output devices that match these criteria and show the most price-effective and promising phones at the top."
            f"{batch_text}"
        )

        # store prompt to file with iso datetime for debugging
        dt = datetime.datetime.now().isoformat().replace(":", "-")
        debug_fn = f"debug_prompt_batch_{batch_num}_{dt}.txt"
        with open(debug_fn, "w", encoding="utf-8") as f:
            f.write(prompt)
        logger.info(f"Saved debug prompt for batch {batch_num} to {debug_fn}")

        try:
            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)..."
            )

            # structured() returns a list of tool outputs
            responses = cli.structured(prompt, PhoneRating)

            # Map back to IDs
            if responses:
                for r in responses:
                    # Normalize ID to string to ensure matching works
                    r_id = str(r.listing_id)
                    results_map[r_id] = {
                        "score": r.score,
                        "ai_reasoning": r.reasoning,
                        "ai_has_5g": r.has_5g,
                        "ai_screen": r.screen_condition,
                    }
                logger.success(
                    f"Batch {batch_num} completed: {len(responses)} evaluations."
                )
            else:
                logger.warning(f"Batch {batch_num} returned no structured data.")
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            # If we hit a quota error here, we might want to sleep longer
            if "429" in str(e):
                logger.warning("Quota hit. Sleeping for 60 seconds...")
                time.sleep(60)
            else:
                time.sleep(2)

        # Rate limiting delay
        if batch_num < total_batches:
            time.sleep(DELAY_SECONDS)

    # --- 4. Merge Results back to DataFrame (assign/overwrite columns directly) ---
    ai_cols_list = []
    for idx, row in df.iterrows():
        lid = str(row["id"])
        # If we evaluated this row, use the AI result
        if lid in results_map:
            ai_cols_list.append(results_map[lid])
        else:
            # Keep existing scoring/AI columns if present, otherwise defaults
            ai_cols_list.append(
                {
                    "score": row.get("score", -1),
                    "ai_reasoning": row.get("ai_reasoning", "Not evaluated / Error"),
                    "ai_has_5g": row.get("ai_has_5g", False),
                    "ai_screen": row.get("ai_screen", "?"),
                }
            )

    metrics_df = pd.DataFrame(ai_cols_list, index=df.index)

    final_df = df.reset_index(drop=True).copy()
    # Overwrite or create the AI columns (this avoids duplicate column labels)
    for c in metrics_df.columns:
        final_df[c] = metrics_df[c].values

    # Ensure single 'score' column exists and is numeric for sorting
    final_df["score"] = pd.to_numeric(final_df["score"], errors="coerce").fillna(-1)

    final_df.sort_values(by="score", ascending=False, inplace=True)
    return final_df


def _timestamp_filename(prefix: str, ext: str) -> str:
    dt = datetime.datetime.now().isoformat().replace(":", "-")
    return f"{prefix}_{dt}.{ext}"


def main():
    parser = argparse.ArgumentParser(description="Fetch and score Tutti listings.")
    parser.add_argument(
        "-p",
        "--pages",
        type=int,
        default=100,
        help="Number of Tutti pages to download (0 = reuse latest candidates CSV).",
    )
    parser.add_argument(
        "-m",
        "--min-price",
        type=float,
        default=7.0,
        help="Minimum price (CHF) to include when evaluating listings.",
    )
    parser.add_argument(
        "-M",
        "--max-price",
        type=float,
        default=127.0,
        help="Maximum price (CHF) to include when evaluating listings.",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        type=str,
        help="Optional log file path. If provided, logs will also be written to this file.",
    )
    args = parser.parse_args()

    # Configure loguru with timestamp
    logger.remove()
    logger.add(
        lambda m: print(m, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    if args.log_file:
        logger.add(
            args.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB",
        )
        logger.info(f"Logging to file: {args.log_file}")

    if args.pages == 0:
        csv_path = find_latest_candidates_csv()
        df = pd.read_csv(csv_path)
    else:
        df = download_tutti_json(pages=args.pages)
        fn = _timestamp_filename("tutti_phones", "csv")
        df.to_csv(fn, index=False)
        logger.info(f"Raw data saved to {fn}")

    if df is not None and not df.empty:
        logger.info(f"Extracted {len(df)} items. Starting AI evaluation...")
        scored_df = evaluate_phones(
            df, min_price=args.min_price, max_price=args.max_price
        )

        logger.info("=== Top Recommendations ===")
        cols = ["score", "title", "price", "ai_has_5g", "ai_reasoning", "link"]

        if "score" in scored_df.columns:
            pd.set_option("display.max_colwidth", 100)
            valid_results = scored_df[scored_df["score"] > 0]
            if not valid_results.empty:
                print(valid_results[cols].head(10).to_string(index=False))
                fn = _timestamp_filename("tutti_router_candidates", "csv")
                scored_df.to_csv(fn, index=False)
                logger.success(f"Full results saved to {fn}")
            else:
                logger.warning("No valid results with score > 0.")
        else:
            logger.error("Scoring failed.")
    else:
        logger.warning("No listings found to evaluate.")


# >>> df
#           id                                              title    price                location   zip                       date                                        description                                 link
# 0   72023485                     Schöne Hülle für iPhone 16 Pro     10.-                    Lyss  3250  2025-11-25T07:50:43+01:00                       Inkl. Porto, TWINT moeglich.  https://www.tutti.ch/de/vi/72023485
# 1   76543915                               Iphone 14 pro 256 gb    499.-                 Locarno  6600  2025-11-25T07:44:19+01:00  Da tip top natel locarno Telefono in buono sta...  https://www.tutti.ch/de/vi/76543915
# 2   77691845  Samsung Galaxy S20 Ultra 5G Neuwertig mit Sams...    200.-           Bremgarten AG  5620  2025-11-25T07:38:56+01:00  SAMSUNG Galaxy S20 Ultra 5G 128GB, Pearl White...  https://www.tutti.ch/de/vi/77691845

# From the dataframe a prompt for an LLM containing listings like this for each phone
# Use a function that allows setting and upper limit on the price
# Idx=1 Titel="Iphone 14 pro 256 gb" Price=499 Description="Da tip top natel ..."


# export GEMINI_API_KEY=`cat ~/api_key.txt` ;  uv run python -i e02_next.py

if __name__ == "__main__":
    main()
