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

try:
    from gaspard import Client
except ImportError:
    print("Gaspard library not found. Ensure it is installed via pip.")
    exit(1)


def get_build_id(session, base_url):
    """Fetch the current Next.js build ID from the homepage."""
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


def download_tutti_json(pages=2, category="phones"):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "*/*",
            "x-nextjs-data": "1",
        }
    )

    # Configuration based on category
    if category == "phones":
        # Slug: handys / ID
        slug_path = ["handys", "Ak8CqY2VsbFBob25lc5TAwMDA"]
        query_params = {}
    elif category == "watches":
        # Slug: suche / ID (from your HTML dump)
        slug_path = ["suche", "Ak6thcHBsZSB3YXRjaMCUwMDAwA"]
        query_params = {"query": "apple watch"}
    else:
        raise ValueError("Unknown category")

    category_slug_url = f"{slug_path[0]}/{slug_path[1]}"
    base_url = f"https://www.tutti.ch/de/q/{category_slug_url}"

    try:
        build_id = get_build_id(session, "https://www.tutti.ch/de")
    except Exception as e:
        logger.error(f"Failed to init: {e}")
        return pd.DataFrame()

    all_items = []

    for page in range(1, pages + 1):
        # Next.js internal data URL
        json_url = (
            f"https://www.tutti.ch/_next/data/{build_id}/de/q/{category_slug_url}.json"
        )

        # Merge basic params with category specific ones (like query=apple+watch)
        params = {"sorting": "newest", "page": page, "slug": slug_path}
        params.update(query_params)

        logger.info(f"Downloading JSON for page {page} ({category})...")
        try:
            resp = session.get(json_url, params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch page {page} (Status: {resp.status_code})"
                )
                continue

            data = resp.json()

            try:
                # Path traversal to find listings
                queries = data["pageProps"]["dehydratedState"]["queries"]
                listing_query = next(
                    (q for q in queries if "listings" in q["state"]["data"]), None
                )

                if not listing_query:
                    logger.warning("No listing data found in JSON.")
                    break

                edges = listing_query["state"]["data"]["listings"]["edges"]

                for edge in edges:
                    node = edge["node"]
                    all_items.append(
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
                # Stop if no items returned for this page
                if len(edges) == 0:
                    logger.info("No items returned. Stopping pagination.")
                    break
            except KeyError as e:
                logger.error(f"JSON structure changed, key not found: {e}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error on page {page}: {e}")
    return pd.DataFrame(all_items)


def find_latest_candidates_csv(pattern="tutti_*.csv"):
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found.")
    latest = max(files, key=os.path.getmtime)
    logger.info(f"Using cached CSV: {latest}")
    return latest


def evaluate_items(df, category="phones", min_price=None, max_price=None, skip_scored=True):
    """
    Filters items by price and uses Gaspard (Gemini) to rate them based on category logic.
    """
    if df.empty:
        return df

    # 1. Parse and filter price
    def clean_price(p):
        if not p or not isinstance(p, str):
            return 99999.0
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

    if min_price is not None or max_price is not None:
        initial_count = len(df)
        if min_price is not None:
            df = df[df["price_numeric"] >= min_price].copy()
        if max_price is not None:
            df = df[df["price_numeric"] <= max_price].copy()
        logger.info(
            f"Filtered {initial_count} items down to {len(df)} based on price."
        )

    if df.empty:
        return df

    if "score" in df.columns and skip_scored:
        to_eval_mask = df["score"].fillna(-1) <= 0
    else:
        to_eval_mask = pd.Series([True] * len(df), index=df.index)

    to_eval_df = df[to_eval_mask].copy()
    if to_eval_df.empty:
        logger.info("No items to evaluate.")
        return df

    # --- 2. Setup Gaspard ---
    if "GEMINI_API_KEY" not in os.environ:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    model_name = "gemini-flash-latest"
    logger.info(f"Initializing Gaspard with model: {model_name}")
    cli = Client(model_name)

    # Generalized Class
    class ItemRating(BasicRepr):
        "Evaluation of an item's suitability."
        def __init__(
                self,
                listing_id: str,
                score: int,  # 0-100
                reasoning: str,
                key_feature: bool,  # Phone: 5G? Watch: Cellular?
                condition_rating: str, # "Excellent", "Good", "Bad"
        ):
            store_attr()

    # --- 3. Prompt Construction ---
    if category == "phones":
        system_prompt = (
            "I need phones to use as permanently plugged-in 5G routers. Requirements:\n"
            "- MUST have 5G (fast preferred).\n"
            "- Display/Touch must work (scratches fine, cracks bad).\n"
            "- If USB-C to HDMI works, screen state matters less.\n"
            "- Prefer Pixel over iPhone, then LineageOS supported phones.\n"
            "- key_feature = True if it definitely has 5G."
        )
    elif category == "watches":
        system_prompt = (
            "I need an Apple Watch. Requirements:\n"
            "- Prefer newer Series (Ultra > 10 > 9 > 8 > 7 > SE).\n"
            "- I want to give watches to my grandmas so that her health is monitored and we know when she falls.\n"
            "- If a seller sells two or more, that is an advantage.\n"
            "- Screen must be readable. Scratches ok if price is low, cracks are bad.\n"
            "- key_feature = True if it has Cellular/LTE support.\n"
            "- Battery health is important.\n"
            "- Accessories alone (bands/chargers) get score 0.\n"
            "- Ignore non-Apple watches."
        )
    else:
        system_prompt = "Evaluate these items for resale value."

    # --- 4. Batch Processing ---
    results_map = {}
    BATCH_SIZE = 3000 # Keep small for context window
    DELAY_SECONDS = 5

    listings_data = to_eval_df.to_dict("records")
    total_batches = math.ceil(len(listings_data) / BATCH_SIZE)
    logger.info(f"Starting AI evaluation in {total_batches} batches...")

    for i in range(0, len(listings_data), BATCH_SIZE):
        batch = listings_data[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        batch_text = ""
        for item in batch:
            batch_text += (
                f"--- ITEM {item['id']} ---\n"
                f"Title: {item['title']}\n"
                f"Price: {item['price']}\n"
                f"Description: {item['description']}\n"
                f"------------------\n"
            )

        final_prompt = (
            f"{system_prompt}\n"
            "Evaluate ALL listings below individually.\n"
            f"{batch_text}"
        )

        try:
            logger.info(f"Processing batch {batch_num}/{total_batches}...")
            responses = cli.structured(final_prompt, ItemRating)

            if responses:
                for r in responses:
                    r_id = str(r.listing_id)
                    results_map[r_id] = {
                        "score": r.score,
                        "ai_reasoning": r.reasoning,
                        "ai_key_feature": r.key_feature,
                        "ai_condition": r.condition_rating,
                    }
            else:
                logger.warning(f"Batch {batch_num} returned no data.")
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            if "429" in str(e):
                time.sleep(60)
            else:
                time.sleep(2)

        if batch_num < total_batches:
            time.sleep(DELAY_SECONDS)

    # --- 5. Merge ---
    ai_cols_list = []
    for idx, row in df.iterrows():
        lid = str(row["id"])
        if lid in results_map:
            ai_cols_list.append(results_map[lid])
        else:
            ai_cols_list.append(
                {
                    "score": row.get("score", -1),
                    "ai_reasoning": row.get("ai_reasoning", "Not evaluated"),
                    "ai_key_feature": row.get("ai_key_feature", False),
                    "ai_condition": row.get("ai_condition", "?"),
                }
            )

    metrics_df = pd.DataFrame(ai_cols_list, index=df.index)
    final_df = df.reset_index(drop=True).copy()
    for c in metrics_df.columns:
        final_df[c] = metrics_df[c].values

    final_df["score"] = pd.to_numeric(final_df["score"], errors="coerce").fillna(-1)
    final_df.sort_values(by="score", ascending=False, inplace=True)
    return final_df


def _timestamp_filename(prefix: str, ext: str) -> str:
    dt = datetime.datetime.now().isoformat().replace(":", "-").split(".")[0]
    return f"{prefix}_{dt}.{ext}"


def main():
    parser = argparse.ArgumentParser(description="Fetch and score Tutti listings.")
    parser.add_argument("-p", "--pages", type=int, default=5, help="Pages to download.")
    parser.add_argument("-c", "--category", type=str, default="phones", choices=["phones", "watches"], help="Category: phones or watches")
    parser.add_argument("-m", "--min-price", type=float, default=7.0)
    parser.add_argument("-M", "--max-price", type=float, default=300.0)
    parser.add_argument("-l", "--log-file", type=str)

    args = parser.parse_args()

    # Loguru config
    logger.remove()
    logger.add(lambda m: print(m, end=""), format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
    if args.log_file:
        logger.add(args.log_file, rotation="10 MB")

    if args.pages == 0:
        # Dry run or resume not fully implemented for mixed categories in this snippet
        # assuming user generates fresh data mostly
        csv_path = find_latest_candidates_csv(f"tutti_{args.category}_*.csv")
        df = pd.read_csv(csv_path)
    else:
        df = download_tutti_json(pages=args.pages, category=args.category)
        if not df.empty:
            fn = _timestamp_filename(f"tutti_{args.category}_raw", "csv")
            df.to_csv(fn, index=False)
            logger.info(f"Saved raw data to {fn}")

    if df is not None and not df.empty:
        logger.info(f"Evaluating {len(df)} items for {args.category}...")
        scored_df = evaluate_items(
            df, category=args.category, min_price=args.min_price, max_price=args.max_price
        )

        logger.info("=== Top Recommendations ===")
        cols = ["score", "title", "price", "ai_key_feature", "ai_reasoning", "link"]

        if "score" in scored_df.columns:
            pd.set_option("display.max_colwidth", 100)
            valid = scored_df[scored_df["score"] > 0]
            if not valid.empty:
                print(valid[cols].head(10).to_string(index=False))
                fn = _timestamp_filename(f"tutti_{args.category}_candidates", "csv")
                scored_df.to_csv(fn, index=False)
                logger.success(f"Saved results to {fn}")
            else:
                logger.warning("No valid results found.")
    else:
        logger.warning("No listings found.")

if __name__ == "__main__":
    main()