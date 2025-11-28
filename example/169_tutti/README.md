

**Tutti 5G Router Scout**

This tool automates the search for cheap smartphones on Tutti.ch to repurpose as dedicated 5G routers. Instead of fragile HTML parsing, it extracts the Next.js build ID to query the internal JSON API directly.

It filters listings by price and pipes the data to Google Gemini (via the `gaspard` library) to evaluate technical suitability. The AI scores devices based on:
*   **5G Connectivity:** Mandatory requirement.
*   **OS Support:** Preference for Pixel and Samsung devices (LineageOS compatibility).
*   **Condition:** Ignores cosmetic damage or battery health; filters out broken touchscreens.

Results are saved to CSV with a viability score (0-100) and reasoning.
