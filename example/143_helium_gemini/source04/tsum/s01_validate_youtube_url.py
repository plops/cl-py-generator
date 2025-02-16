#!/usr/bin/env python3
import re
def validate_youtube_url(url):
    """Validates various YouTube URL formats. Returns normalized YouTube video identifier as a string where unneccessary information has been removed. False if the Link doesn't match acceptable patterns."""
    patterns=[r"""^https://(www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11}).*""", r"""^https://(www\.)?youtube\.com/live/([A-Za-z0-9_-]{11}).*""", r"""^https://(www\.)?youtu\.be/([A-Za-z0-9_-]{11}).*"""]
    for pattern in patterns:
        match=re.match(pattern, url)
        if ( match ):
            return match.groups()[1]
    print("Error: Invalid YouTube URL")
    return False