#!/usr/bin/env python3
import re
def validate_youtube_url(url):
    """Validates various YouTube URL formats."""
    patterns=[r"""^https://(www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]{11}.*""", r"""^https://(www\.)?youtube\.com/live/[A-Za-z0-9_-]{11}.*""", r"""^https://(www\.)?youtu\.be/[A-Za-z0-9_-]{11}.*"""]
    for pattern in patterns:
        if ( re.match(pattern, url) ):
            return True
    print("Error: Invalid YouTube URL")
    return False