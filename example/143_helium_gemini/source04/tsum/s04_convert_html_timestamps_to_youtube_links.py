from urllib.parse import urlencode
import re
from s01_validate_youtube_url import validate_youtube_url

def _youtube_url_with_t(video_id: str, seconds: int) -> str:
    """Return canonical youtube URL with t=<seconds>s parameter added."""
    qs = {'v': video_id, 't': f"{int(seconds)}s"}
    return "https://www.youtube.com/watch?" + urlencode(qs)

def replace_timestamps_in_html(html: str, youtube_url: str) -> str:
    """
    Replace timestamps in HTML (mm:ss or hh:mm:ss) with links to youtube_url at that timestamp.
    If the provided URL is not a valid YouTube URL, do NOT modify the HTML and return it unchanged.
    """
    # Normalize URL to video id (this removes any existing t or other params)
    video_id = validate_youtube_url(youtube_url)
    if not video_id:
        # Do not modify timestamps if the URL isn't a recognized YouTube URL
        return html

    # Match mm:ss or hh:mm:ss where mm and ss are 0-59, hours optional 1-2 digits.
    pattern = re.compile(r'\b(?:\d{1,2}:)?[0-5]?\d:[0-5]\d\b')

    def repl(m):
        ts_text = m.group(0)
        parts = ts_text.split(':')
        if len(parts) == 3:
            h, mm, ss = (int(parts[0]), int(parts[1]), int(parts[2]))
            total = h * 3600 + mm * 60 + ss
        else:
            mm, ss = (int(parts[0]), int(parts[1]))
            total = mm * 60 + ss
        link = _youtube_url_with_t(video_id, total)
        return f'<a href="{link}">{ts_text}</a>'

    return pattern.sub(repl, html)
