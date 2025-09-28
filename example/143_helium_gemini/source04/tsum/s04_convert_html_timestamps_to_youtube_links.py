from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import re

def _youtube_url_with_t(youtube_url: str, seconds: int) -> str:
    """Return youtube_url with t=<seconds>s parameter added or replaced."""
    parsed = urlparse(youtube_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    # YouTube accepts 't' as seconds with trailing 's'
    qs['t'] = f"{int(seconds)}s"
    new_query = urlencode(qs, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)

def replace_timestamps_in_html(html: str, youtube_url: str) -> str:
    """
    Replace timestamps in HTML (mm:ss or hh:mm:ss) with links to youtube_url at that timestamp.
    Example: "00:14:58" -> <a href="...&t=898s">00:14:58</a>
    """
    # Match either H:MM:SS or MM:SS where hours are optional; allow 1-2 digits for fields
    pattern = re.compile(r'\b(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)\b')

    def _to_seconds(match):
        hours = match.group(1)
        mins = match.group(2)
        secs = match.group(3)
        h = int(hours[:-1]) if hours else 0  # hours group includes trailing ':' in this regex, remove it
        # Note: the regex captures hours without trailing colon, so above strip is not needed; safe fallback
        try:
            # If hours captured as e.g. "01" (no colon), convert directly
            h = int(hours) if hours else 0
        except Exception:
            h = 0
        m = int(mins)
        s = int(secs)
        return h * 3600 + m * 60 + s

    def repl(m):
        ts_text = m.group(0)
        # compute seconds robustly handling presence/absence of hours
        parts = ts_text.split(':')
        if len(parts) == 3:
            h, mm, ss = (int(parts[0]), int(parts[1]), int(parts[2]))
            total = h * 3600 + mm * 60 + ss
        else:
            mm, ss = (int(parts[0]), int(parts[1]))
            total = mm * 60 + ss
        link = _youtube_url_with_t(youtube_url, total)
        return f'<a href="{link}">{ts_text}</a>'

    return pattern.sub(repl, html)

