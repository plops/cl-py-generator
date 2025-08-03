from s03_convert_markdown_to_youtube_format import *

r = convert_markdown_to_youtube_format(r"""**Title:**
Let's **go** to http://www.google.com/search?q=hello.""")
assert (
    r"""*Title:*
Let's *go* to http://www.google-dot-com/search?q=hello."""
) == (r)
