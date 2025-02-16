from s03_convert_markdown_to_youtube_format import *
assert(((r"""*Title:*
Let's *go* to http://www.google-dot-com/search?q=hello.""")), convert_markdown_to_youtube_format(r"""**Title:**
Let's **go** to http://www.google.com/search?q=hello."""))