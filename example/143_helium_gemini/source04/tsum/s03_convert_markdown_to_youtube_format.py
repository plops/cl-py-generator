#!/usr/bin/env python3
import re
def convert_markdown_to_youtube_format(text):
    r"""In its comments YouTube only allows *word* for bold text, not **word**. Colons or comma can not be fat (e.g. *Description:* must be written as *Description*: to be formatted properly. YouTube comments containing links seem to cause severe censoring. So we replace links."""
    # adapt the markdown to YouTube formatting
    text=text.replace("**:", ":**")
    text=text.replace("**,", ",**")
    text=text.replace("**.", ".**")
    text=text.replace("**", "*")
    # markdown title starting with ## with fat text
    text=re.sub(r"""^##\s*(.*)""", r"""*\1*""", text)
    # find any text that looks like an url and replace the . with -dot-
    text=re.sub(r"""((?:https?://)?(?:www\.)?\S+)\.(com|org|de|us|gov|net|edu|info|io|co\.uk|ca|fr|au|jp|ru|ch|it|nl|se|es|br|mx|in|kr)""", r"""\1-dot-\2""", text)
    return text