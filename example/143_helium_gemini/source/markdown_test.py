
import markdown

doc = """
# Hello World

This is a **great** tutorial about using Markdown in [Python](https://python.org).

This is a list:

- Item 1
- Item 2
- Item 3


This is a numbered list:

1. Item 1
2. Item 2
3. Item 3


This is a code block:
    
    ```python
    print("Hello World")
    ```

This is a table:

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

This is an image:

![Python Logo](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)

This is a blockquote:

> This is a blockquote

This is a horizontal rule:

---

This is a line break:

This is a line break  

This is another line break  




""" 

html = markdown.markdown(doc)