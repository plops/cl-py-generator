# run with:  export GEMINI_API_KEY=`cat ~/api_key.txt`; uv run python -i p02_fs_tools.py
#
# docs: lisette.answer.ai
from __future__ import annotations
from fastcore.tools import *
from lisette import *

fc_tool_info()
model = "gemini/gemini-2.5-flash"
chat = Chat(model)
r = chat(r"""tell me about the tools""")
print(r)
