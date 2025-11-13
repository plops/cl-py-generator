# run with:  export ANTHROPIC_API_KEY=`cat ~/anthropic.key`; uv run python -i p01_client.py
from __future__ import annotations
from claudette import *
from fastcore.xtras import dumps

print(models)
# ['claude-opus-4-1-20250805', 'claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-20250514', 'claude-3-opus-20240229', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022']
# haiku is cheapest
m = "claude-haiku-4-5"
c = Client(m, log=True)


def adder(a: int, b: int):
    "Add two numbers"
    return (a) + (b)


def multer(a: int, b: int):
    "Multiply two numbers"
    return (a) * (b)


tools = [adder, multer]
pr = "I want to test my tools. Can you use &`[adder,multer]` to solve 42427928*548749+547892?"


def doloop(prompt, tools=None, debug=False):
    msgs = prompt
    while True:
        r = c(msgs, tools=tools)
        if debug:
            print(r)
        if (r.stop_reason) == ("end_turn"):
            return r
        tub = r.content[1]
        func = globals()[tub.name]
        res = func(**tub.input)
        trc = dict(type="tool_result", tool_use_id=tub.id, content=dumps(res))
        msgs = mk_msgs([pr, r.content, [trc]])


r = doloop(pr, tools=tools, debug=True)
