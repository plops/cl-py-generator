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
r = c(pr, tools=tools)
print(r)
tub = r.content[1]
func = globals()[tub.name]
res = func(**tub.input)
trc = dict(type="tool_result", tool_use_id=tub.id, content=dumps(res))
#
#
# The call of mk_msgs goes to the function mk_msgs_anthropic, which creates a list of messages compatible with the Anthropic API. It uses the @delegates decorator to inherit parameters from mk_msgs, allowing flexible argument passing.
# It calls mk_msgs with api='anthropic' to generate the initial list of messages.
# If cache_last_ckpt_only is True, it applies _remove_cache_ckpts to each message to strip unnecessary cache checkpoints.
# If the message list is empty, it returns early.
# Otherwise, it modifies the last message by adding cache control via _add_cache_control, using the cache and ttl parameters.
# Finally, it returns the processed list of messages.
#
#
# The general function, `mk_msgs` (called by mk_msgs_anthropic), creates a list of messages compatible with the OpenAI or Anthropic APIs. It takes a list of message contents (or a single string, which it converts to a list), along with optional arguments and keyword arguments.
#
# - If `msgs` is a string, it wraps it in a list.
# - It generates a list `mm` by calling `mk_msg` for each item in `msgs`, assigning alternating roles ('user' for even indices, 'assistant' for odd) and passing through the `api`, `*args`, and `**kw`.
# - It then flattens `mm` into `res`: if an item in `mm` is a list, it extends `res` with its elements; otherwise, it appends the item.
# - Finally, it returns the flattened list `res`.
#
msgs = mk_msgs([pr, r.content, [trc]])
r2 = c(msgs, tools=tools)
#
# Message(id='msg_01QndYW2KPiotRHiHcUxFjXt', content=[TextBlock(citations=None, text='Now I'll add 547892 to that result:n', type='text')], model='claude-haiku-4-5-20251001', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=In: 168; Out: 21; Cache create: 0; Cache read: 0; Total Tokens: 189; Search: 0)
#
