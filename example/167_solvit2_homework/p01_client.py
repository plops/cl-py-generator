# run with:  export ANTHROPIC_API_KEY=`cat ~/anthropic.key`; uv run python -i p01_client.py
from __future__ import annotations
from claudette import *

print(models)
# ['claude-opus-4-1-20250805', 'claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-20250514', 'claude-3-opus-20240229', 'claude-sonnet-4-20250514', 'claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022']
# haiku is cheapest
m = "claude-haiku-4-5"
c = Client(m)
r = c("Hi there, I am jeremy.")
print(r)
# Message(id='msg_01QeZAHVGFTUUiW6DjCZ3umV', content=[TextBlock(citations=None, text='Hi Jeremy! Nice to meet you. How can I help you today?', type='text')], model='claude-haiku-4-5-20251001', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=In: 14; Out: 18; Cache create: 0; Cache read: 0; Total Tokens: 32; Search: 0)
