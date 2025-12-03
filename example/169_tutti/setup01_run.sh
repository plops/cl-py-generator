export GEMINI_API_KEY=`cat ~/api_key.txt`
uv run python  e03_watch.py -l w.log -p 300 --debug --dry-run -m 13 -M 90 --category phones
uv run python  e03_watch.py -l w.log -p 300 --debug --dry-run -m 13 -M 90 --category watches
