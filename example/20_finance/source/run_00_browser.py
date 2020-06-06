import yfinance as yf
import numpy as np
import collections
import pandas as pd
import pathlib
# %%
_code_git_version="ab09a57061fcf75cb9fa85daca2203b50d2aa363"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/20_finance/source/run_00_browser.py"
_code_generation_time="11:36:01 of Saturday, 2020-06-06 (GMT+1)"
# %%
msft=yf.Ticker("MSFT")