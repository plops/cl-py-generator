#!/usr/bin/python3
# pip3 install --user helium
import numpy as np
import pandas as pd
import pathlib
from helium import *
# %%
_code_git_version="2317ebe426fc803d19a96a5fb7edd74634e003ba"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/22_helium/source/run_00_show.py"
_code_generation_time="23:29:03 of Saturday, 2020-07-04 (GMT+1)"
# %%
def run():
    start_firefox()
    go_to("www.google.com")
def reload():
    exec(open("run_00_show.py").read())
write("site:nvidia.com nvrtc")