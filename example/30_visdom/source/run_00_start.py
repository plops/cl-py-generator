# %% imports
import pathlib
import numpy as np
import pandas as pd
import visdom
_code_git_version="7ab2392faef1465f652e168b36b1a6b75899084c"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="22:16:59 of Tuesday, 2020-12-01 (GMT+1)"
vis=visdom.Visdom()
trace={("x"):([1, 2, 3]),("y"):([4, 5, 6]),("mode"):("markers+lines"),("type"):("custom"),("marker"):({("color"):("red"),("symbol"):(104),("size"):(10)}),("text"):(["one", "two", "three"]),("name"):("first trace")}
layout={("title"):("first plot"),("xaxis"):({("title"):("x1")}),("yaxis"):({("title"):("x2")})}
vis._send({("data"):([trace]),("layout"):(layout),("win"):("mywin")})