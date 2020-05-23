# change gui font size in linux: xrandr --output HDMI-0 --dpi 55
# https://www.youtube.com/watch?v=3HSh_eSGf4c
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebkit import *
from PyQt5.QtWebkitWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
import numpy as np
import collections
import pandas as pd
import pathlib
# %%
output_path="/dev/shm"
_code_git_version="4ef34fcca3d068199bbdb2e46e3cdbc9dcc7d987"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/18_qt_webkit/source/run_00_browser.py"
_code_generation_time="09:39:39 of Saturday, 2020-05-23 (GMT+1)"
# %% open gui windows
app=QApplication([""])
web=QWebview()
web.Load(QUrl("https://youtube.com"))
web.show()