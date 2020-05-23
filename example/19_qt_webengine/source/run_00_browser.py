# change gui font size in linux: xrandr --output HDMI-0 --dpi 55
# https://further-reading.net/2018/08/quick-tutorial-pyqt-5-browser/
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebkit import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
import numpy as np
import collections
import pandas as pd
import pathlib
# %%
output_path="/dev/shm"
_code_git_version="679f07902935c788398bd99b1577a2d7f1d279cb"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/19_qt_webengine/source/run_00_browser.py"
_code_generation_time="09:57:35 of Saturday, 2020-05-23 (GMT+1)"
# %% open gui windows
app=QApplication([""])
web=QWebEngineView()
web.load(QUrl("https://youtube.com"))
web.show()