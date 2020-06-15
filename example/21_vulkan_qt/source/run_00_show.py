import numpy as np
import pandas as pd
import pathlib
from vulkan import *
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QAbstractTableModel, Qt
# %%
_code_git_version="41f65b9491e73eab469411b9242898a5b6d62baa"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/21_vulkan_qt/source/run_00_show.py"
_code_generation_time="20:57:20 of Monday, 2020-06-15 (GMT+1)"
app=QApplication([""])
win=QWidget()
appinfo=VkApplicationInfo(pApplicationName="python vk", applicationVersion=VK_MAKE_VERSION(1, 0, 0), pEngineName="pyvulkan", engineVersion=VK_MAKE_VERSION(1, 0, 0), apiVersion=VK_API_VERSION)
extensions=[e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]
instanceinfo=VkInstanceCreateInfo(pApplicationInfo=appinfo, enabledLayerCount=0, enabledExtensionCount=len(extensions), ppEnabledExtensionNames=extensions)
instance=vkCreateInstance(instanceinfo, None)
win.show()
def cleanup():
    global win
    del(win)
app.aboutToQuit.connect(cleanup)
def run():
    sys.exit(app.exec_())