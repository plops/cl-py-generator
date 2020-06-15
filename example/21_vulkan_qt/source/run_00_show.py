import numpy as np
import pandas as pd
import pathlib
from vulkan import *
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QAbstractTableModel, Qt
# %%
_code_git_version="6ce88b64d27f011c132668389d189060789a8b1f"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/21_vulkan_qt/source/run_00_show.py"
_code_generation_time="21:07:20 of Monday, 2020-06-15 (GMT+1)"
validation_layer=["VK_LAYER_LUNARG_standard_validation"]
enable_validation_layers=True
class InstanceProcAddr(object):
    def __init__(self, func):
        self.__func=func
    def __call__(self, *args, **kwargs):
        func_name=self.__func.__name__
        func=vkGetInstanceProcAddr(args[0], func_name)
        if ( func ):
            return func(*args, **kwargs)
        else:
            return VK_ERROR_EXTENSION_NOT_PRESENT
@InstanceProcAddr
def vkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass
@InstanceProcAddr
def vkDestroyDebugReportCallbackEXT(instance, pCreateInfo, pAllocator):
    pass
def debug_callback(*args):
    print("debug: {} {}".format(args[5], args[6]))
    return 0
app=QApplication([""])
win=QWidget()
appinfo=VkApplicationInfo(pApplicationName="python vk", applicationVersion=VK_MAKE_VERSION(1, 0, 0), pEngineName="pyvulkan", engineVersion=VK_MAKE_VERSION(1, 0, 0), apiVersion=VK_API_VERSION)
extensions=[e.extensionName for e in vkEnumerateInstanceExtensionProperties(None)]
instanceinfo=VkInstanceCreateInfo(pApplicationInfo=appinfo, enabledLayerCount=0, enabledExtensionCount=len(extensions), ppEnabledExtensionNames=extensions)
instance=vkCreateInstance(instanceinfo, None)
if ( enable_validation_layers ):
    createinfo=VkDebugReportCallbackCreateInfoEXT(flags=((VK_DEBUG_REPORT_WARNING_BIT_EXT) | (VK_DEBUG_REPORT_ERROR_BIT_EXT)), pfnCallback=debug_callback)
    callback=vkCreateDebugReportCallbackEXT(instance, createinfo, None)
win.show()
def cleanup():
    global win, instance
    vkDestroyInstance(instance, None)
    del(win)
app.aboutToQuit.connect(cleanup)
def run():
    sys.exit(app.exec_())