import numpy as np
import pandas as pd
import pathlib
from vulkan import *
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QAbstractTableModel, Qt
# %%
_code_git_version="bf53682a539909e755ccde8f35246998027342f9"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/21_vulkan_qt/source/run_00_show.py"
_code_generation_time="21:21:21 of Monday, 2020-06-15 (GMT+1)"
validation_layers=["VK_LAYER_KHRONOS_validation"]
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
instanceinfo=VkInstanceCreateInfo(pApplicationInfo=appinfo, enabledLayerCount=len(validation_layers)  if  enable_validation_layers else 0, ppEnabledLayerNames=validation_layers  if  enable_validation_layers else None, enabledExtensionCount=len(extensions), ppEnabledExtensionNames=extensions)
instance=vkCreateInstance(instanceinfo, None)
# setup debug callback
if ( enable_validation_layers ):
    createinfo=VkDebugReportCallbackCreateInfoEXT(flags=((VK_DEBUG_REPORT_WARNING_BIT_EXT) | (VK_DEBUG_REPORT_ERROR_BIT_EXT)), pfnCallback=debug_callback)
    callback=vkCreateDebugReportCallbackEXT(instance, createinfo, None)
# pick physical device
class QueueFamilyIndices(object):
    def __init__(self):
        self.graphicsFamily=-1
    @property
    def isComplete(self):
        return ((0)<=(self.graphicsFamily))
def find_queue_families(dev):
    indices=QueueFamilyIndices()
    family_props=vkGetPhysicalDeviceQueueFamilyProperties(dev)
    for i, prop in enumerate(family_props):
        if ( ((((0)<(prop.queueCount))) and (((prop.queueFlags) & (VK_QUEUE_GRAPHICS_BIT)))) ):
            indices.graphicsFamily=i
        if ( indices.isComplete ):
            break
    return indices
def is_device_suitable(dev):
    indices=find_queue_families(dev)
    return indices.isComplete
physical_devices=vkEnumeratePhysicalDevices(instance)
physical_device=None
for d in physical_devices:
    if ( is_device_suitable(d) ):
        physical_device=d
        break
assert(((None)!=(physical_device)))
win.show()
def cleanup():
    global win, instance
    vkDestroyInstance(instance, None)
    del(win)
app.aboutToQuit.connect(cleanup)
def run():
    sys.exit(app.exec_())