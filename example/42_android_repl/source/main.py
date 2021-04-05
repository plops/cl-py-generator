import time
import subprocess
import os
import IPython
import jupyter_core.command
import sys
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
_code_git_version="5bf6be7d8aeb6463002fc2468f73a574c54ea972"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="20:38:43 of Monday, 2021-04-05 (GMT+1)"
class MainLayout(BoxLayout):
    pass
class MainApp(App):
    def build(self):
        return MainLayout()
    def on_start(self):
        Clock.schedule_once(self.detect, 5)
    def detect(self, nap):
        ctime=time.ctime()[11:19]
        print("{:s}".format(ctime))
        Clock.schedule_once(self.detect, 1)
def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in ((folders)+(files)):
            yield(os.path.join(root, filename))
def find(dir):
    for filename in listfiles(dir):
        print(filename)
if ( ((__name__)==("__main__")) ):
    sys.argv=["bla", "notebook"]
    jupyter_core.command.main()
    app=MainApp()
    app.run()