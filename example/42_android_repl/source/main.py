import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
_code_git_version="24799c9fa6f60ea3213bf4036a74212ee58d6cce"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="08:32:13 of Monday, 2021-04-05 (GMT+1)"
class MainLayout(BoxLayout):
    pass
class MainApp(App):
    def build(self):
        return MainLayout()
    def on_start(self):
        Clock.schedule_once(self.detect, 5)
    def detect(self, nap):
        ctime=time.ctime()[11:19]
        print("{:s} {}x{} image".format(ctime, rows, cols))
        Clock.schedule_once(self.detect, 1)
if ( ((__name__)==("__main__")) ):
    app=MainApp()
    app.run()