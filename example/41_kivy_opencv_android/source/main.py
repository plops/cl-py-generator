import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from camera import Camera2
_code_git_version="ab22595738935642489147d86dc781adcbd67280"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="09:18:05 of Sunday, 2021-04-04 (GMT+1)"
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