import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from camera import Camera2
_code_git_version="60d7227c0cb829dec0f31c8b46587053a829fd91"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="21:33:44 of Saturday, 2021-04-03 (GMT+1)"
class MainLayout(BoxLayout):
    pass
class MainApp(App):
    def build(self):
        return MainLayout()
    def on_start(self):
        Clock.schedule_once(self.detect, 5)
    def detect(self, nap):
        image=self.root.ids.camera.image
        rows, cols=image.shape[:2]
        ctime=time.ctime()[11:19]
        Clock.schedule_once(self.detect, 1)
if ( ((__name__)==("__main__")) ):
    app=MainApp()
    app.run()