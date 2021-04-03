import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from camera import Camera2
_code_git_version="879cb56fa01c7ee9dc2f7093b6a7b04656e7f96e"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="21:28:12 of Saturday, 2021-04-03 (GMT+1)"
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
        self.root.ids.label.text="{:s} {}x{} image".format(ctime, rows, cols)
        Clock.schedule_once(self.detect, 1)
if ( ((__name__)==("__main__")) ):
    app=MainApp()
    app.run()