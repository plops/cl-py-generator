from kivy.app import App
from kivy.uix.widget import Widget
_code_git_version="b18e0b2c769c533c0019ebba36266f34db93f93b"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="10:39:59 of Saturday, 2021-04-03 (GMT+1)"
class PongGame(Widget):
    pass
class PongApp(App):
    def build(self):
        return PongGame()
if ( ((__name__)==("__main__")) ):
    PongApp().run()