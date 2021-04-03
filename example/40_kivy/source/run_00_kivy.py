from kivy.app import App
from kivy.uix.widget import Widget
_code_git_version="f21161ca42846b070ae51b27ccee46da2ce381c3"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="10:47:06 of Saturday, 2021-04-03 (GMT+1)"
class PongGame(Widget):
    pass
class PongApp(App):
    def build(self):
        return PongGame()
class PongBall(Widget):
    vx=NumericProperty(0)
    vy=NumericProperty(0)
    v=ReferenceListProperty(vx, vy)
    def move(self):
        self.pos=((Vector(*self.v))+(self.pos))
if ( ((__name__)==("__main__")) ):
    PongApp().run()