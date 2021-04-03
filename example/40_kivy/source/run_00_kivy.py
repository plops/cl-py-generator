from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.vector import Vector
_code_git_version="288c4e009dd06fde973eb798b1c4369d13d531c5"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="11:01:46 of Saturday, 2021-04-03 (GMT+1)"
class PongBall(Widget):
    vx=NumericProperty(0)
    vy=NumericProperty(0)
    v=ReferenceListProperty(vx, vy)
    def move(self):
        self.pos=((Vector(*self.v))+(self.pos))
class PongGame(Widget):
    pass
class PongApp(App):
    def build(self):
        return PongGame()
if ( ((__name__)==("__main__")) ):
    PongApp().run()