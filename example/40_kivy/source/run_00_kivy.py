from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
_code_git_version="f50266e2a07d031a048eddffd3940d4c9f585ba9"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="11:20:29 of Saturday, 2021-04-03 (GMT+1)"
class PongBall(Widget):
    vx=NumericProperty(0)
    vy=NumericProperty(0)
    v=ReferenceListProperty(vx, vy)
    def move(self):
        self.pos=((Vector(*self.v))+(self.pos))
class PongGame(Widget):
    ball=ObjectProperty(None)
    def serve_ball(self):
        self.ball.center=self.center
        self.ball.v=Vector(4, 0).rotate(randint(0, 360))
    def update(self, dt):
        self.ball.move()
        if ( ((((self.ball.y)<(0))) or (((self.height)<(self.ball.top)))) ):
            self.ball.vy=((-1)*(self.ball.vy))
        if ( ((((self.ball.x)<(0))) or (((self.width)<(self.ball.right)))) ):
            self.ball.vx=((-1)*(self.ball.vx))
        pass
class PongApp(App):
    def build(self):
        game=PongGame()
        Clock.schedule_interval(game.update, (((1.0    ))/(60)))
        return game
if ( ((__name__)==("__main__")) ):
    PongApp().run()