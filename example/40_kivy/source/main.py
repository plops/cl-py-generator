import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
_code_git_version="79ec8fe5fe8ec95d3ac6026eb1e502bbada7c7ba"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/29_ondrejs_challenge/source/run_00_start.py"
_code_generation_time="18:16:02 of Saturday, 2021-04-03 (GMT+1)"
class PongPaddle(Widget):
    score=NumericProperty(0)
    def bounce_ball(self, ball):
        if ( self.collide_widget(ball) ):
            vx, vy=ball.v
            offset=((((ball.center_y)-(self.center_y)))/((((0.50    ))*(self.height))))
            bounced=Vector(((-1)*(vx)), vy)
            vel=((bounced)*((1.10    )))
            ball.v=vel.x, ((vel.y)+(offset))
class PongBall(Widget):
    vx=NumericProperty(0)
    vy=NumericProperty(0)
    v=ReferenceListProperty(vx, vy)
    def move(self):
        self.pos=((Vector(*self.v))+(self.pos))
class PongGame(Widget):
    ball=ObjectProperty(None)
    player1=ObjectProperty(None)
    player2=ObjectProperty(None)
    def serve_ball(self, vel=(4,0,)):
        self.ball.center=self.center
        self.ball.v=vel
    def update(self, dt):
        self.ball.move()
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)
        if ( ((((self.ball.y)<(self.y))) or (((self.top)<(self.ball.top)))) ):
            self.ball.vy=((-1)*(self.ball.vy))
        if ( ((self.ball.x)<(self.x)) ):
            self.player2.score=((1)+(self.player2.score))
            self.serve_ball(vel=(4,0,))
        if ( ((self.width)<(self.ball.x)) ):
            self.player1.score=((1)+(self.player1.score))
            self.serve_ball(vel=(-4,0,))
    def on_touch_move(self, touch):
        if ( ((touch.x)<(((self.width)/(3)))) ):
            self.player1.center_y=touch.y
        if ( ((((self.width)-(((self.width)/(3)))))<(touch.x)) ):
            self.player2.center_y=touch.y
class PongApp(App):
    game=None
    def build(self):
        self.game=PongGame()
        self.game.serve_ball()
        Clock.schedule_interval(self.game.update, (((1.0    ))/(60)))
        return self.game
if ( ((__name__)==("__main__")) ):
    app=PongApp()
    app.run()