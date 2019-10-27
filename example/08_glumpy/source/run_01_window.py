# https://github.com/rougier/python-opengl/blob/master/code/chapter-03/glumpy-quad-solid.py
# https://www.labri.fr/perso/nrougier/python-opengl/#id7
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from glumpy import app, gloo, gl
app.use("glfw")
window=app.Window()
vertex="""        attribute vec2 position;
        attribute vec4 color;
        varying vec4 v_color;
        void main (){
                        gl_Position=vec4(position, (0.0e+0f), (1.e+0f));
                        v_color=color;
}"""
fragment="""        varying vec4 v_color;
        void main (){
                        gl_FragColor=v_color;
}"""
quad=gloo.Program(vertex, fragment, count=4)
quad["position"]=(-1,1,), (1,1,), (-1,-1,), (1,-1,)
quad["color"]=(1,1,0,1,), (1,0,0,1,), (0,0,1,1,), (0,1,0,1,)
@window.event
def on_draw(dt):
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
app.run()