import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from glumpy import app, gloo, gl
app.use("glfw")
window=app.Window()
vertex="""        attribute vec2 position;
        void main (){
                        gl_Position=vec4(position, (0.0e+0f), (1.e+0f));
}"""
fragment="""        void main (){
                        gl_FragColor=vec4((1.e+0f), (0.0e+0f), (0.0e+0f), (1.e+0f));
}"""
quad=gloo.Program(vertex, fragment, count=4)
quad["position"]=(-1,1,), (1,1,), (-1,-1,), (1,-1,)
@window.event
def on_draw(dt):
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
app.run()