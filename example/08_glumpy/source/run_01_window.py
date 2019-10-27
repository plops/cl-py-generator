# https://github.com/rougier/python-opengl/blob/master/code/chapter-03/glumpy-quad-solid.py
# https://www.labri.fr/perso/nrougier/python-opengl/#id7
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from glumpy import app, gloo, gl
app.use("glfw")
window=app.Window()
vertex="""        attribute vec2 position;
        varying vec2 v_position;
        void main (){
                        gl_Position=vec4(position, (0.0e+0f), (1.e+0f));
                        v_position=position;
}"""
fragment="""        varying vec4 v_position;
        float distance (vec2 p, vec2 center, float radius){
                return ((length(((p)-(center))))-(radius));
}
        vec4 color (float d){
                        vec3 white  = vec3(1, 1, 1);
        vec3 blue  = vec3((1.e-1f), (4.e-1f), (7.e-1f));
        vec3 color  = ((white)-(((sign(d))*(blue))));
                color=((color)*((((1.e+0f))-(((exp((((-4.e+0f))*(abs(d)))))*((((8.e-1f))+((((2.e-1f))*(cos((((1.4e+2f))*(d)))))))))))));
                color=mix(color, white, (((1.e+0f))-(smoothstep((0.0e+0f), (1.9999999e-2f), abs(d)))));
        return vec4(color, (1.e+0f));
}
        void main (){
                        const float epsilon  = (5.e-3f);
        float d  = distance(v_position.xy, vec2((0.0e+0f)), (5.e-1f));
                gl_FragColor=color(d);
}"""
quad=gloo.Program(vertex, fragment, count=4)
quad["position"]=(-1,1,), (1,1,), (-1,-1,), (1,-1,)
@window.event
def on_draw(dt):
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
app.run()