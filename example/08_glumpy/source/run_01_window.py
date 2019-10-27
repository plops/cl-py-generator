# https://github.com/rougier/python-opengl/blob/master/code/chapter-03/glumpy-quad-solid.py
# https://www.labri.fr/perso/nrougier/python-opengl/#id7
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from glumpy import app, gloo, gl
app.use("glfw")
V=np.zeros(16, [("center",np.float32,2,), ("radius",np.float32,1,)])
V["center"]=np.dstack([np.linspace(32, ((512)-(32)), len(V)), np.linspace(25, 28, len(V))])
V["radius"]=15
window=app.Window(512, 50, color=(1,1,1,1,))
vertex="""        uniform vec2 resolution;
        attribute vec2 center;
        attribute float radius;
        varying vec2 v_center;
        varying float v_radius;
        void main (){
                        v_radius=radius;
        v_center=center;
        gl_PointSize=(((2.e+0f))+(ceil((((2.e+0f))*(radius)))));
                        gl_Position=vec4((((-1.e+0f))+((((2.e+0f))*(((center)/(resolution)))))), (0.0e+0f), (1.e+0f));
}"""
fragment="""        varying vec2 v_center;
        varying float v_radius;
        void main (){
                        vec2 p  = ((gl_FragCoord.xy)-(v_center));
        float a  = (1.e+0f);
        float d  = ((length(p))+((-(v_radius)))+((1.e+0f)));
                d=abs(d);
        if ( (0.0e+0f)<d ) {
                                                a=exp(((-1)*(d)*(d)));
};
                gl_FragColor=vec4(vec3((0.0e+0f)), a);
}"""
points=gloo.Program(vertex, fragment)
points.bind(V.view(gloo.VertexBuffer))
@window.event
def on_resize(width, height):
    points["resolution"]=(width,height,)
@window.event
def on_draw(dt):
    window.clear()
    points.draw(gl.GL_POINTS)
app.run()