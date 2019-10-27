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
        void main (){
                        const float epsilon  = (5.e-3f);
        float d  = distance(v_position.xy, vec2((0.0e+0f)), (5.e-1f));
        if ( d<(-(epsilon)) ) {
                                    gl_FragColor=vec4((((1.e+0f))-(abs(d))), 0, 0, 1);
} else {
                        if ( epsilon<d ) {
                                                gl_FragColor=vec4(0, 0, (((1.e+0f))-(abs(d))), 1);
} else {
                                                gl_FragColor=vec4(1, 1, 1, 1);
}
};
}"""
quad=gloo.Program(vertex, fragment, count=4)
quad["position"]=(-1,1,), (1,1,), (-1,-1,), (1,-1,)
@window.event
def on_draw(dt):
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
app.run()