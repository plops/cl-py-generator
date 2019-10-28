# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/agg-segments.py 
# https://www.labri.fr/perso/nrougier/python-opengl/#id39
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from glumpy import app, gloo, gl
app.use("glfw")
V=np.zeros(16, [("center",np.float32,2,), ("radius",np.float32,1,)])
V["center"]=np.dstack([np.linspace(32, ((512)-(32)), len(V)), np.linspace(25, 28, len(V))])
V["radius"]=np.linspace(1, 15, len(V))
window=app.Window(512, 50, color=(1,1,1,1,))
vertex="""        uniform vec2 resolution;
        uniform float antialias;
        attribute float thickenss;
        attribute vec2 p0, p1, uv;
        varying float v_alpha, v_thickness;
        varying vec2 v_p0, v_p1, v_p;
        void main (){
                if ( abs(thickness)<(1.e+0f) ) {
                                    v_thickness=(1.e+0f);
            v_alpha=abs(thickness);
} else {
                                    v_thickness=abs(thickness);
            v_alpha=(1.e+0f);
}
                        float tt  = ((antialias)+(((thickness)/((2.e+0f)))));
        float l  = distance(p1, p0);
        float u  = (((((2.e+0f))*(uv.x)))-((1.e+0f)));
        float v  = (((((2.e+0f))*(uv.y)))-((1.e+0f)));
        vec2 TT  = normalize(((p1)-(p0)));
        vec2 O  = vec2(-TT.y, TT.x);
        vec2 p  = ((p0)+(((uv.x)*(TT)*(l)))+(((u)*(TT)*(tt)))+(((v)*(O)*(tt))));
                gl_Position=vec4((((((((2.e+0f))*(p)))/(resolution)))-((1.e+0f))), (0.0e+0f), (1.e+0f));
                // local space
                TT=vec2((1.e+0f), (0.0e+0f));
        O=vec2((0.0e+0f), (1.e+0f));
        p=((((uv.x)*(TT)*(l)))+(((u)*(TT)*(tt)))+(((v)*(O)*(tt))));
                v_p0=vec2((0.0e+0f), (0.0e+0f));
        v_p1=vec2((1.e+0f), (0.0e+0f));
        v_p=p;
}"""
fragment="""        uniform float antialias;
        varying float v_thickness, v_alpha;
        varying vec2 v_p0, v_p1, v_p;
        void main (){
                        float d  = (0.0e+0f);
        float offset  = ((((v_thickness)/((-2.e+0f))))+(((antialias)/((2.e+0f)))));
        if ( v_p.x<0 ) {
                                    d=((distance(v_p, v_p0))+(offset));
} else {
                        if ( distance(v_p1, v_p0)<v_p.x ) {
                                                d=((distance(v_p, v_p1))+(offset));
} else {
                                                d=((abs(v_p.y))+(offset));
}
}
        if ( d<0 ) {
                                    gl_FragColor=vec4((0.0e+0f), (0.0e+0f), (0.0e+0f), v_alpha);
} else {
                        if ( d<antialias ) {
                                                                d=exp(((-d)*(d)));
                gl_FragColor=vec4((0.0e+0f), (0.0e+0f), (0.0e+0f), ((v_alpha)*(d)));
};
};
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