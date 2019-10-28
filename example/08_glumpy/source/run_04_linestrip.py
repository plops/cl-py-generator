# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/linestrip.py
# https://www.labri.fr/perso/nrougier/python-opengl/#id39
import numpy as np
from glumpy import app, gloo, gl
vertex="""        uniform vec2 resolution;
        uniform float antialias, thickness, linelength;
        attribute vec4 prev, curr, next;
        varying vec2 v_uv;
        void main (){
                        float w  = ((((thickness)/((2.e+0f))))+(antialias));
        vec2 p ;
        if ( (prev.xy)==(curr.xy) ) {
                                    vec2 t1  = normalize(((next.xy)-(curr.xy)));
            vec2 n1  = vec2(-t1.y, t1.x);
                        v_uv=vec2(-w, ((curr.z)*(w)));
            p=((curr.xy)+(((-w)*(t1)))+(((curr.z)*(w)*(n1))));
} else {
                        if ( (curr.xy)==(next.xy) ) {
                                                vec2 t0  = normalize(((curr.xy)-(prev.xy)));
                vec2 n0  = vec2(-t0.y, t0.x);
                                v_uv=vec2(((w)+(linelength)), ((curr.z)*(w)));
                p=((curr.xy)+(((w)*(t0)))+(((curr.z)*(w)*(n0))));
} else {
                                                vec2 t0  = normalize(((curr.xy)-(prev.xy)));
                vec2 n0  = vec2(-t0.y, t0.x);
                vec2 t1  = normalize(((next.xy)-(curr.xy)));
                vec2 n1  = vec2(-t1.y, t1.x);
                vec2 miter  = normalize(((n0)+(n1)));
                float dy  = ((w)/(dot(miter, n1)));
                                v_uv=vec2(curr.w, ((curr.z)*(w)));
                p=((curr.xy)+(((dy)*(curr.z)*(miter))));
}
}
                gl_Position=vec4((((((((2.e+0f))*(p)))/(resolution)))-((1.e+0f))), (0.0e+0f), (1.e+0f));
}"""
fragment="""        uniform float antialias, thickness, linelength;
        varying vec2 v_uv;
        void main (){
                        float d  = (0.0e+0f);
        float w  = ((((thickness)/((2.e+0f))))-(antialias));
        if ( v_uv.x<0 ) {
                                    d=((length(v_uv))-(w));
} else {
                        if ( (linelength)<=(v_uv.x) ) {
                                                d=((distance(v_uv, vec2(linelength, 0)))-(w));
} else {
                                                d=((abs(v_uv.y))-(w));
}
}
        if ( d<0 ) {
                                    gl_FragColor=vec4((0.0e+0f), (0.0e+0f), (0.0e+0f), (1.e+0f));
} else {
                                    d=((d)/(antialias));
            gl_FragColor=vec4((0.0e+0f), (0.0e+0f), (0.0e+0f), exp(((-d)*(d))));
};
}"""
app.use("glfw")
window=app.Window(1200, 400, color=(1,1,1,1,))
def bake(P, closed=False):
    epsilon=(1.000000013351432e-10)
    n=len(P)
    if ( ((closed) and (((epsilon)<(((((P[0])-(P[-1])))**(2)).sum())))) ):
        P=np.append(P, P[0])
        P=P.reshape(((n)+(1)), 2)
        n=((n)+(1))
    V=np.zeros((((1)+(n)+(1)),2,4,), dtype=np.float32)
    V_prev, V_curr, V_next=(V[:-2],V[1:-1],V[2:],)
    V_curr[...,0]=P[:,np.newaxis,0]
    V_curr[...,1]=P[:,np.newaxis,1]
    V_curr[...,2]=(1,-1,)
    L=np.cumsum(np.sqrt(((((P[1:])-(P[:-1])))**(2)).sum(axis=1))).reshape(((n)-(1)), 1)
    V_curr[1:,:,3]=L
    if ( closed ):
        V[0], V[-1]=(V[-3],V[2],)
    else:
        V[0], V[-1]=(V[1],V[-2],)
    return (V_prev,V_curr,V_next,L[-1],)
n=1024
TT=np.linspace(0, ((12)*(2)*(np.pi)), n, dtype=np.float32)
R=np.linspace(10, 246, n, dtype=np.float32)
P=np.dstack((((256)+(((np.cos(TT))*(R)))),((256)+(((np.sin(TT))*(R)))),)).squeeze()
V_prev, V_curr, V_next, length=bake(P)
segments=gloo.Program(vertex, fragment)
segments["prev"]=V_prev
segments["curr"]=V_curr
segments["next"]=V_next
segments["thickness"]=(9.e+0)
segments["antialias"]=(1.4999999999999997e+0)
segments["linelength"]=length
@window.event
def on_resize(width, height):
    segments["resolution"]=(width,height,)
@window.event
def on_draw(dt):
    window.clear()
    segments.draw(gl.GL_TRIANGLE_STRIP)
app.run()