# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/agg-segments.py 
# https://www.labri.fr/perso/nrougier/python-opengl/#id39
import numpy as np
from glumpy import app, gloo, gl
vertex="""        uniform vec2 resolution;
        uniform float antialias;
        attribute float thickness;
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
                        float tt  = ((((v_thickness)/((2.e+0f))))+(antialias));
        float l  = distance(p0, p1);
        float u  = (((((2.e+0f))*(uv.x)))-((1.e+0f)));
        float v  = (((((2.e+0f))*(uv.y)))-((1.e+0f)));
        vec2 TT  = normalize(((p1)-(p0)));
        vec2 O  = vec2(-TT.y, TT.x);
        vec2 p  = ((p0)+(vec2((5.e-1f), (5.e-1f)))+(((uv.x)*(TT)*(l)))+(((u)*(TT)*(tt)))+(((v)*(O)*(tt))));
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
        varying float v_alpha, v_thickness;
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
                                    gl_FragColor=vec4((1.e+0f), (0.0e+0f), (0.0e+0f), v_alpha);
} else {
                        if ( d<antialias ) {
                                                                d=exp(((-d)*(d)));
                gl_FragColor=vec4((0.0e+0f), (2.e-1f), (0.0e+0f), ((v_alpha)*(d)));
};
};
}"""
app.use("glfw")
window=app.Window(1200, 400, color=(1,1,1,1,))
n=100
V=np.zeros((n,4,), dtype=[("p0",np.float32,2,), ("p1",np.float32,2,), ("uv",np.float32,2,), ("thickness",np.float32,1,)])
V["uv"]=((0,0,),(0,1,),(1,0,),(1,1,),)
V["thickness"]=np.linspace((1.0000000149011612e-1), (8.e+0), n).reshape(n, 1)
V["p0"]=np.dstack((np.linspace(100, 1100, n),((np.ones(n))*(50)),)).reshape(n, 1, 2)
V["p1"]=np.dstack((np.linspace(100, 1110, n),((np.ones(n))*(350)),)).reshape(n, 1, 2)
segments=gloo.Program(vertex, fragment, count=((4)*(n)))
segments.bind(V.ravel().view(gloo.VertexBuffer))
segments["antialias"]=(2.e+0)
I=np.zeros((n,6,), dtype=np.uint32)
I[:]=[0, 1, 2, 1, 2, 3]
I=((I)+(((4)*(np.arange(n, dtype=np.uint32).reshape(n, 1)))))
I=I.ravel().view(gloo.IndexBuffer)
@window.event
def on_resize(width, height):
    segments["resolution"]=(width,height,)
@window.event
def on_draw(dt):
    window.clear()
    segments.draw(gl.GL_TRIANGLES, I)
app.run()