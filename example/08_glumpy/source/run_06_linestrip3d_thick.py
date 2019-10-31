# https://www.labri.fr/perso/nrougier/python-opengl/code/chapter-09/linestrip-3d-better.py
# https://www.labri.fr/perso/nrougier/python-opengl/#id39
import numpy as np
from glumpy import app, gloo, gl, glm
vertex="""        uniform vec2 viewport;
        uniform mat4 model, view, projection;
        uniform float antialias, thickness, linelength;
        attribute vec3 prev, curr, next;
        attribute vec2 uv;
        varying vec2 v_uv;
        varying vec3 v_normal;
        varying float v_thickness;
        void main (){
                        vec4 NDC_prev  = ((projection)*(view)*(model)*(vec4(prev.xyz, (1.e+0f))));
                        vec4 NDC_curr  = ((projection)*(view)*(model)*(vec4(curr.xyz, (1.e+0f))));
                        vec4 NDC_next  = ((projection)*(view)*(model)*(vec4(next.xyz, (1.e+0f))));
                        vec2 screen_prev  = ((viewport)*(((((((NDC_prev.xy)/(NDC_prev.w)))+((1.e+0f))))/((2.e+0f)))));
                        vec2 screen_curr  = ((viewport)*(((((((NDC_curr.xy)/(NDC_curr.w)))+((1.e+0f))))/((2.e+0f)))));
                        vec2 screen_next  = ((viewport)*(((((((NDC_next.xy)/(NDC_next.w)))+((1.e+0f))))/((2.e+0f)))));
                        vec4 normal  = ((model)*(vec4(curr.xyz, (1.e+0f))));
                v_normal=normal.xyz;
        if ( normal.z<0 ) {
                                    v_thickness=((thickness)/((2.e+0f)));
} else {
                                    v_thickness=((((thickness)*(((pow(normal.z, (5.e-1f)))+(1)))))/((2.e+0f)));
};
                        float w  = ((((thickness)/((2.e+0f))))+(antialias));
        vec2 position ;
        vec2 t0  = normalize(((screen_curr.xy)-(screen_prev.xy)));
        vec2 n0  = vec2(-t0.y, t0.x);
        vec2 t1  = normalize(((screen_next.xy)-(screen_curr.xy)));
        vec2 n1  = vec2(-t1.y, t1.x);
                v_uv=vec2(uv.x, ((uv.y)*(w)));
        if ( (prev.xy)==(curr.xy) ) {
                                    v_uv.x=-w;
            position=((screen_curr.xy)+(((-w)*(t1)))+(((uv.y)*(w)*(n1))));
} else {
                        if ( (curr.xy)==(next.xy) ) {
                                                v_uv.x=((w)+(linelength));
                position=((screen_curr.xy)+(((w)*(t0)))+(((uv.y)*(w)*(n0))));
} else {
                                                vec2 miter  = normalize(((n0)+(n1)));
                float dy  = ((w)/(max(dot(miter, n1), (1.e+0f))));
                                position=((screen_curr.xy)+(((dy)*(uv.y)*(miter))));
}
}
                gl_Position=vec4((((((((2.e+0f))*(position)))/(viewport)))-((1.e+0f))), ((NDC_curr.z)/(NDC_curr.w)), (1.e+0f));
}"""
fragment="""        uniform float antialias, thickness, linelength;
        varying vec2 v_uv;
        varying float v_thickness;
        varying vec3 v_normal;
        void main (){
                        float d  = (0.0e+0f);
        float w  = ((((v_thickness)/((2.e+0f))))-(antialias));
        vec3 color  = vec3((0.0e+0f), (0.0e+0f), (0.0e+0f));
        if ( v_normal.z<0 ) {
                                                color=(((7.5e-1f))*(vec3(pow(abs(v_normal.z), (5.e-1f)))));
};
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
                                    gl_FragColor=vec4(color, (1.e+0f));
} else {
                                    d=((d)/(antialias));
            gl_FragColor=vec4(color, exp(((-d)*(d))));
};
}"""
app.use("glfw")
window=app.Window(1920, 1080, color=(1,1,1,1,))
def bake(P, closed=False):
    epsilon=(1.000000013351432e-10)
    n=len(P)
    if ( ((closed) and (((epsilon)<(((((P[0])-(P[-1])))**(2)).sum())))) ):
        P=np.append(P, P[0])
        P=P.reshape(((n)+(1)), 3)
        n=((n)+(1))
    V=np.zeros((((1)+(n)+(1)),2,3,), dtype=np.float32)
    UV=np.zeros((n,2,2,), dtype=np.float32)
    V_prev, V_curr, V_next=(V[:-2],V[1:-1],V[2:],)
    V_curr[...,0]=P[:,np.newaxis,0]
    V_curr[...,1]=P[:,np.newaxis,1]
    V_curr[...,2]=P[:,np.newaxis,2]
    L=np.cumsum(np.sqrt(((((P[1:])-(P[:-1])))**(2)).sum(axis=-1))).reshape(((n)-(1)), 1)
    UV[1:,:,0]=L
    UV[...,1]=(1,-1,)
    if ( closed ):
        V[0]=V[-3]
        V[-1]=V[2]
    else:
        V[0]=V[1]
        V[-1]=V[-2]
    return (V_prev,V_curr,V_next,UV,L[-1],)
n=2048
TT=np.linspace(0, ((20)*(2)*(np.pi)), n, dtype=np.float32)
R=np.linspace((1.0000000149011612e-1), ((np.pi)-((1.0000000149011612e-1))), n, dtype=np.float32)
X=((np.cos(TT))*(np.sin(R)))
Y=((np.sin(TT))*(np.sin(R)))
Z=np.cos(R)
P=np.dstack((X,Y,Z,)).squeeze()
V_prev, V_curr, V_next, UV, length=bake(P)
segments=gloo.Program(vertex, fragment)
segments["prev"]=V_prev
segments["curr"]=V_curr
segments["next"]=V_next
segments["uv"]=UV
segments["thickness"]=(1.5e+1)
segments["antialias"]=(1.4999999999999997e+0)
segments["linelength"]=length
segments["model"]=np.eye(4, dtype=np.float32)
segments["view"]=glm.translation(0, 0, -5)
phi=0
theta=0
@window.event
def on_resize(width, height):
    segments["projection"]=glm.perspective((3.e+1), ((width)/(float(height))), (2.e+0), (1.e+2))
    segments["viewport"]=(width,height,)
@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
@window.event
def on_draw(dt):
    global phi, theta, duration
    window.clear()
    segments.draw(gl.GL_TRIANGLE_STRIP)
    theta=((theta)+((1.0000000149011612e-1)))
    phi=((phi)+((2.0000000298023224e-1)))
    model=np.eye(4, dtype=np.float32)
    glm.rotate(model, theta, 0, 1, 0)
    glm.rotate(model, phi, 1, 0, 0)
    segments["model"]=model
app.run(framerate=60)