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
fragment="""        varying vec4 v_center;
        varying float v_radius;
        float distance (vec2 p, vec2 center, float radius){
                return ((length(((p)-(center))))-(radius));
}
        float SDF_circle (vec2 p, float radius){
                return ((length(p))-(radius));
}
        float SDF_plane (vec2 p, vec2 p0, vec2 p1){
                        vec2 tt  = ((p1)-(p0));
        vec2 o  = normalize(vec2(tt.y, (-(tt.x))));
        return dot(o, ((p0)-(p)));
}
        float SDF_box (vec2 p, vec2 size){
                        vec2 d  = ((abs(p))-(size));
        return ((min(max(d.x, d.y), (0.0e+0f)))+(length(max(d, (0.0e+0f)))));
}
        float SDF_round_box (vec2 p, vec2 size, float radius){
                return ((SDF_box(p, size))-(radius));
}
        float SDF_fake_box (vec2 p, vec2 size){
                return max(((abs(p.x))-(size.x)), ((abs(p.y))-(size.y)));
}
        float SDF_triangle (vec2 p, vec2 p0, vec2 p1, vec2 p2){
                        vec2 e0  = ((p1)-(p0));
                        vec2 e1  = ((p2)-(p1));
                        vec2 e2  = ((p0)-(p2));
                        vec2 v0  = ((p)-(p0));
                        vec2 v1  = ((p)-(p1));
                        vec2 v2  = ((p)-(p2));
                        vec2 pq0  = ((v0)-(((e0)*(clamp(((dot(v0, e0))/(dot(e0, e0))), (0.0e+0f), (1.e+0f))))));
                        vec2 pq1  = ((v1)-(((e1)*(clamp(((dot(v1, e1))/(dot(e1, e1))), (0.0e+0f), (1.e+0f))))));
                        vec2 pq2  = ((v2)-(((e2)*(clamp(((dot(v2, e2))/(dot(e2, e2))), (0.0e+0f), (1.e+0f))))));
                        auto s  = sign(((((e0.x)*(e2.y)))-(((e0.y)*(e2.x)))));
        y;
        declare(type(float, s));
                        vec2 vv0  = vec2(dot(pq0, pq0), ((s)*(((((v0.x)*(e0.y)))-(((v0.y)*(e0.x)))))));
                        vec2 vv1  = vec2(dot(pq1, pq1), ((s)*(((((v1.x)*(e1.y)))-(((v1.y)*(e1.x)))))));
                        vec2 vv2  = vec2(dot(pq2, pq2), ((s)*(((((v2.x)*(e2.y)))-(((v2.y)*(e2.x)))))));
                        vec2 d  = min(min(vv0, vv1), vv2);
        return (((-(sqrt(d.x))))*(sign(d.y)));
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
                        vec2 p  = ((gl_FragCoord.xy)-(v_center));
        float a  = (1.e+0f);
        float d  = ((length(p))+((-(v_radius)))+((1.e+0f)));
                d=abs(d);
        if ( (0.0e+0f)<d ) {
                                                a=exp(((-1)*(d)*(d)));
};
                gl_FragColor=vec3(vec3((0.0e+0f)), a);
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