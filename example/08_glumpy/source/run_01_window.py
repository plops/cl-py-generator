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
                        float s  = sign(((((e0.x)*(e2.y)))-(((e0.y)*(e2.x)))));
                        vec2 vv0  = vec2(dot(pq0, pq0), ((s)*(((((dot(v0, x))*(dot(e0, y))))-(((dot(v0, y))*(dot(e0, x))))))));
                        vec2 vv1  = vec2(dot(pq1, pq1), ((s)*(((((dot(v1, x))*(dot(e1, y))))-(((dot(v1, y))*(dot(e1, x))))))));
                        vec2 vv2  = vec2(dot(pq2, pq2), ((s)*(((((dot(v2, x))*(dot(e2, y))))-(((dot(v2, y))*(dot(e2, x))))))));
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