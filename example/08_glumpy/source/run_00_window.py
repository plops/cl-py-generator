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