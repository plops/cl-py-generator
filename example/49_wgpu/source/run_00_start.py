import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import time
import pathlib
import pandas as pd
import glfw
import wgpu
import math
import wgpu.gui.glfw
import wgpu.backends.rs
from pyshader import python2shader, RES_INPUT, RES_OUTPUT, vec2, vec3, vec4, i32
# https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
class Timer(object):
    def __init__(self, name = None):
        self.name=name
    def __enter__(self):
        self.tstart=time.time()
    def __exit__(self, type, value, traceback):
        print("[{}] elapsed: {}s".format(self.name, ((time.time())-(self.tstart))))
def main(canvas):
    adapter=wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device=adapter.request_device()
    return _main(canvas, device)
@python2shader
def vertex_shader(index = (RES_INPUT,"VertexId",i32,), pos = (RES_OUTPUT,"Position",vec4,), color = (RES_OUTPUT,0,vec3,)):
    positions=[vec2((0.    ), (0.50    )), vec2((0.50    ), (0.50    )), vec2((-0.50    ), (0.70    ))]
    p=positions[index]
    pos=vec4(p, (0.    ), 1)
    color=vec3(p, (0.50    ))
@python2shader
def fragment_shader(in_color = (RES_INPUT,0,vec3,), out_color = (RES_OUTPUT,0,vec4,)):
    out_color=vec4(in_color, (1.0    ))
def _main(canvas, device):
    vshader=device.create_shader_module(code=vertex_shader)
    fshader=device.create_shader_module(code=fragment_shader)
    bind_group_layout=device.create_bind_group_layout(entries=[])
    bind_group=device.create_bind_group(layout=bind_group_layout, entries=[])
    pipeline_layout=device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    render_pipeline=device.create_render_pipeline(layout=pipeline_layout, vertex=dict(module=vshader, entry_point="main", buffers=[]), primitive=dict(topology=wgpu.PrimitiveTopology.triangle_list, strip_index_format=wgpu.IndexFormat.uint32, front_face=wgpu.FrontFace.ccw, cull_mode=wgpu.CullMode.none), depth_stencil=None, multisample=dict(count=1, mask=0xFFFFffff, alpha_to_coverage_enabled=False), fragment=dict(module=fshader, entry_point="main", targets=[dict(format=wgpu.TextureFormat.bgra8unorm_srgb, blend=dict(color=(wgpu.BlendFactor.one,wgpu.BlendFactor.zero,wgpu.BlendOperation.add,), alpha=(wgpu.BlendFactor.one,wgpu.BlendFactor.zero,wgpu.BlendOperation.add,)))]))
    swap_chain=canvas.configure_swap_chain(device=device)
    def draw_frame():
        with swap_chain as current_texture_view:
            command_encoder=device.create_command_encoder()
            render_pass=command_encoder.begin_render_pass(color_attachments=[dict(view=current_texture_view, resolve_target=None, load_value=(0,0,0,1,), store_op=wgpu.StoreOp.store)])
            render_pass.set_pipeline(render_pipeline)
            render_pass.set_bind_group(0, bind_group, [], 0, 999_999)
            render_pass.draw(3, 1, 0, 0)
            render_pass.end_pass()
            device.queue.submit([command_encoder.finish])
    canvas.request_draw(draw_frame)
glfw.init()
glfw.ERROR_REPORTING="warn"
canvas=wgpu.gui.glfw.WgpuCanvas(title="wgpu triangle with glfw")
main(canvas)
def better_event_loop(max_fps = 60):
    td=(((1.0    ))/(max_fps))
    while (wgpu.gui.glfw.update_glfw_canvasses()):
        now=time.perf_counter()
        tnext=((math.ceil(((now)/(td))))*(td))
        while (((now)<(tnext))):
            glfw.wait_events_timeout(((tnext)-(now)))
            now=time.perf_counter()
better_event_loop()
glfw.terminate()