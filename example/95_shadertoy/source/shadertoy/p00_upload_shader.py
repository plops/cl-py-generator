# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_upload_shader.ipynb.

# %% auto 0
__all__ = ['start_time', 'debug', 'parser', 'args', 'cm']

# %% ../00_upload_shader.ipynb 0
# |export
#|default_exp p00_upload_shader


# %% ../00_upload_shader.ipynb 1
import time
import argparse
from helium import *



# %% ../00_upload_shader.ipynb 2
start_time=time.time()
debug=True
_code_git_version="148ebadf170c1d594ee15ed84170e201497916e8"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time="21:14:44 of Wednesday, 2022-08-17 (GMT+1)"
start_time=time.time()
debug=True


# %% ../00_upload_shader.ipynb 3
parser=argparse.ArgumentParser()
parser.add_argument("-p", "--password", help="password", required="True", action=None)
parser.add_argument("-v", "--verbose", help="enable verbose output", required="True", action="store_true")
args=parser.parse_args()


# %% ../00_upload_shader.ipynb 4
start_chrome("https://www.shadertoy.com/view/7t3cDs", headless=False)


# %% ../00_upload_shader.ipynb 5
if ( debug ):
    print("{} login with password ".format(((time.time())-(start_time))))
click("Accept")
click("Sign In")
write("plops")
press(TAB)
write(args.password)
click("Sign In")


# %% ../00_upload_shader.ipynb 6
if ( debug ):
    print("{} clear text ".format(((time.time())-(start_time))))
cm=S("//div[contains(@class,'CodeMirror')]")
click(cm)
list(map(lambda x: press(ARROW_UP), range(12)))
list(map(lambda x: press(((SHIFT)+(DELETE))), range(12)))
if ( debug ):
    print("{} update the text ".format(((time.time())-(start_time))))
write(r"""void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.1+ 0.25*cos(iTime+uv.xyx+vec3(1,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}""")


# %% ../00_upload_shader.ipynb 7
if ( debug ):
    print("{} compile code ".format(((time.time())-(start_time))))
click(S("#compileButton"))
click("Save")

