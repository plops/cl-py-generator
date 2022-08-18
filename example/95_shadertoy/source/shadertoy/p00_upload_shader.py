# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_upload_shader.ipynb.

# %% auto 0
__all__ = ['start_time', 'debug', 'parser', 'args', 'url', 'cm']

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
_code_git_version="5f605373cd928b4deb4b483e8197cf4bbadb536c"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/56_myhdl/source/04_tang_lcd/run_04_lcd.py"
_code_generation_time="17:04:03 of Thursday, 2022-08-18 (GMT+1)"
start_time=time.time()
debug=True


# %% ../00_upload_shader.ipynb 3
parser=argparse.ArgumentParser()
parser.add_argument("-p", "--password", help="password", action=None)
parser.add_argument("-i", "--input", help="input file", action=None)
parser.add_argument("-H", "--headless", help="enable headless modex", action="store_true")
parser.add_argument("-k", "--kill", help="kill browser at the end", action="store_true")
parser.add_argument("-v", "--verbose", help="enable verbose output", action="store_true")
args=parser.parse_args()


# %% ../00_upload_shader.ipynb 4
if ( args.verbose ):
    print("{}  args={}".format(((time.time())-(start_time)), args))
if ( args.verbose ):
    print("{} start chrome args.headless={}".format(((time.time())-(start_time)), args.headless))
url="https://www.shadertoy.com/view/7t3cDs"
if ( args.verbose ):
    print("{} go to url={}".format(((time.time())-(start_time)), url))
start_chrome(url, headless=args.headless)


# %% ../00_upload_shader.ipynb 5
if ( args.verbose ):
    print("{} wait for cookie banner ".format(((time.time())-(start_time))))
wait_until(Button("Accept").exists)
click("Accept")
if ( args.verbose ):
    print("{} login with password ".format(((time.time())-(start_time))))
click("Sign In")
write("plops")
press(TAB)
write(args.password)
click("Sign In")


# %% ../00_upload_shader.ipynb 6
if ( args.verbose ):
    print("{} clear text ".format(((time.time())-(start_time))))
cm=S("//div[contains(@class,'CodeMirror')]")
click(cm)
if ( args.verbose ):
    print("{} select all ".format(((time.time())-(start_time))))
press(((CONTROL)+("a")))
if ( args.verbose ):
    print("{} delete ".format(((time.time())-(start_time))))
press(DELETE)
if ( args.verbose ):
    print("{} load source from args.input={}".format(((time.time())-(start_time)), args.input))
with open(args.input) as f:
    s=f.read()
if ( args.verbose ):
    print("{} update the text s={}".format(((time.time())-(start_time)), s))
write(s)


# %% ../00_upload_shader.ipynb 7
if ( args.verbose ):
    print("{} compile code ".format(((time.time())-(start_time))))
click(S("#compileButton"))
if ( args.verbose ):
    print("{} save ".format(((time.time())-(start_time))))
click("Save")
if ( args.verbose ):
    print("{} wait for save to finish ".format(((time.time())-(start_time))))
wait_until(Button("Save").exists)


# %% ../00_upload_shader.ipynb 8
if ( args.kill ):
    if ( args.verbose ):
        print("{} close browser ".format(((time.time())-(start_time))))
    kill_browser()

