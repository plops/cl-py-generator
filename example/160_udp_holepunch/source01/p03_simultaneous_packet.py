#!/usr/bin/env python3
import subprocess
import sys
import os
def run_self_on_remote(host):
    with open(__file__, "rb") as f:
        script_content=f.read()
    cmd=["ssh", host, "python3", "-"]
    print(f"run {cmd}")
    result=subprocess.run(cmd, input=script_content, capture_output=True, check=False, text=False)
    print(f"remote script {result.returncode}")
if ( ((__name__)==("__main__")) ):
    print(f"start {sys.argv}")
    if ( ((sys.argv[0])==("-")) ):
        # remote running script name is '-'
        print("remote script")
    else:
        print("local script")
        run_self_on_remote("tux")