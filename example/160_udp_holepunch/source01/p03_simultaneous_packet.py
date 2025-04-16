#!/usr/bin/env python3
import subprocess
import sys
import os
import socket
port=60001
msg=b"A"
client_ip="24.40.61.10"
server_ip="3.5.1.16"
any="0.0.0.0"
def emit(src, dst):
    try:
        sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((src,port,))
        sock.sendto(msg, (dst,port,))
    except Exception as e:
        print(f"exception {e}")
    finally:
        sock.close()
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
        emit(any, client_ip)
    else:
        print("local script")
        run_self_on_remote("tux")
        emit(any, server_ip)