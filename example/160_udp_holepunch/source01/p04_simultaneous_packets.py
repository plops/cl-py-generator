#!/usr/bin/env python3
import subprocess
import sys
import os
import socket
import time
from subprocess import Popen, PIPE, DEVNULL
port=60001
msg=b"A"
client_ip="14.40.11.250"
server_ip="13.4.0.26"
any="0.0.0.0"
def emit(src, dst):
    try:
        sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((src,port,))
        for i in range(33):
            sock.sendto(msg, (dst,port,))
            time.sleep((0.10    ))
    except Exception as e:
        print(f"exception {e}")
    finally:
        sock.close()
def run_self_on_remote(host):
    with open(__file__, "rb") as f:
        script_content=f.read()
    cmd=["ssh", host, "python3", "-"]
    print(f"run {cmd}")
    process=Popen(cmd, stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
    process.stdin.write(script_content)
    process.stdin.close()
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