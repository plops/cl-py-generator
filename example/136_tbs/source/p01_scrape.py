#!/usr/bin/env python3
import time
import shutil
import argparse
import keyboard
from tbselenium.tbdriver import TorBrowserDriver
start_time=time.time()
debug=True
_code_git_version="f1aaa8ef189309530adfd91535f1a02add55c163"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/136_tbs/source/"
_code_generation_time="00:03:10 of Tuesday, 2024-05-28 (GMT+1)"
parser=argparse.ArgumentParser(description="Scrape url.")
parser.add_argument("-s", "--socks_port", help="SOCKS port.", default=9150, type=int, action=None)
parser.add_argument("-c", "--control_port", help="Control port.", default=9151, type=int, action=None)
parser.add_argument("-b", "--browser-path", help="Path to browser.", default="/", type=str, action=None)
parser.add_argument("-D", "--download-path", help="Path where to store downloads.", default="/dev/shm/", type=str, action=None)
parser.add_argument("-g", "--geckodriver", help="Path to browser.", default="geckodriver", type=str, action=None)
parser.add_argument("-u", "--url", help="URL to scrape.", default="http://news.ycombinator.com/", type=str, action=None)
args=parser.parse_args()
gd=(shutil.which("geckodriver")) if (((args.geckodriver)==("geckodriver"))) else (args.geckodriver)
driver=TorBrowserDriver(args.browser_path, socks_port=args.socks_port, control_port=args.control_port, executable_path=gd, tbb_logfile_path="/dev/shm/ttb.log")
print("{} download args.url={}".format(((time.time())-(start_time)), args.url))
driver.get(args.url)
print("{} press enter ".format(((time.time())-(start_time))))
keyboard.press_and_release("enter")