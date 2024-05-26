#!/usr/bin/env python3
import time
import shutil
import argparse
from tbselenium.tbdriver import TorBrowserDriver
start_time=time.time()
debug=True
_code_git_version="2991e1950480b44df8c1bd949e01914268a19f69"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/136_tbs/source/"
_code_generation_time="00:55:34 of Monday, 2024-05-27 (GMT+1)"
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
driver.capabilities["se:downloadsEnabled"]=True
driver.download_file(args.url, args.download_path)