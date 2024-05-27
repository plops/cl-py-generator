#!/usr/bin/env python3
import os
import time
import shutil
import argparse
import keyboard
from tbselenium.tbdriver import TorBrowserDriver
start_time=time.time()
debug=True
_code_git_version="aa569b9313a90c92660dd7eb5d1e445a2a009935"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/136_tbs/source/"
_code_generation_time="00:41:02 of Tuesday, 2024-05-28 (GMT+1)"
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
rev
print("{} download args.url={}".format(((time.time())-(start_time)), args.url))
# Large files may cause a timeout after 300s, resulting in a selenium.common.exceptions.TimeoutException.
# To handle this, we catch the exception and wait for the download to complete by monitoring the downloads directory.
# During the download, the browser creates a temporary file with the same stem as the downloaded file. This filename may also contain arbitrary characters behind the stem and ends with the extension .part.
# Once the download is complete, the temporary file is renamed to the final filename.
# The script waits until this renaming process is complete before terminating.
try:
    driver.get(args.url)
except Exception as e:
    print("{} nil e={}".format(((time.time())-(start_time)), e))
    print("{} waiting for download to finish ".format(((time.time())-(start_time))))
    file_stem=args.url.split("/")[-1].split(".")[0]
    print("{} nil file_stem={}".format(((time.time())-(start_time)), file_stem))
    while (True):
        if ( any([(((file_stem in f)) and (f.endswith(".part"))) for f in os.listdir(args.download_path)]) ):
            time.sleep(1)
        else:
            break