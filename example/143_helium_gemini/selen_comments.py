# we will need helium and chrome driver

# micromamba activate
# pip install helium


from helium import *
from selenium.webdriver import ChromeOptions
import argparse
import time
url = "https://myactivity.google.com/page?page=youtube_comments"

# start a new chrome browser
# use my profile, so that I can use my saved passwords

options = ChromeOptions()
options.add_argument("user-data-dir=/home/martin/.config/google-chrome/")

# this fails for me if another chrome is running
driver = start_chrome(url, headless=False, options=options)

scroll_down(1000)

while True:
    try:
        scroll_down(num_pixels=10000)
        time.sleep(1)  # Wait for the page to load
    except:
        break  # No more comments to load


# There is no exception at the end.

# There is "Looks like you've reached the end"

#<div jsname="usC3We" class="SHC0R" tabindex="-1"><div jsname="jOfkMb" class="hV1B3e"><div>Looks like you've reached the end</div></div><div class="dkOpcb"><div class="EmVfjc Bqk8Ac" data-loadingmessage="Loading…" jscontroller="qAKInc" jsaction="animationend:kWijWc;dyRcpb:dyRcpb" jsname="aZ2wEe" data-active="false"><div class="Cg7hO" aria-live="assertive" jsname="vyyg5"></div><div jsname="Hxlbvc" class="xu46lf"><div class="ir3uv uWlRce co39ub"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div><div class="ir3uv GFoASc Cn087"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div><div class="ir3uv WpeOqd hfsr6b"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div><div class="ir3uv rHV3jf EjXFBf"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div></div></div></div><div jsname="PJEsad" class="jghBLc Bqk8Ac" aria-live="off" role="status"></div><div jsaction="JIbuQc:QyG3L"><button class="VfPpkd-LgbsSe ksBjEc lKxP2d LQeN7 Bqk8Ac" jscontroller="soHxf" jsaction="click:cOuCgd; mousedown:UX7yZ; mouseup:lbsD7e; mouseenter:tfO1Yc; mouseleave:JywGue; touchstart:p6p2H; touchmove:FwuNnf; touchend:yfqBxc; touchcancel:JMtRjd; focus:AHmuwe; blur:O22p3e; contextmenu:mg9Pef;mlnRJb:fLiPzd;" data-idom-class="ksBjEc lKxP2d LQeN7 Bqk8Ac" jsname="T8gEfd" jslog="73883; track:JIbuQc"><div class="VfPpkd-Jh9lGc"></div><div class="VfPpkd-J1Ukfc-LhBDec"></div><span jsname="V67aGc" class="VfPpkd-vQzf8d">Load more</span></button></div></div>