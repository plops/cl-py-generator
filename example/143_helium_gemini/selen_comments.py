# we will need helium and chrome driver

# micromamba activate
# pip install helium


from helium import *
from selenium.webdriver import ChromeOptions
import argparse
import time
from selenium.webdriver.common.by import By
import re
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

# pip install beautifulsoup4

from bs4 import BeautifulSoup


src = 'Google - My Activity.html' 

# Open the file and read the content
with open(src, 'r') as file:
    content = file.read()

# Parse the content with BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')

# find all links in the page
links = soup.find_all('a')

# Print the links that contain "&lc=" in the href attribute
clinks = []
for link in links:
    href = link.get('href')
    if href and '&lc=' in href:
        print(href)
        clinks.append(href)

href = clinks[200]

# Get the link
response = driver.get(href)
# scroll down so that the comments are loaded
scroll_down(num_pixels=10000)

# The second like button is for my comment

second_like = find_all(Button("Like"))[1]

# <button class="yt-spec-button-shape-next yt-spec-button-shape-next--text yt-spec-button-shape-next--mono yt-spec-button-shape-next--size-s yt-spec-button-shape-next--icon-button yt-spec-button-shape-next--override-small-size-icon" aria-pressed="false" aria-label="Like this comment along with 6 other people" title="" style="">...</button>

# I want the number from the aria-label attribute

label = second_like.web_element.get_attribute('aria-label')

# 'Like this comment along with 6 other people'

# I want the number 6

number_of_likes = label.split(' ')[-3]


# The parent of the like button is the comment get the outer html of the parent
outer = second_like.web_element.find_element(By.XPATH,'../../../../../../../..').get_attribute('outerHTML')

# print any number followed by " replies" in the outer html

replies = re.findall(r'(\d+) replies', outer)

number_of_replies = replies[0]
