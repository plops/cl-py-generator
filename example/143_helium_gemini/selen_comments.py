# we will need helium and chrome driver

# micromamba activate
# pip install helium


from helium import *
from selenium.webdriver import ChromeOptions
import argparse
import time
import tqdm
import pandas as pd
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

import re
url = "https://myactivity.google.com/page?page=youtube_comments"

# start a new chrome browser
# use my profile, so that I can use my saved passwords

options = ChromeOptions()
options.add_argument("user-data-dir=/home/martin/.config/google-chrome/")

# maybe this prevents automatically playing each video
# --autoplay-policy=user-required

# this fails for me if another chrome is running
driver = start_chrome(url, headless=False, options=options)

scroll_down(1000)

# while True:
#     try:
#         scroll_down(num_pixels=10000)
#         time.sleep(1)  # Wait for the page to load
#     except:
#         break  # No more comments to load



# There is no exception at the end.

# There is "Looks like you've reached the end"

#<div jsname="usC3We" class="SHC0R" tabindex="-1"><div jsname="jOfkMb" class="hV1B3e"><div>Looks like you've reached the end</div></div><div class="dkOpcb"><div class="EmVfjc Bqk8Ac" data-loadingmessage="Loading…" jscontroller="qAKInc" jsaction="animationend:kWijWc;dyRcpb:dyRcpb" jsname="aZ2wEe" data-active="false"><div class="Cg7hO" aria-live="assertive" jsname="vyyg5"></div><div jsname="Hxlbvc" class="xu46lf"><div class="ir3uv uWlRce co39ub"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div><div class="ir3uv GFoASc Cn087"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div><div class="ir3uv WpeOqd hfsr6b"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div><div class="ir3uv rHV3jf EjXFBf"><div class="xq3j6 ERcjC"><div class="X6jHbb GOJTSe"></div></div><div class="HBnAAc"><div class="X6jHbb GOJTSe"></div></div><div class="xq3j6 dj3yTd"><div class="X6jHbb GOJTSe"></div></div></div></div></div></div><div jsname="PJEsad" class="jghBLc Bqk8Ac" aria-live="off" role="status"></div><div jsaction="JIbuQc:QyG3L"><button class="VfPpkd-LgbsSe ksBjEc lKxP2d LQeN7 Bqk8Ac" jscontroller="soHxf" jsaction="click:cOuCgd; mousedown:UX7yZ; mouseup:lbsD7e; mouseenter:tfO1Yc; mouseleave:JywGue; touchstart:p6p2H; touchmove:FwuNnf; touchend:yfqBxc; touchcancel:JMtRjd; focus:AHmuwe; blur:O22p3e; contextmenu:mg9Pef;mlnRJb:fLiPzd;" data-idom-class="ksBjEc lKxP2d LQeN7 Bqk8Ac" jsname="T8gEfd" jslog="73883; track:JIbuQc"><div class="VfPpkd-Jh9lGc"></div><div class="VfPpkd-J1Ukfc-LhBDec"></div><span jsname="V67aGc" class="VfPpkd-vQzf8d">Load more</span></button></div></div>

# pip install beautifulsoup4



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
        clinks.append(dict(title=link.text, href= href))

# elem = clinks[200]
# href = clinks[200]['href']

# print(f"Processing {elem['title']} with link {elem['href']}")
# href = elem['href']
# d = dict(title=elem['title'], href=elem['href'])
# # Get the link
# response = driver.get(href)
# # Wait for Download button to appear
# wait_until(Text("Download").exists)

# # scroll down so that the comments are loaded
# # this is with 50% zoom
# scroll_down(num_pixels=1500)
# time.sleep(.2)
# scroll_up(num_pixels=500)
# # Wait for text that contains "wolpumba" to appear
# wait_until(Text("@wolpumba4099").exists)

res = []
for elem in tqdm.tqdm(clinks):
    try:
        print(f"Processing {elem['title']} with link {elem['href']}")
        href = elem['href']
        d = dict(title=elem['title'], href=elem['href'])
        # Get the link
        response = driver.get(href)
        # Wait for Download button to appear
        wait_until(Text("Download").exists)

        # scroll down so that the comments are loaded
        # this seems to work with 50% zoom
        # i set the zoom to 25%, then the comments are loaded automatically without scrolling (i think)
        scroll_down(num_pixels=1500)
        time.sleep(.2)
        scroll_up(num_pixels=500)
        # Wait for text that contains "wolpumba" to appear
        wait_until(Text("@wolpumba4099").exists)

        # The second like button is for my comment

        second_like = find_all(Button("Like"))[1]

        # <button class="yt-spec-button-shape-next yt-spec-button-shape-next--text yt-spec-button-shape-next--mono yt-spec-button-shape-next--size-s yt-spec-button-shape-next--icon-button yt-spec-button-shape-next--override-small-size-icon" aria-pressed="false" aria-label="Like this comment along with 6 other people" title="" style="">...</button>

        # I want the number from the aria-label attribute

        label = second_like.web_element.get_attribute('aria-label')

        # 'Like this comment along with 6 other people'

        # I want the number 6

        number_of_likes = label.split(' ')[-3]
        d['number_of_likes'] = number_of_likes

        # The parent of the like button is the comment get the outer html of the parent
        outer = second_like.web_element.find_element(By.XPATH,'../../../../../../../..').get_attribute('outerHTML')

        # print any number followed by " replies" in the outer html

        replies = re.findall(r'(\d+) replies', outer)

        number_of_replies = replies[0]

        d['number_of_replies'] = number_of_replies
        res.append(d)
    except Exception as e:
        # print the exception and the line number of the exception
        print(e)

        res.append(d)

df = pd.DataFrame(res)
df.to_csv('comments.csv', index=False)

df['number_of_likes2'] = pd.to_numeric(df.number_of_likes)
df['number_of_replies2'] = pd.to_numeric(df.number_of_replies)

# >>> df.sort_values(by='number_of_replies',ascending=False)
#                                                   title                                               href number_of_likes number_of_replies  number_of_likes2  number_of_replies2
# 524               Beyond the Noise #37: Measles is back  https://www.youtube.com/watch?v=Ee7irrt6v9c&lc...               7                 7               7.0                 7.0
# 1554  How has NO ONE thought of this BEFORE? - WC3 -...  https://www.youtube.com/watch?v=sw894lgzvgE&lc...              59                 7              59.0                 7.0
# 365                   Das russische Momentum ist zurück  https://www.youtube.com/watch?v=XnZ5WsQTE88&lc...             118                 7             118.0                 7.0
# 1523  How has NO ONE thought of this BEFORE? - WC3 -...  https://www.youtube.com/watch?v=sw894lgzvgE&lc...              59                 7              59.0                 7.0
# 1519  The Massive Molniya Satellites - How The Sovie...  https://www.youtube.com/watch?v=Nmhf2VK3xBE&lc...             176                 6             176.0                 6.0
# ...                                                 ...                                                ...             ...               ...               ...                 ...

# [1856 rows x 6 columns]
# >>> df.sort_values(by='number_of_likes',ascending=False)
#                                                   title                                               href number_of_likes number_of_replies  number_of_likes2  number_of_replies2
# 1596  The heat may not kill you, but the global food...  https://www.youtube.com/watch?v=kQkyouPOrD4&lc...              94                 3              94.0                 3.0
# 1349   🔴LIVE - DR DISRESPECT - WARZONE 3 - NUKE ATTEMPT  https://www.youtube.com/watch?v=ipM2VgLoEPc&lc...              90                 3              90.0                 3.0
# 438                                 Platonic Hypothesis  https://www.youtube.com/watch?v=Q9DCL_m_haw&lc...               9               NaN               9.0                 NaN
# 202          Is the Kar98 STILL GOOD After Season 4.5??  https://www.youtube.com/watch?v=meG6a2NFJes&lc...               9               NaN               9.0                 NaN
# 849   TNP #47 - SHF 100CP 20GHz Broadband GaAs FET A...  https://www.youtube.com/watch?v=nz43MzAaJWI&lc...               9               NaN               9.0                 NaN
# ...                                                 ...                                                ...             ...               ...               ...                 ...

# [1856 rows x 6 columns]
