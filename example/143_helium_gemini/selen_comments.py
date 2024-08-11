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

# [1856 rows x 6 columns]


# >>> df.sort_values(by='number_of_likes2',ascending=False).iloc[0:30]
#                                                   title                                               href number_of_likes number_of_replies  number_of_likes2  number_of_replies2
# 1819  X-ray timelapse of fluid movement in plants, s...  https://www.youtube.com/watch?v=j-FHbHoiwNk&lc...             505                18             505.0                18.0
# 1643  Pushing a "legal" Mini Solar System to its LIMIT!  https://www.youtube.com/watch?v=KEru5cODA-M&lc...             445                20             445.0                20.0
# 1519  The Massive Molniya Satellites - How The Sovie...  https://www.youtube.com/watch?v=Nmhf2VK3xBE&lc...             176                 6             176.0                 6.0
# 1575  The Best Unintentional ASMR Medical Exam EVER ...  https://www.youtube.com/watch?v=8Y6eo-LHWo8&lc...             166                 5             166.0                 5.0
# 1573                A Hackers' Guide to Language Models  https://www.youtube.com/watch?v=jkrNMKz9pWU&lc...             140                 4             140.0                 4.0
# 365                   Das russische Momentum ist zurück  https://www.youtube.com/watch?v=XnZ5WsQTE88&lc...             118                 7             118.0                 7.0
# 1333  Die ukrainische Offensive ist gescheitert - Wa...  https://www.youtube.com/watch?v=WoP1i9ZzwOs&lc...             117                 3             117.0                 3.0
# 624   The Great Misinterpretation: How Palestinians ...  https://www.youtube.com/watch?v=QlK2mfYYm4U&lc...             101                12             101.0                12.0
# 1345    Colossal sea monster unearthed in UK - BBC News  https://www.youtube.com/watch?v=PDMIFXW5QB0&lc...             100                21             100.0                21.0
# 1343    Colossal sea monster unearthed in UK - BBC News  https://www.youtube.com/watch?v=PDMIFXW5QB0&lc...             100                21             100.0                21.0
# 1342    Colossal sea monster unearthed in UK - BBC News  https://www.youtube.com/watch?v=PDMIFXW5QB0&lc...             100                21             100.0                21.0
# 1596  The heat may not kill you, but the global food...  https://www.youtube.com/watch?v=kQkyouPOrD4&lc...              94                 3              94.0                 3.0
# 1349   🔴LIVE - DR DISRESPECT - WARZONE 3 - NUKE ATTEMPT  https://www.youtube.com/watch?v=ipM2VgLoEPc&lc...              90                 3              90.0                 3.0
# 1309  LDM #356: Teardown of two Russian missile sensors  https://www.youtube.com/watch?v=Ac2ioGwfsbI&lc...              78                 3              78.0                 3.0
# 1804                             Handheld DNA eraser?!?  https://www.youtube.com/watch?v=EXBAdajRuYw&lc...              72                 5              72.0                 5.0
# 631   Chef Wang shares: "Spicy Stir-fried Beef" & "S...  https://www.youtube.com/watch?v=fFUieUu3eE4&lc...              68                 5              68.0                 5.0
# 420   Sodium-ion batteries in the USA. Beating China...  https://www.youtube.com/watch?v=UmW2D_At1PY&lc...              60               NaN              60.0                 NaN
# 1554  How has NO ONE thought of this BEFORE? - WC3 -...  https://www.youtube.com/watch?v=sw894lgzvgE&lc...              59                 7              59.0                 7.0
# 1523  How has NO ONE thought of this BEFORE? - WC3 -...  https://www.youtube.com/watch?v=sw894lgzvgE&lc...              59                 7              59.0                 7.0
# 1668              Beating the UV step!! - Cubane Ep. 16  https://www.youtube.com/watch?v=GOZ-me235Mo&lc...              57                 5              57.0                 5.0
# 1502  Gideon Levy: Israel has no clue what it’s doin...  https://www.youtube.com/watch?v=lGg9QPdkfcs&lc...              55               NaN              55.0                 NaN
# 149   Scalable MatMul-free Language Modeling (Paper ...  https://www.youtube.com/watch?v=B45FlSQ8ITo&lc...              51                 2              51.0                 2.0
# 1578                  3 Ways to Make Sulfur Dioxide Gas  https://www.youtube.com/watch?v=2AFKlLSwUZ4&lc...              47                 5              47.0                 5.0
# 1267  Antikythera Fragment #11  World’s First Precis...  https://www.youtube.com/watch?v=4pK3O43Jddg&lc...              46                 3              46.0                 3.0
# 1271  Antikythera Fragment #11  World’s First Precis...  https://www.youtube.com/watch?v=4pK3O43Jddg&lc...              46                 3              46.0                 3.0
# 1614                     AI Reads Minds of 29 Patients!  https://www.youtube.com/watch?v=qcfC53c3tSc&lc...              37                 3              37.0                 3.0


# >>> df.sort_values(by='number_of_replies2',ascending=False).iloc[0:30]
#                                                   title                                               href number_of_likes number_of_replies  number_of_likes2  number_of_replies2
# 1342    Colossal sea monster unearthed in UK - BBC News  https://www.youtube.com/watch?v=PDMIFXW5QB0&lc...             100                21             100.0                21.0
# 1343    Colossal sea monster unearthed in UK - BBC News  https://www.youtube.com/watch?v=PDMIFXW5QB0&lc...             100                21             100.0                21.0
# 1345    Colossal sea monster unearthed in UK - BBC News  https://www.youtube.com/watch?v=PDMIFXW5QB0&lc...             100                21             100.0                21.0
# 1643  Pushing a "legal" Mini Solar System to its LIMIT!  https://www.youtube.com/watch?v=KEru5cODA-M&lc...             445                20             445.0                20.0
# 1819  X-ray timelapse of fluid movement in plants, s...  https://www.youtube.com/watch?v=j-FHbHoiwNk&lc...             505                18             505.0                18.0
# 624   The Great Misinterpretation: How Palestinians ...  https://www.youtube.com/watch?v=QlK2mfYYm4U&lc...             101                12             101.0                12.0
# 1644  My New Linear Actuators are SO MUCH BETTER! | ...  https://www.youtube.com/watch?v=bQl6gj_6oa8&lc...              17                10              17.0                10.0
# 1554  How has NO ONE thought of this BEFORE? - WC3 -...  https://www.youtube.com/watch?v=sw894lgzvgE&lc...              59                 7              59.0                 7.0
# 1523  How has NO ONE thought of this BEFORE? - WC3 -...  https://www.youtube.com/watch?v=sw894lgzvgE&lc...              59                 7              59.0                 7.0
# 524               Beyond the Noise #37: Measles is back  https://www.youtube.com/watch?v=Ee7irrt6v9c&lc...               7                 7               7.0                 7.0
# 365                   Das russische Momentum ist zurück  https://www.youtube.com/watch?v=XnZ5WsQTE88&lc...             118                 7             118.0                 7.0
# 569               Metal Casting a Hook with a Microwave  https://www.youtube.com/watch?v=hlMfovJ8BvQ&lc...               6                 6               6.0                 6.0
# 1285     How Chips That Power AI Work | WSJ Tech Behind  https://www.youtube.com/watch?v=njyZR0Ceke0&lc...              16                 6              16.0                 6.0
# 1519  The Massive Molniya Satellites - How The Sovie...  https://www.youtube.com/watch?v=Nmhf2VK3xBE&lc...             176                 6             176.0                 6.0
# 341              xLSTM: Extended Long Short-Term Memory  https://www.youtube.com/watch?v=0OaEv1a5jUM&lc...              26                 6              26.0                 6.0
# 347              xLSTM: Extended Long Short-Term Memory  https://www.youtube.com/watch?v=0OaEv1a5jUM&lc...              26                 6              26.0                 6.0
# 496   (483) ESP32 precision GPS receiver (incl. RTK-...  https://www.youtube.com/watch?v=Oc1LBFDj2MA&lc...              18                 6              18.0                 6.0
# 631   Chef Wang shares: "Spicy Stir-fried Beef" & "S...  https://www.youtube.com/watch?v=fFUieUu3eE4&lc...              68                 5              68.0                 5.0
# 370   Nuclear War Expert: 72 Minutes To Wipe Out 60%...  https://www.youtube.com/watch?v=asmaLnhaFiY&lc...              20                 5              20.0                 5.0
# 422   Nuclear War Expert: 72 Minutes To Wipe Out 60%...  https://www.youtube.com/watch?v=asmaLnhaFiY&lc...              20                 5              20.0                 5.0
# 1512                 Pumpkin Implosion at 600ft Deep???  https://www.youtube.com/watch?v=VDAzu0SHhgs&lc...              22                 5              22.0                 5.0
# 1521                 Pumpkin Implosion at 600ft Deep???  https://www.youtube.com/watch?v=VDAzu0SHhgs&lc...              22                 5              22.0                 5.0
# 1110  Restoring a Rare 1970s Heuer - The Ultimate Wa...  https://www.youtube.com/watch?v=BIbYg4ik-9Y&lc...               1                 5               1.0                 5.0
# 386   From 0 to Production - The Modern React Tutori...  https://www.youtube.com/watch?v=d5x0JCZbAJs&lc...               2                 5               2.0                 5.0
# 1108  Restoring a Rare 1970s Heuer - The Ultimate Wa...  https://www.youtube.com/watch?v=BIbYg4ik-9Y&lc...               1                 5               1.0                 5.0
# 1107  Restoring a Rare 1970s Heuer - The Ultimate Wa...  https://www.youtube.com/watch?v=BIbYg4ik-9Y&lc...               1                 5               1.0                 5.0
# 366   Nuclear War Expert: 72 Minutes To Wipe Out 60%...  https://www.youtube.com/watch?v=asmaLnhaFiY&lc...              20                 5              20.0                 5.0
# 1621                Make Potassium Chlorate from Bleach  https://www.youtube.com/watch?v=4Kq9r5vFoQU&lc...              29                 5              29.0                 5.0
# 1804                             Handheld DNA eraser?!?  https://www.youtube.com/watch?v=EXBAdajRuYw&lc...              72                 5              72.0                 5.0
# 242   From 0 to Production - The Modern React Tutori...  https://www.youtube.com/watch?v=d5x0JCZbAJs&lc...               2                 5               2.0                 5.0
