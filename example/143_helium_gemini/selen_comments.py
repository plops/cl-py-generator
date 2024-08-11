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
