#|default_exp p00_get_comments
# python -m venv ~/scraper_env; . ~/scraper_env/bin/activate; pip install pandas helium beautifulsoup4
# helium-3.2.5 beautifulsoup4-4.12.2 pandas-2.1.0
# deactivate
import os
import time
import pandas as pd
import datetime
from helium import *
from bs4 import BeautifulSoup
start_time=time.time()
debug=True
_code_git_version="c4e9cfdde50d6135bed970993b5a0acac807556a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/116_yt_comment_scraper/source/"
_code_generation_time="23:23:59 of Thursday, 2024-05-09 (GMT+1)"
url="https://myactivity.google.com/page?hl=en&utm_medium=web&utm_source=youtube&page=youtube_comments"
start_chrome()
go_to(url)
try:
    wait_until(Button("Sign In").exists, timeout_secs=3)
except Exception as e:
    print(e)
click("Sign In")
email_str="aestiae@gmail.com"
write(email_str)
press(ENTER)
# This browser or app may not be secure. Learn more
# How to get cookies
# Developer Tools (right-click -> Inspect)
# Application tab -> Storage -> Cookies 
# https://stackoverflow.com/questions/48869775/how-can-we-use-local-chromes-cookies-to-login-using-selenium
# how to use youtube api https://youtu.be/m0RWSHdS77E
# https://developers.google.com/youtube/v3
