# we will need helium and chrome driver

# micromamba activate
# pip install helium


from helium import *
from selenium.webdriver import ChromeOptions
import argparse
import time
url = "https://aistudio.google.com/app/prompts/new_chat"

# start a new chrome browser
# use my profile, so that I can use my saved passwords


# https://developer.chrome.com/docs/chromedriver/capabilities
# https://peter.sh/experiments/chromium-command-line-switches/

options = ChromeOptions()
options.add_argument("user-data-dir=/home/martin/.config/google-chrome/")

# this fails for me if another chrome is running
driver = start_chrome(url, headless=False, options=options)


# parse command line argument "-f <file>", read file into prompt2 variable
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="file to read")
args = parser.parse_args()
if args.file:
    with open(args.file) as f:
        prompt2 = f.read()

#print ("increase temperature")
#time.sleep(6)

#write("2", into="1")

time.sleep(4)
prompt1 = "I don't want to watch the video. Create a self-contained bullet list summary from the following transcript that I can understand without watching the video. "
print("Inserting prompt1: ", prompt1)
tf = TextField("Type something")
write(prompt1, into=tf)
# split prompt2 into chunks of 128 characters and submit in sequence
for i in range(0, len(prompt2), 128):
    write(prompt2[i:i+128], into=tf)
    time.sleep(.2)


#click("Edit safety settings")
#click("slider")
#click("X")



#time.sleep(4)
#wait_until(Button('Run').exists)

#click("Run")

#write("Add starting (not stopping) timestamp to each bullet point", into="Type something")

# kill_browser()