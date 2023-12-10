#|default_exp p00_use_bard
# python -m venv ~/bardapi_env; . ~/bardapi_env/bin/activate; pip install bardapi toml
# 
# deactivate
import os
import time
import toml
from bardapi import BardCookies
start_time=time.time()
debug=True
_code_git_version="8e29c746ccaeea65fcf0c8050e6fb7258c9c6ba6"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/122_bard/source/"
_code_generation_time="11:22:17 of Sunday, 2023-12-10 (GMT+1)"
# either set cookies here or read them from env.toml file
config_path="env.toml"
if ( os.path.exists(config_path) ):
    with open(config_path, "r") as f:
        data=toml.load(f)
    cookies=data["cookies"]
    print("{} Read cookies from config file config_path={} cookies={}".format(((time.time())-(start_time)), config_path, cookies))
else:
    print("Warning: No .env file found. Please provide API cookies manually.")
    cookies={("__Secure-1PSID"):("stub"),("__Secure-1PSIDCC"):("stub"),("__Secure-1PAPISID"):("stub")}
bard=BardCookies(cookie_dict=cookies)
with open("transcript", "r") as f:
    text=f.read()
request=(("Summarize as bullet list: ```")+(text)+("```"))
print("{} request request={}".format(((time.time())-(start_time)), request))
print(bard.get_answer(request)["content"])