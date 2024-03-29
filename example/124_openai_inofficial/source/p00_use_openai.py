#|default_exp p00_use_openai
# python -m venv ~/bardapi_venv
# python -m venv ~/bardapi_env; . ~/bardapi_env/bin/activate; pip install pyChatGPT toml
# 
# deactivate
# https://github.com/terry3041/pyChatGPT
# env.toml contains this:
# [cookies]
# session_token = " ... " 
import os
import time
import toml
from pyChatGPT import ChatGPT
start_time=time.time()
debug=True
_code_git_version="75d87cea30200bf0d57682aef14089fef09e2e15"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/124_openai_inofficial/source/"
_code_generation_time="09:54:22 of Sunday, 2023-12-17 (GMT+1)"
# either set cookies here or read them from env.toml file
config_path="env.toml"
if ( os.path.exists(config_path) ):
    with open(config_path, "r") as f:
        data=toml.load(f)
    cookies=data["cookies"]
else:
    print("Warning: No env.toml file found.")
api=ChatGPT(cookies["session_token"], verbose=True)
response=api.send_message("tell me a joke")
print("{} vars response={}".format(((time.time())-(start_time)), response))