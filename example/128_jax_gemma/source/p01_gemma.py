#!/usr/bin/env python3
# https://youtu.be/1RcORri2ZJg?t=418
import os
import kagglehub
from google.colab import userdata
start_time=time.time()
debug=True
_code_git_version="afe65391c4f1ef87e294ca5233b5ccd182abf18a"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/128_jax_gemma/source/"
_code_generation_time="13:26:46 of Saturday, 2024-04-20 (GMT+1)"
os.environ["KAGGLE_USERNAME"]=userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]=userdata.get("KAGGLE_KEY")
# Enable GPU in Colab: Click on Edit > Notebook settings > Select T4 GPU
# !pip install -q git+https://github.com/google-deepmind/gemma.git 
# gemma-2b-it is 3.7Gb in size
GEMMA_VARIANT="2b-it"
GEMMA_PATH=kagglehub.model_download(f"google/gemma/flax/{GEMMA_VARIANT}")