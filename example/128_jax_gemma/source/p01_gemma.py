#!/usr/bin/env python3
# https://youtu.be/1RcORri2ZJg?t=418
!pip install -q git+https://github.com/google-deepmind/gemma.git 
import os
import time
import kagglehub
import gemma
import gemma.params
import gemma.transformer
import gemma.sampler
import sentencepiece as spm
from google.colab import userdata
start_time=time.time()
debug=True
_code_git_version="f5d544d9e51c9eea6c0933a4bb37b27afc93547c"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/128_jax_gemma/source/"
_code_generation_time="15:44:09 of Saturday, 2024-04-20 (GMT+1)"
kagglehub.login()
# Enable GPU in Colab: Click on Edit > Notebook settings > Select T4 GPU
# gemma-2b-it is 3.7Gb in size
GEMMA_VARIANT="2b-it"
GEMMA_PATH=kagglehub.model_download(f"google/gemma/flax/{GEMMA_VARIANT}")
# in addition to logging in with an api key into kaggle, i also had to manually submit a consent form on the kaggle website before i was able to download the gemma data
print("{} nil GEMMA_PATH={}".format(((time.time())-(start_time)), GEMMA_PATH))
# specify tokenizer model file and checkpoint
CKPT_PATH=os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH=os.path.join(GEMMA_PATH, "tokenizer.model")
print("{} nil CKPT_PATH={}".format(((time.time())-(start_time)), CKPT_PATH))
print("{} nil TOKENIZER_PATH={}".format(((time.time())-(start_time)), TOKENIZER_PATH))
params=gemma.params.load_and_format_params(CKPT_PATH)
# load tokenizer
vocab=spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)
transformer_config=gemma.transformer.TransformerConfig.from_params(params=params, cache_size=1024)
transformer=gemma.transformer.Transformer(transformer_config)
# create sampler
sampler=gemma.sampler.Sampler(transformer=transformer, vocab=vocab, params=params["transformer"])
# write prompt in input_batch and perform inference. total_generation_steps is limited to 100 here to preserve host memory
prompt=["\n# What is the meaning of life?"]
reply=sampler(input_strings=prompt, total_generation_steps=100)
print("{} nil reply.text={}".format(((time.time())-(start_time)), reply.text))