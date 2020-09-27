#!/usr/bin/python3
from fastai.vision.all import *
# %%
_code_git_version="c5557a12bcc9003c00cb2eaffedd98c5e1b4913f"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/25_fastai2_again/source/run_00_pets.py"
_code_generation_time="21:54:20 of Sunday, 2020-09-27 (GMT+1)"
path=((untar_data(URLs.PETS))/("images"))
def is_cat(x):
    return x[0].isupper()
dls=ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=(0.20    ), seed=42, label_func=is_cat, item_tfms=Resize(224))
learn=cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)