# AUTOGENERATED! DO NOT EDIT! File to edit: ../00_titanic_from_scratch.ipynb.

# %% auto 0
__all__ = ['start_time', 'debug', 'parser', 'args', 'path', 'line_char_width', 'df', 'modes', 't_dep', 'indep_columns', 't_indep',
           'trn', 'val', 'trn_indep', 'val_indep', 'trn_dep', 'val_dep', 'calc_preds', 'calc_loss', 'update_coeffs',
           'init_coeffs', 'one_epoch', 'train_model']

# %% ../00_titanic_from_scratch.ipynb 0
# |export
#|default_exp p00_titanic_from_scratch


# %% ../00_titanic_from_scratch.ipynb 1
# this file is based on https://github.com/fastai/course22/blob/master/05-linear-model-and-neural-net-from-scratch.ipynb
import os
import time
import pathlib
import argparse
import torch
from torch import tensor



# %% ../00_titanic_from_scratch.ipynb 2
start_time=time.time()
debug=True
_code_git_version="2e67ef303ddc9a764e1c6d00f33bb2ee183a6034"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/96_colab_fastai/source/"
_code_generation_time="20:26:10 of Sunday, 2022-08-28 (GMT+1)"
start_time=time.time()
debug=True


# %% ../00_titanic_from_scratch.ipynb 3
parser=argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="enable verbose output", action="store_true")
args=parser.parse_args()


# %% ../00_titanic_from_scratch.ipynb 4
# i want to run this on google colab. annoyingly i can't seem to access the titanic.zip file. it seems to be necessary to supply some kaggle login information in a json file. rather than doing this i downloaded the titanic.zip file into my google drive
import google.colab.drive
google.colab.drive.mount("/content/drive")


# %% ../00_titanic_from_scratch.ipynb 5
path=pathlib.Path("titanic")
if ( not(path.exists()) ):
    import zipfile
    zipfile.ZipFile(f"/content/drive/MyDrive/{path}.zip").extractall(path)


# %% ../00_titanic_from_scratch.ipynb 6
path=pathlib.Path("titanic")
if ( not(path.exists()) ):
    import zipfile
    import kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f"{path}.zip").extractall(path)


# %% ../00_titanic_from_scratch.ipynb 7
import torch
import numpy as np
import pandas as pd
line_char_width=140
np.set_print_options(linewidth=line_char_width)
torch.set_print_options(linewidth=line_char_width, sci_mode=False, edgeitems=7)
pd.set_option("display_width", line_char_width)


# %% ../00_titanic_from_scratch.ipynb 8
df=pd.read_csv(((path)/("train.csv")))
df


# %% ../00_titanic_from_scratch.ipynb 10
modes=df.mode().iloc[0]


# %% ../00_titanic_from_scratch.ipynb 11
df.fillna(modes, inplace=True)


# %% ../00_titanic_from_scratch.ipynb 15
df["LogFare"]=np.log(((1)+(df.Fare)))


# %% ../00_titanic_from_scratch.ipynb 19
# replace non-numeric values with numbers by introducing new columns (dummies). The dummy columns will be added to the dataframe df and the 3 original columns are dropped.
# Cabin, Name and Ticket contain too many unique values for this approach to be useful
df=pd.get_dummies(df, columns=["Sex", "Pclass", "Embarked"])
df.columns


# %% ../00_titanic_from_scratch.ipynb 21
# create dependent variable as tensor
t_dep=tensor(df.Survived)


# %% ../00_titanic_from_scratch.ipynb 22
# independent variables are all continuous variables of interest and the newly created columns
indep_columns=((["Age", "SibSp", "Parch", "LogFare"])+(added_columns))
t_indep=tensor(df[indep_columns].values, dtype=torch.float)
t_indep


# %% ../00_titanic_from_scratch.ipynb 30
# using what we learned in the previous cells create functions to compute predictions and loss
def calc_preds(coeffs=None, indeps=None):
    return ((indeps)*(coeffs)).sum(axis=1)
def calc_loss(coeffs=None, indeps=None, deps=None):
    preds=calc_preds(coeffs=coeffs, indeps=indeps)
    loss=torch.abs(((preds)-(deps))).mean()
    return loss


# %% ../00_titanic_from_scratch.ipynb 36
# before we can perform training, we have to create a validation dataset
# we do that in the same way as the fastai library does
import fastai.data.transforms
# get training (trn) and validation indices (val)
trn, val=(fastai.data.transforms.RandomSplitter(seed=42))((df))


# %% ../00_titanic_from_scratch.ipynb 37
trn_indep=t_indep[trn]
val_indep=t_indep[val]
trn_dep=t_dep[trn]
val_dep=t_dep[val]
len(trn_indep), len(val_indep)


# %% ../00_titanic_from_scratch.ipynb 38
# create 3 functions for the operations that were introduced in the previous cells
def update_coeffs(coeffs=None, learning_rate=None):
    coeffs.sub_(((coeffs.grad)*(learning_rate)))
    coeffs.grad.zero_()
def init_coeffs():
    coeffs=((torch.rand(n_coeffs))-((0.50    )))
    coeffs.requires_grad_()
    return coeffs
def one_epoch(coeffs=None, learning_rate=None):
    loss=calc_loss(coeffs=coeffs, indeps=trn_indep, deps=trn_dep)
    loss.backward()
    with torch.no_grad():
        update_coeffs(coeffs=coeffs, learning_rate=learning_rate)
    print(f"{loss:.3f}", end="; ")


# %% ../00_titanic_from_scratch.ipynb 39
# now use these functions to train the model
def train_model(epochs=30, learning_rate=(1.00e-2)):
    torch.manual_seed(442)
    coeffs=init_coeffs()
    for i in range(epochs):
        one_epoch(coeffs=coeffs, learning_rate=learning_rate)
    return coeffs

