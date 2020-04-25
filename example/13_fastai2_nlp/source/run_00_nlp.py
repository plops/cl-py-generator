# https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb
# export LANG=en_US.utf8
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from fastai2.text.all import *
path=untar_data(URLs.IMDB)
# => Path('/home/martin/.fastai/data/imdb')
files=get_text_files(path, folders=["train", "test", "unsup"])
spacy=WordTokenizer()
tkn=Tokenizer(spacy)
txts=L(o.open().read() for o in files[:2000])