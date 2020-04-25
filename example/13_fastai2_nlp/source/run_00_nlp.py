# https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb
# export LANG=en_US.utf8
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from fastai2.text.all import *
path=untar_data(URLs.IMDB)
# => Path('/home/martin/.fastai/data/imdb')
files=get_text_files(path, folders=["train", "test", "unsup"])
txts=L(o.open().read() for o in files[:2000])
spacy=WordTokenizer()
tkn=Tokenizer(spacy)
sp=SubwordTokenizer(vocab_sz=1000)
sp.setup(txts)
toks200=txts[:200].map(tkn)
num=Numericalize()
num.setup(toks200)
nums200=toks200.map(num)
dl=LMDataLoader(nums200)
# From the book: At every epoch we shuffle our collection of documents and concatenate them into a stream of tokens. We then cut that stream into a batch of fixed-size consecutive mini-streams. Our model will then read the mini-streams in order, and thanks to an inner state, it will produce the same activation whatever sequence length you picked.
