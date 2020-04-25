# https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb
# export LANG=en_US.utf8
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from fastai2.text.all import *
path=untar_data(URLs.IMDB)
# => Path('/home/martin/.fastai/data/imdb')
get_imdb=partial(get_text_files, folders=["train", "test", "unsup"])
dls_lm=DataBlock(blocks=TextBlock.from_folder(path, is_lm=True), get_items=get_imdb, splitter=RandomSplitter((0.10    ))).dataloaders(path, path=path, bs=64, seq_len=80)
learn=language_model_learner(dls_lm, AWD_LSTM, drop_mult=(0.30    ), metrics=[accuracy, Perplexity()]).to_fp16()
learn.fit_one_cycle(1, (2.00e-2))
learn.save("1epoch")