# https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb
# export LANG=en_US.utf8
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import pathlib
from fastai2.text.all import *
path=untar_data(URLs.IMDB)
# => Path('/home/martin/.fastai/data/imdb')
get_imdb=partial(get_text_files, folders=["train", "test", "unsup"])
dls_lm=DataBlock(blocks=TextBlock.from_folder(path, is_lm=True), get_items=get_imdb, splitter=RandomSplitter((0.10    ))).dataloaders(path, path=path, bs=64, seq_len=80)
learn=language_model_learner(dls_lm, AWD_LSTM, drop_mult=(0.30    ), metrics=[accuracy, Perplexity()]).to_fp16()
problem="imdb"
fn_1epoch="{}_1epoch".format(problem)
path_1epoch=pathlib.Path("/home/martin/.fastai/data/imdb/models/{}.pth".format(fn_1epoch))
if ( path_1epoch.is_file() ):
    learn=learn.load(fn_1epoch)
else:
    learn.fit_one_cycle(1, (2.00e-2))
    # => 16min45sec, 1min
    # epoch     train_loss  valid_loss  accuracy  perplexity  time
    # 0         4.152357    3.935240    0.297858  51.174419   17:51
    learn.save(fn_1epoch)
learn.unfreeze()
learn.fit_one_cycle(10, (2.00e-3))
learn.save_encoder("{}_finetuned".format(problem))