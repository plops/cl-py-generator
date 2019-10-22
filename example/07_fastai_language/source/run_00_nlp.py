# lesson 4: transferlearning nlp
# export LANG=en_US.utf8
# https://www.youtube.com/watch?time_continue=60&v=qqt3aMPB81c
# predict next word of a sentence
# https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from fastai.text import *
path=untar_data(URLs.IMDB)
fn=pathlib.Path("/home/martin/.fastai/data/imdb_sample/data_save.pkl")
bs=48
if ( fn.is_file() ):
    data_lm=load_data(path, "data_lm.pkl")
else:
    data_lm=TextList.from_folder(path).filter_by_folder(include=["train", "test", "unsup"]).split_by_rand_pct((1.0000000149011612e-1)).label_for_lm().databunch(bs=bs)
    data_lm.save("data_lm.pkl")
learn=language_model_learner(data_lm, AWD_LSTM, drop_mult=(3.0000001192092896e-1))
fn_head=pathlib.Path("/home/martin/.fastai/data/imdb_sample/fit_head")
if ( fn_head.is_file() ):
    learn.load("fit_head")
else:
    learn.fit_one_cycle(1, (9.999999776482582e-3), moms=((8.00000011920929e-1),(6.99999988079071e-1),))
    learn.save("fit_head")