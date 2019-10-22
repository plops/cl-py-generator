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
fn=pathlib.Path("/home/martin/.fastai/data/imdb/data_lm.pkl")
bs=48
if ( fn.is_file() ):
    print("load lm data from pkl")
    data_lm=load_data(path, "data_lm.pkl")
else:
    print("load lm data from disk")
    data_lm=TextList.from_folder(path).filter_by_folder(include=["train", "test", "unsup"]).split_by_rand_pct((1.0000000149011612e-1)).label_for_lm().databunch(bs=bs)
    data_lm.save("data_lm.pkl")
learn=language_model_learner(data_lm, AWD_LSTM, drop_mult=(3.0000001192092896e-1))
fn_head=pathlib.Path("/home/martin/.fastai/data/imdb/models/fit_head.pth")
if ( fn_head.is_file() ):
    print("load language model")
    learn.load("fit_head")
else:
    print("train language model")
    learn.fit_one_cycle(1, (9.999999776482582e-3), moms=((8.00000011920929e-1),(6.99999988079071e-1),))
    learn.save("fit_head")
fn_fine=pathlib.Path("/home/martin/.fastai/data/imdb/models/fine_tuned.pth")
if ( fn_fine.is_file() ):
    print("load fine tuned language model")
    learn.load("fine_tuned")
else:
    print("unfreeze and refine language model")
    learn.unfreeze()
    learn.fit_one_cycle(10, (1.0000000474974513e-3), moms=((8.00000011920929e-1),(6.99999988079071e-1),))
    learn.save("fine_tuned")
    learn.save_encoder("fine_tuned_enc")
# %% load data for classification
fn=pathlib.Path("/home/martin/.fastai/data/imdb/data_class.pkl")
if ( fn.is_file() ):
    print("load data for classification from pkl")
    data_class=load_data(path, "data_class.pkl")
else:
    print("load data for classification from disk")
    path=untar_data(URLs.IMDB)
    data_class=TextList.from_folder(path, vocab=data_lm.vocab).split_by_folder(valid="test").label_from_folder(classes=["neg"]).databunch(bs=bs)
    data_class.save("data_class.pkl")
# %%
print("learn classifier")
learn=text_classifier_learner(data_class, AWD_LSTM, drop_mult=(5.e-1))
learn.load_encoder("fine_tuned_enc")
learn.fit_one_cycle(1, (1.9999999552965164e-2), moms=((8.00000011920929e-1),(6.99999988079071e-1),))