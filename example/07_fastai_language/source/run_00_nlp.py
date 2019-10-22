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
print("unfreeze 1 fine_tuned")
fn_fine=pathlib.Path("/home/martin/.fastai/data/imdb/models/fine_tuned.pth")
if ( fn_fine.is_file() ):
    learn.load("fine_tuned")
else:
    learn.unfreeze()
    learn.fit_one_cycle(10, (1.0000000474974513e-3), moms=((8.00000011920929e-1),(6.99999988079071e-1),))
    learn.save("fine_tuned")
    learn.save_encoder("fine_tuned_enc")
text="I liked this movie because"
n_words=40
n_sentences=2
sentences=[]
for _ in range(n_sentences):
    sentences.append(learn.predict(text, n_words, temperature=(7.499999999999999e-1)))
print("\n".join(sentences))
# %% load data for classification
fn=pathlib.Path("/home/martin/.fastai/data/imdb/data_class.pkl")
if ( fn.is_file() ):
    data_class=load_data(path, "data_class.pkl")
else:
    path=untar_data(URLs.IMDB)
    data_class=TextList.from_folder(path, vocab=data_lm.vocab).split_by_folder(valid="test").label_from_folder(classes=["neg"]).databunch(bs=bs)
    data_class.save("data_class.pkl")
# %%
learn=text_classifier_learner(data_class, AWD_LSTM, drop_mult=(5.e-1))
learn.load_encoder("fine_tuned_enc")
learn.lr_find()
learn.recorder.plot()