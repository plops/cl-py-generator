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
fn_finetuned="{}_finetuned".format(problem)
path_finetuned=pathlib.Path("/home/martin/.fastai/data/imdb/models/{}.pth".format(fn_finetuned))
fn_classifier="{}_classifier".format(problem)
if ( path_finetuned.is_file() ):
    print("load finetuned encoder")
    learn=learn.load(fn_1epoch)
    learn=learn.load_encoder(fn_finetuned)
else:
    if ( path_1epoch.is_file() ):
        print("load preexisting 1epoch")
        learn=learn.load(fn_1epoch)
        print("finetune encoder will take 10x 20min")
        learn.unfreeze()
        learn.fit_one_cycle(10, (2.00e-3))
        learn.save_encoder(fn_finetuned)
    else:
        print("compute 1epoch (takes 18min)")
        learn.fit_one_cycle(1, (2.00e-2))
        # => 16min45sec, 1min
        # epoch     train_loss  valid_loss  accuracy  perplexity  time
        # 0         4.152357    3.935240    0.297858  51.174419   17:51
        learn.save(fn_1epoch)
print("create classifier")
dls_class=DataBlock(blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock,), get_y=parent_label, get_items=partial(get_text_files, folders=["train", "test"]), splitter=GrandparentSplitter(valid_name="test")).dataloaders(path, path=path, bs=412, seq_len=72)
learn=text_classifier_learner(dls_class, AWD_LSTM, drop_mult=(0.50    ), metrics=accuracy).to_fp16()
learn=learn.load_encoder(fn_finetuned)
learn.fit_one_cycle(1, (2.00e-2))
learn.freeze_to(-2)
val=(1.00e-2)
learn.fit_one_cycle(1, slice(((val)/((((2.60    ))**(4)))), val))
learn.freeze_to(-3)
val=(5.00e-3)
learn.fit_one_cycle(1, slice(((val)/((((2.60    ))**(4)))), val))
learn.unfreeze()
val=(1.00e-3)
learn.fit_one_cycle(2, slice(((val)/((((2.60    ))**(4)))), val))
learn.save(fn_classifier)