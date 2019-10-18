# lesson 1: image classification
# export LANG=en_US.utf8
# https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
path=untar_data(URLs.PETS)
print("using image data from: {}".format(path))
np.random.seed(2)
bs=64
# %%
path_anno, path_img=path.ls()
fnames=get_image_files(path_img)
pat=r"""/([^/]+)_\d+.jpg$"""
data=ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs)
data.normalize(imagenet_stats)
def look():
    data.show_batch(rows=3)
    print(data.classes)
# %%
learn=cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save("save-1")
# %%
interp=ClassificationInterpretation.from_learner(learn)
losses, idxs=interp.top_losses()
interp.plot_top_losses(9)
interp.plot_confusion_matrix()
interp.most_confused(min_val=2)