# lesson 1: image classification
# export LANG=en_US.utf8
from fastai import *
from fastai.vision import *
from PIL import Image
path=untar_data(URLs.PETS)
print("using image data from: {}".format(path))
np.random.seed(2)
# %%
path_anno, path_img=path.ls()
fnames=get_image_files(path_img)
pat=r"""/([^/]+)_\d+.jpg$"""
data=ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)