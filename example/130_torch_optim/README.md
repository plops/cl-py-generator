
use pytorch to represent a image processing model (color conversion
from camera rgb to sRGB with gamma 2.2, brigthness, offset, hue and
conversion to yuv). torch shall be used to represent the model and to
determine the gradient. the model shall be called to fill a pandas
dataframe with rgb values and corresponding yuv values. lmfit shall
call the model and gradient to find the original parameters of the
model using optimization.
use MSE loss between predicted and target YUV


i think torch doesn't work well together with lmfit
