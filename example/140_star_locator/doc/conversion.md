# non-linear optimizing star locator


i want to create a 2d spot locator (e.g. measure position of stars in
an image). we assume with a roi that contains only one spot and is
centered to within one pixel of the center of the star. the brightness
of the star covers a finite amount of pixel (e.g. 3x3 or 9x9). this is
defined by the PSF of the telescope. for the complete method we assume
that the PSF is a rotationally symmetric gaussian. for the fast
locator method we will use a rotationally symmetric parabolic model to
fit the top of the brightness spot.

we can also make assumptions about the noise in the image (brightness
values can be gaussian distributed or follow a poissonian
distribution). i want to write python or R source code to simulate and
optimize the problem. i also want to write a white paper in latex to
describe the findings. and finally i want a modern C++ implementation
of a fast method (possibly using ceres-solver or nlopt). the questions
that i want to address are: 1) how many pixels and of what pixel shape
footprint (disk or box) should be included to obtain best fit. how is
this influenced by the noise of the brightness values (normally
distributed or poissonian)? how is this influenced by the model
(gaussian or parabolic). 2) can we use Gauss-Newton method (instead of
Levenberg-Marquardt) to refine the result from one frame to the next
(where only noise changes or the spot center shifts by less than a
pixel), or will this not converge (numerical simulation)

lmfit or some R optimization libraries provide tools to analyze the
fitting error and covariance between fit parameters. i think they look
at the hessian of the optimization problem to understand the
neighborhood of the fit optimization solution. i think this will be
useful. describe how we can measure the influence of noise or fitting
mask footprint.

