import numpy as np
from scipy.optimize import curve_fit, least_squares
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
plt.ion()

# Define Gaussian and Parabolic PSF models
def gaussian_psf(xy, x0, y0, A, sigma):
    x, y = xy
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def parabolic_psf(xy, x0, y0, a, b):
    x, y = xy
    return a * ((x - x0)**2 + (y - y0)**2) + b

# Generate simulated star image
def generate_star_image(psf_model, noise_model, shape=(50, 50), **kwargs):
    x, y = np.indices(shape)
    xy = (x.ravel(), y.ravel(),) #np.array([x.ravel(), y.ravel()]).T
    image = psf_model(xy, **kwargs).reshape(shape)
    if noise_model == 'gaussian':
        image += np.random.normal(scale=kwargs.get('noise_sigma', 0.1), size=shape)
    elif noise_model == 'poisson':
        image = np.random.poisson(lam=image)
    return image

# Fit PSF model to image data
def fit_psf(image, psf_model, footprint_shape='disk', footprint_size=5):
    # Extract ROI based on footprint
    center = np.array(image.shape) // 2
    if footprint_shape == 'disk':
        x, y = np.indices(image.shape)
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = distances <= footprint_size
    elif footprint_shape == 'box':
        mask = np.zeros_like(image, dtype=bool)
        mask[center[0] - footprint_size // 2:center[0] + footprint_size // 2 + 1,
             center[1] - footprint_size // 2:center[1] + footprint_size // 2 + 1] = True
    else:
        raise ValueError("Invalid footprint shape.")
    roi = image[mask]
    xy = np.array(np.where(mask)).T - center

    # Initial parameter guesses
    p0 = [0, 0, np.max(roi), 1]  # Adjust based on model
    popt, pcov = curve_fit(psf_model, xy.T, roi.ravel(), p0=p0)
    return popt, pcov

# Example usage:
image = generate_star_image(gaussian_psf, 'gaussian', x0=25.2, y0=25, A=100, sigma=2)

plt.imshow(image, origin='lower')

popt, pcov = fit_psf(image, gaussian_psf)
print("Fitted center:", popt[:2])