# non-linear optimizing star locator

the following is a conversion with gemini 1.5 pro:

User: i want to create a 2d spot locator (e.g. measure position of stars in
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

## Star Spot Locator Project: A Comprehensive Guide

This project aims to develop and analyze a 2D star spot locator, focusing on accuracy, efficiency, and robustness against noise. We'll explore various aspects, including the optimal fitting region, the influence of noise models, and the performance of different optimization techniques.

**1. Simulation and Optimization in Python:**

```python
import numpy as np
from scipy.optimize import curve_fit, least_squares
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt

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
    xy = np.array([x.ravel(), y.ravel()]).T
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
image = generate_star_image(gaussian_psf, 'gaussian', x0=25, y0=25, A=100, sigma=2)
popt, pcov = fit_psf(image, gaussian_psf)
print("Fitted center:", popt[:2])
```

**2. Analysis and Visualization in Python:**

```python
# Analyze fitting errors and covariance
perr = np.sqrt(np.diag(pcov))
print("Parameter errors:", perr)

# Visualize results
plt.imshow(image, cmap='gray')
plt.scatter(popt[1], popt[0], marker='x', color='red', label='Fitted Center')
plt.legend()
plt.show()
```

**3.  Modern C++ Implementation with Ceres-Solver:**

```cpp
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

// Define cost functor for Ceres optimization
struct ParabolicCostFunctor {
  ParabolicCostFunctor(double x, double y, double intensity) : x_(x), y_(y), intensity_(intensity) {}

  template <typename T>
  bool operator()(const T* const parameters, T* residual) const {
    const T x0 = parameters[0];
    const T y0 = parameters[1];
    const T a = parameters[2];
    const T b = parameters[3];

    residual[0] = intensity_ - (a * ceres::pow((x_ - x0), 2) + a * ceres::pow((y_ - y0), 2) + b);
    return true;
  }

 private:
  const double x_;
  const double y_;
  const double intensity_;
};

// ... Load and process image, extract ROI ...

// Create Ceres problem and add residuals
ceres::Problem problem;
for (int i = 0; i < roi.rows; ++i) {
  for (int j = 0; j < roi.cols; ++j) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ParabolicCostFunctor, 1, 4>(
            new ParabolicCostFunctor(j, i, roi.at<double>(i, j))),
        nullptr, parameters);
  }
}

// Set solver options and solve
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_QR;
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);

// Extract and display results
std::cout << summary.BriefReport() << std::endl;
double x0 = parameters[0];
double y0 = parameters[1];
std::cout << "Fitted center: (" << x0 << ", " << y0 << ")" << std::endl;
```

**4.  LaTeX White Paper Outline:**

**Title:**  Accurate and Efficient Star Spot Localization in Astronomical Images

**Abstract:** This paper presents a comprehensive study of star spot localization in astronomical images, focusing on the optimization of fitting regions, the influence of noise models, and the performance of different optimization techniques. 

**1. Introduction:**
   - Background on astronomical image analysis
   - Importance of precise star spot localization
   - Overview of existing methods and their limitations

**2. Methodology:**
   - Description of the proposed approach
   - Gaussian and parabolic PSF models
   - Gaussian and Poissonian noise models
   - Optimization techniques: Levenberg-Marquardt and Gauss-Newton
   - Metrics for evaluating performance

**3. Results and Discussion:**
   - Analysis of optimal fitting region size and shape
   - Impact of noise models on localization accuracy
   - Comparison of optimization techniques and convergence behavior
   - Investigation of fitting error and covariance using lmfit

**4. Conclusion:**
   - Summary of key findings
   - Advantages and limitations of the proposed approach
   - Future work and potential applications

**5. References**

**Addressing Specific Questions:**

1. **Optimal Footprint:**
   - Systematically vary footprint size and shape (disk/box).
   - Quantify localization accuracy (e.g., RMSE) for each combination.
   - Analyze the influence of Gaussian and Poissonian noise on optimal footprint selection.
   - Investigate the impact of using Gaussian vs. parabolic models.

2. **Gauss-Newton Convergence:**
   - Implement both Levenberg-Marquardt and Gauss-Newton methods.
   - Simulate scenarios with varying noise levels and spot displacements.
   - Evaluate convergence properties (iterations, success rate).
   - Analyze conditions under which Gauss-Newton is a viable alternative.

3. **Analyzing Fitting Error and Covariance:**
   - Utilize lmfit or R optimization libraries to obtain covariance matrices.
   - Interpret covariance values to understand parameter correlations.
   - Quantify the influence of noise and footprint size on parameter uncertainties.

By following this detailed guide, you'll be able to develop, analyze, and document your star spot locator project effectively, achieving both accuracy and efficiency in your star localization tasks. 


User: Elaborate on this: Analyzing Fitting Error and Covariance:
Utilize lmfit or R optimization libraries to obtain covariance
matrices.  Interpret covariance values to understand parameter
correlations.  Quantify the influence of noise and footprint size on
parameter uncertainties.


## Delving Deeper into Analyzing Fitting Error and Covariance:

Analyzing the fitting error and covariance matrix provides crucial insights into the reliability and robustness of your star spot localization. Here's how you can leverage `lmfit` (Python) or similar optimization libraries to quantify the influence of noise and footprint size:

**1. Obtaining the Covariance Matrix:**

Both `lmfit` and R optimization libraries (like `optim` or `nls`) provide access to the estimated covariance matrix after fitting. 

* **lmfit (Python):**
    ```python
    from lmfit import minimize, Parameters, report_fit

    # ... (Define your model, data, and perform fitting as before) ...

    result = minimize(residual_function, params, args=(xdata, ydata))
    print(report_fit(result)) # This will print the covariance matrix

    # Accessing specific elements:
    covariance_matrix = result.covar
    parameter_a_variance = covariance_matrix[0, 0] 
    parameter_a_b_covariance = covariance_matrix[0, 1]
    ```

* **R:**
    ```R
    # ... (Define your model, data, and perform fitting) ...
    fit <- nls(y ~ model_function(x, a, b), data = data, start = list(a = a_init, b = b_init))

    # Accessing covariance matrix
    cov_matrix <- vcov(fit)
    print(cov_matrix)
    ```

**2. Interpreting Covariance Values:**

* **Diagonal Elements (Variances):** The diagonal elements of the
  covariance matrix represent the variances of the estimated
  parameters. A larger variance indicates greater uncertainty in the
  parameter estimate.
* **Off-Diagonal Elements (Covariances):** Off-diagonal elements
  represent covariances between pairs of parameters.
    - **Positive Covariance:** A positive value suggests that when one
      parameter is overestimated, the other is also likely to be
      overestimated.
    - **Negative Covariance:** A negative value implies that
      overestimating one parameter is associated with underestimating
      the other.
    - **Near-Zero Covariance:** Values close to zero suggest weak or
      no linear correlation between parameters.

**3. Quantifying Influence of Noise and Footprint Size:**

**a) Varying Noise Levels:**

1.  **Simulate Data:** Generate multiple datasets with varying noise
    levels (different standard deviations for Gaussian noise or lambda
    values for Poisson noise) while keeping the true spot location and
    other parameters constant.
2. **Fit and Analyze:** Fit your model to each dataset, obtain the
   covariance matrices, and analyze:
    - **Parameter Uncertainties:** How do the variances (diagonal
      elements) of the estimated spot location parameters change with
      increasing noise?
    - **Parameter Correlations:** Do the covariances between
      parameters become stronger or weaker with higher noise?

**b) Varying Footprint Size:**

1.  **Adjust ROI:** Systematically change the size of the footprint
    (disk radius or box dimensions) used to extract the region of
    interest around the estimated spot center.
2. **Fit and Compare:** Fit your model for each footprint size, obtain the covariance matrix, and assess:
    - **Impact on Uncertainties:** Does a larger or smaller footprint
      lead to lower variances in the estimated spot location
      parameters?
    - **Trade-off:** Consider the trade-off between including more
      data points (larger footprint) and potentially introducing more
      background noise or deviations from the assumed PSF model.

**Visualizations:**

- **Correlation Matrix:** Calculate the correlation matrix from the
  covariance matrix to visualize parameter correlations more
  intuitively (values range from -1 to 1). Use heatmaps or pair plots
  to display the matrix.
- **Confidence Ellipses:** For 2D spot locations, plot confidence
  ellipses derived from the covariance matrix to visualize the joint
  uncertainty in the estimated (x, y) position.

**Key Insights:**

- Understanding how noise and footprint size impact parameter
  uncertainties helps determine the optimal balance for your specific
  application and data characteristics.
- Strong correlations between parameters can indicate potential issues
  with model identifiability, suggesting a need for more data or model
  adjustments.

By systematically analyzing the covariance matrix, you can gain
valuable insights into the reliability of your star spot localization,
identify potential issues, and make informed decisions to improve the
accuracy and robustness of your analysis.
