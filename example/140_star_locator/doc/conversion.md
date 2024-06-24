# non-linear optimizing star locator

the following is a conversion with gemini 1.5 pro:

## Question:
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

## Answer: Star Spot Locator Project: A Comprehensive Guide

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

By following this detailed guide, you'll be able to develop, analyze,
and document your star spot locator project effectively, achieving
both accuracy and efficiency in your star localization tasks.


## Question:
User: Elaborate on this: Analyzing Fitting Error and Covariance:
Utilize lmfit or R optimization libraries to obtain covariance
matrices.  Interpret covariance values to understand parameter
correlations.  Quantify the influence of noise and footprint size on
parameter uncertainties.


## Answer: Delving Deeper into Analyzing Fitting Error and Covariance:

Analyzing the fitting error and covariance matrix provides crucial
insights into the reliability and robustness of your star spot
localization. Here's how you can leverage `lmfit` (Python) or similar
optimization libraries to quantify the influence of noise and
footprint size:

**1. Obtaining the Covariance Matrix:**

Both `lmfit` and R optimization libraries (like `optim` or `nls`)
provide access to the estimated covariance matrix after fitting.

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
2. **Fit and Compare:** Fit your model for each footprint size, obtain
   the covariance matrix, and assess:
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


## Question:
User: can i represent the search for the best disk diameter (shape
footprint) using an optimization? if yes, what is a good approach



## Answer:
Yes, you can absolutely represent the search for the best disk
diameter (footprint size) as an optimization problem. Here's a
breakdown of how you can approach this:

**1. Define Your Objective Function**

The objective function quantifies how "good" a given disk diameter is for your star spot localization.  You want to minimize or maximize this function depending on the metric you choose:

* **Minimize RMSE (Root Mean Squared Error):** A common choice is to minimize the RMSE of the estimated spot center location over multiple simulated images or different regions within a single image. This represents overall accuracy.
    ```python
    def objective_function(diameter, images, true_locations):
        """
        Calculates the RMSE for a given disk diameter.

        Args:
            diameter: Disk diameter to evaluate.
            images: A list of simulated star images or image regions.
            true_locations: The corresponding true (x, y) locations of the star spots.

        Returns:
            RMSE for the given diameter.
        """
        estimated_locations = []
        for image, true_loc in zip(images, true_locations):
            # ... (Extract ROI using the given diameter, fit model, get estimated location)
            estimated_locations.append(estimated_location)  
        rmse = calculate_rmse(estimated_locations, true_locations)
        return rmse
    ```

* **Maximize Signal-to-Noise Ratio (SNR):**  You could aim to maximize the SNR of the extracted spot signal, which might be desirable if you're dealing with faint stars.
* **Other Metrics:** You can also explore other metrics like the variance of the estimated spot location (lower variance is better), indicating more precise localization.

**2. Optimization Algorithm**

* **Golden-Section Search or Brent's Method:** If your objective function is relatively well-behaved (unimodal, meaning it has one clear minimum/maximum), these methods are efficient for finding the optimal diameter. 
    - **Advantages:** Guaranteed convergence for unimodal functions, relatively few function evaluations.
    - **Limitations:** May not perform well for more complex objective functions. 

* **Grid Search:** A simple but potentially computationally expensive approach is to evaluate your objective function over a predefined range of diameters.
    - **Advantages:** Straightforward to implement.
    - **Limitations:**  Can be slow, especially with a fine grid, and might miss the true optimum if the grid spacing is too large.

* **Global Optimization (if necessary):** If you suspect your objective function might have multiple local minima/maxima, use a global optimization algorithm.
    - **Examples:** Simulated annealing, genetic algorithms, particle swarm optimization.
    - **Advantages:**  Can handle more complex objective functions.
    - **Limitations:** Often require more function evaluations and might be slower.

**3. Python Implementation (using `scipy.optimize.minimize_scalar`)**

```python
from scipy.optimize import minimize_scalar

# ... (Define your objective_function, generate simulated data or load real data, etc.) ...

# Set bounds for the disk diameter search
diameter_bounds = (2, 20)  # Example bounds

# Perform optimization
result = minimize_scalar(objective_function, 
                      args=(images, true_locations), 
                      bounds=diameter_bounds,
                      method='bounded') # Or use 'golden' for golden section search

optimal_diameter = result.x
print(f"Optimal Disk Diameter: {optimal_diameter}")
```

**Important Considerations**

* **Data Characteristics:** The optimal diameter will depend on the characteristics of your star images (star brightness, PSF size, noise level).  Simulations or testing on a representative dataset are crucial.
* **Computational Cost:** Balance the complexity of your optimization approach with the computational cost, especially if you need to process many images.
* **Robustness:**  Evaluate the robustness of your optimized diameter on a diverse set of images to ensure it generalizes well.

By formulating the diameter search as an optimization problem and carefully considering the objective function and optimization algorithm, you can automate the process of finding the best disk diameter for your star spot localization task. 


## Question:

User: your suggestion of using RMSE of the estimated spot center
location over multiple simulated images sounds good. but i have to
define how many simulations should be collated into one. there is the
risk that too little simulations will create unreliable data. i
believe it could help to look at the variance of the results to
estimate at which point we have a reliable estimate of the
RMSE. complete this thought


## Answer:
You're absolutely right, simply choosing an arbitrary number of
simulations could lead to unreliable RMSE estimates. Analyzing the
variance of the RMSE values themselves is a great approach to
determine a sufficient number of simulations.

Here's how you can incorporate variance analysis into your optimization process:

**1. Iterative Approach with Variance Threshold:**

Instead of fixing the number of simulations beforehand, use an iterative approach with a variance threshold:

   ```python
   import numpy as np
   from scipy.optimize import minimize_scalar

   def objective_function(diameter, num_simulations, true_locations):
       """Calculates the RMSE for a given diameter and number of simulations."""
       rmse_values = []
       for _ in range(num_simulations):
           # ... Generate a simulated image, fit model, calculate RMSE ...
           rmse_values.append(rmse)
       return np.mean(rmse_values), np.var(rmse_values)

   def find_optimal_diameter(variance_threshold):
       num_simulations = 10  # Initial number of simulations
       rmse = float('inf')  # Initialize with a large RMSE
       variance = float('inf')  

       while variance > variance_threshold:
           # Optimize diameter using the current number of simulations
           result = minimize_scalar(lambda diameter: objective_function(diameter, num_simulations, true_locations)[0], 
                                     bounds=diameter_bounds, method='bounded') 
           optimal_diameter = result.x
           rmse, variance = objective_function(optimal_diameter, num_simulations, true_locations)

           # Increase the number of simulations for the next iteration
           num_simulations += 5  # Adjust the increment as needed

       return optimal_diameter, rmse, variance, num_simulations
   ```

**Explanation:**

- **`objective_function` now returns both mean RMSE and variance.**
- **Iterate until variance is below the threshold:**
    - Start with a reasonable initial number of simulations (`num_simulations`).
    - Optimize the diameter using the current simulation count.
    - Calculate the mean RMSE and variance of the RMSE values.
    - If the variance is above the `variance_threshold`, increase the `num_simulations` and repeat. 

**2. Choosing the Variance Threshold:**

- **Trade-off:**  A lower `variance_threshold` means more simulations (potentially higher computation time) but a more reliable RMSE estimate. 
- **Start with Visual Inspection:**  Initially, run the optimization with a few different `variance_threshold` values and plot the RMSE convergence against the number of simulations.  This can give you an idea of how the variance decreases and a suitable threshold for your application.
- **Domain Knowledge:**  Consider the desired level of precision for your RMSE estimate based on your specific problem and the impact of even small errors in spot location.

**3. Additional Tips:**

- **Seed Random Number Generator:**  For reproducibility, use a fixed seed for your random number generator within each optimization iteration.
- **Parallelization:** If your simulations are computationally expensive, consider parallelizing the simulation and fitting process within each iteration to speed up the optimization. 
- **Adaptive Increments:** You could potentially make the increment for `num_simulations` adaptive, increasing it by larger steps initially and then refining with smaller steps as the variance decreases.

By incorporating variance analysis into your diameter optimization,
you can make the process more robust and ensure that your final RMSE
estimate, and consequently, your optimal disk diameter, are based on a
sufficiently large and reliable set of simulations.



## Question:
User: what is the correct optimizer to use if the gray values of the
image are poisson distributed? least squares can be used for normal
distributed noise.

## Answer:

You're right, using ordinary least squares (OLS) might not be the most
appropriate choice when your image data follows a Poisson
distribution. OLS assumes normally distributed errors, which doesn't
hold true for Poisson data, especially at low signal levels where the
Poisson distribution becomes skewed.

Here are more suitable optimizers for handling Poisson-distributed image data:

**1. Poisson Loss Function (Maximum Likelihood Estimation)**

The most statistically principled approach is to use a loss function that directly models the Poisson noise. This is equivalent to performing Maximum Likelihood Estimation (MLE).

* **Mathematical Formulation:**  For Poisson distributed data, the negative log-likelihood (which you want to minimize) is:

   ```
   Loss = sum(y_pred - y_true * log(y_pred)) 
   ```
   where:
     - `y_true` are your observed pixel values (counts)
     - `y_pred` are the predicted pixel values from your star spot model

* **Implementation:**
    - **`scipy.optimize` (Python):** You can use the general-purpose optimizers in `scipy.optimize` (e.g., `minimize`) and pass in a custom loss function that calculates the Poisson negative log-likelihood.
    - **GLM with Poisson Family (R):** In R, Generalized Linear Models (GLMs) are ideal. Use `glm(..., family = poisson())`.
    - **Other Libraries:** Many optimization and statistical modeling libraries have built-in support for Poisson regression or MLE with Poisson data. 

**2. Weighted Least Squares**

If you want to stay within the framework of least squares, you can use *weighted least squares (WLS)* to account for the unequal variances of the Poisson distribution.

* **Weighting Scheme:** In WLS, you assign weights to each data point inversely proportional to its variance. For Poisson data, the variance is equal to the mean. Since you don't know the true mean, you can approximate it with the observed counts: 

   ```
   weights = 1 / y_true 
   ```
   (Be careful with zero counts; add a small constant to avoid division by zero).

* **Implementation:**
    - **`curve_fit` (Python):** The `curve_fit` function in `scipy.optimize` allows you to provide weights.
    - **`lm` (R):** Use the `weights` argument in the `lm()` function. 

**Which Method to Choose:**

- **Poisson Loss Function (MLE):**
    - **Pros:** Statistically most appropriate for Poisson data, provides more accurate confidence intervals and statistical tests.
    - **Cons:** Might be slightly more complex to implement than WLS.

- **Weighted Least Squares:**
    - **Pros:** Simpler to implement, often computationally faster than full MLE.
    - **Cons:**  Might provide less accurate results than MLE, especially for low count values.

**In summary**, using a Poisson loss function (MLE) is generally recommended for optimal accuracy when dealing with Poisson-distributed image data. However, weighted least squares can be a good alternative if you need a simpler or faster approach. 
