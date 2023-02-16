Explain Maximum Likelihood
Maximum likelihood is a method used in statistics to estimate the parameters of a probability distribution that best explain a set of observed data. The idea is to find the values of the parameters that maximize the likelihood of the data, given the assumed distribution. The likelihood function expresses the probability of observing the data for a given set of parameter values, and the maximum likelihood estimate (MLE) is the set of parameter values that maximize this probability. MLE is widely used to estimate parameters in various models such as linear regression and probability distribution models.

￼
￼
￼
￼
How is means squared error related to gaussian noise?
Mean Squared Error (MSE) is a commonly used measure of the difference between two sets of data, such as an estimated value and the true value. It is calculated by taking the average of the squared differences between the estimated and true values.

In the context of Gaussian noise, MSE can be used as a measure of the quality of a signal that has been corrupted by additive Gaussian noise. If the noise is Gaussian and the signal is being modeled as a linear function of some unknown parameters, the MSE can be used as a measure of how well the unknown parameters are estimated. The MSE is also a popular cost function used in optimization problems such as linear regression, where the goal is to minimize the MSE between the predicted and actual values.

In summary, MSE is a measure of the difference between the estimated and true values and it's often used as a measure of the quality of a signal corrupted by additive Gaussian noise and a cost function in optimization problem where the goal is to minimize the MSE.

￼
￼
￼
￼
So MSE can only be used as a cost function if the noise is gaussian? Would estimated parameters be wrong if the noise is poisson distributed?
MSE is a measure of the difference between the estimated and true values, it can be used as a cost function regardless of the noise distribution as long as it's differentiable and convex. However, if the noise is not Gaussian, the MSE may not be the optimal measure of the difference between the estimated and true values, and other cost functions may be more appropriate. For example, if the noise is Poisson distributed, the Negative Log-Likelihood (NLL) or the Cramer-Rao Lower Bound (CRLB) may be more appropriate as cost functions.

Regarding the parameters, the MLE can still be used to estimate the parameters of the model regardless of the noise distribution as long as the likelihood of the data can be expressed in terms of the parameters. However, if the noise is not Gaussian, the MLE may not be the optimal estimator and other estimators may be more appropriate.

In summary, MSE can be used as a cost function regardless of the noise distribution, but other cost functions may be more appropriate if the noise distribution is not Gaussian. The MLE can be used to estimate the parameters regardless of the noise distribution, but other estimators may be more appropriate if the noise distribution is not Gaussian.

￼
￼
￼
￼
Explain NLL and CRLB
Negative Log-Likelihood (NLL) is a measure of the difference between the estimated and true values that is commonly used when the noise is Poisson distributed. It is calculated by taking the negative logarithm of the likelihood function, which expresses the probability of observing the data for a given set of parameter values. The NLL is used as a cost function in optimization problems, with the goal of minimizing the NLL between the predicted and actual values.

Cramer-Rao Lower Bound (CRLB) is a measure of the lower bound on the variance of an unbiased estimator. It is a measure of the precision of an estimator, and is commonly used to evaluate the performance of different estimators. The CRLB is calculated by taking the inverse of the Fisher Information Matrix (FIM), which expresses the information about the parameters contained in the data. The CRLB is used to determine the optimal estimator among different estimators, with the goal of minimizing the variance of the estimator.

In summary, NLL is a measure of the difference between the estimated and true values that is commonly used when the noise is Poisson distributed and it's used as a cost function in optimization problems. CRLB is a measure of the lower bound on the variance of an unbiased estimator, it expresses the precision of an estimator and it's used to evaluate the performance of different estimators.

￼
￼
￼
￼
Consider the problem of fitting a decaying exponential function (as one would see in a fluorescence lifetime experiment). The noise, i believe, would be Poisson distributed. Show the FIM, CRLB and NLL for this problem.
In the problem of fitting a decaying exponential function, the signal is modeled as

S(t) = A * exp(-λt)

where A and λ are the parameters to be estimated and t is the independent variable.

The likelihood function for this problem, assuming Poisson noise, is given by:

p(D|A,λ) = (S(t)^D) * exp(-S(t)) / D!

Where D are the observed data.

The Fisher Information Matrix (FIM) is the matrix of second derivatives of the log-likelihood function with respect to the parameters A and λ:

FIM = [ ∂^2(ln(p(D|A,λ))/∂A^2 ∂^2(ln(p(D|A,λ))/∂A∂λ ]
[ ∂^2(ln(p(D|A,λ))/∂λ∂A ∂^2(ln(p(D|A,λ))/∂λ^2 ]

It's hard to provide the actual FIM due to its complexity, but it can be calculated numerically or analytically.

The Cramer-Rao Lower Bound (CRLB) is given by the inverse of the FIM. It expresses the lower bound on the variance of an unbiased estimator for the parameters A and λ:

CRLB = [ var(A) cov(A, λ) ]
[ cov(A, λ) var(λ) ]

Where var(A) and var(λ) are the lower bounds on the variances of the estimators for A and λ, respectively, and cov(A, λ) is the lower bound on the covariance between the estimators for A and λ.

Finally, the Negative Log-Likelihood (NLL) is given by:

NLL = -ln(p(D|A,λ))

Minimizing NLL with respect to A and λ would give us the maximum likelihood estimates of A and λ.

In summary, FIM is a matrix of second derivatives of the log-likelihood function with respect to the parameters, CRLB is the inverse of the FIM and expresses the lower bound on the variance of an unbiased estimator for the parameters, and NLL is the negative logarithm of the likelihood function and it's used as a cost function to find the maximum likelihood estimates of A and λ.