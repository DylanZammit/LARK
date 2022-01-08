# LARK for volatility estimation

This repo implements Lévy Adaptive Regression Kernels (LARK): a Lévy-driven non-parametric Bayesian model. It uses reversible-jump Monte Carlo Markov Chain (RJ-MCMC) for trans-dimensional jumps when proposing a new path.

## Model Specification
The current code is implemented for the following model

<img src="https://render.githubusercontent.com/render/math?math=dX_t=\mu_t dt%2B\sigma(t, X_t)dB_t,">

and our aim is to non-parametrically estimate the volatility function <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t)">. So far, this function can have one of the following forms:
- <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t) = \sigma(t)">,
- <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t) = \sigma(X_t)">,
- <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t) = \sigma(t)\sigma(X_t)"> (not so promising so far).
Future work might be done on the form of <img src="https://render.githubusercontent.com/render/math?math=\sigma(B_t)">, in other words, the volatility depends on the underlying Brownian Motion process.
## Likelihood
If our sample consists of <img src="https://render.githubusercontent.com/render/math?math=X_{t_1}, \cdots, X_{t_n}">, then we can write our log-likelihood in the following manner.

<img src="https://render.githubusercontent.com/render/math?math=\ell(X|\Theta)=-\frac{n}{2}\log(2\pi \Delta t)-\frac{n}{2}\sum_{i=1}^n\log(\eta(X_{t_i}|\Theta))-\frac{1}{2\Delta t}\sum_{i=1}^n\frac{(\mu_{t_i}-X_{t_i})^2}{\eta(X_{t_i}|\Theta)}">

Since this is quite expensive to compute, especially for large sample sizes, multiprocessing was used to increase efficiency.
## Prior
For the Lévy process we take the Gamma process which has Lévy measure <img src="https://render.githubusercontent.com/render/math?math=\nu"> given by 

<img src="https://render.githubusercontent.com/render/math?math=\nu(x)=\alpha e^{-\beta x}x^{-1}.">
 
This measure is not finite (<img src="https://render.githubusercontent.com/render/math?math=\int_{-\infty}^\infty \nu(x)dx=\infty)"> and thus would have paths of infinite jumps and giving us computational problems. However since small jumps (<img src="https://render.githubusercontent.com/render/math?math=<\epsilon">) are absolutely summable, we can consider

<img src="https://render.githubusercontent.com/render/math?math=\nu_\epsilon(x)=\alpha e^{-\beta x}x^{-1}I_{x\geq \epsilon}(x)">
 
instead, and treat small jumps as a death process.

[Reference papers properly by Chong Tu, Wolpert etc]
