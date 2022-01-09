# LARK for volatility estimation

This repo implements Lévy Adaptive Regression Kernels (LARK): a Lévy-driven non-parametric Bayesian model. It uses reversible-jump Monte Carlo Markov Chain (RJ-MCMC) for trans-dimensional jumps when proposing a new path.

## Model Specification
The current code is implemented for the following model

<img src="https://render.githubusercontent.com/render/math?math=dX_t=\mu_t dt%2B\sigma(t, X_t)dB_t,">

and our aim is to non-parametrically estimate the volatility function <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t)">. So far, this function can have one of the following forms:
- <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t) = \sigma(t)">,
- <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t) = \sigma(X_t)">,
- <img src="https://render.githubusercontent.com/render/math?math=\sigma(t, X_t) = \sigma(t)\sigma(X_t)"> (not so promising so far).

The main difference between these different types of methods would be the formulation of the likelihood function, which are all very similar. Future work might be done on the form of <img src="https://render.githubusercontent.com/render/math?math=\sigma(B_t)">, in other words, the volatility depends on the underlying Brownian Motion process. The likelihood function would again be similar (at least visually), but this introduces a new problem. We would have to treat the underlying BM as a latent variable since it is not directly observable. To do this we would treat each BM increment as an unknown "parameter" of interest in our Bayesian model and hence infer a posterior distribution over this BM. This would be computationally much more intensive since we would have introduced as many new parameters as sample size. 
## Likelihood
If our sample consists of <img src="https://render.githubusercontent.com/render/math?math=X_{t_1}, \cdots, X_{t_n}">, then we can write our log-likelihood in the following manner.

<img src="https://render.githubusercontent.com/render/math?math=\ell(X|\Theta)=-\frac{n}{2}\log(2\pi \Delta t)-\frac{n}{2}\sum_{i=1}^n\log(\eta(X_{t_i}|\Theta))-\frac{1}{2\Delta t}\sum_{i=1}^n\frac{(\mu_{t_i}-X_{t_i})^2}{\eta(X_{t_i}|\Theta)}">

Since this is quite computationally expensive, especially for large sample sizes, multiprocessing is exploited to increase efficiency.
## Prior
For the Lévy process we take the Gamma process which has Lévy measure <img src="https://render.githubusercontent.com/render/math?math=\nu"> given by 

<img src="https://render.githubusercontent.com/render/math?math=\nu(x)=\alpha e^{-\beta x}x^{-1}.">
 
This measure is not finite (<img src="https://render.githubusercontent.com/render/math?math=\int_{-\infty}^\infty \nu(x)dx=\infty)"> and thus would have paths of infinite jumps and giving us computational problems. However since small jumps (<img src="https://render.githubusercontent.com/render/math?math=<\epsilon">) are absolutely summable, we can consider

<img src="https://render.githubusercontent.com/render/math?math=\nu_\epsilon(x)=\alpha e^{-\beta x}x^{-1}I_{x\geq \epsilon}(x)">
 
instead, and treat small jumps as a death process.

## Synthetic Example

Here we generate 1000 random data points with no drift and some volatility function depending on time. Below is a plot
of the increments, and their cumulative sum.
![sim_data](https://github.com/DylanZammit/LARK/blob/master/img/sim_data.png)
Below is the result of the non-parametric LARK estimate along with its 95% credible interval with just 1000 iterations and no burn-in-period. As kernels, the exponential kernel and Haar kernels were used, and their parameters were left free to be chosen by the Bayesian model.
![sim_res](https://github.com/DylanZammit/LARK/blob/master/img/sim_res.png)


[Reference papers properly by Chong Tu, Wolpert etc]

