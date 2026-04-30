import numpy as np
from scipy.special import stdtr, stdtrit
from scipy.optimize import brentq

def q_mixture_t_root(k, w, mu, s, df, prob, xtol=1e-8, maxiter=100):
    """
    Computes quantiles for a mixture of location-scale t-distributions.

    Inputs:
    k       : (int) Number of components
    w       : (array) Vector of mixture weights (will be normalized)
    mu      : (array) Vector of location parameters
    s       : (array) Vector of scale parameters
    df      : (scalar or array) Degrees of freedom
    prob    : (array) Quantiles to solve for (e.g., [0.025, 0.5, 0.975])
    xtol    : (float, optional) Tolerance (absolute error) for the root finder. Default is 1e-8.
    maxiter : (int, optional) Maximum iterations for the root finder. Default is 100.
    """
    # 1. Sanitize and broadcast inputs
    w = np.asarray(w, dtype=np.float64).ravel()
    w = w / np.sum(w)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    prob = np.atleast_1d(prob)

    # Recycle df if a single value is provided
    if np.isscalar(df) or len(np.atleast_1d(df)) == 1:
        df = np.full(k, df, dtype=np.float64)
    else:
        df = np.asarray(df, dtype=np.float64)

    quantiles = np.zeros_like(prob, dtype=np.float64)

    # 2. Define the objective function (closure over component arrays)
    def obj_func(x, target_p):
        # Vectorized evaluation of the standard t CDF across all k components
        t_stats = (x - mu) / s
        mix_cdf = np.sum(w * stdtr(df, t_stats))
        return mix_cdf - target_p

    # 3. Solve for each probability
    for i, p in enumerate(prob):
        # Generate guaranteed brackets using the inverse CDF (stdtrit)
        component_quantiles = mu + s * stdtrit(df, p)
        lower_bound = np.min(component_quantiles)
        upper_bound = np.max(component_quantiles)

        # If the bounds are identical, the root is already found
        if np.isclose(lower_bound, upper_bound):
            quantiles[i] = lower_bound
            continue

        # Nudge outward to absorb floating-point error in stdtr/stdtrit inverses.
        # f is monotone (mixture CDF), so this preserves the sign bracket.
        eps = max(abs(upper_bound - lower_bound), 1.0) * 1e-8
        lower_bound -= eps
        upper_bound += eps

        # Find the root using Brent's method
        quantiles[i] = brentq(
            obj_func,
            a=lower_bound,
            b=upper_bound,
            args=(p,),
            xtol=xtol,
            maxiter=maxiter
        )

    return quantiles


def q_mixture_t_mc(k, w, mu, s, df, prob, n_samples=1_000_000, seed=None):
    """
    Computes empirical quantiles for a mixture of location-scale t-distributions
    using a highly optimized Monte Carlo block-sampling approach.

    Inputs:
    k         : (int) Number of components
    w         : (array) Vector of mixture weights (will be normalized)
    mu        : (array) Vector of location parameters
    s         : (array) Vector of scale parameters
    df        : (scalar or array) Degrees of freedom
    prob      : (array) Quantiles to solve for (e.g., [0.025, 0.5, 0.975])
    n_samples : (int) Total number of Monte Carlo draws (default: 1,000,000)
    seed      : (int, optional) Random seed for reproducibility
    """
    # 1. Sanitize inputs
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)  # Ensure weights sum to 1
    mu = np.asarray(mu, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    prob = np.atleast_1d(prob)

    # Recycle df if a single value is provided
    if np.isscalar(df) or len(np.atleast_1d(df)) == 1:
        df = np.full(k, df, dtype=np.float64)
    else:
        df = np.asarray(df, dtype=np.float64)

    # Initialize the modern NumPy random number generator
    rng = np.random.default_rng(seed)

    # 2. Determine how many samples to draw from each component
    # This is O(K) and avoids O(N) array indexing
    counts = rng.multinomial(n_samples, w)

    # 3. Pre-allocate a single contiguous array for all samples
    mixture_samples = np.empty(n_samples, dtype=np.float64)

    # 4. Block-fill the array
    current_idx = 0
    for i in range(k):
        n_i = counts[i]
        if n_i > 0:
            # Draw standard t, scale, shift, and insert in one vectorized step
            end_idx = current_idx + n_i
            mixture_samples[current_idx:end_idx] = mu[i] + s[i] * rng.standard_t(df[i], size=n_i)
            current_idx = end_idx

    # 5. Extract empirical quantiles
    # 'method="linear"' is the NumPy default, equivalent to R's default type=7
    quantiles = np.quantile(mixture_samples, prob)

    return quantiles