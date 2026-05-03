import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from rashomon import counter
from rashomon.extract_pools import extract_pools, get_trt_ctl_pooled_partition


def compute_policy_means(D: np.ndarray, y: np.ndarray, num_policies: int) -> np.ndarray:
    """
    Compute an array that makes it easy to find means for each policy

    Arguments:
    D (np.ndarray): Dataset
    y (np.ndarray): Outcomes
    num_policies (int): Number of policies

    Returns:
    policy_means (np.ndarray): Size (num_policies,2)
        policy_means[i, 0] = sum of all y where D[i,0] = i
        policy_means[i, 1] = count of where D[i,0] = i
    """
    policy_means = np.ndarray(shape=(num_policies, 2))
    for policy_id in range(num_policies):
        idx = np.where(D == policy_id)
        policy_means[policy_id, 0] = np.sum(y[idx])
        policy_means[policy_id, 1] = len(idx[0])
    return policy_means


def compute_pool_means(policy_means: np.ndarray, pi_pools: dict[int, list[int]]) -> np.ndarray:
    """
    Compute the mean value in each pool

    Arguments:
    policy_means (np.ndarray): Size (num_policies,2)
        policy_means[i, 0] = sum of all y where D[i,0] = i
        policy_means[i, 1] = count of where D[i,0] = i
    pi_pools (dict[int, list[int]]): Dictionary mapping pools to policies

    Returns:
    mu_pools (np.ndarray): Size (H,) where H is the number of pools
        mu_pools[i] = mean value in pool i
    """
    H = len(pi_pools.keys())
    mu_pools_temp = np.ndarray(shape=(H, 2))
    for pool_id, pool in pi_pools.items():
        policy_subset = policy_means[pool, :]
        nodata_idx = np.where(policy_subset[:, 1] == 0)[0]
        policy_subset[:, 0][nodata_idx] = 0
        mu_pools_temp[pool_id, :] = np.sum(policy_subset, axis=0)
    sums = mu_pools_temp[:, 0]
    counts = mu_pools_temp[:, 1]
    nodata_idx = np.where(counts == 0)[0]
    sums[nodata_idx] = 0
    out_format = np.zeros_like(sums)
    mu_pools = np.divide(sums, counts, out=out_format, where=counts != 0)
    return mu_pools


def partition_sigma(sigma: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Maximally split policies in arm i starting at dosage j
    All other existing splits are maintained
    """
    sigma_fix = np.copy(sigma)
    sigma_fix[i, j:] = 0
    sigma_fix[np.isinf(sigma)] = np.inf
    return sigma_fix


def predict(D: np.ndarray, sigma: np.ndarray, policies: list, policy_means: np.ndarray,
            lattice_edges: list[tuple[int, int]] | None = None,
            return_num_pools: bool = False) -> np.ndarray:
    """
    Predict outcomes based on the given dataset, policies, and partition matrix.

    Arguments:
    D (np.ndarray): Dataset of shape (N, 2) where N is the number of samples.
     The first column contains policy indices and the second column contains dosages.
    sigma (np.ndarray): Partition matrix of shape (P, Q) where P is the number of policies
        and Q is the number of dosages.
    policies (list): List of policies.
    policy_means (np.ndarray): Array of shape (P, 2) where P is the number of policies.
        The first column contains the sum of outcomes and the second column contains the count of samples.
    lattice_edges (list[tuple[int, int]] | None): List of edges in the lattice structure,
        where each edge is a tuple of two integers. Default is None.
    return_num_pools (bool): Whether to return the number of pools. Default is False.

    Returns:
    np.ndarray: Predicted outcomes of shape (N,).
    If return_num_pools is True, returns a tuple of predicted outcomes and the number of pools.
    """
    pi_pools, pi_policies = extract_pools(policies, sigma, lattice_edges)
    mu_pools = compute_pool_means(policy_means, pi_pools)
    D_pool = [pi_policies[pol_id] for pol_id in D[:, 0]]
    y_hat = mu_pools[D_pool]

    if return_num_pools:
        return y_hat, mu_pools.shape[0]
    return y_hat


def compute_B(D: np.ndarray, y: np.ndarray, sigma: np.ndarray, i: int, j: int,
              policies: list, policy_means: np.ndarray, reg: float = 1, normalize: int = 0,
              lattice_edges: list[tuple[int, int]] | None = None) -> float:
    """
    The B function in Theorem \ref{thm:rashomon-equivalent-bound}

    Arguments:
    D (np.ndarray): Dataset
    y (np.ndarray): Outcomes
    sigma (np.ndarray): Partition matrix
    i (int): Feature index
    j (int): Factor level index
    policies (list): List of policies
    policy_means (np.ndarray): Size (num_policies, 2). See `compute_policy_means`
    reg (float): Regularization parameter. Defaults 1
    normalize (int): Normalization factor. Defaults 0
    lattice_edges (list[tuple[int, int]]): List of edges in the Hasse. Defaults to None

    Returns:
    B (int): The lower bound on the loss function
    """

    # Split maximally in arm i from dosage j
    sigma_fix = partition_sigma(sigma, i, j)

    # Compute squared loss for this maximal split
    # This loss is B minus the regularization term
    mu_D = predict(D, sigma_fix, policies, policy_means, lattice_edges)
    mse = mean_squared_error(y[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    # The least number of pools
    # The number of pools when the splittable policies are pooled maximally
    sigma_fix[i, (j + 1):] = 1
    sigma_fix[np.isinf(sigma)] = np.inf
    h = counter.num_pools(sigma_fix)

    B = mse + reg * h

    return B


def compute_Q(D: np.ndarray, y: np.ndarray, sigma: np.ndarray, policies: list,
              policy_means: np.ndarray, reg: float = 1, normalize: int = 0,
              lattice_edges: list[tuple[int, int]] | None = None) -> float:
    """
    Compute the loss Q

    Arguments:
    D (np.ndarray): Dataset
    y (np.ndarray): Outcomes
    sigma (np.ndarray): Partition matrix
    policies (list): List of policies
    policy_means (np.ndarray): Size (num_policies, 2). See `compute_policy_means`
    reg (float): Regularization parameter. Defaults 1
    normalize (int): Normalization factor. Defaults 0
    lattice_edges (list[tuple[int, int]]): List of edges in the Hasse. Defaults to None

    Returns:
    Q (int): The loss function
    """

    mu_D, h = predict(D, sigma, policies, policy_means, lattice_edges, return_num_pools=True)
    mse = mean_squared_error(y[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    Q = mse + reg * h

    return Q


def compute_Q_bayes(D: np.ndarray, y: np.ndarray, sigma: np.ndarray, policies: list,
                    policy_means: np.ndarray, reg: float = 1.0,
                    lattice_edges=None) -> float:
    """
    Bayesian marginal posterior loss from Lemma A2:
    Q(Pi) = (n-H)/2 * log(SSE) + lambda'*H + c(Pi)
    where lambda' = reg - 0.5*log(pi), c(Pi) = 0.5*sum_h log(n_h) - logGamma((n-H)/2)
    """
    import math
    from scipy.special import gammaln

    mu_D, H = predict(D, sigma, policies, policy_means, lattice_edges, return_num_pools=True)
    n = D.shape[0]
    SSE = float(np.sum((y[:, 0] - mu_D) ** 2))

    if SSE <= 0.0 or n <= H:
        return float('inf')

    pi_pools, _ = extract_pools(policies, sigma, lattice_edges)
    n_h = np.array([float(np.sum(policy_means[pi_pools[h], 1])) for h in range(H)], dtype=float)

    lamb_prime = reg - 0.5 * math.log(math.pi)
    c_Pi = 0.5 * float(np.sum(np.log(np.maximum(n_h, 1)))) - gammaln((n - H) / 2.0)
    return (n - H) / 2.0 * math.log(SSE) + lamb_prime * H + c_Pi


def compute_B_slopes(D, X, y, sigma, i, j, policies, reg=1, normalize=0):
    """
    The B function in Theorem 6.3 \ref{thm:rashomon-equivalent-bound}
    """

    # Split maximally in arm i from dosage j
    sigma_fix = partition_sigma(sigma, i, j)
    pi_fixed_pools, pi_fixed_policies = extract_pools(policies, sigma_fix)

    # Compute squared loss for this maximal split
    y_est = np.zeros(shape=D.shape) + np.inf
    for pi_k, pol_list_k in pi_fixed_pools.items():

        # Extract the X matrix
        k_idx = [i for i, p in enumerate(D[:, 0]) if p in pol_list_k]
        X_k = X[k_idx, :]
        y_k = y[k_idx, :]

        # Run regression and estimate outcomes
        model_k = LinearRegression().fit(X_k, y_k)
        y_est_k = model_k.predict(X_k)
        y_est[k_idx, :] = y_est_k

    mse = mean_squared_error(y[:, 0], y_est)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    # The least number of pools
    # The number of pools when the splittable policies are pooled maximally
    sigma_fix[i, (j + 1):] = 1
    sigma_fix[np.isinf(sigma)] = np.inf
    h = counter.num_pools(sigma_fix)

    B = mse + reg * h

    return B


def compute_Q_slopes(D, X, y, sigma, policies, reg=1, normalize=0):
    """
    Compute the loss Q
    """

    pi_pools, pi_policies = extract_pools(policies, sigma)

    y_est = np.zeros(shape=D.shape) + np.inf
    for pi_k, pol_list_k in pi_pools.items():

        # Extract the X matrix
        k_idx = [i for i, p in enumerate(D[:, 0]) if p in pol_list_k]
        X_k = X[k_idx, :]
        y_k = y[k_idx, :]

        # Run regression and estimate outcomes
        model_k = LinearRegression().fit(X_k, y_k)
        y_est_k = model_k.predict(X_k)
        y_est[k_idx, :] = y_est_k

    mse = mean_squared_error(y[:, 0], y_est)

    if normalize > 0:
        mse = mse * D.shape[0] / normalize

    h = len(pi_pools.keys())
    Q = mse + reg * h

    return Q


def compute_het_Q(D_tc, y_tc, sigma_int, trt_pools, ctl_pools, policy_means, reg=1, normalize=0):
    """
    Compute the loss after pooling across treatment and control as per sigma_int
    D_tc indices need not be re-indexed. The indicies should match policy_means
    policy_means is for the entire dataset
    """

    sigma_pools, sigma_policies = get_trt_ctl_pooled_partition(trt_pools, ctl_pools, sigma_int)
    mu_pools = compute_pool_means(policy_means, sigma_pools)
    D_tc_pool = [sigma_policies[pol_id] for pol_id in D_tc[:, 0]]
    mu_D = mu_pools[D_tc_pool]
    mse = mean_squared_error(y_tc[:, 0], mu_D)

    if normalize > 0:
        mse = mse * D_tc.shape[0] / normalize

    h = mu_pools.shape[0]
    Q = mse + reg * h

    return Q
