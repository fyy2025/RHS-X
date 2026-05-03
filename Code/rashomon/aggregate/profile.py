import numpy as np

from collections import deque

from .. import loss
from .. import counter
from ..hasse import enumerate_policies, policy_to_profile, is_policies_sorted
from ..sets import RashomonSet, RashomonProblemCache, RashomonSubproblemCache
from ..extract_pools import lattice_edges


def initialize_sigma(M: int, R: int | np.ndarray) -> np.ndarray:
    """ Initialize the sigma matrix with M arms and R dosage levels."""
    if isinstance(R, int):
        sigma = np.ndarray(shape=(M, R - 2))
        sigma[:, :] = 1
    else:
        sigma = np.ndarray(shape=(M, np.max(R) - 2))
        sigma[:, :] = np.inf
        for idx, R_idx in enumerate(R):
            sigma[idx, :(R_idx-2)] = 1
    return sigma


def RAggregate_profile(M: int, R: int | np.ndarray, H: int, D: np.ndarray,
                       y: np.ndarray, theta: float, profile: tuple, reg: float = 1,
                       policies: list | None = None, policy_means: np.ndarray | None = None,
                       normalize: int = 0) -> RashomonSet:
    """
    RPS enumeration algorithm for a single profile

    Parameters:
    M (int): Number of features
    R (int or np.ndarray of int): Number of factor levels per feature
    H (int): Maximum number of pools in this profile
    D (np.ndarray): Data i.e., policy integers
    y (np.ndarray): Outcomes
    theta (float): Rashomon threshold
    profile (tuple): Profile under consideration
    reg (float): Regularization parameter. Defaults to 1
    policies (list), policy_means (np.ndarray) - Optional precomputed values. Defaults to None
    normalize (int): Normalization parameter. Defaults to 0

    Returns:
    RashomonSet: Set of near-optimal poolings
    """

    # If R is fixed across, make it a list for compatbility later on
    if isinstance(R, int):
        R = np.array([R] * M)

    if policies is None or policy_means is None:
        all_policies = enumerate_policies(M, R)
        policies = [x for x in all_policies if policy_to_profile(x) == profile]
        policy_means = loss.compute_policy_means(D, y, len(policies))

    if np.max(R) == 2:
        sigma = np.zeros(shape=(M, 1)) + np.inf
        P_qe = RashomonSet(sigma.shape)
        P_qe.insert(sigma)
        return P_qe

    sigma = initialize_sigma(M, R)
    policies_sorted = is_policies_sorted(policies)
    hasse_edges = lattice_edges(policies, sorted=policies_sorted, M=M, R=R-1)

    P_qe = RashomonSet(sigma.shape)
    Q_seen = RashomonProblemCache(sigma.shape)
    problems = RashomonSubproblemCache(sigma.shape)

    for i in range(M):
        if not np.isinf(sigma[i, 0]):
            queue = deque([(sigma, i, 0)])
            break

    while len(queue) > 0:

        (sigma, i, j) = queue.popleft()
        sigma = np.copy(sigma)

        # Cache problem
        if problems.seen(sigma, i, j):
            continue
        problems.insert(sigma, i, j)

        if counter.num_pools(sigma) > H:
            continue

        sigma_0 = np.copy(sigma)
        sigma_1 = np.copy(sigma)
        sigma_1[i, j] = 1
        sigma_0[i, j] = 0

        # Add problem variants to queue
        for m in range(M):
            R_m = R[m]

            j1 = 0
            while problems.seen(sigma_1, m, j1) and j1 < R_m - 3:
                j1 += 1
            if j1 <= R_m - 3 and not problems.seen(sigma_1, m, j1):
                queue.append((sigma_1, m, j1))

            j0 = 0
            while problems.seen(sigma_0, m, j0) and j0 < R_m - 3:
                j0 += 1
            if j0 <= R_m - 3 and not problems.seen(sigma_0, m, j0):
                queue.append((sigma_0, m, j0))

        # Check if further splits in arm i is feasible
        # B = loss.compute_B(D, y, sigma, i, j, policies, policy_means, reg, normalize)
        B = loss.compute_B(D, y, sigma, i, j, policies, policy_means, reg, normalize, hasse_edges)
        if B > theta:
            continue

        # Check if the pooling already satisfies the Rashomon threshold
        if not Q_seen.seen(sigma_1):
            Q_seen.insert(sigma_1)
            # Q = loss.compute_Q(D, y, sigma_1, policies, policy_means, reg, normalize)
            Q = loss.compute_Q(D, y, sigma_1, policies, policy_means, reg, normalize, hasse_edges)
            if Q <= theta:
                P_qe.insert(sigma_1)

        if not Q_seen.seen(sigma_0) and counter.num_pools(sigma_0) <= H:
            Q_seen.insert(sigma_0)
            # Q = loss.compute_Q(D, y, sigma_0, policies, policy_means, reg, normalize)
            Q = loss.compute_Q(D, y, sigma_0, policies, policy_means, reg, normalize, hasse_edges)
            if Q <= theta:
                P_qe.insert(sigma_0)

        # Add children problems to the queue
        if j < R[i] - 3:  # j < R_i - 2 in math notation
            if not problems.seen(sigma_1, i, j + 1):
                queue.append((sigma_1, i, j + 1))
            if not problems.seen(sigma_0, i, j + 1):
                queue.append((sigma_0, i, j + 1))

    return P_qe


def _brute_RAggregate_profile(M: int, R: int | np.ndarray, H: int, D: np.ndarray,
                              y: np.ndarray, theta: float, profile: tuple, reg: float = 1,
                              policies: list | None = None, policy_means: np.ndarray | None = None,
                              normalize: int = 0) -> RashomonSet:
    """
    Brute force RPS enumeration for a single profile

    Parameters:
    M (int): Number of features
    R (int or np.ndarray of int): Number of factor levels per feature
    H (int): Maximum number of pools in this profile
    D (np.ndarray): Data i.e., policy integers
    y (np.ndarray): Outcomes
    theta (float): Rashomon threshold
    profile (tuple): Profile under consideration
    reg (float): Regularization parameter. Defaults to 1
    policies (list), policy_means (np.ndarray) - Optional precomputed values. Defaults to None
    normalize (int): Normalization parameter. Defaults to 0

    Returns:
    RashomonSet: Set of near-optimal poolings
    """

    if policies is None or policy_means is None:
        all_policies = enumerate_policies(M, R)
        policies = [x for x in all_policies if policy_to_profile(x) == profile]
        policy_means = loss.compute_policy_means(D, y, len(policies))

    if np.max(R) == 2:
        sigma = np.zeros(shape=(M, 1)) + np.inf
        P_qe = RashomonSet(sigma.shape)
        P_qe.insert(sigma)
        P_qe.calculate_loss(D, y, policies, policy_means, reg, normalize=normalize)
        return P_qe

    sigma = initialize_sigma(M, R)

    # If R is fixed across, make it a list for compatbility later on
    if isinstance(R, int):
        R = [R] * M

    P_qe = RashomonSet(sigma.shape)

    indices_raw = np.where(sigma == 1)
    idx_rows = indices_raw[0]
    idx_cols = indices_raw[1]
    indices = []
    for i in range(len(idx_rows)):
        indices.append((idx_rows[i], idx_cols[i]))

    # t1_ctr = 0
    # t2_ctr = 0
    # ctr = 0
    policies_sorted = is_policies_sorted(policies)
    hasse_edges = lattice_edges(policies, sorted=policies_sorted, M=M, R=R-1)

    for x in counter.powerset(indices):
        sigma_x = sigma.copy()
        for i, j in x:
            sigma_x[i, j] = 0

        # Q, t1, t2 = loss.compute_Q(D, y, sigma_x, policies, policy_means, reg, normalize, hasse_edges)
        Q = loss.compute_Q(D, y, sigma_x, policies, policy_means, reg, normalize, hasse_edges)
        if Q <= theta:
            P_qe.insert(sigma_x)
            P_qe.Q = np.append(P_qe.Q, Q)

        # ctr += 1
        # t1_ctr += t1
        # t2_ctr += t2

    # print(f"\tLattice took {t1_ctr / ctr} ({t1_ctr}) s on average")
    # print(f"\tConnected components took {t2_ctr / ctr} ({t2_ctr}) s on average")

    return P_qe


def RAggregate_profile_bayes(M: int, R: int | np.ndarray, H: int, D: np.ndarray,
                             y: np.ndarray, theta: float, profile: tuple, reg: float = 1,
                             policies: list | None = None,
                             policy_means: np.ndarray | None = None) -> RashomonSet:
    """
    RPS enumeration using the Bayesian Lemma A2 loss.
    B-based pruning is skipped (no valid lower bound for the Bayesian formula).
    """

    if isinstance(R, int):
        R = np.array([R] * M)

    if policies is None or policy_means is None:
        all_policies = enumerate_policies(M, R)
        policies = [x for x in all_policies if policy_to_profile(x) == profile]
        policy_means = loss.compute_policy_means(D, y, len(policies))

    if np.max(R) == 2:
        sigma = np.zeros(shape=(M, 1)) + np.inf
        P_qe = RashomonSet(sigma.shape)
        P_qe.insert(sigma)
        return P_qe

    sigma = initialize_sigma(M, R)
    policies_sorted = is_policies_sorted(policies)
    hasse_edges = lattice_edges(policies, sorted=policies_sorted, M=M, R=R-1)

    P_qe = RashomonSet(sigma.shape)
    Q_seen = RashomonProblemCache(sigma.shape)
    problems = RashomonSubproblemCache(sigma.shape)

    for i in range(M):
        if not np.isinf(sigma[i, 0]):
            queue = deque([(sigma, i, 0)])
            break

    while len(queue) > 0:

        (sigma, i, j) = queue.popleft()
        sigma = np.copy(sigma)

        if problems.seen(sigma, i, j):
            continue
        problems.insert(sigma, i, j)

        if counter.num_pools(sigma) > H:
            continue

        sigma_0 = np.copy(sigma)
        sigma_1 = np.copy(sigma)
        sigma_1[i, j] = 1
        sigma_0[i, j] = 0

        for m in range(M):
            R_m = R[m]

            j1 = 0
            while problems.seen(sigma_1, m, j1) and j1 < R_m - 3:
                j1 += 1
            if j1 <= R_m - 3 and not problems.seen(sigma_1, m, j1):
                queue.append((sigma_1, m, j1))

            j0 = 0
            while problems.seen(sigma_0, m, j0) and j0 < R_m - 3:
                j0 += 1
            if j0 <= R_m - 3 and not problems.seen(sigma_0, m, j0):
                queue.append((sigma_0, m, j0))

        # No B-pruning for Bayesian loss (no valid lower bound)

        if not Q_seen.seen(sigma_1):
            Q_seen.insert(sigma_1)
            Q = loss.compute_Q_bayes(D, y, sigma_1, policies, policy_means, reg, hasse_edges)
            if Q <= theta:
                P_qe.insert(sigma_1)

        if not Q_seen.seen(sigma_0) and counter.num_pools(sigma_0) <= H:
            Q_seen.insert(sigma_0)
            Q = loss.compute_Q_bayes(D, y, sigma_0, policies, policy_means, reg, hasse_edges)
            if Q <= theta:
                P_qe.insert(sigma_0)

        if j < R[i] - 3:
            if not problems.seen(sigma_1, i, j + 1):
                queue.append((sigma_1, i, j + 1))
            if not problems.seen(sigma_0, i, j + 1):
                queue.append((sigma_0, i, j + 1))

    return P_qe


def _brute_RAggregate_profile_bayes(M: int, R: int | np.ndarray, H: int, D: np.ndarray,
                                    y: np.ndarray, theta: float, profile: tuple, reg: float = 1,
                                    policies: list | None = None,
                                    policy_means: np.ndarray | None = None) -> RashomonSet:
    """
    Brute-force RPS enumeration using the Bayesian Lemma A2 loss.
    """

    if policies is None or policy_means is None:
        all_policies = enumerate_policies(M, R)
        policies = [x for x in all_policies if policy_to_profile(x) == profile]
        policy_means = loss.compute_policy_means(D, y, len(policies))

    if np.max(R) == 2:
        sigma = np.zeros(shape=(M, 1)) + np.inf
        P_qe = RashomonSet(sigma.shape)
        P_qe.insert(sigma)
        P_qe.calculate_loss_bayes(D, y, policies, policy_means, reg)
        return P_qe

    sigma = initialize_sigma(M, R)

    if isinstance(R, int):
        R = [R] * M

    P_qe = RashomonSet(sigma.shape)

    indices_raw = np.where(sigma == 1)
    idx_rows = indices_raw[0]
    idx_cols = indices_raw[1]
    indices = []
    for i in range(len(idx_rows)):
        indices.append((idx_rows[i], idx_cols[i]))

    policies_sorted = is_policies_sorted(policies)
    hasse_edges = lattice_edges(policies, sorted=policies_sorted, M=M, R=R-1)

    for x in counter.powerset(indices):
        sigma_x = sigma.copy()
        for i, j in x:
            sigma_x[i, j] = 0

        Q = loss.compute_Q_bayes(D, y, sigma_x, policies, policy_means, reg, hasse_edges)
        if Q <= theta:
            P_qe.insert(sigma_x)
            P_qe.Q = np.append(P_qe.Q, Q)

    return P_qe


if __name__ == "__init__":

    # Fix random seed
    np.random.seed(3)

    # Imports for testing only
    from rashomon.extract_pools import extract_pools

    #
    # Setup matrix
    #
    sigma = np.array([[1, 1, 0], [0, 1, 1]], dtype="float64")
    sigma_profile = (1, 1)

    M, n = sigma.shape
    R = n + 2
    num_policies = (R - 1) ** M
    all_policies = enumerate_policies(M, R)
    policies = [x for x in all_policies if policy_to_profile(x) == sigma_profile]
    pi_pools, pi_policies = extract_pools(policies, sigma)

    #
    # Generate data
    #
    num_pools = len(pi_pools)
    mu = np.random.uniform(0, 4, size=num_pools)
    var = [1] * num_pools

    n_per_pol = 10

    num_data = num_policies * n_per_pol
    X = np.ndarray(shape=(num_data, M))
    D = np.ndarray(shape=(num_data, 1), dtype="int_")
    y = np.ndarray(shape=(num_data, 1))

    for idx, policy in enumerate(policies):
        pool_i = pi_policies[idx]
        mu_i = mu[pool_i]
        var_i = var[pool_i]
        y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

        start_idx = idx * n_per_pol
        end_idx = (idx + 1) * n_per_pol

        X[start_idx:end_idx,] = policy
        D[start_idx:end_idx,] = idx
        y[start_idx:end_idx,] = y_i

    #
    # Aggregate
    #
    P_set = RAggregate_profile(2, 5, 4, D, y, 5, sigma_profile, reg=1)
    print(f"There are {P_set.size} poolings in the Rashomon set.")
    print(f"Original matrix in P_set: {P_set.seen(sigma)}.")

    print("The poolings are:")
    for pooling in P_set:
        print(pooling, "--\n")
