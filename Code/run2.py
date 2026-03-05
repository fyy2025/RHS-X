import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC


def main():

    N_ITER = 300000
    N_BURN = 50000
    N_THIN = 10

    M = 3
    R = np.array([4, 3, 3])

    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)

    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

    # Profile (0, 0, 0)
    sigma_000 = None
    mu_000 = np.array([0])
    var_000 = np.array([1])

    # Profile (0, 0, 1)
    sigma_001 = np.array([[1]])
    mu_001 = np.array([-2])
    var_001 = np.array([1])

    # Profile (0, 1, 0)
    sigma_010 = np.array([[1]])
    mu_010 = np.array([1])
    var_010 = np.array([1])

    # Profile (0, 1, 1)
    sigma_011 = np.array([[1], [0]])
    mu_011 = np.array([1, -2])
    var_011 = np.array([1, 1])

    # Profile (1, 0, 0)
    sigma_100 = np.array([[0, 1]])
    mu_100 = np.array([0, 2])
    var_100 = np.array([1, 1])

    # Profile (1, 0, 1)
    sigma_101 = np.array([[0, 1], [0, np.inf]])
    mu_101 = np.array([0, 2, 1, -2])
    var_101 = np.array([1, 1, 1, 1])

    # Profile (1, 1, 0)
    sigma_110 = np.array([[0, 1], [1, np.inf]])
    mu_110 = np.array([0, -2])
    var_110 = np.array([1, 1])

    # Profile (1, 1, 1)
    sigma_111 = np.array([[0, 1], [1, np.inf], [0, np.inf]])
    mu_111 = np.array([0, 2, 4, -2])
    var_111 = np.array([1, 1, 1, 1])

    sigma = [sigma_000, sigma_001, sigma_010, sigma_011,
             sigma_100, sigma_101, sigma_110, sigma_111]

    mu = [mu_000, mu_001, mu_010, mu_011,
          mu_100, mu_101, mu_110, mu_111]

    var = [var_000, var_001, var_010, var_011,
           var_100, var_101, var_110, var_111]

    policies_profiles = {}
    policies_profiles_masked = {}
    policies_ids_profiles = {}
    pi_policies = {}
    pi_pools = {}

    for k, profile in enumerate(profiles):

        policies_temp = [(i, x) for i, x in enumerate(all_policies)
                         if hasse.policy_to_profile(x) == profile]

        unzipped_temp = list(zip(*policies_temp))
        policies_ids_k = list(unzipped_temp[0])
        policies_k = list(unzipped_temp[1])

        policies_profiles[k] = deepcopy(policies_k)
        policies_ids_profiles[k] = policies_ids_k

        profile_mask = list(map(bool, profile))

        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M)
                                     if profile_mask[i]])

        policies_profiles_masked[k] = policies_k

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(
                policies_k, sigma[k]
            )

            if len(pi_pools_k.keys()) != mu[k].shape[0]:
                print(
                    f"Profile {k}. Expected {len(pi_pools_k.keys())} pools. "
                    f"Received {mu[k].shape[0]} means."
                )

            pi_policies[k] = pi_policies_k
            pi_pools[k] = {}

            for x, y in pi_pools_k.items():
                y_full = [policies_profiles[k][i] for i in y]
                y_agg = [all_policies.index(i) for i in y_full]
                pi_pools[k][x] = y_agg
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k] = {0: [0]}

    def generate_data(mu, var, n_per_pol):
        num_data = num_policies * n_per_pol

        X = np.zeros((num_data, M))
        D = np.zeros((num_data, 1), dtype="int_")
        y = np.zeros((num_data, 1))

        idx_ctr = 0

        for k, profile in enumerate(profiles):
            policies_k = policies_profiles[k]

            for idx, policy in enumerate(policies_k):

                policy_idx = [
                    i for i, x in enumerate(all_policies) if x == policy
                ]

                pool_id = pi_policies[k][idx]
                mu_i = mu[k][pool_id]
                var_i = var[k][pool_id]

                y_i = np.random.normal(
                    mu_i, var_i, size=(n_per_pol, 1)
                )

                start_idx = idx_ctr * n_per_pol
                end_idx = (idx_ctr + 1) * n_per_pol

                X[start_idx:end_idx, :] = policy
                D[start_idx:end_idx, :] = policy_idx[0]
                y[start_idx:end_idx, :] = y_i

                idx_ctr += 1

        return X, D, y

    num_samples_per_feature = 500

    np.random.seed(721)

    X, D, y = generate_data(mu, var, num_samples_per_feature)

    policy_means = loss.compute_policy_means(D, y, num_policies)

    H = np.inf
    theta = 13
    lamb = 1

    R_set, R_profiles = aggregate.RAggregate(
        M, R, H, D, y, theta, reg=lamb, verbose=True
    )

    anchors = AIS.build_anchor_states(R_set, R_profiles, M, R)

    prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(
        all_policies, hasse.policy_to_profile
    )

    RPS_states = AIS.raggregate_to_states(
        (R_set, R_profiles), profiles
    )

    score_s = AIS.make_score_s_expneg_raw(
        D=D,
        y=y,
        M=M,
        R=R,
        prof_idx_of_policy=prof_idx_of_policy,
        policies=all_policies,
        policy_means=policy_means,
        reg=lamb,
        lattice_edges=None,
        beta=1.0,
        prior_logprob=lambda state: 0.0,
    )

    log_alpha = [score_s(A) for A in anchors]
    print(log_alpha)

    res = MCMC.run_mcmc_streaming(
        RPS_states,
        log_alpha,
        score_s,
        steps=N_ITER,
        burnin=N_BURN,
        thin=N_THIN,
        out_jsonl="mcmc_run1.jsonl",
        progress_json="mcmc_run1_progress.json",
    )


if __name__ == "__main__":
    main()