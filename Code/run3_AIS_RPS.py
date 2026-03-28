import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC


def main():

    N_CHAIN = 20
    theta = 13.5
    lamb = 1

    n_paths=300
    n_levels=20
    moves_per_level=5


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
    mu_010 = np.array([-3])
    var_010 = np.array([1])

    # Profile (0, 1, 1)
    sigma_011 = np.array([[1], [0]])
    mu_011 = np.array([-1, 2])
    var_011 = np.array([1, 1])

    # Profile (1, 0, 0)
    sigma_100 = np.array([[0, 1]])
    mu_100 = np.array([3, 4])
    var_100 = np.array([1, 1])

    # Profile (1, 0, 1)
    sigma_101 = np.array([[0, 1], [0, np.inf]])
    mu_101 = np.array([-5, 2.5, 1.5, -2.5])
    var_101 = np.array([1, 1, 1, 1])

    # Profile (1, 1, 0)
    sigma_110 = np.array([[0, 1], [1, np.inf]])
    mu_110 = np.array([0, -2.5])
    var_110 = np.array([1, 1])

    # Profile (1, 1, 1)
    sigma_111 = np.array([[0, 1], [1, np.inf], [0, np.inf]])
    mu_111 = np.array([3.5, -0.5, -1.5, -3.5])
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

    RPS_mean = AIS.estimate_policy_means_from_RPS(
        RPS_states,                     # dict with key "samples": List[State]
        log_alpha,
        all_policies,                     # global policy list (length P)
        policy_means,                 # np.ndarray [P,2] = [sum_y, count]
        prof_idx_of_policy,           # length-P array: policy_id -> profile k, for 36 policies, which profile is each policy in
        R,                        # np.ndarray of arm levels (includes control)
        M,
        lattice_edges=None            # optional lattice; pass None if unused
    )

    buckets = AIS.make_p0_buckets_weighted_S0(RPS_states, np.asarray(R,int), log_alpha,
                                        eps1=0.05, eps2=0.25, min_len=1)
    log_p0 = lambda z: AIS.log_p0_distance_weighted_S0(z, buckets)

    ladder, ess_ratios = AIS.pilot_adaptive_ladder(
        init_sampler=lambda N: MCMC.init_from_RPS_batch(RPS_states, log_alpha, N, rng_seed=777),
        log_p0=log_p0,
        score_s=score_s,
        N=512,
        ess_target=0.80,
        beta0=0.0, beta1=1.0,
        initial_delta=1/n_levels,
        min_delta=1e-3,
        moves_per_probe=3,   # small mixing at each accepted β (optional)
        min_len=1,
        rng_seed=777
    )
    print("Adaptive ladder has", len(ladder), "levels; first 10:", ladder[:10])
    print("Per-step ESS ratios (len =", len(ess_ratios), "):", ess_ratios[:10])

    AIS_RPS_diff = []
    for iter in range(N_CHAIN):
        cfg = AIS.AISConfig(n_paths=n_paths, n_levels=n_levels, moves_per_level=moves_per_level, min_len=1, seed=iter*42)
        out = AIS.run_ais_state_streaming(
            anchors=R_set,          # not used by q0, but you may keep for consistency
            score_s=score_s,        # returns exp(-loss(state)) or similar
            cfg=cfg,
            RPS=RPS_states,              # your Rashomon partitions as a list of State
            R_per=R,  # levels per arm (includes control)
            eps1=0.5, eps2=0.75,
            out_jsonl = f'AIS_samples_diff_theta{theta}_lambda{lamb}.jsonl',
            ladder=ladder
        )

        mu_hat = AIS.estimate_policy_means_from_ais(
            ais_out=out,                 # from your run_ais_state
            all_policies=all_policies,
            policy_means=policy_means,
            prof_idx_of_policy=prof_idx_of_policy,
            lattice_edges=None,                  # or the hasse edges if you use them
            R_per = R,
            M=M
        )

        diff_mean = mu_hat - RPS_mean
        AIS_RPS_diff.append(diff_mean)

    diff_quants = {q: np.quantile(AIS_RPS_diff, q, axis=0) for q in (0.1, 0.5, 0.9)}
    out = pd.DataFrame.from_dict(diff_quants)
    print(out)
    out.to_csv(f'AIS_RPS_diff_n{theta}_lambda{lamb}')
if __name__ == "__main__":
    main()