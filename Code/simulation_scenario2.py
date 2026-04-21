import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run AIS MCMC simulation")
    parser.add_argument('--epoch', type=int, required=True, help="Epoch number to run")
    parser.add_argument('--eps1', type=float, required=True, help="Probability of sampling within RPS")
    parser.add_argument('--eps2', type=float, required=True, help="Probability of sampling within RPS and its 1-edit neighbourhood")
    args = parser.parse_args()

    epoch = args.epoch
    eps1 = args.eps1
    eps2 = args.eps2
    print(f"Starting execution for epoch: {epoch}")

    N_ITER = 50000
    N_BURN = 20000
    N_THIN = 10
    n_paths = 300
    n_levels = 20
    ladder = [0, 0.05, 0.15, 0.35, 0.55, 0.75, 0.875, 1]
    # N_ITER = 110
    # N_BURN = 10
    # N_THIN = 1
    # n_paths=30
    # n_levels=5

    moves_per_level=5
    num_samples_per_feature=500

    lamb=1

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
    mu_010 = np.array([-1.5])
    var_010 = np.array([1])

    # Profile (0, 1, 1)
    sigma_011 = np.array([[1], [0]])
    mu_011 = np.array([-1, 2])
    var_011 = np.array([1, 1])

    # Profile (1, 0, 0)
    sigma_100 = np.array([[0, 1]])
    mu_100 = np.array([-1.5, 1])
    var_100 = np.array([1, 1])

    # Profile (1, 0, 1)
    sigma_101 = np.array([[0, 1], [0, np.inf]])
    mu_101 = np.array([-0.5, 2.5, 1.5, -2.5])
    var_101 = np.array([1, 1, 1, 1])

    # Profile (1, 1, 0)
    sigma_110 = np.array([[0, 1], [1, np.inf]])
    mu_110 = np.array([0, -2.5])
    var_110 = np.array([1, 1])

    # Profile (1, 1, 1)
    sigma_111 = np.array([[0, 1], [1, np.inf], [0, np.inf]])
    mu_111 = np.array([3, -0.5, -1.5, -2])
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

        policies_temp = [(i, x) for i, x in enumerate(all_policies) if hasse.policy_to_profile(x) == profile]
        unzipped_temp = list(zip(*policies_temp))
        policies_ids_k = list(unzipped_temp[0])
        policies_k = list(unzipped_temp[1])
        policies_profiles[k] = deepcopy(policies_k)
        policies_ids_profiles[k] = policies_ids_k

        profile_mask = list(map(bool, profile))

        # Mask the empty arms
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        policies_profiles_masked[k] = policies_k

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            if len(pi_pools_k.keys()) != mu[k].shape[0]:
                print(f"Profile {k}. Expected {len(pi_pools_k.keys())} pools. Received {mu[k].shape[0]} means.")
            pi_policies[k] = pi_policies_k
            # pi_pools_k has indicies that match with policies_profiles[k]
            # Need to map those indices back to all_policies
            pi_pools[k] = {}
            for x, y in pi_pools_k.items():
                y_full = [policies_profiles[k][i] for i in y]
                y_agg = [all_policies.index(i) for i in y_full]
                pi_pools[k][x] = y_agg
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k] = {0: [0]}

    def generate_data(mu, var, n_per_pol, all_policies, pi_policies, M):
        num_data = num_policies * n_per_pol
        X = np.zeros(shape=(num_data, M))
        D = np.zeros(shape=(num_data, 1), dtype='int_')
        y = np.zeros(shape=(num_data, 1))

        idx_ctr = 0
        for k, profile in enumerate(profiles):
            policies_k = policies_profiles[k]

            for idx, policy in enumerate(policies_k):
                policy_idx = [i for i, x in enumerate(all_policies) if x == policy]

                pool_id = pi_policies[k][idx]
                mu_i = mu[k][pool_id]
                var_i = var[k][pool_id]
                y_i = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))

                start_idx = idx_ctr * n_per_pol
                end_idx = (idx_ctr + 1) * n_per_pol

                X[start_idx:end_idx, ] = policy
                D[start_idx:end_idx, ] = policy_idx[0]
                y[start_idx:end_idx, ] = y_i

                idx_ctr += 1

        return X, D, y

    num_samples_per_feature = num_samples_per_feature

    overall_result = []

    # use iter = 1:10, 11:20 ...
    
    for iter in range(10*(epoch-1)+1,10*epoch+1):
        np.random.seed(iter*42)
        X, D, y = generate_data(mu, var, num_samples_per_feature, all_policies, pi_policies, M)
        policy_means = loss.compute_policy_means(D, y, num_policies)
        prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(all_policies, hasse.policy_to_profile)


        def score_s(state):
            Q = AIS.global_loss_raw(
                state=state,
                D=D, y=y, M=M, R=R,
                prof_idx_of_policy = prof_idx_of_policy,
                policies=all_policies,
                policy_means=policy_means,
                reg=lamb, normalize=0,
                lattice_edges=None,
            )
            return float(np.exp(-Q))

        result = dict()

        ### MCMC
        start = time.time()
        MCMC.run_mcmc_streaming_rand_start(
            profiles = profiles,
            M = M,
            R = R,
            score_s = score_s,
            seed = None,
            steps = N_ITER,
            burnin = N_BURN,
            thin = N_THIN,
            min_len = 1,
            out_jsonl = f"./output_files2/mcmc_samples{epoch}.jsonl",
            progress_json = f"./output_files2/mcmc_progress{epoch}.json"
        )

        mcmc_res = MCMC.load_mcmc_res_from_jsonl(f"./output_files2/mcmc_samples{epoch}.jsonl")

        MCMC_post_mean = MCMC.policy_means_matrix_from_mcmc(
            mcmc_res["samples"],                      # mcmc_res["samples"]
            all_policies,                     # global list/array of policies (length P)
            policy_means,                 # np.ndarray [P,2] = [sum_y, count]
            prof_idx_of_policy,           # length-P array: policy_id -> profile k
            R,                        # np.ndarray of arm levels (includes control)
            M,
            lattice_edges=None,           # pass None if unused
            policy_labels=None            # optional column names; defaults to policy indices (0..P-1)
        )

        log_weight = np.log(np.repeat(1/len(mcmc_res["samples"]), len(mcmc_res["samples"])))

        mcmc_summ = MCMC.quantiles_for_all_policies_root(
            mcmc_res["samples"],
            log_weight,
            D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
            M,
            all_policies,                 # global policies list (len P)
            prof_idx_of_policy,       # length-P array: global policy id -> profile k
            R,                    # per-arm levels (len M)
            lattice_edges=None,
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
            p=[0.025,0.5,0.975],
            seed=None
        )

        result["MCMC_quantiles"] = mcmc_summ

        end = time.time()

        result["MCMC_time"] = end - start
        result["MCMC"] = np.mean(MCMC_post_mean, axis=0)

        ### Exact:
        start = time.time()

        all_partitions, losses = AIS.enumerate_all_states_and_losses(
            profiles=profiles,
            R=R,
            M=M,
            policies=all_policies,
            policy_means=policy_means,
            prof_idx_of_policy=prof_idx_of_policy,
            D=D, y=y,
            reg=lamb, normalize=0,
            lattice_edges=None,
            max_states=None  # or an integer cap to safeguard
        )

        true_log_post = [-i[1] for i in losses]

        exact_post_mean = AIS.estimate_policy_means_from_RPS(
            all_partitions,                     # dict with key "samples": List[State]
            true_log_post,
            all_policies,                     # global policy list (length P)
            policy_means,                 # np.ndarray [P,2] = [sum_y, count]
            prof_idx_of_policy,           # length-P array: policy_id -> profile k, for 36 policies, which profile is each policy in
            R,                        # np.ndarray of arm levels (includes control)
            M,
            lattice_edges=None            # optional lattice; pass None if unused
        )

        exact_summ = MCMC.quantiles_for_all_policies_root(
            all_partitions,
            true_log_post,
            D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
            M,
            all_policies,                 # global policies list (len P)
            prof_idx_of_policy,       # length-P array: global policy id -> profile k
            R,                    # per-arm levels (len M)
            lattice_edges=None,
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
            p=[0.025,0.5,0.975],
            seed=None
        )

        result["exact_quantiles"] = exact_summ

        end = time.time()

        result["exact_time"] = end-start
        result["exact"] = exact_post_mean

        ### AIS / RPS

        for theta in [13, 13.1, 13.2, 13.3, 13.4]:
            start = time.time()
            H = np.inf
            R_set, R_profiles = aggregate.RAggregate(M, R, H, D, y, theta, reg=lamb, verbose=True)

            anchors = AIS.build_anchor_states(R_set, R_profiles, M, R)
            prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(all_policies, hasse.policy_to_profile)

            RPS_states = AIS.raggregate_to_states((R_set, R_profiles), profiles)

            log_alpha = [np.log(max(1e-300, score_s(A))) for A in anchors]

            RPS_post_mean = AIS.estimate_policy_means_from_RPS(
                RPS_states,                     # dict with key "samples": List[State]
                log_alpha,
                all_policies,                     # global policy list (length P)
                policy_means,                 # np.ndarray [P,2] = [sum_y, count]
                prof_idx_of_policy,           # length-P array: policy_id -> profile k, for 36 policies, which profile is each policy in
                R,                        # np.ndarray of arm levels (includes control)
                M,
                lattice_edges=None            # optional lattice; pass None if unused
            )

            RPS_summ = MCMC.quantiles_for_all_policies_root(
                RPS_states,
                log_alpha,
                D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
                M,
                all_policies,                 # global policies list (len P)
                prof_idx_of_policy,       # length-P array: global policy id -> profile k
                R,                    # per-arm levels (len M)
                lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025,0.5,0.975],
                seed=None
            )

            result[f"RPS_{theta}_quantiles"] = RPS_summ

            end = time.time()

            result[f"RPS_{theta}_time"] = end-start
            result[f"RPS_{theta}"] = RPS_post_mean


            start = time.time()
            cfg = AIS.AISConfig(n_paths=n_paths, n_levels=n_levels, moves_per_level=moves_per_level, min_len=1, seed=2)
            # ais_out = AIS.run_ais_streaming_from_data_parallel(
            #     epoch,
            #     M,
            #     R,
            #     H,
            #     x,
            #     D,
            #     y,
            #     theta,
            #     reg=lamb,
            #     eps1=eps1,
            #     eps2=eps2,
            #     all_policies=all_policies,
            #     policy_means=policy_means,
            #     score_s=score_s,
            #     out_dir="./output_files2",
            #     num_workers=1,
            #     cfg = cfg,
            # )

            ais_out = AIS.run_ais_streaming_from_data_parallel_fixed_ladder(
                epoch,
                M,
                R,
                H,
                x,
                D,
                y,
                theta,
                reg=lamb,
                eps1=eps1,
                eps2=eps2,
                all_policies=all_policies,
                policy_means=policy_means,
                score_s=score_s,
                out_dir="./output_files2",
                num_workers=1,
                cfg = cfg,
                ladder = ladder
            )

            summ = AIS.ais_quantiles_for_all_policies_root(
                ais_out,
                D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
                M,
                all_policies,                 # global policies list (len P)
                prof_idx_of_policy,       # length-P array: global policy id -> profile k
                R,                    # per-arm levels (len M)
                lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025,0.5,0.975], # make this a vector
                seed=None
            )

            result[f"AIS_{theta}_quantiles"] = summ

            mu_hat = AIS.estimate_policy_means_from_ais(
                ais_out=ais_out,                 # from your run_ais_state
                all_policies=all_policies,
                policy_means=policy_means,
                prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None,                  # or the hasse edges if you use them
                R_per = R,
                M=M
            )

            end = time.time()

            result[f"AIS_{theta}_time"] = end - start
            result[f"AIS_{theta}"] = mu_hat

        overall_result.append(result)

    with open(f"./output2/sim1_result{epoch}.pkl", "wb") as f:
        pickle.dump(overall_result, f)

if __name__ == "__main__":
    main()
