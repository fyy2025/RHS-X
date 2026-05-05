import argparse
import math
import os
import pickle
import time
from copy import deepcopy

import numpy as np

from rashomon import AIS, MCMC, aggregate, extract_pools, hasse, loss


def parse_ais_cfg_grid(cfg_grid):
    configs = []
    for raw in cfg_grid.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Each AIS config must be n_samples:moves_per_level:n_rungs, "
                f"got {raw!r}"
            )
        n_samples, moves_per_level, n_rungs = map(int, parts)
        if n_samples <= 0 or moves_per_level <= 0 or n_rungs < 2:
            raise ValueError(
                "AIS config values must satisfy n_samples > 0, "
                "moves_per_level > 0, and n_rungs >= 2."
            )
        configs.append(
            {
                "n_samples": n_samples,
                "moves_per_level": moves_per_level,
                "n_rungs": n_rungs,
            }
        )
    if not configs:
        raise ValueError("At least one AIS config is required.")
    return configs


def make_log_ladder(n_rungs, min_log10=-0.9):
    """Geometric AIS ladder with strictly positive starts and endpoint 1."""
    n_rungs = int(n_rungs)
    if n_rungs < 1:
        raise ValueError("n_rungs must be at least 1.")
    return np.logspace(min_log10, 0.0, n_rungs)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scenario 2 AIS config sweep: one simulated dataset, one MCMC run, "
            "multiple AIS runs with different AISConfig parameters."
        )
    )
    parser.add_argument("--iter", type=int, default=1,
                        help="Dataset index; seed is iter * 42.")
    parser.add_argument("--theta_init", type=float, default=13.0,
                        help="Initial MSE-scale theta to find the MAP state.")
    parser.add_argument("--epsilons", type=str,
                        default="0.16,0.17,0.18,0.19,0.20",
                        help="Comma-separated Rashomon ratio offsets, theta = MAP_Q * (1 + epsilon).")
    parser.add_argument("--eps1", type=float, default=0.4)
    parser.add_argument("--eps2", type=float, default=0.7)
    parser.add_argument("--ais_cfg_grid", type=str,
                        default="300:3:5,300:5:5,300:12:5,300:3:10,300:5:10,300:12:10",
                        help="Comma-separated AIS configs: n_samples:moves_per_level:n_rungs.")
    parser.add_argument("--ladder_min_log10", type=float, default=-0.9,
                        help="Smallest log10 beta for the geometric AIS ladder.")
    parser.add_argument("--mcmc_steps", type=int, default=50000)
    parser.add_argument("--mcmc_burnin", type=int, default=20000)
    parser.add_argument("--mcmc_thin", type=int, default=10)
    parser.add_argument("--n_per_pol", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="./output2")
    parser.add_argument("--out_name", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ais_cfgs = parse_ais_cfg_grid(args.ais_cfg_grid)
    epsilons = [float(e) for e in args.epsilons.split(",")]

    lamb = 1
    M = 3
    R = np.array([4, 3, 3])
    profiles, _ = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)
    g = num_policies * args.n_per_pol

    sigma_000 = None
    mu_000 = np.array([0])
    var_000 = np.array([1])

    sigma_001 = np.array([[1]])
    mu_001 = np.array([-2])
    var_001 = np.array([1])

    sigma_010 = np.array([[1]])
    mu_010 = np.array([-1.5])
    var_010 = np.array([1])

    sigma_011 = np.array([[1], [0]])
    mu_011 = np.array([-1, 2])
    var_011 = np.array([1, 1])

    sigma_100 = np.array([[0, 1]])
    mu_100 = np.array([-1.5, 1])
    var_100 = np.array([1, 1])

    sigma_101 = np.array([[0, 1], [0, np.inf]])
    mu_101 = np.array([-0.5, 2.5, 1.5, -2.5])
    var_101 = np.array([1, 1, 1, 1])

    sigma_110 = np.array([[0, 1], [1, np.inf]])
    mu_110 = np.array([0, -2.5])
    var_110 = np.array([1, 1])

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
    pi_policies = {}
    pi_pools = {}
    for k, profile in enumerate(profiles):
        policies_temp = [
            (i, x) for i, x in enumerate(all_policies)
            if hasse.policy_to_profile(x) == profile
        ]
        policy_ids_k, policies_k = zip(*policies_temp)
        policy_ids_k = list(policy_ids_k)
        policies_k = list(policies_k)
        policies_profiles[k] = deepcopy(policies_k)

        profile_mask = list(map(bool, profile))
        policies_masked_k = [
            tuple(pol[i] for i in range(M) if profile_mask[i])
            for pol in policies_k
        ]

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(
                policies_masked_k, sigma[k]
            )
            pi_policies[k] = pi_policies_k
            pi_pools[k] = {}
            for pool_id, local_idxs in pi_pools_k.items():
                global_policies = [policies_profiles[k][i] for i in local_idxs]
                pi_pools[k][pool_id] = [all_policies.index(p) for p in global_policies]
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k] = {0: [policy_ids_k[0]]}

    def generate_data():
        num_data = num_policies * args.n_per_pol
        X = np.zeros((num_data, M))
        D = np.zeros((num_data, 1), dtype="int_")
        y = np.zeros((num_data, 1))
        idx_ctr = 0
        for k, _profile in enumerate(profiles):
            for idx, policy in enumerate(policies_profiles[k]):
                policy_idx = all_policies.index(policy)
                pool_id = pi_policies[k][idx]
                mu_i = mu[k][pool_id]
                var_i = var[k][pool_id]
                y_i = np.random.normal(mu_i, var_i, size=(args.n_per_pol, 1))
                start_idx = idx_ctr * args.n_per_pol
                end_idx = (idx_ctr + 1) * args.n_per_pol
                X[start_idx:end_idx] = policy
                D[start_idx:end_idx] = policy_idx
                y[start_idx:end_idx] = y_i
                idx_ctr += 1
        return X, D, y

    np.random.seed(args.iter * 42)
    X, D, y = generate_data()
    policy_means = loss.compute_policy_means(D, y, num_policies)
    prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(
        all_policies, hasse.policy_to_profile
    )

    def score_s(state):
        Q = AIS.global_loss_raw(
            state=state, D=D, y=y, M=M, R=R,
            prof_idx_of_policy=prof_idx_of_policy,
            policies=all_policies, policy_means=policy_means,
            reg=lamb, lattice_edges=None,
        )
        return float(np.exp(-Q))

    result = {
        "iter": args.iter,
        "seed": args.iter * 42,
        "epsilons": epsilons,
        "eps1": args.eps1,
        "eps2": args.eps2,
        "ladder_min_log10": args.ladder_min_log10,
        "theta_init": args.theta_init,
        "ais_cfg_grid": ais_cfgs,
        "D": D,
        "y": y,
    }

    sigma2 = AIS.compute_sigma2_saturated(D, y, all_policies)
    normal_kw = dict(
        D=D, y=y, M=M,
        policies=all_policies,
        prof_idx_of_policy=prof_idx_of_policy,
        R=R, g=g, sigma2=sigma2,
        lattice_edges=None,
        p=[0.025, 0.5, 0.975],
    )
    result["sigma2"] = sigma2

    # One MCMC baseline for this dataset.
    print(
        f"Running one MCMC baseline: iter={args.iter}, "
        f"steps={args.mcmc_steps}, burnin={args.mcmc_burnin}, thin={args.mcmc_thin}"
    )
    start = time.time()
    mcmc_jsonl = os.path.join(args.out_dir, f"mcmc_ais_cfg_iter{args.iter}.jsonl")
    mcmc_progress = os.path.join(args.out_dir, f"mcmc_ais_cfg_iter{args.iter}.json")
    MCMC.run_mcmc_streaming_rand_start(
        profiles=profiles, M=M, R=R, score_s=score_s,
        seed=args.iter * 42, steps=args.mcmc_steps,
        burnin=args.mcmc_burnin, thin=args.mcmc_thin, min_len=1,
        out_jsonl=mcmc_jsonl, progress_json=mcmc_progress,
    )
    mcmc_res = MCMC.load_mcmc_res_from_jsonl(mcmc_jsonl)
    log_weight = np.log(
        np.repeat(1.0 / len(mcmc_res["samples"]), len(mcmc_res["samples"]))
    )
    mcmc_mean_matrix = MCMC.policy_means_matrix_from_mcmc(
        mcmc_res["samples"], all_policies, policy_means,
        prof_idx_of_policy, R, M, lattice_edges=None, policy_labels=None,
    )
    result["MCMC_states"] = mcmc_res["samples"]
    result["MCMC_logw"] = log_weight
    result["MCMC"] = np.mean(mcmc_mean_matrix, axis=0)
    result["MCMC_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
        mcmc_res["samples"], log_weight, **normal_kw
    )
    result["MCMC_time"] = time.time() - start

    # One RPS seed set for all AIS configs.
    R_set_init, R_profiles_init = aggregate.RAggregate(
        M, R, np.inf, D, y, args.theta_init, reg=lamb, verbose=False
    )
    rps_init = AIS.raggregate_to_states((R_set_init, R_profiles_init), profiles)
    log_init = [math.log(max(1e-300, score_s(s))) for s in rps_init]
    map_q = -max(log_init) if log_init else float("inf")
    result["map_q"] = map_q

    # Five RPS seed sets; for each one, run the full AIS config grid.
    for eps_idx, epsilon in enumerate(epsilons, start=1):
        theta = map_q * (1 + epsilon)
        print(f"MAP_Q={map_q:.6g}; epsilon={epsilon}; theta={theta:.6g}")

        R_set, R_profiles = aggregate.RAggregate(
            M, R, np.inf, D, y, theta, reg=lamb, verbose=False
        )
        anchors = AIS.build_anchor_states(R_set, R_profiles, M, R)
        if len(anchors) == 0:
            print(f"  epsilon={epsilon}: empty RPS set, skipping")
            continue
        log_alpha = [math.log(max(1e-300, score_s(a))) for a in anchors]
        RPS_states = AIS.raggregate_to_states((R_set, R_profiles), profiles)
        RPS_post_mean = AIS.estimate_policy_means_from_RPS(
            RPS_states, log_alpha, all_policies, policy_means,
            prof_idx_of_policy, R, M, lattice_edges=None,
        )

        eps_key = f"{epsilon:g}"
        result[f"RPS_{eps_key}_theta"] = theta
        result[f"RPS_{eps_key}_states"] = RPS_states
        result[f"RPS_{eps_key}_logw"] = log_alpha
        result[f"RPS_{eps_key}"] = RPS_post_mean
        result[f"RPS_{eps_key}_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
            RPS_states, log_alpha, **normal_kw
        )
        result[f"RPS_{eps_key}_n_states"] = len(RPS_states)

        for cfg_idx, cfg_params in enumerate(ais_cfgs, start=1):
            ladder = make_log_ladder(cfg_params["n_rungs"], args.ladder_min_log10)
            cfg = AIS.AISConfig(
                n_paths=cfg_params["n_samples"],
                n_levels=cfg_params["n_rungs"],
                moves_per_level=cfg_params["moves_per_level"],
                min_len=1,
                seed=args.iter * 100000 + eps_idx * 1000 + cfg_idx,
            )
            label = (
                f"iter{args.iter}_eps{epsilon:g}_cfg{cfg_idx}_"
                f"s{cfg.n_paths}_m{cfg.moves_per_level}_r{cfg.n_levels}"
            )
            print(
                f"Running AIS epsilon {eps_idx}/{len(epsilons)} "
                f"config {cfg_idx}/{len(ais_cfgs)}: "
                f"epsilon={epsilon}, n_samples={cfg.n_paths}, "
                f"moves_per_level={cfg.moves_per_level}, n_rungs={cfg.n_levels}"
            )
            start = time.time()
            ais_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=RPS_states,
                init_log_alpha=log_alpha,
                R=R,
                eps1=args.eps1,
                eps2=args.eps2,
                score_s=score_s,
                cfg=cfg,
                out_dir=args.out_dir,
                label=label,
            )
            ais_post_mean = AIS.estimate_policy_means_from_ais(
                ais_out=ais_out,
                all_policies=all_policies,
                policy_means=policy_means,
                prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None,
                R_per=R,
                M=M,
            )
            key = f"AIS_{eps_key}_cfg{cfg_idx}"
            result[f"{key}_params"] = cfg_params
            result[f"{key}_ladder"] = ladder
            result[f"{key}_label"] = label
            result[f"{key}_output"] = ais_out
            result[f"{key}"] = ais_post_mean
            result[f"{key}_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(
                ais_out, **normal_kw
            )
            result[f"{key}_time"] = time.time() - start

    out_name = args.out_name or f"sim2_ais_cfg_iter{args.iter}.pkl"
    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved AIS config sweep -> {out_path}")


if __name__ == "__main__":
    main()
