import time
import math
import pickle
import argparse
import os
import numpy as np

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC


def pac_bayes_bound(log_scores, n_obs, n_prior, delta=0.05):
    ls = np.asarray(log_scores, dtype=float)
    ls = ls[np.isfinite(ls)]
    if len(ls) == 0:
        return {"bound": np.inf, "E_Q_risk": np.inf, "kl": np.inf, "H_Q": 0.0}
    lw = ls - ls.max()
    q = np.exp(lw)
    q /= q.sum()
    e_q_loss = -ls.max() - float(np.dot(q, lw))
    e_q_risk = e_q_loss / float(n_obs)
    h_q = float(-np.dot(q, np.log(np.maximum(q, 1e-300))))
    kl = max(float(np.log(n_prior) - h_q), 0.0)
    penalty = np.log(2.0 * np.sqrt(float(n_obs)) / delta)
    bound = e_q_risk + float(np.sqrt((kl + penalty) / (2.0 * n_obs)))
    return {"bound": bound, "E_Q_risk": e_q_risk, "kl": kl, "H_Q": h_q}


def run_pac_bayes_explorer(
    init_states, init_log_scores, score_s, n_obs, n_prior,
    n_steps=300, delta=0.05, min_len=1, frontier_cap=500, seed=None,
):
    rng = np.random.default_rng(seed)
    visited = {}
    log_scores = []

    def add_visited(state, log_score):
        sig = AIS.state_signature(state)
        if sig in visited:
            return False
        visited[sig] = (state, float(log_score))
        log_scores.append(float(log_score))
        return True

    def score_state(state):
        return math.log(max(1e-300, score_s(state)))

    for state, log_score in zip(init_states, init_log_scores):
        add_visited(state, log_score)

    frontier = {}

    def add_neighbors(state):
        for nb in AIS.state_neighbors_ubs(state, min_len=min_len):
            sig = AIS.state_signature(nb)
            if sig not in visited and sig not in frontier:
                frontier[sig] = nb

    for state, _ in visited.values():
        add_neighbors(state)

    trace = [pac_bayes_bound(log_scores, n_obs, n_prior, delta=delta)]

    for _ in range(n_steps):
        if not frontier:
            break
        items = list(frontier.items())
        if frontier_cap is not None and len(items) > frontier_cap:
            idx = rng.choice(len(items), size=frontier_cap, replace=False)
            items = [items[i] for i in idx]
        chosen_sig = None
        chosen_state = None
        chosen_log_score = None
        chosen_bound = None
        for sig, state in items:
            ls = score_state(state)
            candidate_bound = pac_bayes_bound(log_scores + [ls], n_obs, n_prior, delta=delta)
            if chosen_bound is None or candidate_bound["bound"] < chosen_bound["bound"]:
                chosen_sig = sig
                chosen_state = state
                chosen_log_score = ls
                chosen_bound = candidate_bound
        frontier.pop(chosen_sig, None)
        add_visited(chosen_state, chosen_log_score)
        add_neighbors(chosen_state)
        trace.append(chosen_bound)

    states = [state for state, _ in visited.values()]
    logs = [log_score for _, log_score in visited.values()]
    return states, logs, trace


def main():
    parser = argparse.ArgumentParser(
        description="Scenario 2: MCMC, RPS, AIS-RPS, AIS-SSS, SSS, PB — Bayesian loss (Lemma A2)")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number")
    parser.add_argument("--eps1", type=float, default=0.5, help="AIS eps1")
    parser.add_argument("--eps2", type=float, default=0.75, help="AIS eps2")
    parser.add_argument("--n_sss_steps", type=int, default=200, help="SSS steps per theta")
    parser.add_argument("--pb_steps", type=int, default=300, help="PB exploration steps per theta")
    parser.add_argument("--pb_delta", type=float, default=0.05, help="PB confidence parameter")
    parser.add_argument("--frontier_cap", type=int, default=500,
                        help="Max frontier candidates per PB step")
    parser.add_argument("--thetas", type=str, default="15355,15365,15380,15400,15430",
                        help="Comma-separated Bayesian-loss RPS thresholds")
    args = parser.parse_args()

    epoch = args.epoch
    eps1 = args.eps1
    eps2 = args.eps2
    n_sss_steps = args.n_sss_steps
    pb_steps = args.pb_steps
    pb_delta = args.pb_delta
    frontier_cap = args.frontier_cap
    thetas = [float(t) for t in args.thetas.split(",")]

    print(
        f"Epoch {epoch} | eps1={eps1} eps2={eps2} | SSS steps={n_sss_steps} | "
        f"PB steps={pb_steps} delta={pb_delta} | thetas={thetas}"
    )
    os.makedirs("./output_all2", exist_ok=True)

    N_ITER = 50000
    N_BURN = 20000
    N_THIN = 10
    n_paths = 300
    n_levels = 20
    moves_per_level = 5
    ladder = [pow(10, i / 10) for i in range(-9, 1, 1)]
    num_samples_per_feature = 500

    lamb = 1
    M = 3
    R = np.array([4, 3, 3])

    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

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
    mu    = [mu_000, mu_001, mu_010, mu_011,
             mu_100, mu_101, mu_110, mu_111]
    var   = [var_000, var_001, var_010, var_011,
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
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        policies_profiles_masked[k] = policies_k

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            pi_policies[k] = pi_policies_k
            pi_pools[k] = {}
            for x, y_val in pi_pools_k.items():
                y_full = [policies_profiles[k][i] for i in y_val]
                y_agg = [all_policies.index(i) for i in y_full]
                pi_pools[k][x] = y_agg
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k] = {0: [0]}

    def generate_data(mu, var, n_per_pol, all_policies, pi_policies, M):
        num_data = num_policies * n_per_pol
        X = np.zeros(shape=(num_data, M))
        D = np.zeros(shape=(num_data, 1), dtype="int_")
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
                X[start_idx:end_idx,] = policy
                D[start_idx:end_idx,] = policy_idx[0]
                y[start_idx:end_idx,] = y_i
                idx_ctr += 1
        return X, D, y

    overall_result = []

    for iter in range(10 * (epoch - 1) + 1, 10 * epoch + 1):
        np.random.seed(iter * 42)
        X, D, y = generate_data(mu, var, num_samples_per_feature, all_policies, pi_policies, M)
        policy_means = loss.compute_policy_means(D, y, num_policies)
        prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(
            all_policies, hasse.policy_to_profile)

        # score_s uses Bayesian marginal posterior from Lemma A2:
        # exp(-Q(Π)) where Q = (n-H)/2*log(SSE) + λ'H + c(Π)
        def score_s(state):
            Q = AIS.global_loss_bayes(
                state=state, D=D, y=y, M=M, R=R,
                prof_idx_of_policy=prof_idx_of_policy,
                policies=all_policies, policy_means=policy_means,
                reg=lamb, lattice_edges=None,
            )
            return float(np.exp(-Q))

        result = dict()

        # MCMC
        start = time.time()
        MCMC.run_mcmc_streaming_rand_start(
            profiles=profiles, M=M, R=R, score_s=score_s,
            seed=None, steps=N_ITER, burnin=N_BURN, thin=N_THIN, min_len=1,
            out_jsonl=f"./output_all2/mcmc_samples_epoch{epoch}_iter{iter}.jsonl",
            progress_json=f"./output_all2/mcmc_progress_epoch{epoch}_iter{iter}.json",
        )
        mcmc_res = MCMC.load_mcmc_res_from_jsonl(
            f"./output_all2/mcmc_samples_epoch{epoch}_iter{iter}.jsonl")
        MCMC_post_mean = MCMC.policy_means_matrix_from_mcmc(
            mcmc_res["samples"], all_policies, policy_means,
            prof_idx_of_policy, R, M, lattice_edges=None, policy_labels=None,
        )
        log_weight_mcmc = np.log(
            np.repeat(1.0 / len(mcmc_res["samples"]), len(mcmc_res["samples"])))
        mcmc_summ = MCMC.quantiles_for_all_policies_root(
            mcmc_res["samples"], log_weight_mcmc, D, y, M, all_policies,
            prof_idx_of_policy, R, lattice_edges=None,
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0, p=[0.025, 0.5, 0.975], seed=None,
        )
        result["MCMC_quantiles"] = mcmc_summ
        result["MCMC_time"] = time.time() - start
        result["MCMC"] = np.mean(MCMC_post_mean, axis=0)

        # Exact: enumerate all partitions, score with Bayesian loss
        start = time.time()
        all_partitions, _ = AIS.enumerate_all_states_and_losses(
            profiles=profiles, R=R, M=M, policies=all_policies,
            policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
            D=D, y=y, reg=lamb, normalize=0, lattice_edges=None, max_states=None,
        )
        true_log_post = [
            -AIS.global_loss_bayes(
                state=s, D=D, y=y, M=M, R=R,
                prof_idx_of_policy=prof_idx_of_policy,
                policies=all_policies, policy_means=policy_means,
                reg=lamb, lattice_edges=None,
            )
            for s in all_partitions
        ]
        exact_post_mean = AIS.estimate_policy_means_from_RPS(
            all_partitions, true_log_post, all_policies, policy_means,
            prof_idx_of_policy, R, M, lattice_edges=None,
        )
        exact_summ = MCMC.quantiles_for_all_policies_root(
            all_partitions, true_log_post, D, y, M, all_policies,
            prof_idx_of_policy, R, lattice_edges=None,
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0, p=[0.025, 0.5, 0.975], seed=None,
        )
        result["exact_quantiles"] = exact_summ
        result["exact_time"] = time.time() - start
        result["exact"] = exact_post_mean
        result["n_exact_states"] = len(all_partitions)

        n_obs = int(y.shape[0])
        n_prior = int(len(all_partitions))

        cfg = AIS.AISConfig(
            n_paths=n_paths, n_levels=n_levels,
            moves_per_level=moves_per_level, min_len=1, seed=None,
        )

        # Compute thresholds in Bayesian-loss scale relative to the MAP
        map_q = -max(true_log_post)
        theta_offsets = [10, 15, 20, 35, 50]

        for theta_offset in theta_offsets:
            theta = map_q + theta_offset
            H = np.inf
            R_set, R_profiles = aggregate.RAggregate(
                M, R, H, D, y, theta, reg=lamb, verbose=False, use_bayes=True)
            anchors = AIS.build_anchor_states(R_set, R_profiles, M, R)
            if len(anchors) == 0:
                continue
            log_alpha = [math.log(max(1e-300, score_s(A))) for A in anchors]
            RPS_states = AIS.raggregate_to_states((R_set, R_profiles), profiles)

            d = theta_offset
            result[f"RPS_{d}_coverage"] = MCMC.posterior_mass_coverage(
                log_alpha, true_log_post)
            result[f"RPS_{d}_n_states"] = len(RPS_states)
            result[f"RPS_{d}"] = AIS.estimate_policy_means_from_RPS(
                RPS_states, log_alpha, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"RPS_{d}_quantiles"] = MCMC.quantiles_for_all_policies_root(
                RPS_states, log_alpha, D, y, M, all_policies,
                prof_idx_of_policy, R, lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )

            # SSS warm-started from the highest-scoring RPS anchor
            start = time.time()
            best_anchor = anchors[int(np.argmax(log_alpha))]
            sss_visited = MCMC.run_sss(
                profiles=profiles, M=M, R=R, score_s=score_s,
                n_steps=n_sss_steps, min_len=1, seed=None,
                start_state=best_anchor,
            )
            sss_states = [s for s, _ in sss_visited]
            sss_log_scores = [ls for _, ls in sss_visited]
            result[f"SSS_{d}_coverage"] = MCMC.posterior_mass_coverage(
                sss_log_scores, true_log_post)
            result[f"SSS_{d}_n_visited"] = len(sss_visited)
            result[f"SSS_{d}"] = AIS.estimate_policy_means_from_RPS(
                sss_states, sss_log_scores, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"SSS_{d}_quantiles"] = MCMC.quantiles_for_all_policies_root(
                sss_states, sss_log_scores, D, y, M, all_policies,
                prof_idx_of_policy, R, lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )
            result[f"SSS_{d}_time"] = time.time() - start

            # AIS seeded from RPS
            start = time.time()
            ais_rps_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=RPS_states,
                init_log_alpha=log_alpha,
                R=R, eps1=eps1, eps2=eps2,
                score_s=score_s, cfg=cfg,
                out_dir="./output_all2",
                label=f"epoch{epoch}_iter{iter}_RPS_{d}",
            )
            result[f"AIS_RPS_{d}"] = AIS.estimate_policy_means_from_ais(
                ais_out=ais_rps_out, all_policies=all_policies,
                policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None, R_per=R, M=M,
            )
            result[f"AIS_RPS_{d}_quantiles"] = AIS.ais_quantiles_for_all_policies_root(
                ais_rps_out, D, y, M, all_policies, prof_idx_of_policy, R,
                lattice_edges=None, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )
            result[f"AIS_RPS_{d}_coverage"] = MCMC.posterior_mass_coverage(
                [math.log(max(1e-300, score_s(s))) for s in ais_rps_out.terminals],
                true_log_post,
            )
            result[f"AIS_RPS_{d}_time"] = time.time() - start

            # AIS seeded from SSS
            start = time.time()
            ais_sss_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=sss_states,
                init_log_alpha=sss_log_scores,
                R=R, eps1=eps1, eps2=eps2,
                score_s=score_s, cfg=cfg,
                out_dir="./output_all2",
                label=f"epoch{epoch}_iter{iter}_SSS_{d}",
            )
            result[f"AIS_SSS_{d}"] = AIS.estimate_policy_means_from_ais(
                ais_out=ais_sss_out, all_policies=all_policies,
                policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None, R_per=R, M=M,
            )
            result[f"AIS_SSS_{d}_quantiles"] = AIS.ais_quantiles_for_all_policies_root(
                ais_sss_out, D, y, M, all_policies, prof_idx_of_policy, R,
                lattice_edges=None, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )
            result[f"AIS_SSS_{d}_coverage"] = MCMC.posterior_mass_coverage(
                [math.log(max(1e-300, score_s(s))) for s in ais_sss_out.terminals],
                true_log_post,
            )
            result[f"AIS_SSS_{d}_time"] = time.time() - start

            # PB greedy explorer
            start = time.time()
            pb_states, pb_log_scores, pb_trace = run_pac_bayes_explorer(
                init_states=RPS_states,
                init_log_scores=log_alpha,
                score_s=score_s,
                n_obs=n_obs,
                n_prior=n_prior,
                n_steps=pb_steps,
                delta=pb_delta,
                min_len=1,
                frontier_cap=frontier_cap,
                seed=iter * 42,
            )
            result[f"PB_{d}"] = AIS.estimate_policy_means_from_RPS(
                pb_states, pb_log_scores, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"PB_{d}_quantiles"] = MCMC.quantiles_for_all_policies_root(
                pb_states, pb_log_scores, D, y, M, all_policies,
                prof_idx_of_policy, R, lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )
            result[f"PB_{d}_coverage"] = MCMC.posterior_mass_coverage(
                pb_log_scores, true_log_post)
            result[f"PB_{d}_n_states"] = len(pb_states)
            result[f"PB_{d}_state_fraction"] = len(pb_states) / float(n_prior)
            result[f"PB_{d}_exhausted_state_space"] = len(pb_states) >= n_prior
            result[f"PB_{d}_bound"] = pb_trace[-1]["bound"]
            result[f"PB_{d}_risk"] = pb_trace[-1]["E_Q_risk"]
            result[f"PB_{d}_kl"] = pb_trace[-1]["kl"]
            result[f"PB_{d}_entropy"] = pb_trace[-1]["H_Q"]
            result[f"PB_{d}_trace"] = pb_trace
            result[f"PB_{d}_time"] = time.time() - start

        overall_result.append(result)

    with open(f"./output_all2/sim2_all_result{epoch}.pkl", "wb") as f:
        pickle.dump(overall_result, f)
    print(f"Saved epoch {epoch} to ./output_all2/sim2_all_result{epoch}.pkl")


if __name__ == "__main__":
    main()
