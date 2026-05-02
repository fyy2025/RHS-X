import time
import math
import pickle
import argparse
import os
import numpy as np

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC


def pac_bayes_bound(log_scores, n_obs, n_prior, delta=0.05):
    """
    Rockova-style PAC-Bayes bound for the Gibbs posterior over a visited
    state set.

    The empirical risk is loss / n_obs, where score_s(state)=exp(-loss).
    The prior is uniform over n_prior exact states.
    """
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
    init_states,
    init_log_scores,
    score_s,
    n_obs,
    n_prior,
    n_steps=300,
    delta=0.05,
    min_len=1,
    frontier_cap=500,
    seed=None,
):
    """
    Greedy PAC-Bayes state-space explorer.

    Starting from RPS states, maintain a visited set and a neighbor frontier.
    At each step, add the frontier state that gives the smallest PAC-Bayes
    bound for the Gibbs posterior over visited + candidate states.

    This chooses states within one fixed theta run. It does not choose theta.
    """
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
            candidate_bound = pac_bayes_bound(
                log_scores + [ls], n_obs, n_prior, delta=delta)
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
        description="Scenario 1 with PAC-Bayes state-space exploration compared with AIS+RPS")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number")
    parser.add_argument("--eps1", type=float, default=0.5, help="AIS eps1")
    parser.add_argument("--eps2", type=float, default=0.75, help="AIS eps2")
    parser.add_argument("--pb_steps", type=int, default=300,
                        help="PAC-Bayes exploration steps per theta")
    parser.add_argument("--pb_delta", type=float, default=0.05,
                        help="PAC-Bayes confidence parameter")
    parser.add_argument("--frontier_cap", type=int, default=500,
                        help="Max frontier candidates scored per PB step")
    args = parser.parse_args()

    epoch = args.epoch
    eps1 = args.eps1
    eps2 = args.eps2
    pb_steps = args.pb_steps
    pb_delta = args.pb_delta
    frontier_cap = args.frontier_cap

    print(
        f"Epoch {epoch} | AIS eps1={eps1} eps2={eps2} | "
        f"PB steps={pb_steps} delta={pb_delta} frontier_cap={frontier_cap}"
    )
    os.makedirs("./output_pb", exist_ok=True)

    n_paths = 300
    n_levels = 20
    moves_per_level = 5
    ladder = [pow(10, i / 10) for i in range(-9, 1, 1)]
    num_samples_per_feature = 500

    lamb = 1
    M = 2
    R = np.array([4, 3])

    num_profiles = 2**M
    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)

    sigma_00 = None
    mu_00 = np.array([0])
    var_00 = np.array([1])
    sigma_01 = np.array([[1]])
    mu_01 = np.array([-1])
    var_01 = np.array([1])
    sigma_10 = np.array([[1, 0]])
    mu_10 = np.array([-2, -3])
    var_10 = np.array([1, 1])
    sigma_11 = np.array([[0, 1], [0, np.inf]])
    mu_11 = np.array([2, 3, -1, 1])
    var_11 = np.array([1, 1, 1, 1])

    sigma = [sigma_00, sigma_01, sigma_10, sigma_11]
    mu = [mu_00, mu_01, mu_10, mu_11]
    var = [var_00, var_01, var_10, var_11]

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
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(
                policies_k, sigma[k])
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
        X, D, y = generate_data(
            mu, var, num_samples_per_feature, all_policies, pi_policies, M)
        policy_means = loss.compute_policy_means(D, y, num_policies)
        prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(
            all_policies, hasse.policy_to_profile)

        def score_s(state):
            Q = AIS.global_loss_raw(
                state=state, D=D, y=y, M=M, R=R,
                prof_idx_of_policy=prof_idx_of_policy,
                policies=all_policies, policy_means=policy_means,
                reg=lamb, normalize=0, lattice_edges=None,
            )
            return float(np.exp(-Q))

        result = dict()

        start = time.time()
        all_partitions, losses = AIS.enumerate_all_states_and_losses(
            profiles=profiles, R=R, M=M, policies=all_policies,
            policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
            D=D, y=y, reg=lamb, normalize=0, lattice_edges=None, max_states=None,
        )
        true_log_post = [-i[1] for i in losses]
        exact_post_mean = AIS.estimate_policy_means_from_RPS(
            all_partitions, true_log_post, all_policies, policy_means,
            prof_idx_of_policy, R, M, lattice_edges=None,
        )
        exact_summ = MCMC.quantiles_for_all_policies_root(
            all_partitions, true_log_post, D, y, M, all_policies,
            prof_idx_of_policy, R, lattice_edges=None,
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
            p=[0.025, 0.5, 0.975], seed=None,
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

        # PB/Rockova is run separately at each theta, matching AIS+RPS.
        # The PB bound guides state exploration within theta; it is not used
        # to select a best theta.
        for theta in [7.8, 8, 8.2, 8.4, 8.6]:
            H = np.inf
            R_set, R_profiles = aggregate.RAggregate(
                M, R, H, D, y, theta, reg=lamb, verbose=False)
            anchors = AIS.build_anchor_states(R_set, R_profiles, M, R)
            log_alpha = [math.log(max(1e-300, score_s(A))) for A in anchors]
            RPS_states = AIS.raggregate_to_states((R_set, R_profiles), profiles)

            result[f"RPS_{theta}_coverage"] = MCMC.posterior_mass_coverage(
                log_alpha, true_log_post)
            result[f"RPS_{theta}_n_states"] = len(RPS_states)
            result[f"RPS_{theta}"] = AIS.estimate_policy_means_from_RPS(
                RPS_states, log_alpha, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"RPS_{theta}_quantiles"] = MCMC.quantiles_for_all_policies_root(
                RPS_states, log_alpha, D, y, M, all_policies,
                prof_idx_of_policy, R, lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )

            start = time.time()
            ais_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=RPS_states,
                init_log_alpha=log_alpha,
                R=R, eps1=eps1, eps2=eps2,
                score_s=score_s, cfg=cfg,
                out_dir="./output_pb",
                label=f"epoch{epoch}_iter{iter}_RPS_{theta}",
            )
            result[f"AIS_RPS_{theta}"] = AIS.estimate_policy_means_from_ais(
                ais_out=ais_out, all_policies=all_policies,
                policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None, R_per=R, M=M,
            )
            result[f"AIS_RPS_{theta}_quantiles"] = AIS.ais_quantiles_for_all_policies_root(
                ais_out, D, y, M, all_policies, prof_idx_of_policy, R,
                lattice_edges=None, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )
            result[f"AIS_RPS_{theta}_coverage"] = MCMC.posterior_mass_coverage(
                [math.log(max(1e-300, score_s(s))) for s in ais_out.terminals],
                true_log_post,
            )
            result[f"AIS_RPS_{theta}_time"] = time.time() - start

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
            result[f"PB_{theta}"] = AIS.estimate_policy_means_from_RPS(
                pb_states, pb_log_scores, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"PB_{theta}_quantiles"] = MCMC.quantiles_for_all_policies_root(
                pb_states, pb_log_scores, D, y, M, all_policies,
                prof_idx_of_policy, R, lattice_edges=None,
                mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
                p=[0.025, 0.5, 0.975], seed=None,
            )
            result[f"PB_{theta}_coverage"] = MCMC.posterior_mass_coverage(
                pb_log_scores, true_log_post)
            result[f"PB_{theta}_n_states"] = len(pb_states)
            result[f"PB_{theta}_bound"] = pb_trace[-1]["bound"]
            result[f"PB_{theta}_risk"] = pb_trace[-1]["E_Q_risk"]
            result[f"PB_{theta}_kl"] = pb_trace[-1]["kl"]
            result[f"PB_{theta}_entropy"] = pb_trace[-1]["H_Q"]
            result[f"PB_{theta}_trace"] = pb_trace
            result[f"PB_{theta}_time"] = time.time() - start

        overall_result.append(result)

    with open(f"./output_pb/sim1_pb_result{epoch}.pkl", "wb") as f:
        pickle.dump(overall_result, f)
    print(f"Saved epoch {epoch} to ./output_pb/sim1_pb_result{epoch}.pkl")


if __name__ == "__main__":
    main()
