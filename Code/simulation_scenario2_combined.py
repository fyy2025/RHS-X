import time
import math
import pickle
import argparse
import os
import numpy as np

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC


# ---------------------------------------------------------------------------
# Scenario 2 — combined method comparison in a SINGLE pkl per epoch:
#
#     RPS                : Rashomon partition set (reference set)
#     AIS   (RPS seed)   : AIS annealed from the RPS states
#     AIS   (random seed): AIS annealed from uniformly-random states,
#                          matched in count to the RPS
#
# MCMC, exact enumeration and PAC-Bayes are dropped. The two AIS variants share
# the same data, seeds, score, annealing ladder, eps1/eps2 and seed *count* — so
# the AIS vs AIS_rand comparison isolates seed *quality* (RPS vs random). The
# annealing ladder length is configurable (--n_ladder): a shorter ladder starves
# the annealing so terminal states stay near the seed p0, which is what exposes
# the RPS-vs-random seed effect. Results are keyed:
#
#     RPS_{theta}_*        AIS_{theta}_*        AIS_rand_{theta}_*
# ---------------------------------------------------------------------------

def _sample_random_seed_states(n_seeds, M, R, profiles, score_s, base_seed):
    """Draw `n_seeds` distinct uniformly-random states and score them.

    Returns (states, log_scores). Deduplicates by state signature so the p0
    buckets are not skewed by repeated draws; tops up with fresh seeds until
    `n_seeds` unique states are collected or the attempt budget is exhausted.
    """
    states, log_scores, seen = [], [], set()
    max_attempts = max(50 * n_seeds, 1000)
    attempt = 0
    while len(states) < n_seeds and attempt < max_attempts:
        st = AIS.random_partition_state_uniform_bits(
            M, R, seed=base_seed + attempt, profiles=profiles)
        attempt += 1
        sig = AIS.state_signature(st)
        if sig in seen:
            continue
        seen.add(sig)
        states.append(st)
        log_scores.append(math.log(max(1e-300, score_s(st))))
    return states, log_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scenario 2 — RPS + AIS(RPS seed) + AIS(random seed), configurable ladder")
    parser.add_argument("--epoch",    type=int,   required=True)
    parser.add_argument("--eps1",     type=float, default=0.4)
    parser.add_argument("--eps2",     type=float, default=0.7)
    parser.add_argument("--n_ladder", type=int,   default=4,
                        help="Number of annealing ladder rungs (10**-0.9 .. 1.0). "
                             "Fewer rungs = less annealing = stronger seed effect.")
    parser.add_argument("--thetas",   type=str,
                        default="13.0,13.1,13.2,13.3,13.4",
                        help="Comma-separated theta values for the Rashomon set")
    args = parser.parse_args()

    epoch    = args.epoch
    eps1     = args.eps1
    eps2     = args.eps2
    n_ladder = args.n_ladder
    thetas   = [float(t) for t in args.thetas.split(",")]

    print(f"Epoch {epoch} | eps1={eps1} eps2={eps2} | "
          f"n_ladder={n_ladder} | thetas={thetas}")

    os.makedirs("./output2_combined", exist_ok=True)

    # ── AIS hyper-parameters ────────────────────────────────────────────────
    n_paths          = 300
    n_levels         = 20
    moves_per_level  = 5
    # Annealing ladder from 10**-0.9 up to 1.0 with `n_ladder` rungs. Fewer rungs
    # = coarser bridging, so terminal states stay closer to the seed p0. The old
    # fixed ladder was 10 rungs: list(np.logspace(-0.9, 0.0, 10)).
    ladder           = list(np.logspace(-0.9, 0.0, n_ladder))

    num_samples_per_feature = 500
    lamb = 1

    # ── Problem structure ───────────────────────────────────────────────────
    M = 3
    R = np.array([4, 3, 3])

    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies  = hasse.enumerate_policies(M, R)
    num_policies  = len(all_policies)

    g = num_policies * num_samples_per_feature  # g = n (unit information prior)

    # Profile (0, 0, 0)
    sigma_000 = None
    mu_000    = np.array([0])
    var_000   = np.array([1])

    # Profile (0, 0, 1)
    sigma_001 = np.array([[1]])
    mu_001    = np.array([-2])
    var_001   = np.array([1])

    # Profile (0, 1, 0)
    sigma_010 = np.array([[1]])
    mu_010    = np.array([-1.5])
    var_010   = np.array([1])

    # Profile (0, 1, 1)
    sigma_011 = np.array([[1], [0]])
    mu_011    = np.array([-1, 2])
    var_011   = np.array([1, 1])

    # Profile (1, 0, 0)
    sigma_100 = np.array([[0, 1]])
    mu_100    = np.array([-1.5, 1])
    var_100   = np.array([1, 1])

    # Profile (1, 0, 1)
    sigma_101 = np.array([[0, 1], [0, np.inf]])
    mu_101    = np.array([-0.5, 2.5, 1.5, -2.5])
    var_101   = np.array([1, 1, 1, 1])

    # Profile (1, 1, 0)
    sigma_110 = np.array([[0, 1], [1, np.inf]])
    mu_110    = np.array([0, -2.5])
    var_110   = np.array([1, 1])

    # Profile (1, 1, 1)
    sigma_111 = np.array([[0, 1], [1, np.inf], [0, np.inf]])
    mu_111    = np.array([3, -0.5, -1.5, -2])
    var_111   = np.array([1, 1, 1, 1])

    sigma = [sigma_000, sigma_001, sigma_010, sigma_011,
             sigma_100, sigma_101, sigma_110, sigma_111]
    mu    = [mu_000,    mu_001,    mu_010,    mu_011,
             mu_100,    mu_101,    mu_110,    mu_111]
    var   = [var_000,   var_001,   var_010,   var_011,
             var_100,   var_101,   var_110,   var_111]

    policies_profiles        = {}
    policies_profiles_masked = {}
    policies_ids_profiles    = {}
    pi_policies              = {}
    pi_pools                 = {}

    for k, profile in enumerate(profiles):
        policies_temp  = [(i, x) for i, x in enumerate(all_policies)
                          if hasse.policy_to_profile(x) == profile]
        unzipped_temp  = list(zip(*policies_temp))
        policies_ids_k = list(unzipped_temp[0])
        policies_k     = list(unzipped_temp[1])
        policies_profiles[k]     = deepcopy(policies_k)
        policies_ids_profiles[k] = policies_ids_k

        profile_mask = list(map(bool, profile))
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        policies_profiles_masked[k] = policies_k

        if np.sum(profile) > 0:
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            pi_policies[k] = pi_policies_k
            pi_pools[k]    = {}
            for x, y_val in pi_pools_k.items():
                y_full         = [policies_profiles[k][i] for i in y_val]
                y_agg          = [all_policies.index(i) for i in y_full]
                pi_pools[k][x] = y_agg
        else:
            pi_policies[k] = {0: 0}
            pi_pools[k]    = {0: [0]}

    def generate_data(mu, var, n_per_pol, all_policies, pi_policies, M):
        num_data = num_policies * n_per_pol
        X = np.zeros(shape=(num_data, M))
        D = np.zeros(shape=(num_data, 1), dtype="int_")
        y = np.zeros(shape=(num_data, 1))
        idx_ctr = 0
        for k, profile in enumerate(profiles):
            for idx, policy in enumerate(policies_profiles[k]):
                policy_idx  = [i for i, x in enumerate(all_policies) if x == policy]
                pool_id     = pi_policies[k][idx]
                mu_i, var_i = mu[k][pool_id], var[k][pool_id]
                y_i         = np.random.normal(mu_i, var_i, size=(n_per_pol, 1))
                start_idx, end_idx = idx_ctr * n_per_pol, (idx_ctr + 1) * n_per_pol
                X[start_idx:end_idx] = policy
                D[start_idx:end_idx] = policy_idx[0]
                y[start_idx:end_idx] = y_i
                idx_ctr += 1
        return X, D, y

    # ── Simulation loop ─────────────────────────────────────────────────────
    overall_result = []

    for iter in range(5 * (epoch - 1) + 1, 5 * epoch + 1):
        np.random.seed(iter * 42)
        X, D, y      = generate_data(mu, var, num_samples_per_feature,
                                      all_policies, pi_policies, M)
        policy_means = loss.compute_policy_means(D, y, num_policies)
        prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(
            all_policies, hasse.policy_to_profile)

        result  = dict()
        sigma2  = AIS.compute_sigma2_saturated(D, y, all_policies)

        score_s = AIS.make_score_s_gprior(
            D=D, y=y, M=M, R=R,
            prof_idx_of_policy=prof_idx_of_policy,
            policies=all_policies, policy_means=policy_means,
            g=g, sigma2=sigma2, lam=lamb,
        )
        normal_kw = dict(
            D=D, y=y, M=M,
            policies=all_policies,
            prof_idx_of_policy=prof_idx_of_policy,
            R=R, g=g, sigma2=sigma2,
            lattice_edges=None,
            p=[0.025, 0.5, 0.975],
        )

        result["D"]        = D
        result["y"]        = y
        result["sigma2"]   = sigma2
        result["thetas"]   = thetas
        result["n_ladder"] = n_ladder
        result["ladder"]   = ladder

        cfg = AIS.AISConfig(
            n_paths=n_paths, n_levels=n_levels,
            moves_per_level=moves_per_level, min_len=1, seed=None,
        )

        # ── Per-theta: RPS / AIS(RPS) / AIS(random) / PAC-Bayes ─────────────
        for theta in thetas:
            H = np.inf
            R_set, R_profiles = aggregate.RAggregate(
                M, R, H, D, y, theta, reg=lamb, verbose=False)
            anchors    = AIS.build_anchor_states(R_set, R_profiles, M, R)
            if len(anchors) == 0:
                print(f"  theta={theta}: empty RPS, skipping")
                continue
            log_alpha  = [math.log(max(1e-300, score_s(A))) for A in anchors]
            RPS_states = AIS.raggregate_to_states((R_set, R_profiles), profiles)

            # -- RPS (reference set / seed-count source) --
            RPS_post_mean = AIS.estimate_policy_means_from_RPS(
                RPS_states, log_alpha, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"RPS_{theta:g}_states"]    = RPS_states
            result[f"RPS_{theta:g}_logw"]      = log_alpha
            result[f"RPS_{theta:g}"]           = RPS_post_mean
            result[f"RPS_{theta:g}_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
                RPS_states, log_alpha, **normal_kw)
            result[f"RPS_{theta:g}_n_states"]  = len(RPS_states)

            # -- AIS seeded from RPS --
            start   = time.time()
            ais_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=RPS_states, init_log_alpha=log_alpha,
                R=R, eps1=eps1, eps2=eps2, score_s=score_s, cfg=cfg,
                out_dir="./output2_combined",
                label=f"RPS_{theta:g}",
            )
            AIS_post_mean = AIS.estimate_policy_means_from_ais(
                ais_out=ais_out, all_policies=all_policies,
                policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None, R_per=R, M=M,
            )
            result[f"AIS_{theta:g}_output"]    = ais_out
            result[f"AIS_{theta:g}"]           = AIS_post_mean
            result[f"AIS_{theta:g}_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(
                ais_out, **normal_kw)
            result[f"AIS_{theta:g}_time"]      = time.time() - start

            # -- AIS seeded from RANDOM states (count matched to the RPS) --
            n_seeds   = len(RPS_states)
            base_seed = iter * 100003 + int(round(theta * 1000)) * 97
            rand_states, rand_log_scores = _sample_random_seed_states(
                n_seeds, M, R, profiles, score_s, base_seed)
            print(f"  theta={theta:g}: RPS={n_seeds} states, "
                  f"random seeds drawn={len(rand_states)}")

            start        = time.time()
            ais_rand_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=rand_states, init_log_alpha=rand_log_scores,
                R=R, eps1=eps1, eps2=eps2, score_s=score_s, cfg=cfg,
                out_dir="./output2_combined",
                label=f"RAND_{theta:g}",
            )
            AIS_rand_post_mean = AIS.estimate_policy_means_from_ais(
                ais_out=ais_rand_out, all_policies=all_policies,
                policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
                lattice_edges=None, R_per=R, M=M,
            )
            result[f"AIS_rand_{theta:g}_output"]    = ais_rand_out
            result[f"AIS_rand_{theta:g}"]           = AIS_rand_post_mean
            result[f"AIS_rand_{theta:g}_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(
                ais_rand_out, **normal_kw)
            result[f"AIS_rand_{theta:g}_time"]      = time.time() - start
            result[f"AIS_rand_{theta:g}_n_seeds"]   = len(rand_states)

        overall_result.append(result)

    out_path = f"./output2_combined/sim2_combined_result{epoch}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(overall_result, f)
    print(f"Saved epoch {epoch} → {out_path}")


if __name__ == "__main__":
    main()
