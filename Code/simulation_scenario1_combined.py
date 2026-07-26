import time
import math
import pickle
import argparse
import os
import numpy as np

from copy import deepcopy
from rashomon import hasse, extract_pools, loss, aggregate, AIS, MCMC


# ---------------------------------------------------------------------------
# Scenario 1 — ALL methods in a SINGLE pkl per epoch:
#
#     MCMC               : gold-standard posterior sample (the reference)
#     RPS                : Rashomon partition set (reference set)
#     AIS   (RPS seed)   : AIS annealed from the RPS states
#     AIS   (random seed): AIS annealed from uniformly-random states (count-matched)
#     PB                 : PAC-Bayes explorer seeded from the RPS
#
# EVERYTHING runs in LOG space. The raw g-prior score is ~exp(-9000) and
# underflows to a constant, which collapses all weights to uniform and turns
# MCMC into a random walk. We therefore use log_score_s (= -A*SSE - B*|Pi|, no
# exp/clamp) for MCMC, the RPS weights, both AIS variants, and PB.
#
# The RPS is built with the g-prior loss: RAggregate is run with reg = lamb_tilde
# (= reg*), so its MSE loss is proportional to the g-prior score, and thresholds
# are on the g-prior theta_RA scale (~1.0x). Results are keyed:
#
#   MCMC_*   RPS_{theta}_*   AIS_{theta}_*   AIS_rand_{theta}_*   PB_{theta}_*
# ---------------------------------------------------------------------------

def _sample_random_seed_states(n_seeds, M, R, profiles, log_score_s, base_seed):
    """Draw `n_seeds` distinct uniformly-random states and score them.

    Returns (states, log_scores). log_scores are LOG g-prior scores (no
    exp/clamp) -- the g-prior score itself underflows. Deduplicates by state
    signature so the p0 buckets are not skewed by repeated draws.
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
        log_scores.append(log_score_s(st))
    return states, log_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scenario 1 — MCMC + RPS + AIS(RPS) + AIS(random) + PAC-Bayes, all log-space")
    parser.add_argument("--epoch",        type=int,   required=True)
    parser.add_argument("--eps1",         type=float, default=0.4)
    parser.add_argument("--eps2",         type=float, default=0.7)
    parser.add_argument("--n_ladder",     type=int,   default=4,
                        help="Number of annealing ladder rungs (10**-0.9 .. 1.0). "
                             "Fewer rungs = less annealing = stronger seed effect.")
    parser.add_argument("--pb_steps",     type=int,   default=300,
                        help="PAC-Bayes exploration steps")
    parser.add_argument("--pb_delta",     type=float, default=0.05)
    parser.add_argument("--frontier_cap", type=int,   default=500)
    parser.add_argument("--thetas",       type=str,
                        default="1.02,1.03,1.04,1.05,1.06",
                        help="Comma-separated g-prior theta_RA thresholds for the RPS")
    args = parser.parse_args()

    epoch        = args.epoch
    eps1         = args.eps1
    eps2         = args.eps2
    n_ladder     = args.n_ladder
    pb_steps     = args.pb_steps
    pb_delta     = args.pb_delta
    frontier_cap = args.frontier_cap
    thetas       = [float(t) for t in args.thetas.split(",")]

    print(f"Epoch {epoch} | eps1={eps1} eps2={eps2} | n_ladder={n_ladder} | "
          f"PB steps={pb_steps} | thetas={thetas}")

    os.makedirs("/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined", exist_ok=True)

    # ── AIS / MCMC hyper-parameters ─────────────────────────────────────────
    n_paths          = 300
    n_levels         = 20
    moves_per_level  = 5
    ladder           = list(np.logspace(-0.9, 0.0, n_ladder))
    N_ITER           = 50000    # MCMC steps
    N_BURN           = 20000
    N_THIN           = 10
    n_prior          = 64    # PAC-Bayes prior support size

    num_samples_per_feature = 500
    lamb = 1

    # ── Problem structure ───────────────────────────────────────────────────
    M = 2
    R = np.array([4, 3])

    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies  = hasse.enumerate_policies(M, R)
    num_policies  = len(all_policies)

    g = num_policies * num_samples_per_feature  # g = n (unit information prior)

    # Profile (0, 0)
    sigma_00 = None
    mu_00    = np.array([0])
    var_00   = np.array([1])

    # Profile (0, 1)
    sigma_01 = np.array([[1]])
    mu_01    = np.array([-1])
    var_01   = np.array([1])

    # Profile (1, 0)
    sigma_10 = np.array([[1, 0]])
    mu_10    = np.array([-2, -3])
    var_10   = np.array([1, 1])

    # Profile (1, 1)
    sigma_11 = np.array([[0, 1], [0, np.inf]])
    mu_11    = np.array([2, 3, -1, 1])
    var_11   = np.array([1, 1, 1, 1])

    sigma = [sigma_00, sigma_01, sigma_10, sigma_11]
    mu    = [mu_00,    mu_01,    mu_10,    mu_11]
    var   = [var_00,   var_01,   var_10,   var_11]

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
        n_obs   = int(y.shape[0])

        # LOG-space g-prior score (no exp, no clamp) -- used EVERYWHERE.
        log_score_s = AIS.make_log_score_s_gprior(
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
        result["n_prior"]  = n_prior

        cfg = AIS.AISConfig(
            n_paths=n_paths, n_levels=n_levels,
            moves_per_level=moves_per_level, min_len=1, seed=None,
        )

        # ── MCMC (reference) — LOG space so it does not random-walk ──────────
        start = time.time()
        MCMC.run_mcmc_streaming_rand_start(
            profiles=profiles, M=M, R=R, log_score_s=log_score_s,
            seed=None, steps=N_ITER, burnin=N_BURN, thin=N_THIN, min_len=1,
            out_jsonl=f"/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined/mcmc_samples_epoch{epoch}_iter{iter}.jsonl",
            progress_json=f"/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined/mcmc_progress_epoch{epoch}_iter{iter}.json",
        )
        mcmc_res     = MCMC.load_mcmc_res_from_jsonl(
            f"/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined/mcmc_samples_epoch{epoch}_iter{iter}.jsonl")
        log_weight   = np.log(
            np.repeat(1.0 / len(mcmc_res["samples"]), len(mcmc_res["samples"])))
        MCMC_post_mean = MCMC.policy_means_matrix_from_mcmc(
            mcmc_res["samples"], all_policies, policy_means,
            prof_idx_of_policy, R, M, lattice_edges=None, policy_labels=None,
        )
        result["MCMC_states"]    = mcmc_res["samples"]
        result["MCMC_logw"]      = log_weight
        result["MCMC"]           = np.mean(MCMC_post_mean, axis=0)
        result["MCMC_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
            mcmc_res["samples"], log_weight, **normal_kw)
        result["MCMC_time"]      = time.time() - start

        # ── Per-theta: RPS / AIS(RPS) / AIS(random) / PAC-Bayes ─────────────
        lamb_tilde = 2*sigma2*(1+g)*(lamb+np.log(1+g)/2)/(g*g)  # reg*, using g = n
        for theta in thetas:
            H = np.inf
            R_set, R_profiles = aggregate.RAggregate(
                M, R, H, D, y, theta, reg=lamb_tilde, verbose=False)
            anchors    = AIS.build_anchor_states(R_set, R_profiles, M, R)
            if len(anchors) == 0:
                print(f"  theta={theta}: empty RPS, skipping")
                continue
            log_alpha  = [log_score_s(A) for A in anchors]        # LOG weights (no clamp)
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
                R=R, eps1=eps1, eps2=eps2, log_score_s=log_score_s, cfg=cfg,
                out_dir="/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined",
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
                n_seeds, M, R, profiles, log_score_s, base_seed)
            print(f"  theta={theta:g}: RPS={n_seeds} states, "
                  f"random seeds drawn={len(rand_states)}")

            start        = time.time()
            ais_rand_out = AIS.run_ais_state_streaming_from_custom_states(
                ladder=ladder,
                init_states=rand_states, init_log_alpha=rand_log_scores,
                R=R, eps1=eps1, eps2=eps2, log_score_s=log_score_s, cfg=cfg,
                out_dir="/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined",
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

            # -- PAC-Bayes explorer seeded from RPS (LOG space) --
            start = time.time()
            pb_states, pb_log_scores, pb_trace = AIS.run_pac_bayes_explorer(
                init_states=RPS_states, init_log_scores=log_alpha,
                log_score_s=log_score_s, n_obs=n_obs, n_prior=n_prior,
                n_steps=pb_steps, delta=pb_delta,
                min_len=1, frontier_cap=frontier_cap, seed=iter * 42,
            )
            PB_post_mean = AIS.estimate_policy_means_from_RPS(
                pb_states, pb_log_scores, all_policies, policy_means,
                prof_idx_of_policy, R, M, lattice_edges=None,
            )
            result[f"PB_{theta:g}_states"]         = pb_states
            result[f"PB_{theta:g}_logw"]           = pb_log_scores
            result[f"PB_{theta:g}"]                = PB_post_mean
            result[f"PB_{theta:g}_quantiles"]      = AIS.states_quantiles_normal_for_all_policies(
                pb_states, pb_log_scores, **normal_kw)
            result[f"PB_{theta:g}_time"]           = time.time() - start
            result[f"PB_{theta:g}_n_states"]       = len(pb_states)
            result[f"PB_{theta:g}_state_fraction"] = len(pb_states) / float(n_prior)
            result[f"PB_{theta:g}_bound"]          = pb_trace[-1]["bound"]
            result[f"PB_{theta:g}_risk"]           = pb_trace[-1]["E_Q_risk"]
            result[f"PB_{theta:g}_kl"]             = pb_trace[-1]["kl"]
            result[f"PB_{theta:g}_entropy"]        = pb_trace[-1]["H_Q"]
            result[f"PB_{theta:g}_trace"]          = pb_trace

        overall_result.append(result)

    out_path = f"/mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined/sim1_combined_result{epoch}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(overall_result, f)
    print(f"Saved epoch {epoch} → {out_path}")


if __name__ == "__main__":
    main()
