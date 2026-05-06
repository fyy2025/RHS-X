"""
NHANES MCMC ground-truth comparison.

Reproduces the data pipeline from NHANES_data_analysis_ais.ipynb, runs MCMC
as ground truth, then computes L1 error and IoU for RPS, AIS, and PB.

Outputs: ./nhanes_mcmc_comparison.pkl
"""
import argparse
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

from copy import deepcopy
from rashomon import AIS, MCMC, hasse, loss
from rashomon.aggregate import (
    RAggregate_profile,
    find_profile_lower_bound,
    find_feasible_combinations,
    remove_unused_poolings,
)
from rashomon.sets import RashomonSet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",    default="../Data/NHANES_telomere.csv")
    p.add_argument("--rps_pkl",     default="../Results/nhanes_pruned_results_outlier.pkl")
    p.add_argument("--ais_jsonl",   default="./AIS_nhanes_samples300.jsonl")
    p.add_argument("--pb_pkl",      default="./PB_nhanes_steps300.pkl")
    p.add_argument("--out_pkl",     default="./nhanes_mcmc_comparison.pkl")
    p.add_argument("--mcmc_steps",  type=int, default=50000)
    p.add_argument("--mcmc_burnin", type=int, default=20000)
    p.add_argument("--mcmc_thin",   type=int, default=10)
    p.add_argument("--mcmc_chains", type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading  (mirrors notebook cells 4–6)
# ---------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["HoursWorked"]     = df["HoursWorked"].map({"<=20": 1, "21-40": 2, ">=41": 3})
    df["Gender"]          = df["Gender"].map({"Female": 1, "Male": 2})
    df["Age"]             = df["Age"].map({"<=18": 1, "19-30": 2, "31-50": 3, "51-70": 4, ">=70": 5})
    df["Race"]            = df["Race"].map({"White": 1, "Black": 2, "Hispanic": 3, "Other": 3})
    df["Education"]       = df["Education"].map({"< GED": 1, "GED": 2, "College": 3})
    df["MaritalStatus"]   = df["MaritalStatus"].map({"Single": 1, "Married": 2, "Divorced/Widowed": 3})
    df["HouseholdIncome"] = df["HouseholdIncome"].map({"<20k": 1, "20k-45k": 2, "45k-75k": 3, ">=75k": 4})
    Z = df.to_numpy()
    Z = np.delete(Z, [4877], axis=0)   # remove known outlier
    X = Z[:, [1, 2, 3, 5]]            # Gender, Age, Education, Income
    y = Z[:, 0].reshape(-1, 1)        # Telomean
    return X, y, Z


# ---------------------------------------------------------------------------
# Race-stratified data setup  (mirrors notebook cells 8–12)
# ---------------------------------------------------------------------------
def setup_race_data(X, y, Z, all_policies, policy_race, policy_race_idx, M):
    race_profiles = np.array([1, 2, 3])
    n_per_race = len(policy_race)
    num_data = X.shape[0]

    D_race = {}; y_race = {}; policy_means_race = {}
    policy_means_combined = np.zeros((n_per_race, 2))

    for race in race_profiles:
        idx = np.where(Z[:, 4] == race)[0]
        X_k = X[idx]; y_k = y[idx]
        D_k = np.zeros((len(y_k), 1), dtype=np.int64)
        for i in range(len(y_k)):
            pol_i = tuple(int(v) for v in X_k[i])
            D_k[i, 0] = next(j for j, p in enumerate(policy_race) if p == pol_i)
        D_race[race] = D_k
        y_race[race] = y_k
        pm_k = loss.compute_policy_means(D_k, y_k, n_per_race)
        nodata = np.where(pm_k[:, 1] == 0)[0]
        pm_k[nodata, 0] = -np.inf; pm_k[nodata, 1] = 1
        policy_means_race[race] = pm_k
        pm_k_copy = pm_k.copy(); pm_k_copy[nodata, 0] = 0
        policy_means_combined += pm_k_copy
        pm_k[nodata, 0] = -np.inf

    # D_remapped: local policy ids for homogeneous model
    D_remapped = np.zeros((num_data, 1), dtype=np.int64)
    for i in range(num_data):
        pol_i = tuple(int(v) for v in X[i])
        D_remapped[i, 0] = next(j for j, p in enumerate(policy_race) if p == pol_i)

    # D_nhanes: race-stratified global ids (270 = 90 * 3 policies)
    D_nhanes = np.zeros((num_data, 1), dtype=np.int64)
    for i in range(num_data):
        race_i = int(Z[i, 4]) - 1
        D_nhanes[i, 0] = race_i * n_per_race + int(D_remapped[i, 0])

    policies_nhanes = policy_race + policy_race + policy_race
    prof_idx_nhanes = np.array([0]*n_per_race + [1]*n_per_race + [2]*n_per_race)
    policy_means_nhanes = np.concatenate(
        [policy_means_race[r] for r in [1, 2, 3]], axis=0
    )

    return (D_race, y_race, policy_means_race, D_remapped,
            D_nhanes, policies_nhanes, prof_idx_nhanes, policy_means_nhanes,
            policy_means_combined)


# ---------------------------------------------------------------------------
# Rashomon set enumeration  (mirrors notebook cells 13–21)
# Re-run to recover sigma matrices, which are not stored in the pruned pkl.
# ---------------------------------------------------------------------------
def enumerate_rashomon(D_race, y_race, policy_means_race, D_remapped, y,
                       policy_race, M, R, reg, theta, num_data):
    H = np.inf
    race_profiles = np.array([1, 2, 3])
    all_active_profile = tuple([1] * M)
    n_per_race = len(policy_race)

    eq_lb_profiles = np.zeros(3)
    for k, race in enumerate(race_profiles):
        eq_lb_profiles[k] = find_profile_lower_bound(
            D_race[race], y_race[race], policy_means_race[race]
        )
    eq_lb_profiles /= num_data
    eq_lb_sum = float(np.sum(eq_lb_profiles))

    H_profile = H - len(race_profiles) + 1
    policies_masked = [tuple(pol) for pol in policy_race]

    rashomon_profiles = []
    for k, race in enumerate(race_profiles):
        theta_k = theta - (eq_lb_sum - eq_lb_profiles[k])
        rk = RAggregate_profile(
            M, R, H_profile,
            D_race[race], y_race[race], theta_k,
            all_active_profile, reg,
            policies_masked, policy_means_race[race],
            normalize=num_data,
        )
        rk.calculate_loss(
            D_race[race], y_race[race], policies_masked,
            policy_means_race[race], reg, normalize=num_data,
        )
        rk.sort()
        rashomon_profiles.append(rk)
        print(f"  Race {race}: {len(rk)} partitions")

    rashomon_homogeneous = RAggregate_profile(
        M, R, H, D_remapped, y, theta,
        all_active_profile, reg,
        policies_masked, policy_means_race[1],
        normalize=num_data,
    )
    rashomon_homogeneous.calculate_loss(
        D_remapped, y, policies_masked, policy_means_race[1], reg, normalize=num_data
    )
    print(f"  Homogeneous: {len(rashomon_homogeneous)} partitions")

    return rashomon_profiles, rashomon_homogeneous


# ---------------------------------------------------------------------------
# Build RPS states  (mirrors notebook cell 27)
# ---------------------------------------------------------------------------
def build_rps_states(R_set_2, rashomon_profiles, rashomon_homogeneous, M, R):
    all_active_profile = tuple([1] * M)
    RPS_states = []
    for r_set in R_set_2:
        state = []
        for race_idx in range(3):
            if r_set[0] == -1:
                sigma = rashomon_homogeneous.sigma[r_set[1]]
            else:
                sigma = rashomon_profiles[race_idx].sigma[r_set[race_idx]]
            B = AIS._as_compact_B(sigma)
            state.append(AIS.ProfilePart(cov_ids=all_active_profile, B=B))
        RPS_states.append(state)
    return RPS_states


# ---------------------------------------------------------------------------
# score_s  (mirrors notebook cell 28)
# ---------------------------------------------------------------------------
def make_score_s(D_race, y_race, policy_race, M, R, num_data, reg):
    n_per_race = len(policy_race)

    def score_s_nhanes(state):
        total_loss = 0.0
        for race_idx, race in enumerate([1, 2, 3]):
            pp = state[race_idx]
            sigma_full = AIS.assemble_sigma_full_for_profile(pp, M, R)
            D_k = D_race[race]
            y_k = y_race[race]
            pm_k = loss.compute_policy_means(D_k, y_k, n_per_race)
            if sigma_full is None:
                y_flat = np.asarray(y_k).ravel()
                mu_k = float(np.mean(y_flat)) if y_flat.size else 0.0
                mse_k = (float(np.mean((y_flat - mu_k) ** 2))
                         * y_flat.size / num_data if y_flat.size else 0.0)
                Q_k = mse_k + reg
            else:
                Q_k = float(loss.compute_Q(
                    D_k, y_k, sigma_full, policy_race, pm_k,
                    reg=reg, normalize=num_data,
                ))
            total_loss += Q_k
        return float(np.exp(-total_loss))

    return score_s_nhanes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # --- Data ---
    print("Loading data...")
    X, y, Z = load_data(args.data_csv)
    num_data = X.shape[0]
    print(f"  {num_data} observations")

    M = 4
    R = np.array([4, 3, 6, 4])
    all_active_profile = tuple([1] * M)
    reg = 1e-5
    best_loss_known = 0.06423696912419681
    theta = best_loss_known * (1 + 0.004)

    profiles, _ = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)

    # Policies for the all-active profile (race-level policies)
    policies_temp = [(i, x) for i, x in enumerate(all_policies)
                     if hasse.policy_to_profile(x) == all_active_profile]
    policy_race_idx = [t[0] for t in policies_temp]
    policy_race     = [t[1] for t in policies_temp]

    # --- Race-stratified setup ---
    print("Setting up race-stratified data...")
    (D_race, y_race, policy_means_race, D_remapped,
     D_nhanes, policies_nhanes, prof_idx_nhanes, policy_means_nhanes,
     _) = setup_race_data(X, y, Z, all_policies, policy_race, policy_race_idx, M)

    # --- Load pruned RPS pickle ---
    print(f"Loading RPS pickle from {args.rps_pkl}...")
    with open(args.rps_pkl, "rb") as f:
        rps_data = pickle.load(f)
    R_set_2 = rps_data["R_set"]
    model_losses_2 = np.asarray(rps_data["losses"], float)
    print(f"  {len(R_set_2)} RPS states in pickle")

    # --- Re-enumerate Rashomon set to recover sigma matrices ---
    print("Enumerating Rashomon set per race (needed for sigma matrices)...")
    rashomon_profiles, rashomon_homogeneous = enumerate_rashomon(
        D_race, y_race, policy_means_race, D_remapped, y,
        policy_race, M, R, reg, theta, num_data,
    )

    # --- Build RPS states and score_s ---
    RPS_states = build_rps_states(R_set_2, rashomon_profiles, rashomon_homogeneous, M, R)
    log_alpha  = list(-model_losses_2)
    print(f"Built {len(RPS_states)} RPS states")

    score_s = make_score_s(D_race, y_race, policy_race, M, R, num_data, reg)

    # Shared quantile kwargs
    sigma2 = AIS.compute_sigma2_saturated(D_nhanes, y, policies_nhanes)
    normal_kw = dict(
        D=D_nhanes, y=y, M=M,
        policies=policies_nhanes,
        prof_idx_of_policy=prof_idx_nhanes,
        R=R, g=float(num_data), sigma2=sigma2,
        lattice_edges=None,
        p=[0.025, 0.5, 0.975],
    )

    result = {}

    # --- RPS ---
    print("Computing RPS estimates...")
    rps_logw = np.asarray(log_alpha, float)
    rps_normw = np.exp(rps_logw - np.max(rps_logw))
    rps_normw /= rps_normw.sum()
    result["RPS_mean"] = AIS.estimate_policy_means_from_RPS(
        RPS_states, rps_logw, policies_nhanes, policy_means_nhanes,
        prof_idx_nhanes, R, M, lattice_edges=None,
    )
    result["RPS_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
        RPS_states, rps_logw, **normal_kw
    )
    result["RPS_n_states"] = len(RPS_states)
    result["RPS_ESS"] = float(1.0 / np.sum(rps_normw ** 2))

    # --- MCMC ---
    print(f"Running MCMC ({args.mcmc_chains} chains × {args.mcmc_steps} steps, "
          f"burnin={args.mcmc_burnin}, thin={args.mcmc_thin})...")
    t0 = time.time()
    mcmc_res = MCMC.run_mcmc(
        score_s, RPS_states, log_alpha,
        n_chains=args.mcmc_chains,
        steps=args.mcmc_steps,
        burnin=args.mcmc_burnin,
        thin=args.mcmc_thin,
        seed=args.seed,
        min_len=1,
    )
    result["MCMC_time"] = time.time() - t0
    n_mcmc = len(mcmc_res["samples"])
    print(f"  Done in {result['MCMC_time']:.1f}s  |  {n_mcmc} samples")

    mcmc_logw = np.log(np.full(n_mcmc, 1.0 / n_mcmc))
    result["MCMC_mean"] = AIS.estimate_policy_means_from_RPS(
        mcmc_res["samples"], mcmc_logw, policies_nhanes, policy_means_nhanes,
        prof_idx_nhanes, R, M, lattice_edges=None,
    )
    result["MCMC_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
        mcmc_res["samples"], mcmc_logw, **normal_kw
    )
    result["MCMC_n_samples"] = n_mcmc

    # --- AIS ---
    if os.path.exists(args.ais_jsonl):
        print(f"Loading AIS from {args.ais_jsonl}...")
        _r = AIS.load_ais_from_jsonl(args.ais_jsonl)
        ais_logw = np.asarray(_r["logw"], float)
        ais_normw = np.exp(ais_logw - np.max(ais_logw))
        ais_normw /= ais_normw.sum()
        ais_out = AIS.AISOutput(
            terminals=_r["terminals"], logw=ais_logw, normw=ais_normw, ladder=None
        )
        result["AIS_mean"] = AIS.estimate_policy_means_from_ais(
            ais_out=ais_out, all_policies=policies_nhanes,
            policy_means=policy_means_nhanes,
            prof_idx_of_policy=prof_idx_nhanes,
            lattice_edges=None, R_per=R, M=M,
        )
        result["AIS_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(
            ais_out, **normal_kw
        )
        result["AIS_n_states"] = len(ais_out.terminals)
        result["AIS_ESS"] = float(1.0 / np.sum(ais_normw ** 2))
        print(f"  AIS: {len(ais_out.terminals)} terminals, ESS={result['AIS_ESS']:.1f}")
    else:
        print(f"  WARNING: {args.ais_jsonl} not found, skipping AIS.")

    # --- PB ---
    if os.path.exists(args.pb_pkl):
        print(f"Loading PB from {args.pb_pkl}...")
        with open(args.pb_pkl, "rb") as f:
            pb_data = pickle.load(f)
        pb_states  = pb_data["pb_states"]
        pb_logw    = np.asarray(pb_data["pb_log_scores"], float)
        pb_normw   = np.exp(pb_logw - np.max(pb_logw))
        pb_normw  /= pb_normw.sum()
        pb_out = AIS.AISOutput(terminals=pb_states, logw=pb_logw, normw=pb_normw, ladder=None)
        result["PB_mean"] = AIS.estimate_policy_means_from_RPS(
            pb_states, pb_logw, policies_nhanes, policy_means_nhanes,
            prof_idx_nhanes, R, M, lattice_edges=None,
        )
        result["PB_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(pb_out, **normal_kw)
        result["PB_n_states"] = len(pb_states)
        result["PB_ESS"] = float(1.0 / np.sum(pb_normw ** 2))
        result["PB_time"] = pb_data.get("pb_time", float("nan"))
        print(f"  PB: {len(pb_states)} states, ESS={result['PB_ESS']:.1f}")
    else:
        print(f"  WARNING: {args.pb_pkl} not found, skipping PB.")

    # --- Summary table ---
    mc_lo = result["MCMC_quantiles"]["0.025"]
    mc_hi = result["MCMC_quantiles"]["0.975"]

    def l1(a, b):
        return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))

    def ci_iou(lo_m, hi_m, lo_ref, hi_ref):
        inter = np.maximum(0, np.minimum(hi_m, hi_ref) - np.maximum(lo_m, lo_ref))
        ref   = hi_ref - lo_ref
        return float(np.nanmean(np.where(ref > 0, inter / ref, np.nan)))

    rows = []
    for method in ["RPS", "AIS", "PB"]:
        if f"{method}_mean" not in result:
            continue
        qs = result[f"{method}_quantiles"]
        rows.append(dict(
            method=method,
            n_states=result[f"{method}_n_states"],
            ESS=result[f"{method}_ESS"],
            ESS_ratio=result[f"{method}_ESS"] / result[f"{method}_n_states"],
            L1_mean=l1(result[f"{method}_mean"], result["MCMC_mean"]),
            L1_q025=l1(qs["0.025"], mc_lo),
            L1_q975=l1(qs["0.975"], mc_hi),
            IoU=ci_iou(qs["0.025"], qs["0.975"], mc_lo, mc_hi),
            runtime_s=result.get(f"{method}_time", float("nan")),
        ))
    rows.append(dict(
        method="MCMC", n_states=n_mcmc, ESS=float(n_mcmc), ESS_ratio=1.0,
        L1_mean=0.0, L1_q025=0.0, L1_q975=0.0, IoU=1.0,
        runtime_s=result["MCMC_time"],
    ))

    summary = pd.DataFrame(rows).set_index("method").round(4)
    print("\n--- Comparison table ---")
    print(summary.to_string())

    result["summary"] = summary
    with open(args.out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved -> {args.out_pkl}")


if __name__ == "__main__":
    main()
