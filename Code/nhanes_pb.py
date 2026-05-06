"""
NHANES PAC-Bayes explorer.

Reproduces the data pipeline from nhanes_mcmc_comparison.py, then runs PB only.
Output: <out_dir>/PB_nhanes_steps<steps>.pkl
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from copy import deepcopy
from rashomon import AIS, hasse, loss
from rashomon.aggregate import (
    RAggregate_profile,
    find_profile_lower_bound,
)
from rashomon.sets import RashomonSet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",        default="../Dat/NHANES_telomere.csv")
    p.add_argument("--out_dir",         default="./output_nhanes_mcmc")
    p.add_argument("--rps_pkl",         default=None,
                   help="nhanes_pruned_results_outlier.pkl (default: <out_dir>/nhanes_pruned_results_outlier.pkl)")
    p.add_argument("--pb_steps",        type=int,   default=300)
    p.add_argument("--pb_delta",        type=float, default=0.05)
    p.add_argument("--pb_frontier_cap", type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


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
    Z = np.delete(Z, [4877], axis=0)
    X = Z[:, [1, 2, 3, 5]]
    y = Z[:, 0].reshape(-1, 1)
    return X, y, Z


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

    D_remapped = np.zeros((num_data, 1), dtype=np.int64)
    for i in range(num_data):
        pol_i = tuple(int(v) for v in X[i])
        D_remapped[i, 0] = next(j for j, p in enumerate(policy_race) if p == pol_i)

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


def enumerate_rashomon(D_race, y_race, policy_means_race, D_remapped, y,
                       policy_race, M, R, reg, theta, num_data):
    H = np.inf
    race_profiles = np.array([1, 2, 3])
    all_active_profile = tuple([1] * M)
    n_per_race = len(policy_race)

    pm_fresh = {}
    for race in race_profiles:
        pm = loss.compute_policy_means(D_race[race], y_race[race], n_per_race)
        nodata = np.where(pm[:, 1] == 0)[0]
        pm[nodata, 0] = 0.0
        pm_fresh[race] = pm

    eq_lb_profiles = np.zeros(3)
    for k, race in enumerate(race_profiles):
        eq_lb_profiles[k] = find_profile_lower_bound(
            D_race[race], y_race[race], pm_fresh[race]
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
            policies_masked, pm_fresh[race],
            normalize=num_data,
        )
        rk.calculate_loss(
            D_race[race], y_race[race], policies_masked,
            pm_fresh[race], reg, normalize=num_data,
        )
        rk.sort()
        rashomon_profiles.append(rk)
        print(f"  Race {race}: {len(rk)} partitions")

    pm_hom = loss.compute_policy_means(D_remapped, y, n_per_race)
    nodata_h = np.where(pm_hom[:, 1] == 0)[0]
    pm_hom[nodata_h, 0] = 0.0

    rashomon_homogeneous = RAggregate_profile(
        M, R, H, D_remapped, y, theta,
        all_active_profile, reg,
        policies_masked, pm_hom,
        normalize=num_data,
    )
    rashomon_homogeneous.calculate_loss(
        D_remapped, y, policies_masked, pm_hom, reg, normalize=num_data
    )
    print(f"  Homogeneous: {len(rashomon_homogeneous)} partitions")

    return rashomon_profiles, rashomon_homogeneous


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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rps_pkl = "../Results/nhanes_pruned_results_outlier.pkl"
    out_pkl = os.path.join(args.out_dir, f"PB_nhanes_steps{args.pb_steps}.pkl")
    np.random.seed(args.seed)

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

    policies_temp   = [(i, x) for i, x in enumerate(all_policies)
                       if hasse.policy_to_profile(x) == all_active_profile]
    policy_race_idx = [t[0] for t in policies_temp]
    policy_race     = [t[1] for t in policies_temp]

    print("Setting up race-stratified data...")
    (D_race, y_race, policy_means_race, D_remapped,
     D_nhanes, policies_nhanes, prof_idx_nhanes, policy_means_nhanes,
     _) = setup_race_data(X, y, Z, all_policies, policy_race, policy_race_idx, M)

    print(f"Loading RPS pickle from {rps_pkl}...")
    with open(rps_pkl, "rb") as f:
        rps_data = pickle.load(f)
    R_set_2        = rps_data["R_set"]
    model_losses_2 = np.asarray(rps_data["losses"], float)
    print(f"  {len(R_set_2)} RPS states in pickle")

    print("Enumerating Rashomon set per race...")
    rashomon_profiles, rashomon_homogeneous = enumerate_rashomon(
        D_race, y_race, policy_means_race, D_remapped, y,
        policy_race, M, R, reg, theta, num_data,
    )

    RPS_states = build_rps_states(R_set_2, rashomon_profiles, rashomon_homogeneous, M, R)
    log_alpha  = list(-model_losses_2)
    print(f"Built {len(RPS_states)} RPS states")

    sigma2  = AIS.compute_sigma2_saturated(D_nhanes, y, policies_nhanes)
    score_s = AIS.make_score_s_gprior(
        D=D_nhanes, y=y, M=M, R=R,
        prof_idx_of_policy=prof_idx_nhanes,
        policies=policies_nhanes, policy_means=policy_means_nhanes,
        g=float(num_data), sigma2=sigma2, lam=reg,
    )

    print(f"Running PB ({args.pb_steps} steps)...")
    t0 = time.time()
    n_obs   = int(y.shape[0])
    n_prior = float(AIS._total_space_size(RPS_states[0], np.asarray(R, int)) if len(RPS_states) else 1)
    pb_states, pb_log_scores, pb_trace = AIS.run_pac_bayes_explorer(
        init_states=RPS_states,
        init_log_scores=log_alpha,
        score_s=score_s,
        n_obs=n_obs,
        n_prior=n_prior,
        n_steps=args.pb_steps,
        delta=args.pb_delta,
        min_len=1,
        frontier_cap=args.pb_frontier_cap,
        seed=args.seed,
    )
    pb_time = time.time() - t0

    pb_normw = np.exp(np.asarray(pb_log_scores, float) - np.max(pb_log_scores))
    pb_normw /= pb_normw.sum()
    ess = float(1.0 / np.sum(pb_normw ** 2))
    print(f"  {len(pb_states)} states, ESS={ess:.1f}, time={pb_time:.1f}s")

    with open(out_pkl, "wb") as f:
        pickle.dump({
            "pb_states":     pb_states,
            "pb_log_scores": pb_log_scores,
            "pb_trace":      pb_trace,
            "pb_time":       pb_time,
        }, f)
    print(f"Saved -> {out_pkl}")


if __name__ == "__main__":
    main()
