"""
Charitable donations PAC-Bayes explorer.

Sets up data, runs RAggregate, then runs PB only.
Output: <out_dir>/charity_pb_<steps>.pkl
"""
import argparse
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

from rashomon import AIS, hasse, loss
from rashomon.aggregate import RAggregate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dta",        default="../Dat/Does_Price_Matter_AER_merged.dta")
    p.add_argument("--out_dir",         default="./output_charity_mcmc")
    p.add_argument("--pb_steps",        type=int,   default=300)
    p.add_argument("--pb_delta",        type=float, default=0.05)
    p.add_argument("--pb_frontier_cap", type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def load_data(dta_path):
    raw_df = pd.read_stata(dta_path)
    cols_to_keep = [
        "treatment", "control",
        "ratio", "size", "ask",
        "amount", "gave", "amountchange",
        "red0",
    ]
    df = raw_df[cols_to_keep].copy().dropna()

    ratio_map   = {"Control": 0, 1: 1, 2: 2, 3: 3}
    size_map    = {"Control": 0, "$25,000": 1, "$50,000": 2, "$100,000": 3, "Unstated": 4}
    ask_map     = {"Control": 0, "1x": 1, "1.25x": 2, "1.50x": 3}
    redblue_map = {0: 1, 1: 2, np.nan: 0}

    df["ratio"] = df["ratio"].map(ratio_map)
    df["size"]  = df["size"].map(size_map)
    df["ask"]   = df["ask"].map(ask_map)
    df["red0"]  = df["red0"].map(redblue_map)
    df = df.astype({"ratio": np.int64, "size": np.int64, "ask": np.int64, "red0": np.int64})
    df = df.drop(["treatment", "control"], axis=1)

    Z = df.to_numpy()
    X = Z[:, [0, 1, 2, 6]]
    y = (Z[:, 3] / 100).reshape(-1, 1)
    return X, y


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_pkl = os.path.join(args.out_dir, f"charity_pb_{args.pb_steps}.pkl")
    np.random.seed(args.seed)

    print("Loading data...")
    X, y = load_data(args.data_dta)
    num_data = X.shape[0]
    print(f"  {num_data} observations")

    M   = 4
    R   = np.array([4, 5, 4, 3])
    reg = 1e-7
    q   = 0.0075579

    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)
    prof_idx_of_policy, profiles_hasse = AIS.build_profile_index_of_policy(
        all_policies, hasse.policy_to_profile
    )

    print("Building D and policy means...")
    D = np.zeros((num_data, 1), dtype=np.int64)
    for i in range(num_data):
        pol_i   = tuple(int(v) for v in X[i])
        D[i, 0] = next(j for j, p in enumerate(all_policies) if p == pol_i)

    policy_means = loss.compute_policy_means(D, y, num_policies)
    nodata_idx = np.where(policy_means[:, 1] == 0)[0]
    policy_means[nodata_idx, 0] = -np.inf
    policy_means[nodata_idx, 1] = 1

    print(f"Running RAggregate (q={q}, reg={reg})...")
    t0 = time.time()
    R_set, rashomon_profiles = RAggregate(M, R, np.inf, D, y, q, reg=reg)
    print(f"  {len(R_set)} states ({time.time()-t0:.1f}s)")

    RPS_states = AIS.raggregate_to_states((R_set, rashomon_profiles), profiles_hasse)

    sigma2  = AIS.compute_sigma2_saturated(D, y, all_policies)
    score_s = AIS.make_score_s_gprior(
        D=D, y=y, M=M, R=R,
        prof_idx_of_policy=prof_idx_of_policy,
        policies=all_policies, policy_means=policy_means,
        g=float(num_data), sigma2=sigma2, lam=reg,
    )
    log_alpha = [math.log(max(1e-300, score_s(s))) for s in RPS_states]
    print(f"Built {len(RPS_states)} RPS states")

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
