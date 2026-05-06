"""
Charitable donations MCMC ground-truth comparison.

Mirrors real_data_charitable_donations.ipynb cells 0–75.
Runs RAggregate, then streaming MCMC as ground truth, loads AIS/PB if
available, and saves a comparison table to --out_dir.
"""
import argparse
import math
import os
import pickle
import time

import numpy as np
import pandas as pd

from rashomon import AIS, MCMC, hasse, loss
from rashomon.aggregate import RAggregate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dta",    default="../Dat/Does_Price_Matter_AER_merged.dta",
                   help="Path to Does_Price_Matter_AER_merged.dta")
    p.add_argument("--out_dir",     default="./output_charity_mcmc",
                   help="All outputs (and default input lookups) go here")
    p.add_argument("--ais_jsonl",   default=None,
                   help="charity_ais.jsonl (default: <out_dir>/charity_ais.jsonl)")
    p.add_argument("--pb_pkl",      default=None,
                   help="charity_pb.pkl (default: <out_dir>/charity_pb.pkl)")
    p.add_argument("--mcmc_steps",  type=int, default=50000)
    p.add_argument("--mcmc_burnin", type=int, default=20000)
    p.add_argument("--mcmc_thin",   type=int, default=10)
    p.add_argument("--mcmc_chains", type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
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

    # columns after drop: ratio(0) size(1) ask(2) amount(3) gave(4) amountchange(5) red0(6)
    Z = df.to_numpy()
    X = Z[:, [0, 1, 2, 6]]          # ratio, size, ask, red0
    y = (Z[:, 3] / 100).reshape(-1, 1)  # amount / 100
    return X, y


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ais_jsonl = args.ais_jsonl or os.path.join(args.out_dir, "charity_ais.jsonl")
    pb_pkl    = args.pb_pkl    or os.path.join(args.out_dir, "charity_pb.pkl")
    out_pkl   = os.path.join(args.out_dir, "charity_mcmc_comparison.pkl")
    np.random.seed(args.seed)

    # --- Data ---
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

    # --- Assign policy indices and compute policy means ---
    print("Building D and policy means...")
    D = np.zeros((num_data, 1), dtype=np.int64)
    for i in range(num_data):
        pol_i  = tuple(int(v) for v in X[i])
        D[i, 0] = next(j for j, p in enumerate(all_policies) if p == pol_i)

    policy_means = loss.compute_policy_means(D, y, num_policies)
    nodata_idx = np.where(policy_means[:, 1] == 0)[0]
    policy_means[nodata_idx, 0] = -np.inf
    policy_means[nodata_idx, 1] = 1

    # --- RAggregate ---
    print(f"Running RAggregate (q={q}, reg={reg})...")
    t0 = time.time()
    R_set, rashomon_profiles = RAggregate(M, R, np.inf, D, y, q, reg=reg)
    print(f"  {len(R_set)} states in Rashomon set ({time.time()-t0:.1f}s)")

    # --- RPS states + score_s ---
    def score_s(state):
        Q = AIS.global_loss_raw(
            state=state, D=D, y=y, M=M, R=R,
            prof_idx_of_policy=prof_idx_of_policy,
            policies=all_policies, policy_means=policy_means,
            reg=reg, lattice_edges=None,
        )
        return float(np.exp(-Q))

    RPS_states = AIS.raggregate_to_states((R_set, rashomon_profiles), profiles_hasse)
    log_alpha  = [math.log(max(1e-300, score_s(s))) for s in RPS_states]
    print(f"Built {len(RPS_states)} RPS states")

    sigma2 = AIS.compute_sigma2_saturated(D, y, all_policies)
    normal_kw = dict(
        D=D, y=y, M=M,
        policies=all_policies,
        prof_idx_of_policy=prof_idx_of_policy,
        R=R, g=float(num_data), sigma2=sigma2,
        lattice_edges=None,
        p=[0.025, 0.5, 0.975],
    )

    result = {}

    # --- RPS ---
    print("Computing RPS estimates...")
    rps_logw  = np.asarray(log_alpha, float)
    rps_normw = np.exp(rps_logw - np.max(rps_logw))
    rps_normw /= rps_normw.sum()
    result["RPS_mean"] = AIS.estimate_policy_means_from_RPS(
        RPS_states, rps_logw, all_policies, policy_means,
        prof_idx_of_policy, R, M, lattice_edges=None,
    )
    result["RPS_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
        RPS_states, rps_logw, **normal_kw
    )
    result["RPS_n_states"] = len(RPS_states)
    result["RPS_ESS"]      = float(1.0 / np.sum(rps_normw ** 2))

    # --- MCMC (streaming — each chain writes a jsonl to out_dir) ---
    print(f"Running MCMC ({args.mcmc_chains} chains × {args.mcmc_steps} steps, "
          f"burnin={args.mcmc_burnin}, thin={args.mcmc_thin})...")
    t0 = time.time()
    all_samples = []
    for c in range(args.mcmc_chains):
        chain_jsonl    = os.path.join(args.out_dir, f"mcmc_chain_{c}.jsonl")
        chain_progress = os.path.join(args.out_dir, f"mcmc_progress_{c}.json")
        print(f"  Chain {c+1}/{args.mcmc_chains} -> {chain_jsonl}")
        MCMC.run_mcmc_streaming(
            RPS=RPS_states,
            log_alpha=log_alpha,
            score_s=score_s,
            steps=args.mcmc_steps,
            burnin=args.mcmc_burnin,
            thin=args.mcmc_thin,
            min_len=1,
            seed=args.seed + c,
            out_jsonl=chain_jsonl,
            progress_json=chain_progress,
        )
        chain_res = MCMC.load_mcmc_res_from_jsonl(chain_jsonl)
        all_samples.extend(chain_res["samples"])
        print(f"    {len(chain_res['samples'])} samples kept")
    result["MCMC_time"] = time.time() - t0
    n_mcmc = len(all_samples)
    print(f"  Done in {result['MCMC_time']:.1f}s  |  {n_mcmc} total samples")

    mcmc_logw = np.log(np.full(n_mcmc, 1.0 / n_mcmc))
    result["MCMC_mean"] = AIS.estimate_policy_means_from_RPS(
        all_samples, mcmc_logw, all_policies, policy_means,
        prof_idx_of_policy, R, M, lattice_edges=None,
    )
    result["MCMC_quantiles"] = AIS.states_quantiles_normal_for_all_policies(
        all_samples, mcmc_logw, **normal_kw
    )
    result["MCMC_n_samples"] = n_mcmc

    # --- AIS ---
    if os.path.exists(ais_jsonl):
        print(f"Loading AIS from {ais_jsonl}...")
        _r = AIS.load_ais_from_jsonl(ais_jsonl)
        ais_logw  = np.asarray(_r["logw"], float)
        ais_normw = np.exp(ais_logw - np.max(ais_logw))
        ais_normw /= ais_normw.sum()
        ais_out = AIS.AISOutput(
            terminals=_r["terminals"], logw=ais_logw, normw=ais_normw, ladder=None
        )
        result["AIS_mean"] = AIS.estimate_policy_means_from_ais(
            ais_out=ais_out, all_policies=all_policies,
            policy_means=policy_means,
            prof_idx_of_policy=prof_idx_of_policy,
            lattice_edges=None, R_per=R, M=M,
        )
        result["AIS_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(
            ais_out, **normal_kw
        )
        result["AIS_n_states"] = len(ais_out.terminals)
        result["AIS_ESS"]      = float(1.0 / np.sum(ais_normw ** 2))
        print(f"  AIS: {len(ais_out.terminals)} terminals, ESS={result['AIS_ESS']:.1f}")
    else:
        print(f"  WARNING: {ais_jsonl} not found, skipping AIS.")

    # --- PB ---
    if os.path.exists(pb_pkl):
        print(f"Loading PB from {pb_pkl}...")
        with open(pb_pkl, "rb") as f:
            pb_data = pickle.load(f)
        pb_states = pb_data["pb_states"]
        pb_logw   = np.asarray(pb_data["pb_log_scores"], float)
        pb_normw  = np.exp(pb_logw - np.max(pb_logw))
        pb_normw /= pb_normw.sum()
        pb_out = AIS.AISOutput(
            terminals=pb_states, logw=pb_logw, normw=pb_normw, ladder=None
        )
        result["PB_mean"] = AIS.estimate_policy_means_from_RPS(
            pb_states, pb_logw, all_policies, policy_means,
            prof_idx_of_policy, R, M, lattice_edges=None,
        )
        result["PB_quantiles"] = AIS.ais_quantiles_normal_for_all_policies(
            pb_out, **normal_kw
        )
        result["PB_n_states"] = len(pb_states)
        result["PB_ESS"]      = float(1.0 / np.sum(pb_normw ** 2))
        result["PB_time"]     = pb_data.get("pb_time", float("nan"))
        print(f"  PB: {len(pb_states)} states, ESS={result['PB_ESS']:.1f}")
    else:
        print(f"  WARNING: {pb_pkl} not found, skipping PB.")

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
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved -> {out_pkl}")


if __name__ == "__main__":
    main()
