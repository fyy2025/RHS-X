from rashomon import AIS, extract_pools, loss
from rashomon.AIS import State
import math, random, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple
import os
import pickle
from pathlib import Path
import gzip
import lzma
import tempfile


# --- small helpers (match your existing conventions) ---
def init_from_RPS(RPS: List, log_alpha: List[float], rng: np.random.Generator) -> List:
    p = AIS._softmax_logalpha(log_alpha)
    return AIS._copy_state(RPS[rng.choice(len(RPS), p=p)])

def init_from_RPS_batch(RPS, log_alpha, batch_N, rng_seed=None):
    if rng_seed is not None:
        np.random.seed(rng_seed); random.seed(rng_seed)
    # softmax over log_alpha
    z = np.asarray(log_alpha, float); m = float(z.max()); p = np.exp(z-m); p /= p.sum()
    idx = np.random.choice(len(RPS), size=batch_N, p=p)
    return [AIS._copy_state(RPS[i]) for i in idx]

# --- one MH step targeting the true posterior p(x) ∝ score_s(x) ---
def mh_step_true_posterior(x, score_s, min_len=1):
    N_cur = [n for n in AIS.state_neighbors_ubs(x, min_len=min_len) if not AIS.states_equal(n, x)]
    if not N_cur: 
        return x, 0
    prop = random.choice(N_cur)
    N_prop = [n for n in AIS.state_neighbors_ubs(prop, min_len=min_len) if not AIS.states_equal(n, prop)]
    lcur = math.log(max(1e-300, score_s(x)))
    lprop = math.log(max(1e-300, score_s(prop)))
    # uniform-neighbor proposal → include degree correction
    logr = (lprop - lcur) + math.log(max(1, len(N_cur))) - math.log(max(1, len(N_prop)))
    if math.log(random.random() + 1e-300) < min(0.0, logr):
        return prop, 1
    return x, 0

# --- single chain ---
def run_mcmc_chain(score_s,
                   RPS: List, log_alpha: List[float],
                   steps=20000, burnin=2000, thin=10,
                   seed=0, min_len=1):
    np.random.seed(seed); random.seed(seed)
    rng = np.random.default_rng(seed)

    x = init_from_RPS(RPS, log_alpha, rng)  # start in RPS (loss-weighted)
    accepts = 0
    samples = []

    for t in range(steps):
        x, acc = mh_step_true_posterior(x, score_s, min_len=min_len)
        accepts += acc
        if t >= burnin and ((t - burnin) % thin == 0):
            samples.append(AIS._copy_state(x))

    acc_rate = accepts / max(1, steps)
    return samples, acc_rate

# --- multiple chains + aggregation to a posterior over states ---
def run_mcmc(score_s, RPS, log_alpha,
             n_chains=8, steps=20000, burnin=2000, thin=10,
             seed=0, min_len=1):
    all_samples = []
    acc_rates = []
    for c in range(n_chains):
        samples, ar = run_mcmc_chain(score_s, RPS, log_alpha,
                                     steps=steps, burnin=burnin, thin=thin,
                                     seed=seed + 31*c, min_len=min_len)
        all_samples.extend(samples)
        acc_rates.append(ar)
    # empirical posterior over unique states
    counts = defaultdict(int)
    for s in all_samples:
        counts[AIS.state_signature(s)] += 1
    total = sum(counts.values())
    post = {sig: cnt/total for sig, cnt in counts.items()}
    return {"samples": all_samples,
            "acc_rate_mean": float(np.mean(acc_rates)),
            "acc_rate_sd": float(np.std(acc_rates)),
            "posterior": post}


def estimate_policy_means_from_mcmc(
    mcmc_res,                     # dict with key "samples": List[State]
    policies,                     # global policy list (length P)
    policy_means,                 # np.ndarray [P,2] = [sum_y, count]
    prof_idx_of_policy,           # length-P array: policy_id -> profile k, for 36 policies, which profile is each policy in
    R_per,                        # np.ndarray of arm levels (includes control)
    M,
    lattice_edges=None            # optional lattice; pass None if unused
):
    """
    Posterior-weighted mean outcome per policy using MCMC samples.
    Weights default to uniform over retained samples.
    Returns: np.ndarray [P]
    """
    samples = mcmc_res["samples"]
    if len(samples) == 0:
        return np.zeros(len(policies), float)

    P = len(policies)
    mu_hat = np.zeros(P, dtype=float)

    # uniform weights over MCMC samples (empirical posterior)
    w = np.full(len(samples), 1.0 / len(samples), float)

    # build profile→global index map once
    K = int(np.max(prof_idx_of_policy)) + 1
    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[prof_idx_of_policy[pid]].append(pid)

    # per-profile slices (fixed order defines local indices)
    prof_policies  = [[policies[i] for i in idxs] for idxs in prof_to_global]
    prof_means_arr = [
        policy_means[np.array(idxs, int), :] if idxs else np.zeros((0, 2))
        for idxs in prof_to_global
    ]

    R_per = np.asarray(R_per, int)  # ensure array for assemble

    for x_state, ww in zip(samples, w):
        for k in range(K):
            idxs = prof_to_global[k]
            if not idxs:
                continue

            # expand compact profile matrix to full-width expected by extract_pools
            sigma_full_k = AIS.assemble_sigma_full_for_profile(x_state[k], M, R_per)

            # pools + policy→pool mapping on the *local* list for profile k
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)
            mu_pools_k = loss.compute_pool_means(prof_means_arr[k], pi_pools_k)

            # IMPORTANT: pi_policies_k is keyed by LOCAL INDEX, not policy tuple
            for local_idx, pid in enumerate(idxs):
                pool_id = pi_policies_k[local_idx]
                mu_hat[pid] += ww * mu_pools_k[pool_id]

    return mu_hat

def estimate_policy_means_from_RPS(
    RPS_states,                     # dict with key "samples": List[State]
    log_alpha,
    policies,                     # global policy list (length P)
    policy_means,                 # np.ndarray [P,2] = [sum_y, count]
    prof_idx_of_policy,           # length-P array: policy_id -> profile k, for 36 policies, which profile is each policy in
    R_per,                        # np.ndarray of arm levels (includes control)
    M,
    lattice_edges=None            # optional lattice; pass None if unused
):
    """
    Posterior-weighted mean outcome per policy using MCMC samples.
    Weights default to uniform over retained samples.
    Returns: np.ndarray [P]
    """
    samples = RPS_states
    if len(samples) == 0:
        return np.zeros(len(policies), float)

    P = len(policies)
    mu_hat = np.zeros(P, dtype=float)

    # uniform weights over MCMC samples (empirical posterior)
    w = [i/sum(log_alpha) for i in log_alpha]

    # build profile→global index map once
    K = int(np.max(prof_idx_of_policy)) + 1
    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[prof_idx_of_policy[pid]].append(pid)

    # per-profile slices (fixed order defines local indices)
    prof_policies  = [[policies[i] for i in idxs] for idxs in prof_to_global]
    prof_means_arr = [
        policy_means[np.array(idxs, int), :] if idxs else np.zeros((0, 2))
        for idxs in prof_to_global
    ]

    R_per = np.asarray(R_per, int)  # ensure array for assemble

    for x_state, ww in zip(samples, w):
        for k in range(K):
            idxs = prof_to_global[k]
            if not idxs:
                continue

            # expand compact profile matrix to full-width expected by extract_pools
            sigma_full_k = AIS.assemble_sigma_full_for_profile(x_state[k], M, R_per)

            # pools + policy→pool mapping on the *local* list for profile k
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)
            mu_pools_k = loss.compute_pool_means(prof_means_arr[k], pi_pools_k)

            # IMPORTANT: pi_policies_k is keyed by LOCAL INDEX, not policy tuple
            for local_idx, pid in enumerate(idxs):
                pool_id = pi_policies_k[local_idx]
                mu_hat[pid] += ww * mu_pools_k[pool_id]

    return mu_hat

# --- optional: compare MCMC posterior to AIS posterior (over shared support) ---
def ais_posterior_over_signatures(ais_out) -> Dict[Tuple, float]:
    w_by_sig = defaultdict(float)
    for s, w in zip(ais_out.terminals, ais_out.normw):
        w_by_sig[AIS.state_signature(s)] += float(w)
    Z = sum(w_by_sig.values())
    return {sig: w/Z for sig, w in w_by_sig.items()} if Z>0 else {}

def tv_distance(p: Dict[Tuple,float], q: Dict[Tuple,float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k,0.0) - q.get(k,0.0)) for k in keys)

def policy_means_matrix_from_mcmc(
    samples,                      # mcmc_res["samples"]
    policies,                     # global list/array of policies (length P)
    policy_means,                 # np.ndarray [P,2] = [sum_y, count]
    prof_idx_of_policy,           # length-P array: policy_id -> profile k
    R_per,                        # np.ndarray of arm levels (includes control)
    M,
    lattice_edges=None,           # pass None if unused
    policy_labels=None            # optional column names; defaults to policy indices (0..P-1)
):
    """
    Build a DataFrame of per-sample predicted means per policy.
    Shape: (num_samples, P)
    """
    samples = samples
    S = len(samples)
    P = len(policies)
    if S == 0:
        return pd.DataFrame(columns=(policy_labels if policy_labels is not None else list(range(P))))

    # Map profiles -> global policy indices
    K = int(np.max(prof_idx_of_policy)) + 1
    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[prof_idx_of_policy[pid]].append(pid)

    # Per-profile slices (fixed order defines local indices)
    prof_policies  = [[policies[i] for i in idxs] for idxs in prof_to_global]
    prof_means_arr = [policy_means[np.array(idxs, int), :] if idxs else np.zeros((0, 2))
                      for idxs in prof_to_global]

    R_per = np.asarray(R_per, int)

    # Storage
    mat = np.zeros((S, P), dtype=float)

    for s_idx, state in enumerate(samples):
        # Fill this row
        row = np.zeros(P, float)
        for k in range(K):
            idxs = prof_to_global[k]
            if not idxs:
                continue

            sigma_full_k = AIS.assemble_sigma_full_for_profile(state[k], M, R_per)
            # NOTE: extract_pools returns (pools, policy->pool) on the local list
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)
            mu_pools_k = loss.compute_pool_means(prof_means_arr[k], pi_pools_k)

            # pi_policies_k is keyed by LOCAL INDEX
            for local_idx, pid in enumerate(idxs):
                pool_id = pi_policies_k[local_idx]
                row[pid] = mu_pools_k[pool_id]

        mat[s_idx, :] = row

    cols = policy_labels if policy_labels is not None else list(range(P))
    return pd.DataFrame(mat, columns=cols)


def summarize_policy_means(df, quantiles=(0.025, 0.5, 0.975)):
    """
    Summary across MCMC samples for each policy (column).
    Returns a tidy DataFrame with columns: policy, mean, sd, q{quantile}.
    """
    stats = {
        "policy": df.columns,
        "mean":   df.mean(axis=0).to_numpy(),
        "sd":     df.std(axis=0, ddof=1).to_numpy()
    }
    out = pd.DataFrame(stats)
    for q in quantiles:
        out[f"q{int(100*q)}"] = df.quantile(q, axis=0).to_numpy()
    return out

def plot_pred_vs_true(out,                         # output from summarize_policy_means
                      policy_means,
                      point_alpha=0.6,
                      point_color="#0b3d91",     # dark blue
                      bar_color="#87ceeb",       # sky blue
                      line_color="red",
                      figsize=(5,5)):
    """
    Replicates the ggplot: error bars for intervals, points for posterior means,
    and a y = x reference line. Aspect ratio is 1:1.
    """
    y_true = policy_means[:,0] / policy_means[:,1]
    y_mean  = out['q50'].to_numpy()
    y_lower = out['q2'].to_numpy()
    y_upper = out['q97'].to_numpy()

    if not (len(y_true) == len(y_mean) == len(y_lower) == len(y_upper)):
        raise ValueError("All input arrays must have the same length.")

    # Error bar extents
    y_err = np.vstack([y_mean - y_lower, y_upper - y_mean])

    fig, ax = plt.subplots(figsize=figsize)

    # Error bars
    ax.errorbar(y_true, y_mean, yerr=y_err, fmt='none',
                ecolor=bar_color, elinewidth=1, capsize=2, alpha=0.9)

    # Points
    ax.scatter(y_true, y_mean, s=20, alpha=point_alpha, color=point_color)

    # y = x line over the current range
    xmin = np.nanmin(y_true)
    xmax = np.nanmax(y_true)
    ymin = np.nanmin([y_lower.min(), y_mean.min()])
    ymax = np.nanmax([y_upper.max(), y_mean.max()])
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    ax.plot([lo, hi], [lo, hi], color=line_color, linestyle='-')

    # Labels & style
    ax.set_xlabel("True data generating beta")
    ax.set_ylabel("Estimates of y")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Minimal “bw” style
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    return fig, ax

# Now instead of computing posterior mean, use Bayesian framework, start with normal-InvGamma prior 
# and sample from posterior distribution to do the plot

def nig_posterior_params(n, sum_y, sumsq_y, mu0=0.0, kappa0=1.0, alpha0=0.0, beta0=-1.0):
    """
    Normal-Inverse-Gamma prior:
      sigma2 ~ InvGamma(alpha0, beta0)
      mu | sigma2 ~ Normal(mu0, sigma2/kappa0)

    Given sufficient stats in a pool:
      n, sum_y, sumsq_y = sum(y^2)

    Returns posterior params (mu_n, kappa_n, alpha_n, beta_n).
    """
    n = int(n)
    if n <= 0:
        # no data: posterior = prior
        return float(mu0), float(kappa0), float(alpha0), float(beta0)

    ybar = sum_y / n
    sse = sumsq_y - n * (ybar ** 2)  # sum (y - ybar)^2

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * ybar) / kappa_n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * sse + 0.5 * (kappa0 * n / kappa_n) * ((ybar - mu0) ** 2)

    return float(mu_n), float(kappa_n), float(alpha_n), float(beta_n)

def sample_mu_from_nig(mu_n, kappa_n, alpha_n, beta_n, rng, size=1):
    """
    Sample mu marginally by:
      sigma2 ~ InvGamma(alpha_n, beta_n)
      mu | sigma2 ~ Normal(mu_n, sigma2 / kappa_n)

    Returns array of shape (size,).
    """
    # InvGamma(alpha, beta): sample via Gamma(alpha, scale=1/beta) then invert
    g = rng.gamma(shape=alpha_n, scale=1.0 / beta_n, size=size)
    sigma2 = 1.0 / g
    mu = rng.normal(loc=mu_n, scale=np.sqrt(sigma2 / kappa_n), size=size)
    return mu

def policy_mean_draws_from_mcmc_nig(
    mcmc_res,
    D, y,
    policies,
    prof_idx_of_policy,
    R_per,
    M,
    lattice_edges=None,
    n_draws_per_state=3,
    mu0=0.0, kappa0=1.0, alpha0=0.0, beta0=-1.0,
    seed=0,
):
    """
    Returns:
      draws: np.ndarray of shape (S * n_draws_per_state, P)
             Each row is one posterior draw of mean for every policy.

    This includes partition uncertainty (MCMC) + within-pool (NIG) uncertainty.
    """
    rng = np.random.default_rng(seed)

    samples = mcmc_res["samples"]
    S = len(samples)
    P = len(policies)
    if S == 0:
        return np.zeros((0, P), float)

    # y to 1D
    y1 = y[:, 0] if (isinstance(y, np.ndarray) and y.ndim == 2) else np.asarray(y).ravel()
    pid_all = D[:, 0].astype(int)

    K = int(np.max(prof_idx_of_policy)) + 1

    # Profile -> list of global policy ids (fixed order defines local indices)
    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[int(prof_idx_of_policy[pid])].append(pid)

    prof_policies = [[policies[i] for i in idxs] for idxs in prof_to_global]
    prof_pid_to_local = []
    for k in range(K):
        idxs = prof_to_global[k]
        prof_pid_to_local.append({pid: j for j, pid in enumerate(idxs)})

    # Also split data indices by profile (fast masks)
    prof_data_idx = []
    for k in range(K):
        idxs = np.array(prof_to_global[k], dtype=int)
        if idxs.size == 0:
            prof_data_idx.append(np.array([], dtype=int))
            continue
        mask = np.isin(pid_all, idxs)
        prof_data_idx.append(np.flatnonzero(mask))

    # output draws
    out = np.zeros((S * n_draws_per_state, P), float)
    row = 0

    for s_idx, state in enumerate(samples):
        # For each profile, build pools & also pool sufficient stats
        per_prof_poolmap = [None] * K
        per_prof_poolstats = [None] * K
        per_prof_npools = [0] * K

        for k in range(K):
            idxs = prof_to_global[k]
            if not idxs:
                continue

            sigma_full_k = AIS.assemble_sigma_full_for_profile(state[k], M, np.asarray(R_per, int))
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)
            n_pools = len(pi_pools_k)
            per_prof_npools[k] = n_pools
            per_prof_poolmap[k] = pi_policies_k  # local_idx -> pool_id

            # compute sufficient stats by scanning the data rows in this profile
            n = np.zeros(n_pools, int)
            sy = np.zeros(n_pools, float)
            sy2 = np.zeros(n_pools, float)

            didx = prof_data_idx[k]
            pid_k = pid_all[didx]
            y_k = y1[didx]

            pid2loc = prof_pid_to_local[k]
            for pid_obs, y_obs in zip(pid_k, y_k):
                loc = pid2loc[int(pid_obs)]               # local index of this policy in prof_policies[k]
                pool = pi_policies_k[loc]                 # pool id
                n[pool] += 1
                sy[pool] += float(y_obs)
                sy2[pool] += float(y_obs) ** 2

            per_prof_poolstats[k] = (n, sy, sy2)

        # Now draw pool means and assign to policies
        for _ in range(n_draws_per_state):
            mu_draw = np.zeros(P, float)

            for k in range(K):
                idxs = prof_to_global[k]
                if not idxs:
                    continue
                pi_policies_k = per_prof_poolmap[k]
                n, sy, sy2 = per_prof_poolstats[k]
                n_pools = per_prof_npools[k]

                # sample a mean for each pool
                pool_mu = np.zeros(n_pools, float)
                for j in range(n_pools):
                    mu_n, k_n, a_n, b_n = nig_posterior_params(
                        n[j], sy[j], sy2[j],
                        mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0
                    )
                    pool_mu[j] = sample_mu_from_nig(mu_n, k_n, a_n, b_n, rng, size=1)[0]

                # assign to each policy in this profile
                for local_idx, pid in enumerate(idxs):
                    pool_id = pi_policies_k[local_idx]
                    mu_draw[pid] = pool_mu[pool_id]

            out[row, :] = mu_draw
            row += 1

    return out

def summarize_policy_draws(draws, qs=(0.1, 0.5, 0.9)):
    """
    draws: (Ndraw, P)
    Returns dict with mean and quantiles per policy.
    """
    mean = draws.mean(axis=0)
    quants = {q: np.quantile(draws, q, axis=0) for q in qs}
    return {"mean": mean, "quantiles": quants}

def plot_policy_spread(policy_means, summ, q_low=0.1, q_high=0.9, title="Posterior policy means (MCMC + NIG)"):
    """
    Scatter + vertical intervals vs true mu.
    """
    true_mu = policy_means[:,0] / policy_means[:,1]
    mean = summ["mean"]
    lo = summ["quantiles"][q_low]
    hi = summ["quantiles"][q_high]

    x = np.asarray(true_mu).ravel()
    y = mean

    plt.figure()
    plt.errorbar(x, y, yerr=[y - lo, hi - y], fmt='o', capsize=2)
    mn = min(x.min(), y.min(), lo.min())
    mx = max(x.max(), y.max(), hi.max())
    plt.plot([mn, mx], [mn, mx], linestyle='--')
    plt.xlabel("True policy mean")
    plt.ylabel("Posterior mean (with interval)")
    plt.title(title)
    plt.show()



def mcmc_unique_count(mcmc_res) -> int:
    """Number of unique states in mcmc_res['samples']."""
    return len({ AIS.state_signature(s) for s in mcmc_res["samples"] })

def mcmc_unique_stats(mcmc_res, with_prob=True):
    """
    Returns {signature: {"count": c, "prob": p}} (prob=empirical frequency).
    If with_prob=False, omits the prob field.
    """
    counts = defaultdict(int)
    samples = mcmc_res["samples"]
    for s in samples:
        counts[AIS.state_signature(s)] += 1
    total = float(len(samples)) or 1.0
    if with_prob:
        return {sig: {"count": c, "prob": c/total} for sig, c in counts.items()}
    else:
        return {sig: {"count": c} for sig, c in counts.items()}

# Example usage:
# mcmc_res = run_mcmc(score_s, RPS_states, log_alpha, n_chains=8, steps=30000, burnin=5000, thin=10, seed=1, min_len=1)
# mcmc_post = mcmc_res["posterior"]
# ais_post = ais_posterior_over_signatures(ais_out)
# print("Mean acc rate:", mcmc_res["acc_rate_mean"])
# print("TV distance(AIS, MCMC):", tv_distance(ais_post, mcmc_post))

def find_states_matching_targets(mcmc_res, R_per, M, target_by_profile):
    """
    target_by_profile: {k: (cov_ids_tuple, Sigma_k_compact)}
    Returns indices/samples that match ALL specified profiles.
    """
    R_per = np.asarray(R_per, int)
    # pre-expand targets
    tgt_full = {}
    for k, (cov_ids_k, Sigma_k) in target_by_profile.items():
        pp = AIS.ProfilePart(cov_ids=tuple(cov_ids_k), B=np.asarray(Sigma_k, float) if Sigma_k is not None else None)
        tgt_full[k] = AIS.assemble_sigma_full_for_profile(pp, M, R_per)

    matches_idx, matches = [], []
    samples = mcmc_res.get("samples", [])
    for i, s in enumerate(samples):
        ok = True
        for k, target_full in tgt_full.items():
            samp_full = AIS.assemble_sigma_full_for_profile(s[k], M, R_per)
            if not np.array_equal(samp_full, target_full):
                ok = False; break
        if ok:
            matches_idx.append(i); matches.append(s)
    total = max(1, len(samples))
    return {"indices": matches_idx, "samples": matches,
            "count": len(matches_idx), "fraction": len(matches_idx)/total}

def save_mcmc_res(mcmc_res, path, *, compression="gz", protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save an object (e.g., mcmc_res) to disk as a pickle.

    Parameters
    ----------
    mcmc_res : Any
        The object to serialize.
    path : str | os.PathLike
        Output file path. Use .pkl, .pkl.gz, or .pkl.xz (extension not forced).
    compression : {"gz","xz",None}
        Optional compression (gzip or lzma). Default "gz".
    protocol : int
        Pickle protocol; defaults to pickle.HIGHEST_PROTOCOL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Choose opener based on compression
    if compression == "gz" or str(path).endswith(".gz"):
        opener = lambda p: gzip.open(p, "wb")
    elif compression == "xz" or str(path).endswith(".xz"):
        opener = lambda p: lzma.open(p, "wb")
    elif compression is None:
        opener = lambda p: open(p, "wb")
    else:
        raise ValueError("compression must be one of {'gz','xz',None}")

    # Atomic write via temp file in same directory
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp_name = tmp.name
    try:
        with opener(tmp_name) as f:
            pickle.dump(mcmc_res, f, protocol=protocol)
        os.replace(tmp_name, path)  # atomic on POSIX/Windows
    except Exception:
        # clean up temp file if something failed
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise

def load_mcmc_res(path):
    """
    Load a pickle (optionally gz/xz compressed) saved by save_mcmc_res.
    """
    path = Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    elif str(path).endswith(".xz"):
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


##Dynamically streaming trial:
import json, math, random, numpy as np
from typing import List, Optional, Callable

def _state_to_jsonable(state):
    """Convert State -> JSON-safe (replace inf with 'inf')."""
    out = []
    for pp in state:
        cov = None if pp.cov_ids is None else list(pp.cov_ids)  # tuple -> list for JSON
        if pp.B is None:
            B = None
        else:
            A = np.asarray(pp.B, float)
            B_list = A.tolist()
            # JSON doesn't support Infinity; store as "inf"
            def fix(v):
                if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                    return "inf" if math.isinf(v) else "nan"
                return v
            if isinstance(B_list[0], list):  # 2D
                B = [[fix(x) for x in row] for row in B_list]
            else:  # 1D
                B = [fix(x) for x in B_list]
        out.append({"cov_ids": cov, "B": B})
    return out

def _jsonable_to_state(obj):
    """Optional: convert JSON-safe back to State (requires ProfilePart in scope)."""
    out=[]
    for d in obj:
        cov = None if d["cov_ids"] is None else tuple(d["cov_ids"])
        B = d["B"]
        if B is None:
            arr = None
        else:
            # convert "inf" back to np.inf
            def unfix(x):
                if x == "inf": return np.inf
                if x == "nan": return np.nan
                return x
            if isinstance(B[0], list):
                arr = np.array([[unfix(x) for x in row] for row in B], float)
            else:
                arr = np.array([unfix(x) for x in B], float)
        out.append(AIS.ProfilePart(cov_ids=cov, B=arr))
    return out

def mh_step_true_posterior(x: State, score_s: Callable[[State], float], min_len=1):
    """Same MH step you used: uniform over neighbors with degree correction."""
    N_cur = [n for n in AIS.state_neighbors_ubs(x, min_len=min_len) if not AIS.states_equal(n, x)]
    if not N_cur:
        return x, 0
    prop = random.choice(N_cur)
    N_prop = [n for n in AIS.state_neighbors_ubs(prop, min_len=min_len) if not AIS.states_equal(n, prop)]
    lcur = math.log(max(1e-300, score_s(x)))
    lprop = math.log(max(1e-300, score_s(prop)))
    logr = (lprop - lcur) + math.log(max(1, len(N_cur))) - math.log(max(1, len(N_prop)))
    if math.log(random.random() + 1e-300) < min(0.0, logr):
        return prop, 1
    return x, 0

def run_mcmc_streaming(
    RPS: List, 
    log_alpha: List[float],
    score_s: Callable[[State], float],
    steps: int = 50000,
    burnin: int = 5000,
    thin: int = 5,
    min_len: int = 1,
    seed: Optional[int] = 0,
    out_jsonl: str = "mcmc_samples.jsonl",
    progress_json: str = "mcmc_progress.json"
):
    """
    Runs MH and appends every `thin`-th post-burn-in sample to out_jsonl immediately.
    """
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    rng = np.random.default_rng(seed)

    x0 = init_from_RPS(RPS, log_alpha, rng)  # start in RPS (loss-weighted)
    x = x0
    accepts = 0
    kept = 0

    # open in append mode so you can resume / tail the file
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for t in range(steps):
            x, acc = mh_step_true_posterior(x, score_s, min_len=min_len)
            accepts += acc

            # keep sample on schedule
            if t >= burnin and ((t - burnin) % thin == 0):
                rec = {
                    "iter": t,
                    "state": _state_to_jsonable(x),
                    "log_score": float(math.log(max(1e-300, score_s(x))))
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()  # ensures it’s on disk right away
                kept += 1

            # update progress every so often (e.g., every 200 iters)
            if (t % 200) == 0:
                prog = {
                    "iter": t,
                    "accept_rate_so_far": accepts / max(1, t+1),
                    "kept_samples": kept,
                    "burnin": burnin,
                    "thin": thin,
                    "steps": steps
                }
                with open(progress_json, "w", encoding="utf-8") as g:
                    json.dump(prog, g)

    return {"accept_rate": accepts / max(1, steps), "kept": kept, "out_jsonl": out_jsonl}

def load_mcmc_res_from_jsonl(jsonl_path: str):
    """
    Read a streaming MCMC JSONL file (written by run_mcmc_streaming) and return:
      mcmc_res = {"samples": [State,...], "iters": np.array, "log_score": np.array}
    where each State is a list[ProfilePart].
    Requires: _jsonable_to_state(...) and ProfilePart in scope.
    """
    samples = []
    iters = []
    log_score = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            samples.append(_jsonable_to_state(rec["state"]))
            iters.append(int(rec.get("iter", -1)))
            log_score.append(float(rec.get("log_score", np.nan)))

    return {
        "samples": samples,
        "iters": np.asarray(iters, dtype=int),
        "log_score": np.asarray(log_score, dtype=float),
    }