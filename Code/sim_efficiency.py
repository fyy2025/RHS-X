"""
sim_efficiency.py
=================

Illustrative AIS-efficiency study on the tiny (64-partition) scenario-1 model
space. It demonstrates the central efficiency principle of annealed importance
sampling seeded from a proposal p0:

    the number of annealing levels T needed to reach a target relative ESS
    grows with the distance between the seed proposal p0 and the target
    posterior pi, and empirically scales ~linearly with sqrt(chi^2(pi || p0)).

Because the space has only 64 partitions, the exact posterior, the exact chi^2
divergence, and full enumeration are all trivial, so we can measure the
relationship exactly rather than estimate it.

Design / assumptions (the provided spec was a fragment; these are the choices
made, all documented here):

  * Model space  : scenario 1 (M=2, R=[4,3]); the full partition space is
                   enumerated with `AIS.enumerate_all_states_and_losses`.
  * Target pi    : the g-prior posterior, pi(Pi) propto exp(log_score_s_gprior),
                   i.e. the SAME object MCMC / AIS target elsewhere in this repo.
  * Seed set S   : the top-`k` partitions by posterior mass (a stand-in for the
                   near-optimal RPS). L1(S) = its 1-flip (single Sigma-entry)
                   neighbours, obtained from `AIS.state_neighbors_ubs`.
  * Proposal p0  : the bucket proposal of apara2025 -- mass alpha1 on S
                   (distributed proportional to the posterior), alpha2-alpha1 on
                   L1(S) (uniform), 1-alpha2 on the complement (uniform).
                   A single "spread" parameter eps in (0,1] interpolates the
                   bucket masses from the strong seed (eps->0) to the UNIFORM
                   proposal (eps=1, masses proportional to bucket sizes).
  * Path         : geometric, f_j propto p0^{1-b_j} * pi^{b_j}, b_j = j/T.
  * Move         : single-entry Sigma flip = hop to a uniformly-chosen 1-flip
                   neighbour, Metropolis-corrected for neighbour-count asymmetry.
  * Weights      : log-space,  log_w += log_f_j(x) - log_f_{j-1}(x)  (spec).

Outputs (written to ../Figures unless --outdir given):
  * sim_efficiency_ess_vs_T.png   -- relative ESS vs T, one curve per eps + uniform
  * sim_efficiency_Tmin_vs_chi.png-- minimal T vs sqrt(chi^2), the linear scaling
  * a printed summary table (minimal T and chi^2 per seed strength)
"""

import os
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

from rashomon import hasse, extract_pools, loss, AIS


# ---------------------------------------------------------------------------
# 1. Build the 64-partition model space with an exact g-prior posterior
# ---------------------------------------------------------------------------
def build_model_space(num_samples_per_feature=500, lamb=1, data_seed=42):
    """Enumerate the scenario-1 partition space and return, as flat arrays over
    the enumerated states:

        log_pi  : (P,) unnormalised log target (g-prior log posterior)
        neigh   : (P, max_deg) int padded 1-flip neighbour indices (-1 = pad)
        deg     : (P,) neighbour counts
        states  : list of the underlying State objects (for reference)
    """
    M = 2
    R = np.array([4, 3])

    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)
    g = num_policies * num_samples_per_feature  # unit-information prior, g = n

    # --- scenario-1 ground-truth pooling structure (same as simulation_scenario1) ---
    sigma_00 = None;               mu_00 = np.array([0]);      var_00 = np.array([1])
    sigma_01 = np.array([[1]]);    mu_01 = np.array([-1]);     var_01 = np.array([1])
    sigma_10 = np.array([[1, 0]]); mu_10 = np.array([-2, -3]); var_10 = np.array([1, 1])
    sigma_11 = np.array([[0, 1], [0, np.inf]])
    mu_11 = np.array([2, 3, -1, 1]); var_11 = np.array([1, 1, 1, 1])
    sigma = [sigma_00, sigma_01, sigma_10, sigma_11]
    mu    = [mu_00, mu_01, mu_10, mu_11]
    var   = [var_00, var_01, var_10, var_11]

    policies_profiles = {}
    pi_policies = {}
    for k, profile in enumerate(profiles):
        policies_temp = [(i, x) for i, x in enumerate(all_policies)
                         if hasse.policy_to_profile(x) == profile]
        _, policies_k = map(list, zip(*policies_temp))
        policies_profiles[k] = deepcopy(policies_k)
        profile_mask = list(map(bool, profile))
        for idx, pol in enumerate(policies_k):
            policies_k[idx] = tuple([pol[i] for i in range(M) if profile_mask[i]])
        if np.sum(profile) > 0:
            _, pi_policies_k = extract_pools.extract_pools(policies_k, sigma[k])
            pi_policies[k] = pi_policies_k
        else:
            pi_policies[k] = {0: 0}

    def generate_data():
        num_data = num_policies * num_samples_per_feature
        D = np.zeros(shape=(num_data, 1), dtype="int_")
        y = np.zeros(shape=(num_data, 1))
        idx_ctr = 0
        for k, profile in enumerate(profiles):
            for idx, policy in enumerate(policies_profiles[k]):
                policy_idx = [i for i, x in enumerate(all_policies) if x == policy]
                pool_id = pi_policies[k][idx]
                y_i = np.random.normal(mu[k][pool_id], var[k][pool_id],
                                       size=(num_samples_per_feature, 1))
                s, e = idx_ctr * num_samples_per_feature, (idx_ctr + 1) * num_samples_per_feature
                D[s:e] = policy_idx[0]
                y[s:e] = y_i
                idx_ctr += 1
        return D, y

    np.random.seed(data_seed)
    D, y = generate_data()
    policy_means = loss.compute_policy_means(D, y, num_policies)
    prof_idx_of_policy, profiles = AIS.build_profile_index_of_policy(
        all_policies, hasse.policy_to_profile)
    sigma2 = AIS.compute_sigma2_saturated(D, y, all_policies)

    # Enumerate every partition (state) in the space
    states, _ = AIS.enumerate_all_states_and_losses(
        profiles=profiles, R=R, M=M, policies=all_policies,
        policy_means=policy_means, prof_idx_of_policy=prof_idx_of_policy,
        D=D, y=y, reg=lamb, normalize=0, lattice_edges=None, max_states=None)
    P = len(states)

    # Exact (unnormalised) log posterior for each state
    log_pi = np.array([
        AIS.log_score_s_gprior(s, D, y, M, R, prof_idx_of_policy, all_policies,
                               policy_means, g=g, sigma2=sigma2, lam=lamb)
        for s in states])

    # 1-flip neighbour graph (index space)
    sig2idx = {AIS.state_signature(s): i for i, s in enumerate(states)}
    neigh_lists = []
    for s in states:
        nb = []
        for n in AIS.state_neighbors_ubs(s, min_len=1):
            j = sig2idx.get(AIS.state_signature(n))
            if j is not None:
                nb.append(j)
        neigh_lists.append(sorted(set(nb)))
    deg = np.array([len(nb) for nb in neigh_lists])
    max_deg = int(deg.max())
    neigh = -np.ones((P, max_deg), dtype=int)
    for i, nb in enumerate(neigh_lists):
        neigh[i, :len(nb)] = nb

    return dict(states=states, log_pi=log_pi, neigh=neigh, deg=deg, P=P)


# ---------------------------------------------------------------------------
# 2. Seed-set bucket proposal p0(eps) and its exact chi^2 divergence
# ---------------------------------------------------------------------------
def build_seed_buckets(log_pi, neigh, deg, k):
    """Return (S, L1) index arrays: S = top-k posterior states, L1 = their
    1-flip neighbours that are not already in S."""
    order = np.argsort(-log_pi)
    S = set(order[:k].tolist())
    L1 = set()
    for i in S:
        for slot in range(deg[i]):
            j = int(neigh[i, slot])
            if j not in S:
                L1.add(j)
    return np.array(sorted(S)), np.array(sorted(L1))


def make_p0(eps, log_pi, S, L1, P, a_S=0.9, a_L=0.08):
    """Bucket proposal with a single spread parameter eps in (0,1].

    eps = 1  -> masses proportional to bucket sizes == UNIFORM proposal.
    eps -> 0 -> strong seed: alpha1 -> a_S on S, alpha2-alpha1 -> a_L on L1(S).

    Returns a normalised probability vector p0 (length P).
    """
    nS, nL = len(S), len(L1)
    comp_mask = np.ones(P, dtype=bool)
    comp_mask[S] = False
    comp_mask[L1] = False
    nC = int(comp_mask.sum())

    # uniform-equivalent bucket masses (eps=1)
    uS, uL = nS / P, nL / P
    t = 1.0 - eps  # concentration: 0 at eps=1, 1 at eps=0
    alpha1 = uS + t * (a_S - uS)
    alpha2 = (uS + uL) + t * ((a_S + a_L) - (uS + uL))
    alpha1 = float(np.clip(alpha1, 1e-12, 1 - 1e-9))
    alpha2 = float(np.clip(alpha2, alpha1 + 1e-12, 1 - 1e-12))

    p0 = np.zeros(P)
    # S: alpha1 distributed within S, interpolating uniform (t=0) -> posterior (t=1)
    wS = np.exp(t * (log_pi[S] - log_pi[S].max()))
    wS = wS / wS.sum()
    p0[S] = alpha1 * wS
    # L1(S): uniform
    if nL > 0:
        p0[L1] = (alpha2 - alpha1) / nL
    # complement: uniform
    if nC > 0:
        p0[comp_mask] = (1.0 - alpha2) / nC
    p0 = p0 / p0.sum()
    return p0


def chi2_divergence(log_pi, p0):
    """Exact chi^2(pi || p0) = sum_i pi_i^2 / p0_i - 1  (pi normalised)."""
    pi = np.exp(log_pi - log_pi.max())
    pi = pi / pi.sum()
    return float(np.sum(pi ** 2 / p0) - 1.0)


def temper_to_perplexity(log_pi_raw, target_perp, tol=1e-4):
    """Rescale the (unnormalised) log target so the normalised posterior spreads
    over ~`target_perp` effective states (perplexity = exp(entropy)).

    The exact g-prior posterior on 64 states is astronomically peaked (log range
    ~1e4), which makes any geometric annealing path a step function at beta=0 --
    unusable for an *illustration*. Tempering to a moderate perplexity gives a
    target that AIS can actually bridge, without changing the qualitative
    efficiency principle being demonstrated. Returns a normalised-scale log pmf
    (shifted so max = 0). Note: the seed-vs-uniform chi^2 ordering is preserved.
    """
    lp = log_pi_raw - log_pi_raw.max()
    P = len(lp)
    target_perp = min(max(float(target_perp), 1.001), P - 0.001)

    def perplexity(tau):
        z = lp / tau
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        H = -np.sum(w * np.log(w + 1e-300))
        return math.exp(H)

    lo, hi = 1e-4, 1e8            # perplexity is increasing in tau
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        pm = perplexity(mid)
        if pm < target_perp:
            lo = mid
        else:
            hi = mid
        if abs(pm - target_perp) < tol:
            break
    tau = math.sqrt(lo * hi)
    out = lp / tau
    return out - out.max()


# ---------------------------------------------------------------------------
# 3. Vectorised annealed importance sampling on the index space
# ---------------------------------------------------------------------------
def relative_ess_AIS(log_p0, log_pi, neigh, deg, T, n_paths, moves_per_level=1,
                     rng=None, schedule_power=1.0):
    """One AIS run with T geometric-path levels; returns relative ESS in (0, 1].

    `schedule_power` shapes the inverse-temperature ladder beta_j = (j/T)^power.
    power=1 is the naive linear schedule; power>1 packs more levels near beta=0
    (where a peaked target changes fastest), which is what lets the minimal-T
    vs sqrt(chi^2) relationship approach the theoretical linear scaling.
    """
    if rng is None:
        rng = np.random.default_rng()
    P = len(log_pi)
    log_deg = np.log(deg)

    betas = np.linspace(0.0, 1.0, T + 1) ** schedule_power   # beta_0=0 ... beta_T=1
    # log f_j over all states for every level (P,) each
    logf = [(1.0 - b) * log_p0 + b * log_pi for b in betas]

    # x_0 ~ p0
    p0 = np.exp(log_p0 - log_p0.max()); p0 /= p0.sum()
    cur = rng.choice(P, size=n_paths, p=p0)
    log_w = np.zeros(n_paths)

    for j in range(1, T + 1):
        # importance weight increment: f_j(x) / f_{j-1}(x) at the current x
        log_w += logf[j][cur] - logf[j - 1][cur]
        # MH moves targeting f_j (single-flip proposal, degree-corrected)
        fj = logf[j]
        for _ in range(moves_per_level):
            slot = (rng.random(n_paths) * deg[cur]).astype(int)
            prop = neigh[cur, slot]
            log_ratio = (fj[prop] - fj[cur]) + (log_deg[cur] - log_deg[prop])
            acc = np.log(rng.random(n_paths)) < log_ratio
            cur = np.where(acc, prop, cur)

    # relative ESS from the (log) importance weights
    m = log_w.max()
    w = np.exp(log_w - m)
    ess = (w.sum() ** 2) / np.sum(w ** 2)
    return ess / n_paths


# ---------------------------------------------------------------------------
# 4. Driver: sweep T for each eps, find minimal T, make the figures
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="AIS efficiency vs seed quality (64-partition space)")
    ap.add_argument("--k", type=int, default=16,
                    help="seed-set size |S| (top-k posterior); keep >= target_perplexity so S "
                         "covers the target support and chi^2 is monotone in seed strength")
    ap.add_argument("--eps", type=str, default="1.0,0.9,0.8,0.7,0.6,0.5,0.35,0.2,0.1,0.05",
                    help="comma-separated spread values (eps=1 is the uniform proposal)")
    ap.add_argument("--T_grid", type=str,
                    default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256",
                    help="comma-separated numbers of annealing levels to try")
    ap.add_argument("--n_paths", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5, help="repetitions averaged per (eps, T)")
    ap.add_argument("--moves_per_level", type=int, default=5)
    ap.add_argument("--schedule_power", type=float, default=3.0,
                    help="inverse-temp ladder beta_j=(j/T)^power; >1 packs levels near beta=0")
    ap.add_argument("--target_ess", type=float, default=0.5, help="relative-ESS target for minimal T")
    ap.add_argument("--target_perplexity", type=float, default=8.0,
                    help="temper the target posterior to spread over ~this many of the 64 states")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="../Figures")
    args = ap.parse_args()

    eps_list = [float(e) for e in args.eps.split(",")]
    T_grid = [int(t) for t in args.T_grid.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Building 64-partition model space (scenario 1) ...")
    ms = build_model_space()
    neigh, deg, P = ms["neigh"], ms["deg"], ms["P"]
    # Temper the (pathologically peaked) exact posterior to an illustratable spread
    log_pi = temper_to_perplexity(ms["log_pi"], args.target_perplexity)
    perp = math.exp(-np.sum(
        (lambda w: w * np.log(w + 1e-300))(np.exp(log_pi) / np.exp(log_pi).sum())))
    print(f"  |P| = {P} partitions | degrees: min={deg.min()} max={deg.max()} mean={deg.mean():.1f}")
    print(f"  target tempered to perplexity {perp:.1f} of {P} states")

    S, L1 = build_seed_buckets(log_pi, neigh, deg, args.k)
    print(f"  seed set |S|={len(S)}  |L1(S)\\S|={len(L1)}  complement={P-len(S)-len(L1)}")

    # p0 and exact chi^2 per eps
    p0s, chi2s = {}, {}
    for eps in eps_list:
        p0 = make_p0(eps, log_pi, S, L1, P)
        p0s[eps] = p0
        chi2s[eps] = chi2_divergence(log_pi, p0)

    # sweep T x eps (averaged over reps)
    print("\nSweeping annealing levels T for each seed strength ...")
    ess_curve = {eps: [] for eps in eps_list}
    for eps in eps_list:
        log_p0 = np.log(np.maximum(p0s[eps], 1e-300))
        for T in T_grid:
            vals = [relative_ess_AIS(log_p0, log_pi, neigh, deg, T, args.n_paths,
                                     args.moves_per_level, rng,
                                     schedule_power=args.schedule_power)
                    for _ in range(args.reps)]
            ess_curve[eps].append(float(np.mean(vals)))
        print(f"  eps={eps:<5} chi2={chi2s[eps]:9.3f}  relESS@Tmax={ess_curve[eps][-1]:.3f}")

    # minimal T to reach the target relative ESS (linear interp between grid pts)
    def minimal_T(eps):
        ys = np.array(ess_curve[eps]); xs = np.array(T_grid, dtype=float)
        hit = np.where(ys >= args.target_ess)[0]
        if len(hit) == 0:
            return np.nan
        i = hit[0]
        if i == 0:
            return float(xs[0])
        x0, x1, y0, y1 = xs[i-1], xs[i], ys[i-1], ys[i]
        return float(x0 + (args.target_ess - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)

    Tmin = {eps: minimal_T(eps) for eps in eps_list}

    # ---- summary table ----
    print("\n" + "=" * 60)
    print(f"Summary (target relative ESS = {args.target_ess})")
    print("=" * 60)
    print(f"{'eps':>6}{'chi^2(pi||p0)':>16}{'sqrt(chi^2)':>13}{'minimal T':>12}")
    for eps in eps_list:
        tag = "  (uniform)" if abs(eps - 1.0) < 1e-9 else ""
        print(f"{eps:>6}{chi2s[eps]:>16.3f}{math.sqrt(chi2s[eps]):>13.3f}{Tmin[eps]:>12.2f}{tag}")

    # ---- figure 1: relative ESS vs T ----
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eps_list)))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for eps, c in zip(eps_list, colors):
        lbl = f"eps={eps} (uniform)" if abs(eps - 1.0) < 1e-9 else f"eps={eps}  (chi2={chi2s[eps]:.1f})"
        ax.plot(T_grid, ess_curve[eps], "-o", color=c, ms=4, lw=1.6, label=lbl)
    ax.axhline(args.target_ess, color="k", ls="--", lw=1.1, alpha=0.7,
               label=f"target relESS={args.target_ess}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("annealing levels  T")
    ax.set_ylabel("relative ESS")
    ax.set_ylim(0, 1.02)
    ax.set_title("AIS efficiency vs seed quality  (64-partition space)")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    f1 = os.path.join(args.outdir, "sim_efficiency_ess_vs_T.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight")

    # ---- figure 2: minimal T vs sqrt(chi^2) ----
    xs = np.array([math.sqrt(chi2s[e]) for e in eps_list])
    ys = np.array([Tmin[e] for e in eps_list])
    good = np.isfinite(ys)
    fig2, ax2 = plt.subplots(figsize=(6.4, 5))
    ax2.scatter(xs[good], ys[good], c=plt.cm.viridis(np.linspace(0, 0.9, len(eps_list)))[good],
                s=70, edgecolor="k", zorder=5)
    for e, x, yv in zip(eps_list, xs, ys):
        if np.isfinite(yv):
            ax2.annotate(f"eps={e}", (x, yv), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)
    if good.sum() >= 2:
        b, a = np.polyfit(xs[good], ys[good], 1)
        xx = np.linspace(xs[good].min(), xs[good].max(), 50)
        r = np.corrcoef(xs[good], ys[good])[0, 1]
        ax2.plot(xx, a + b * xx, "r--", lw=1.4,
                 label=f"linear fit  (slope={b:.2f}, r={r:.3f})")
        ax2.legend(fontsize=9)
    ax2.set_xlabel(r"$\sqrt{\chi^2(\pi \,\|\, p_0)}$")
    ax2.set_ylabel(f"minimal T for relESS $\\geq$ {args.target_ess}")
    ax2.set_title("Minimal annealing length scales with $\\sqrt{\\chi^2}$")
    ax2.grid(alpha=0.3, ls=":")
    fig2.tight_layout()
    f2 = os.path.join(args.outdir, "sim_efficiency_Tmin_vs_chi.png")
    fig2.savefig(f2, dpi=150, bbox_inches="tight")

    print(f"\nSaved figures:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
