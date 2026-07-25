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
  * Target pi    : the true g-prior posterior, pi(Pi) propto exp(log_score_s_gprior),
                   i.e. the SAME object MCMC / AIS target elsewhere in this repo.
                   Untempered by default (--target_perplexity 0). NOTE: on
                   scenario 1 this posterior is a near point mass (perplexity
                   ~1.1); --target_perplexity>0 tempers it for a cleaner
                   illustration at the cost of faithfulness.
  * Seed set S   : the TRUE g-prior RPS by default (--seed_mode rps): every
                   partition within `rps_gap` log-posterior units of the MAP
                   (posterior-gap Rashomon set under the g-prior score). Optional
                   --seed_mode topk uses the top-k posterior states. L1(S) = its
                   1-flip (single Sigma-entry) neighbours from
                   `AIS.state_neighbors_ubs`.
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
# 1. Build the model space with an exact g-prior posterior
# ---------------------------------------------------------------------------
def scenario_config(scenario, mu_scale=1.0, var_scale=1.0):
    """Return (M, R, sigma, mu, var) for the requested scenario -- the exact
    ground-truth pooling structures used by simulation_scenario{1,2}.

    mu_scale shrinks the pool-mean separations and var_scale inflates the noise
    SD (note: `var` is used as the scale/SD in np.random.normal). Both make the
    poolings harder to tell apart, spreading the g-prior posterior over more
    partitions instead of collapsing it onto a single MAP (point mass)."""
    if scenario == 1:
        M = 2
        R = np.array([4, 3])
        sigma_00 = None;               mu_00 = np.array([0]);      var_00 = np.array([1])
        sigma_01 = np.array([[1]]);    mu_01 = np.array([-1]);     var_01 = np.array([1])
        sigma_10 = np.array([[1, 0]]); mu_10 = np.array([-2, -3]); var_10 = np.array([1, 1])
        sigma_11 = np.array([[0, 1], [0, np.inf]])
        mu_11 = np.array([2, 3, -1, 1]); var_11 = np.array([1, 1, 1, 1])
        sigma = [sigma_00, sigma_01, sigma_10, sigma_11]
        mu    = [mu_00, mu_01, mu_10, mu_11]
        var   = [var_00, var_01, var_10, var_11]
    elif scenario == 2:
        M = 3
        R = np.array([4, 3, 3])
        sigma_000 = None;                 mu_000 = np.array([0]);     var_000 = np.array([1])
        sigma_001 = np.array([[1]]);      mu_001 = np.array([-2]);    var_001 = np.array([1])
        sigma_010 = np.array([[1]]);      mu_010 = np.array([-1.5]);  var_010 = np.array([1])
        sigma_011 = np.array([[1], [0]]); mu_011 = np.array([-1, 2]); var_011 = np.array([1, 1])
        sigma_100 = np.array([[0, 1]]);   mu_100 = np.array([-1.5, 1]); var_100 = np.array([1, 1])
        sigma_101 = np.array([[0, 1], [0, np.inf]])
        mu_101 = np.array([-0.5, 2.5, 1.5, -2.5]); var_101 = np.array([1, 1, 1, 1])
        sigma_110 = np.array([[0, 1], [1, np.inf]]); mu_110 = np.array([0, -2.5]); var_110 = np.array([1, 1])
        sigma_111 = np.array([[0, 1], [1, np.inf], [0, np.inf]])
        mu_111 = np.array([3, -0.5, -1.5, -2]); var_111 = np.array([1, 1, 1, 1])
        sigma = [sigma_000, sigma_001, sigma_010, sigma_011,
                 sigma_100, sigma_101, sigma_110, sigma_111]
        mu    = [mu_000, mu_001, mu_010, mu_011, mu_100, mu_101, mu_110, mu_111]
        var   = [var_000, var_001, var_010, var_011, var_100, var_101, var_110, var_111]
    else:
        raise ValueError(f"scenario must be 1 or 2, got {scenario}")
    mu  = [np.asarray(m, dtype=float) * mu_scale for m in mu]
    var = [np.asarray(v, dtype=float) * var_scale for v in var]
    return M, R, sigma, mu, var


def build_model_space(scenario=1, num_samples_per_feature=500, lamb=1, data_seed=42,
                      mu_scale=1.0, var_scale=1.0, build_neighbors=True):
    """Enumerate the scenario's partition space and return, as flat arrays over
    the enumerated states:

        log_pi  : (P,) unnormalised log target (g-prior log posterior)
        neigh   : (P, max_deg) int padded 1-flip neighbour indices (-1 = pad)
        deg     : (P,) neighbour counts
        states  : list of the underlying State objects (for reference)
    """
    M, R, sigma, mu, var = scenario_config(scenario, mu_scale=mu_scale, var_scale=var_scale)

    profiles, profile_map = hasse.enumerate_profiles(M)
    all_policies = hasse.enumerate_policies(M, R)
    num_policies = len(all_policies)
    g = num_policies * num_samples_per_feature  # unit-information prior, g = n

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

    # 1-flip neighbour graph (index space). Skippable for a fast perplexity/RPS
    # preview (this is the expensive part on the 65536-state scenario 2).
    sig2idx = {AIS.state_signature(s): i for i, s in enumerate(states)}
    if build_neighbors:
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
    else:
        neigh, deg = None, None

    # g-prior loss coefficients L_gp = A*SSE + B*|Pi|  (= -log_score_s_gprior),
    # and reg* that makes RAggregate's MSE loss proportional to L_gp:
    #   Q_mse = SSE/N + reg*.h = L_gp / (A*N)   ->   same RPS as the g-prior.
    N = int(y.shape[0])
    A = g / (2.0 * sigma2 * (1.0 + g))
    B = lamb + 0.5 * math.log(1.0 + g)
    reg_star = B / (A * N)

    return dict(states=states, log_pi=log_pi, neigh=neigh, deg=deg, P=P,
                sig2idx=sig2idx, profiles=profiles, D=D, y=y, M=M, R=R,
                A=A, B=B, N=N, reg_star=reg_star)


# ---------------------------------------------------------------------------
# 2a. RPS from RAggregate at a Rashomon threshold (the real pipeline)
# ---------------------------------------------------------------------------
def rps_via_raggregate(ms, theta_gap=None, theta_ra=None):
    """Compute the RPS at a posterior-gap Rashomon threshold using the ACTUAL
    RAggregate branch-and-bound, run with reg* so its MSE loss equals the
    g-prior loss L_gp = A*SSE + B*|Pi| (up to the constant 1/(A*N)).

    Provide EITHER theta_gap (g-prior nats above the MAP) OR theta_ra (the raw
    absolute threshold you'd pass RAggregate, e.g. the ~1.02 production values).
    Returns (S, L1, theta_RA) where S, L1 are index arrays into the enumerated
    space and theta_RA = (L_gp_min + theta_gap)/(A*N). On the tiny enumerable
    space this is identical to thresholding L_gp directly -- validated
    5-for-5 / 34-for-34 earlier -- but it exercises the real code path.
    """
    from rashomon import aggregate

    A, B, N = ms["A"], ms["B"], ms["N"]
    reg_star = ms["reg_star"]
    # L_gp(Pi) = -log_score_s_gprior; MAP = min over the enumerated space.
    L_gp = -ms["log_pi"]
    MAP = float(L_gp.min())
    # RAggregate thresholds Q_mse <= theta_mse, and Q_mse = L_gp/(A*N):
    if theta_ra is not None:
        theta_mse = theta_ra                        # absolute threshold given directly
    else:
        theta_mse = (MAP + theta_gap) / (A * N)     # from posterior gap above the MAP

    R_set, R_profiles = aggregate.RAggregate(
        ms["M"], ms["R"], np.inf, ms["D"], ms["y"], theta_mse,
        reg=reg_star, verbose=False, num_workers=1)
    rps_states = AIS.raggregate_to_states((R_set, R_profiles), ms["profiles"])

    sig2idx = ms["sig2idx"]
    S = sorted({sig2idx[AIS.state_signature(s)] for s in rps_states
                if AIS.state_signature(s) in sig2idx})
    S = np.array(S, dtype=int)

    neigh, deg = ms["neigh"], ms["deg"]
    Sset = set(S.tolist())
    L1 = set()
    for i in S:
        for slot in range(deg[i]):
            j = int(neigh[i, slot])
            if j not in Sset:
                L1.add(j)
    return S, np.array(sorted(L1), dtype=int), theta_mse


def make_p0_bucket(S, L1, log_pi, P, eps1, eps2):
    """Production-style eps1/eps2 seed proposal (make_p0_buckets_weighted_S0),
    unified so it covers the two- and three-region cases:

        eps1        mass on the RPS S, distributed proportional to the posterior
        eps2 - eps1 mass on L1(S), the 1-flip neighbours, uniform
        1 - eps2    mass on the complement, uniform

    Setting eps2 == eps1 collapses the L1 bucket to zero width -> TWO regions:
    `eps1` on the RPS + `(1 - eps1)` uniform over EVERYTHING else (L1 folded into
    the complement, so it is not a zero-mass hole). eps2 > eps1 gives the full
    three-region production proposal with extra emphasis on the neighbours.
    Always covers the whole space (p0 > 0 everywhere -> finite chi^2)."""
    wS = np.exp(log_pi[S] - log_pi[S].max()); wS /= wS.sum()
    p0 = np.zeros(P)
    p0[S] = eps1 * wS

    if eps2 > eps1 + 1e-12 and len(L1) > 0:
        # three-region: RPS / L1 / complement
        comp = np.ones(P, dtype=bool); comp[S] = False; comp[L1] = False
        nC = int(comp.sum())
        p0[L1] = (eps2 - eps1) / len(L1)
        if nC > 0:
            p0[comp] = (1.0 - eps2) / nC
    else:
        # two-region: RPS + uniform over everything else (L1 in the complement)
        notS = np.ones(P, dtype=bool); notS[S] = False
        nR = int(notS.sum())
        if nR > 0:
            p0[notS] = (1.0 - eps1) / nR
    return p0 / p0.sum()


# ---------------------------------------------------------------------------
# 2b. Seed-set bucket proposal p0(eps) and its exact chi^2 divergence
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


def build_seed_rps(log_pi, neigh, deg, gap):
    """True g-prior RPS: S = {Pi : L_gp(Pi) <= L_gp,min + gap}, i.e. every
    partition within `gap` log-posterior units of the MAP (posterior-gap
    Rashomon set). L1 = its 1-flip neighbours. This is the actual Rashomon set
    under the g-prior score, not a top-k stand-in."""
    L = -log_pi
    L = L - L.min()
    S = set(np.where(L <= gap)[0].tolist())
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
    """Exact chi^2(pi || p0) = sum_i pi_i^2 / p0_i - 1  (pi normalised).
    Clamped at 0 (it is nonnegative; can go slightly negative from rounding when
    p0 == pi, e.g. when the RPS covers the whole space)."""
    pi = np.exp(log_pi - log_pi.max())
    pi = pi / pi.sum()
    return max(0.0, float(np.sum(pi ** 2 / p0) - 1.0))


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
    ap = argparse.ArgumentParser(description="AIS efficiency vs seed quality (g-prior partition space)")
    ap.add_argument("--scenario", type=int, default=1, choices=[1, 2],
                    help="1 = M=2,R=[4,3] (64 states); 2 = M=3,R=[4,3,3] (65536 states)")
    ap.add_argument("--num_samples", type=int, default=4,
                    help="samples per policy. LOW by default so the g-prior posterior is SPREAD "
                         "(not a point mass); n=500 (production) collapses it to the MAP and the "
                         "sweep degenerates (constant chi^2, all T_min=1).")
    ap.add_argument("--mu_scale", type=float, default=0.25,
                    help="shrink pool-mean separations (<1 makes poolings more ambiguous -> spreads "
                         "the posterior); default 0.25 pairs with the low num_samples above.")
    ap.add_argument("--var_scale", type=float, default=1.0,
                    help="inflate noise SD. NOTE: near-no-op -- the g-prior normalises by the "
                         "estimated sigma^2, so uniform var scaling cancels out.")
    ap.add_argument("--theta_gaps", type=str,
                    default="0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4",
                    help="comma-separated Rashomon thresholds (posterior gap from the MAP, in nats). "
                         "Each gives a different RPS via RAggregate(reg*) -> a different seed proposal. "
                         "Keep them in the transition range (RPS grows from 1 to covering the "
                         "posterior); above that chi^2 saturates and T_min floors at 1.")
    ap.add_argument("--theta_RA", type=str, default=None,
                    help="comma-separated ABSOLUTE RAggregate thresholds (e.g. production's "
                         "1.018,1.02,1.022,1.024,1.026). Overrides --theta_gaps. NOTE: theta_RA is "
                         "n-specific -- it only reproduces the production RPS when run at the SAME "
                         "num_samples (=500), which is the point-mass regime.")
    ap.add_argument("--eps1", type=float, default=0.8,
                    help="mass on the RPS S (~ posterior). Same eps1 as the production bucket.")
    ap.add_argument("--eps2", type=float, default=0.8,
                    help="cumulative mass on S+L1. eps2==eps1 (default) -> TWO regions (RPS + "
                         "uniform rest, our current proposal); eps2>eps1 -> production 3-region "
                         "bucket with extra emphasis on the 1-flip neighbours L1.")
    ap.add_argument("--T_grid", type=str,
                    default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256",
                    help="comma-separated numbers of annealing levels to try")
    ap.add_argument("--n_paths", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=5, help="repetitions averaged per (eps, T)")
    ap.add_argument("--moves_per_level", type=int, default=5)
    ap.add_argument("--schedule_power", type=float, default=3.0,
                    help="inverse-temp ladder beta_j=(j/T)^power; >1 packs levels near beta=0")
    ap.add_argument("--target_ess", type=float, default=0.5, help="relative-ESS target for minimal T")
    ap.add_argument("--target_perplexity", type=float, default=0.0,
                    help="0 = FAITHFUL untempered g-prior posterior; >0 = temper the target to "
                         "spread over ~this many states (illustration only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="../Figures")
    args = ap.parse_args()

    use_ra = args.theta_RA is not None
    if use_ra:
        sweep_vals = [float(t) for t in args.theta_RA.split(",")]   # absolute RAggregate thresholds
    else:
        sweep_vals = [float(t) for t in args.theta_gaps.split(",")]  # posterior gaps above the MAP
    T_grid = [int(t) for t in args.T_grid.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Building model space (scenario {args.scenario}) ...")
    ms = build_model_space(scenario=args.scenario, num_samples_per_feature=args.num_samples,
                           mu_scale=args.mu_scale, var_scale=args.var_scale)
    neigh, deg, P = ms["neigh"], ms["deg"], ms["P"]
    if args.target_perplexity and args.target_perplexity > 0:
        log_pi = temper_to_perplexity(ms["log_pi"], args.target_perplexity)
        mode = f"tempered to perplexity {args.target_perplexity:g}"
    else:
        log_pi = ms["log_pi"] - ms["log_pi"].max()   # FAITHFUL: true g-prior posterior
        mode = "FAITHFUL untempered g-prior posterior"
    w = np.exp(log_pi); w /= w.sum()
    perp = math.exp(-np.sum(w * np.log(w + 1e-300)))
    print(f"  |P| = {P} partitions | degrees: min={deg.min()} max={deg.max()} mean={deg.mean():.1f}")
    print(f"  target: {mode}  ->  perplexity {perp:.2f} of {P} states")
    if perp < 2.0:
        print(f"  !! WARNING: posterior is near a POINT MASS (perplexity {perp:.2f}). The theta "
              f"sweep will degenerate (chi^2 ~constant, all T_min=1). Lower --num_samples "
              f"(and/or --mu_scale), or use --target_perplexity to temper.", flush=True)
    print(f"  reg* = {ms['reg_star']:.6e}  (A={ms['A']:.4g}, B={ms['B']:.4g}, N={ms['N']})")
    _regions = "2-region (RPS + uniform rest)" if args.eps2 <= args.eps1 + 1e-12 else "3-region (RPS/L1/complement)"
    print(f"  proposal: bucket eps1={args.eps1} eps2={args.eps2}  -> {_regions}")

    # For each Rashomon threshold theta_gap: RPS via RAggregate(reg*), build the
    # seed proposal from it, and record its exact chi^2 to the posterior.
    print("\nBuilding RPS + proposal per Rashomon threshold (RAggregate reg*)"
          f" [sweeping {'theta_RA' if use_ra else 'theta_gap'}] ...")
    A, N = ms["A"], ms["N"]; MAP = float((-ms["log_pi"]).min())
    theta_gaps = sweep_vals   # dict keys are the swept values (gaps, or theta_RA if --theta_RA)
    p0s, chi2s, rps_size, theta_RA = {}, {}, {}, {}
    for th in sweep_vals:
        if use_ra:
            S, L1, th_RA = rps_via_raggregate(ms, theta_ra=th)
        else:
            S, L1, th_RA = rps_via_raggregate(ms, theta_gap=th)
        gap_true = A * N * th_RA - MAP              # posterior gap (nats) actually realised
        p0 = make_p0_bucket(S, L1, log_pi, P, args.eps1, args.eps2)
        p0s[th] = p0
        chi2s[th] = chi2_divergence(log_pi, p0)
        rps_size[th] = len(S)
        theta_RA[th] = th_RA
        print(f"  theta_RA={th_RA:.6g} gap={gap_true:<8.3g} |RPS|={len(S):<6d} "
              f"|L1|={len(L1):<6d} chi2={chi2s[th]:.3f}", flush=True)

    # sweep T x theta (averaged over reps)
    print("\nSweeping annealing levels T for each threshold ...")
    ess_curve = {th: [] for th in theta_gaps}
    for th in theta_gaps:
        log_p0 = np.log(np.maximum(p0s[th], 1e-300))
        for T in T_grid:
            vals = [relative_ess_AIS(log_p0, log_pi, neigh, deg, T, args.n_paths,
                                     args.moves_per_level, rng,
                                     schedule_power=args.schedule_power)
                    for _ in range(args.reps)]
            ess_curve[th].append(float(np.mean(vals)))
        print(f"  theta={th:<7g} chi2={chi2s[th]:9.3f}  relESS@Tmax={ess_curve[th][-1]:.3f}", flush=True)

    # minimal T to reach the target relative ESS (linear interp between grid pts)
    def minimal_T(th):
        ys = np.array(ess_curve[th]); xs = np.array(T_grid, dtype=float)
        hit = np.where(ys >= args.target_ess)[0]
        if len(hit) == 0:
            return np.nan
        i = hit[0]
        if i == 0:
            return float(xs[0])
        x0, x1, y0, y1 = xs[i-1], xs[i], ys[i-1], ys[i]
        return float(x0 + (args.target_ess - y0) * (x1 - x0) / (y1 - y0)) if y1 > y0 else float(x1)

    Tmin = {th: minimal_T(th) for th in theta_gaps}

    # ---- summary table ----
    print("\n" + "=" * 82)
    print(f"Summary (target relative ESS = {args.target_ess}) | reg*={ms['reg_star']:.4e}, A*N={ms['A']*ms['N']:.1f}")
    print("=" * 82)
    print(f"{'theta_gap':>10}{'theta_RA':>14}{'|RPS|':>7}{'chi^2':>12}{'sqrt(chi^2)':>13}{'minimal T':>12}")
    for th in theta_gaps:
        print(f"{th:>10g}{theta_RA[th]:>14.6g}{rps_size[th]:>7d}"
              f"{chi2s[th]:>12.3f}{math.sqrt(chi2s[th]):>13.3f}{Tmin[th]:>12.2f}")

    # ---- figure 1: relative ESS vs T ----
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(theta_gaps)))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for th, c in zip(theta_gaps, colors):
        ax.plot(T_grid, ess_curve[th], "-o", color=c, ms=4, lw=1.6,
                label=f"theta={th:g}  (|RPS|={rps_size[th]}, chi2={chi2s[th]:.1f})")
    ax.axhline(args.target_ess, color="k", ls="--", lw=1.1, alpha=0.7,
               label=f"target relESS={args.target_ess}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("annealing levels  T")
    ax.set_ylabel("relative ESS")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"AIS efficiency vs Rashomon threshold  (scenario {args.scenario}, {P}-partition space)")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    f1 = os.path.join(args.outdir, f"sim_efficiency_s{args.scenario}_ess_vs_T.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight")

    # ---- figure 2: minimal T vs sqrt(chi^2) ----
    xs = np.array([math.sqrt(chi2s[t]) for t in theta_gaps])
    ys = np.array([Tmin[t] for t in theta_gaps])
    good = np.isfinite(ys)
    fig2, ax2 = plt.subplots(figsize=(6.4, 5))
    ax2.scatter(xs[good], ys[good], c=colors[good], s=70, edgecolor="k", zorder=5)
    for t, x, yv in zip(theta_gaps, xs, ys):
        if np.isfinite(yv):
            ax2.annotate(f"θ={t:g}", (x, yv), textcoords="offset points",
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
    ax2.set_title(f"Minimal annealing length scales with $\\sqrt{{\\chi^2}}$  (scenario {args.scenario})")
    ax2.grid(alpha=0.3, ls=":")
    fig2.tight_layout()
    f2 = os.path.join(args.outdir, f"sim_efficiency_s{args.scenario}_Tmin_vs_chi.png")
    fig2.savefig(f2, dpi=150, bbox_inches="tight")

    print(f"\nSaved figures:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
