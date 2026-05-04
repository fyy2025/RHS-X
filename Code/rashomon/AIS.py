from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, List, Sequence, Set, Dict, Union, Any
import numpy as np
import math, random
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- bring your repo primitives (edit path as needed) ---
# compute_Q(D, y, sigma, policies, policy_means, reg=1, normalize=0, lattice_edges=None) -> float
from rashomon.loss import compute_Q as _compute_Q
from rashomon.aggregate.raggregate import RAggregate          # (variant A)
from rashomon import hasse, extract_pools, loss, quantiles

###
# 1) State + anchor materialization
###

class ProfilePart: 
    """new class for profile partitions"""
    cov_ids: List[int]        # active covariate indices for THIS profile
    B: np.ndarray  

    def __init__(self, cov_ids, B):
        self.cov_ids = cov_ids
        self.B = B

State = List[ProfilePart]  # length P; each item is an (M,R) int matrix

def states_equal(a: State, b: State) -> bool:
    if len(a)!=len(b): return False
    for ap,bp in zip(a,b):
        if ap.cov_ids != bp.cov_ids: return False
        if not np.array_equal(ap.B, bp.B): return False
    return True

def build_anchor_states(R_set, R_profiles, M: int, R: int) -> List[State]:
    """
    R_set[g]: list[int] of length P (which candidate per profile)
    R_profiles[p][i]: (M,R) matrix for profile p, candidate i
    Returns: list of states; each state is [Sigma_0, ..., Sigma_{P-1}], Sigma_p ∈ ℤ^{M×R}.
    """
    P = len(R_profiles)
    anchors: List[State] = []
    profiles, profile_map = hasse.enumerate_profiles(int(math.log2(P)))
    for idx_vec in R_set:
        if len(idx_vec) != P:
            raise ValueError(f"R_set entry length {len(idx_vec)} != P={P}")
        state = []
        for p, sel in enumerate(idx_vec):
            cand = R_profiles[p].sigma[int(sel)]
            state.append(ProfilePart(cov_ids=profiles[p], B=cand))
        anchors.append(state)
        # # dedup by bytes of concatenated matrices
        # key = b"".join(S.tobytes() for S in state)
        # if key not in seen:
        #     anchors.append([S.copy() for S in state])
        #     seen.add(key)
    return anchors

# ---------- Flipping ones/zeros at one position inside a profile ----------

def _unit_shift_row_neighbors(row: np.ndarray, need: Optional[int] = None) -> List[np.ndarray]:
    """
    NEW BEHAVIOR (bit flip):
      Generate neighbors by flipping exactly ONE finite entry in `row`:
      0 -> 1  or  1 -> 0.  +inf entries are NOT touched.

    Args
    ----
    row  : 1D array of length C; dtype float (may contain +inf sentinels)
    need : if provided, only columns j < need are considered flippable
           (useful when different arms have different (R_m - 2))

    Returns
    -------
    neighbors : list of 1D arrays (copies) with exactly one position flipped
    """
    r = np.asarray(row, dtype=float).ravel()
    C = r.size

    # finite positions are flippable; optionally restrict to first `need` columns
    finite = np.isfinite(r)
    if need is not None:
        mask = np.zeros(C, dtype=bool)
        mask[:int(need)] = True
        finite &= mask

    idxs = np.flatnonzero(finite) # which entries are not inf and therefore can be flipped
    if idxs.size == 0:
        return []

    nbrs = []
    for j in idxs:
        new_r = r.copy()
        new_r[j] = 0.0 if (new_r[j] > 0.0) else 1.0
        nbrs.append(new_r)
    return nbrs

def profile_part_neighbors_ubs(part: ProfilePart, min_len=1) -> List[ProfilePart]:
    '''Neighbour of a profile, iterating over all rows of B, extract neighbors of every row'''
    C, R = part.B.shape
    out=[]
    for r in range(C): # repeat of finding neighbour for each row
        b = part.B[r,:]
        # while math.isinf(b[-1]): # get rid of inf to compute boundary
        #     b = b[:-1]
        # if len(b)<=1: continue
        
        for nb_row in _unit_shift_row_neighbors(b):
            if nb_row is None: continue
            nbB = part.B.copy()
            # while (len(nbB[r,:]) > len(nb_row)): # pad inf back to keep format
            #     nb_row.append(float('inf'))
            nbB[r,:] = nb_row
            out.append(ProfilePart(cov_ids=list(part.cov_ids), B=nbB))
    return out

def _copy_B(B):
    if B is None:
        return None
    if isinstance(B, np.ndarray):
        return B.copy()
    if isinstance(B, (list, tuple)):
        return [np.asarray(r).copy() for r in B]
    # fallback: try to array-copy
    return np.asarray(B).copy()

def state_neighbors_ubs(state: List[ProfilePart], min_len: int = 1) -> List[List[ProfilePart]]:
    """Neighborhood of a state, iterate over all profile"""
    neigh = []
    for p, part in enumerate(state):
        if p==0:
            continue
        # enumerate neighbors for THIS profile
        for nb in profile_part_neighbors_ubs(part, min_len=min_len):
            y_new = state.copy()  # new list object (shallow copy)
            # enforce tuple for cov_ids; deep-copy B to avoid aliasing
            cov_tup = None if nb.cov_ids is None else tuple(nb.cov_ids)
            B_copy  = _copy_B(nb.B)
            y_new[p] = ProfilePart(cov_ids=cov_tup, B=B_copy)

            # skip exact self
            if states_equal(state, y_new):
                continue

            neigh.append(y_new)

    # dedupe neighbors (important if multiple flips lead to same global state)
    seen=set(); uniq=[]
    for s in neigh:
        sig=state_signature(s)
        if sig in seen: 
            continue
        seen.add(sig)
        uniq.append(s)
    return uniq


###
# 2) warm start proposal p0
###

# ---------- helpers to make minimal changes ----------
def _softmax_logalpha(log_alpha: List[float], tau: float = 1.0) -> np.ndarray:
    z = np.array(log_alpha, float) / max(1e-12, tau)
    m = float(np.max(z))
    w = np.exp(z - m)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

def _copy_state(x: State) -> State:
    out=[]
    for pp in x:
        cov = None if pp.cov_ids is None else tuple(pp.cov_ids)
        if pp.B is None: Bc=None
        else:
            arr = np.asarray(pp.B)
            Bc = arr.copy() if isinstance(pp.B, np.ndarray) else [np.asarray(r).copy() for r in pp.B]
        out.append(type(pp)(cov_ids=cov, B=Bc))
    return out

def state_signature(s: State) -> Tuple:
    sig=[]
    for pp in s:
        cov = None if pp.cov_ids is None else tuple(pp.cov_ids)
        if pp.B is None: Bsig=('None',)
        else:
            A = np.asarray(pp.B, float)
            Bsig=(A.shape, tuple(A.ravel()))
        sig.append((cov, Bsig))
    return tuple(sig)

def _count_profile_bits(pp, R_per: np.ndarray) -> int: # for each profile, how many non-inf elements in the sigma matrices
    if pp.cov_ids is None: return 0
    cov = np.asarray(pp.cov_ids, int)
    C_per = np.maximum(np.asarray(R_per, int) - 2, 0) #length of P, each element representing # of the possible boundary for one feature
    return int(C_per[cov==1].sum()) # sum of possible boundary slot across all active features

def _total_space_size(example_state: State, R_per: np.ndarray) -> int: # for the whole space, how many possible partitions (distinct sigma matrices)
    tot = 1
    for pp in example_state:
        Ck = _count_profile_bits(pp, R_per)
        tot *= (1 << Ck) # raise to the power of 2
    return tot # sum of possible partitions within each profile

@dataclass
class P0BucketsWeightedS0:
    eps1: float
    eps2: float
    S0_sigs: Set[Tuple]
    S1_sigs: Set[Tuple]
    S1_map: Dict[Tuple, State]          # optional: sig -> concrete neighbor (used if sampling S1)
    size_S1: int
    size_Sgt1: int
    S0_logprob: Dict[Tuple, float]      # NEW: per-RPS state log prob ∝ exp(log_alpha)

# count the size of 1-edit, whole model space, to get prob for warm start proposal
def make_p0_buckets_weighted_S0(RPS: List[State],
                                R_per: np.ndarray,
                                log_alpha: List[float],
                                eps1: float = 0.6,
                                eps2: float = 0.85,
                                min_len: int = 1) -> P0BucketsWeightedS0:
    # S0: RPS signatures
    S0_sigs = [state_signature(s) for s in RPS]
    # normalize weights inside S0 using your existing log_alpha
    p_rps = _softmax_logalpha(log_alpha)                            # CHANGED: reuse log_alpha
    S0_logprob = {sig: math.log(p_rps[i]) for i, sig in enumerate(S0_sigs)}

    # S1: union of 1-edit neighbors (unique), excluding S0
    S0_set = set(S0_sigs)
    S1_sigs: Set[Tuple] = set()
    S1_map: Dict[Tuple, State] = {}
    for s in RPS:
        for n in state_neighbors_ubs(s, min_len=min_len):
            sig = state_signature(n)
            if sig in S0_set or sig in S1_sigs: continue
            S1_sigs.add(sig); S1_map[sig] = n

    size_Omega = _total_space_size(RPS[0], np.asarray(R_per, int))
    size_S1 = len(S1_sigs)
    size_Sgt1 = max(0, size_Omega - len(S0_sigs) - size_S1)

    return P0BucketsWeightedS0(
        eps1=eps1, eps2=eps2,
        S0_sigs=set(S0_sigs),
        S1_sigs=S1_sigs, S1_map=S1_map,
        size_S1=size_S1, size_Sgt1=size_Sgt1,
        S0_logprob=S0_logprob
    )

def log_p0_distance_weighted_S0(x: State, buckets: P0BucketsWeightedS0) -> float:
    sig = state_signature(x)
    if sig in buckets.S0_sigs: # sum of prob for all states within RPS equals eps1
        # mass eps1 distributed by alpha within S0  (not uniform)
        return math.log(buckets.eps1) + buckets.S0_logprob[sig]      # CHANGED: weighted by log_alpha
    if sig in buckets.S1_sigs: # sum of prob for all states within 1-edit neighborhood equals eps2-eps1
        if buckets.size_S1 == 0: return float("-inf")
        return math.log(buckets.eps2 - buckets.eps1) - math.log(buckets.size_S1)
    if buckets.size_Sgt1 == 0: return float("-inf") # sum of prob for all other states equals 1-eps2
    return math.log(1.0 - buckets.eps2) - math.log(buckets.size_Sgt1)

def _editable_indices(pp, R_per: np.ndarray) -> List[Tuple[int,int]]:
    """
    Return list of (row_in_B, j) positions that are real interior bits
    for this profile's compact matrix B (i.e., finite, within width).
    """
    if pp.cov_ids is None:
        return []
    cov = np.asarray(pp.cov_ids, int)
    C_per = np.maximum(np.asarray(R_per, int) - 2, 0)
    idxs = []
    if pp.B is None:
        # no interior bits; return empty
        return idxs
    B = pp.B
    if isinstance(B, np.ndarray):
        B2 = B if B.ndim == 2 else B.reshape(1, -1)
    else:
        rows = [np.asarray(r).ravel() for r in B]
        maxlen = max((r.size for r in rows), default=0)
        rows = [r if r.size == maxlen else np.pad(r, (0, maxlen - r.size)) for r in rows]
        B2 = np.vstack(rows) if rows else np.zeros((0, 0), float)

    r = 0
    for m in range(len(cov)):
        if cov[m] != 1:
            continue
        c = int(C_per[m])
        if c == 0:
            continue
        if r < B2.shape[0]:
            width = min(c, B2.shape[1])
            for j in range(width):
                if not np.isinf(B2[r, j]):
                    idxs.append((r, j))
        r += 1
    return idxs

def random_state_jitter_from_RPS(RPS: List[State], R_per: np.ndarray, k_flips: int = 3) -> State:
    """
    Pick a random RPS state, then flip up to k random interior bits across profiles.
    If a profile has no interior bits, it's skipped.
    """
    if not RPS:
        raise ValueError("RPS is empty.")
    x = _copy_state(random.choice(RPS))
    # Gather editable indices per profile
    editable = []
    for p, pp in enumerate(x):
        idxs = _editable_indices(pp, R_per)
        for (r, j) in idxs:
            editable.append((p, r, j))
    if not editable:
        return x
    k = min(k_flips, len(editable))
    for (p, r, j) in random.sample(editable, k):
        # ensure B is ndarray 2D
        B = x[p].B
        if B is None:
            continue
        if isinstance(B, np.ndarray):
            if B.ndim == 1:
                B = B.reshape(1, -1)
        else:
            # list → ndarray
            rows = [np.asarray(row).ravel() for row in B]
            W = max(rw.size for rw in rows) if rows else 0
            rows = [rw if rw.size == W else np.pad(rw, (0, W - rw.size), constant_values=np.inf) for rw in rows]
            B = np.vstack(rows).astype(float)
        # flip if finite (ignore +inf padding)
        if not np.isinf(B[r, j]):
            B[r, j] = 1.0 - (1.0 if B[r, j] != 0.0 else 0.0)
        x[p] = type(x[p])(cov_ids=x[p].cov_ids, B=B)
    return x

def sample_p0(buckets: P0BucketsWeightedS0, RPS: List[State], R_per: np.ndarray) -> State:
    rng = np.random.default_rng()
    rn = rng.random()
    s = _copy_state(random.choice(RPS))
    if rn > buckets.eps2:
        beyond = False
        while beyond == False:
            s = _copy_state(random_state_jitter_from_RPS(RPS, R_per, 3))
            sig = state_signature(s)
            if sig in buckets.S0_sigs or sig in buckets.S1_sigs: 
                continue
            else:
                beyond = True
    elif rn >= buckets.eps1:
        sig = random.choice(tuple(buckets.S1_sigs))
        s = _copy_state(buckets.S1_map[sig])

    return s


###
# 3) Compute loss of a partition to calculate posterior
###

# ---------- 1) Q when sigma is None: one-pool predictor (mean of y_k) ----------
def _compute_Q_none_mean(D_k: np.ndarray, y_k: np.ndarray, *, reg: float, normalize: int) -> float:
    yv = y_k.ravel()
    if yv.size:
        mu = float(np.mean(yv))
        mse = float(np.mean((yv - mu) ** 2))
        if normalize:
            mse = mse * yv.size / float(normalize)
    else:
        mse = 0.0
    h = 1  # one pool
    return mse + reg * h

# ---------- 2) Global loss: sum per-profile losses; DO NOT touch non-None sigma ----------
RLike = Union[int, Sequence[int], np.ndarray]

## to use compute_Q, need to full (M × C) sigma with C = max(R_m - 2)
def assemble_sigma_full_for_profile(
    part,
    M: int,                            # number of features/arms
    R: Sequence[int],                  # per-arm levels, length M
) -> np.ndarray:
    """
    Build a full (M × C) sigma with C = max(R_m - 2).
    For arm m, only the first (R_m - 2) entries are meaningful; the rest are +inf.
    Active arms (cov_ids[m] == 1) get their row from B (padded/truncated as needed).
    Inactive arms get +inf.
    """
    if part is None:
        return None
    
    B = getattr(part, "B", None)
    cov_ids = getattr(part, "cov_ids", None)

    R = np.asarray(R, dtype=int)
    assert R.shape[0] == M, "R must be length M"
    # internal boundary count per arm
    C_per = np.maximum(R - 2, 0)
    C = int(C_per.max()) if M > 0 else 0

    # full matrix filled with +inf (inactive/no-boundary sentinel)
    Sigma = np.full((M, C), np.inf, dtype=float)

    # handle trivial cases
    if C == 0:
        return Sigma  # nothing to place; all rows are effectively inactive

    # binary mask of actives
    if cov_ids is None:
        active_mask = np.zeros(M, dtype=int)
    else:
        active_mask = np.asarray(cov_ids, dtype=int).reshape(-1)
        if active_mask.size != M:
            raise ValueError("cov_ids must be a binary mask of length M")

    active_idx = np.nonzero(active_mask)[0]
    A = active_idx.size
    if A == 0 or B is None:
        return Sigma  # all inactive or no B provided

    # normalize B to a list of row arrays, one per active arm (in ascending arm index)
    rows = []

    # Case 1: 1D → only one active arm
    if isinstance(B, np.ndarray) and B.ndim == 1:
        if A != 1:
            raise ValueError("B is 1D but multiple active arms in cov_ids")
        rows = [np.asarray(B, dtype=float)]

    # Case 2: 2D → rows correspond to active arms in the order of active_idx
    elif isinstance(B, np.ndarray) and B.ndim == 2:
        if B.shape[0] != A:
            raise ValueError(f"B has {B.shape[0]} rows; expected {A} (number of active arms)")
        rows = [np.asarray(B[r], dtype=float) for r in range(A)]

    # Case 3: ragged list/tuple of rows
    elif isinstance(B, (list, tuple)):
        if len(B) != A:
            raise ValueError(f"len(B)={len(B)}; expected {A} (number of active arms)")
        rows = [np.asarray(row, dtype=float) for row in B]

    else:
        raise TypeError(f"Unsupported B type: {type(B)}")

    # place each active row into its arm with correct per-arm width
    for row_arr, m in zip(rows, active_idx):
        need = int(C_per[m])            # columns that are meaningful for arm m
        if need == 0:
            continue                    # arm has no internal boundary positions
        # trim/pad the source row to 'need', then place into Sigma[m, :need]
        src = row_arr.ravel()
        if src.size < need:
            src = np.pad(src, (0, need - src.size), mode="constant", constant_values=1.0)
        elif src.size > need:
            src = src[:need]
        Sigma[m, :need] = src

    return Sigma

def _compute_Q_none_mean(y_k, *, reg: float, normalize: int) -> float:
    yv = np.asarray(y_k).ravel()
    if yv.size:
        mu = float(np.mean(yv))
        mse = float(np.mean((yv - mu) ** 2))
        if normalize:
            mse = mse * yv.size / float(normalize)
    else:
        mse = 0.0
    return mse + reg * 1  # one pool

def build_profile_index_of_policy(policies, policy_to_profile):
    profs = [policy_to_profile(p) for p in policies]
    uniq = sorted(set(profs))
    prof_to_idx = {pr: i for i, pr in enumerate(uniq)}
    prof_idx = np.array([prof_to_idx[pr] for pr in profs], dtype=int)
    return prof_idx, uniq

def global_loss_raw(
    state,                      # list[ProfilePart or raw sigma/None], length = num_profiles
    D, y,                       # D: (N,1) global policy ids; y: (N,) or (N,1)
    policies,                   # list of ALL policies (len = num_policies)
    policy_means,               # (num_policies, 2)
    prof_idx_of_policy,         # np.ndarray(len=num_policies): profile index for each policy
    M: int,                     # number of features/arms in this profile family
    R,
    reg: float = 1.0,
    normalize: int = 0,
    lattice_edges=None,
) -> float:
    """Compute the loss for one global partition"""
    # coerce arrays
    normalize = D.shape[0]
    D_arr = np.asarray(D);  y_arr = np.asarray(y)
    if D_arr.ndim == 1: D_arr = D_arr.reshape(-1, 1)
    if y_arr.ndim == 1: y_arr = y_arr.reshape(-1, 1)

    total = 0.0
    num_profiles = len(state)
    num_policies = len(policies)

    # which policy ids belong to each profile?
    pol_ids_by_profile = [np.where(prof_idx_of_policy == k)[0] for k in range(num_profiles)]
    pid_global = D_arr[:, 0].astype(int)

    for k in range(num_profiles):
        part_k = state[k]
        pol_ids = pol_ids_by_profile[k]
        if pol_ids.size == 0:
            total += _compute_Q_none_mean([], reg=reg, normalize=normalize)
            continue

        mask = np.isin(pid_global, pol_ids)
        if not np.any(mask):
            total += _compute_Q_none_mean([], reg=reg, normalize=normalize)
            continue

        D_k = D_arr[mask].copy()
        y_k = y_arr[mask].copy()

        # remap policy ids in D_k to local 0..(P_k-1)
        local_map = -np.ones(num_policies, dtype=int)
        local_map[pol_ids] = np.arange(pol_ids.size, dtype=int)
        D_k[:, 0] = local_map[D_k[:, 0].astype(int)]

        policies_k = [policies[i] for i in pol_ids]
        # pm_k = np.asarray(policy_means)[pol_ids, :]
        pm_k = loss.compute_policy_means(D_k, y_k, len(policies_k))

        # Expand per-profile candidate to full M rows for compute_Q
        Sigma_k_full = assemble_sigma_full_for_profile(part_k, M, R)

        if Sigma_k_full is None:
            Q_k = _compute_Q_none_mean(y_k, reg=reg, normalize=normalize)
            # print(Q_k)
            # Sigma_k_full = np.full((M,np.max(R)-2), np.inf, float)
            # Q_k = float(_compute_Q(
            #     D=D_k, y=y_k, sigma=Sigma_k_full,
            #     policies=policies_k, policy_means=pm_k,
            #     reg=reg, normalize=normalize, lattice_edges=lattice_edges
            # ))
            # print(Q_k)
        else:
            Q_k = float(_compute_Q(
                D=D_k, y=y_k, sigma=Sigma_k_full,
                policies=policies_k, policy_means=pm_k,
                reg=reg, normalize=normalize, lattice_edges=lattice_edges
            ))
        total += Q_k
        # print(Q_k)
    return total


def global_loss_bayes(
    state,
    D, y,
    policies,
    policy_means,
    prof_idx_of_policy,
    M: int,
    R,
    reg: float = 1.0,
    lattice_edges=None,
) -> float:
    """Per-observation profile-additive Bayesian marginal log-loss.

    For each profile, compute the Lemma A2 loss
      Q_k(Π_k) = (n_k-H_k)/2 · log(SSE_k) + λ'H_k + c(Π_k),
    then return sum_k Q_k(Π_k) / n. This matches the profile-additive
    objective used by RAggregate(..., use_bayes=True).

    Use as: score_s(state) = exp(-global_loss_bayes(state, ...))
    """
    D_arr = np.asarray(D)
    y_arr = np.asarray(y)
    if D_arr.ndim == 1:
        D_arr = D_arr.reshape(-1, 1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    n = D_arr.shape[0]
    num_profiles = len(state)
    num_policies = len(policies)
    R = np.asarray(R, dtype=int)
    C = max(1, int(np.maximum(R - 2, 0).max()))
    pol_ids_by_profile = [
        np.where(prof_idx_of_policy == k)[0] for k in range(num_profiles)
    ]
    pid_global = D_arr[:, 0].astype(int)

    total_Q = 0.0

    for k in range(num_profiles):
        part_k = state[k]
        pol_ids = pol_ids_by_profile[k]
        mask = np.isin(pid_global, pol_ids)
        if not np.any(mask):
            continue

        D_k = D_arr[mask].copy()
        y_k = y_arr[mask].copy()

        local_map = -np.ones(num_policies, dtype=int)
        local_map[pol_ids] = np.arange(pol_ids.size, dtype=int)
        D_k[:, 0] = local_map[D_k[:, 0].astype(int)]

        policies_k = [policies[int(i)] for i in pol_ids]
        pm_k = loss.compute_policy_means(D_k, y_k, len(policies_k))
        Sigma_k = assemble_sigma_full_for_profile(part_k, M, R)

        if Sigma_k is None:
            Sigma_k = np.full((M, C), np.inf, dtype=float)

        Q_k = loss.compute_Q_bayes(
            D=D_k,
            y=y_k,
            sigma=Sigma_k,
            policies=policies_k,
            policy_means=pm_k,
            reg=reg,
            lattice_edges=lattice_edges,
        )
        if np.isfinite(Q_k):
            total_Q += float(Q_k)
        else:
            return float("inf")

    return total_Q / float(n)


def global_loss_raw2(
    state,                      # list[ProfilePart or raw sigma/None], length = num_profiles
    D, y,                       # D: (N,1) global policy ids; y: (N,) or (N,1)
    policies,                   # list of ALL policies (len = num_policies)
    policy_means,               # (num_policies, 2)  (kept for signature compatibility; NOT used)
    prof_idx_of_policy,         # np.ndarray(len=num_policies): profile index for each policy
    M: int,                     # number of features/arms
    R,
    reg: float = 1.0,
    normalize: int = 0,
    lattice_edges=None,         # can be None or a GLOBAL edge list; we rebuild per-profile if possible
) -> float:
    """
    Repo-faithful global loss:
      total = sum_k compute_Q(D_k_local, y_k, sigma_full_k, policies_k, policy_means_k, ...)
    Key differences vs earlier versions:
      - policy_means_k is recomputed per profile using loss.compute_policy_means (repo process)
      - degenerate sigma is represented by an all-+inf matrix (not None / mean shortcut)
      - D_k[:,0] is remapped to local indices 0..P_k-1 exactly as expected by predict/extract_pools
      - lattice edges are built per profile if you provide a builder; otherwise passed through
    """
    # imports from repo
    from rashomon import loss as _loss
    try:
        from rashomon import hasse as _hasse
    except Exception:
        _hasse = None

    normalize = D.shape[0]
    # coerce arrays
    D_arr = np.asarray(D)
    y_arr = np.asarray(y)
    if D_arr.ndim == 1:
        D_arr = D_arr.reshape(-1, 1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    R = np.asarray(R, dtype=int)
    assert R.shape[0] == M, "R must be length M"

    # width for sigma in repo convention: at least 1 col for all-inf case
    C_per = np.maximum(R - 2, 0)
    C = max(1, int(C_per.max()))

    num_profiles = len(state)
    num_policies = len(policies)

    # global policy ids per profile
    pol_ids_by_profile = [np.where(prof_idx_of_policy == k)[0].astype(int) for k in range(num_profiles)]
    pid_global = D_arr[:, 0].astype(int)

    total = 0.0

    for k in range(num_profiles):
        pol_ids = pol_ids_by_profile[k]
        if pol_ids.size == 0:
            # repo effectively contributes nothing if no policies in this profile
            continue

        mask = np.isin(pid_global, pol_ids)
        if not np.any(mask):
            # repo effectively contributes nothing if no data for this profile
            continue

        # slice data for this profile
        D_k = D_arr[mask].copy()
        y_k = y_arr[mask].copy()

        # localize policy ids 0..P_k-1
        local_map = -np.ones(num_policies, dtype=int)
        local_map[pol_ids] = np.arange(pol_ids.size, dtype=int)
        D_k[:, 0] = local_map[D_k[:, 0].astype(int)]

        # local policy list (order defines local ids)
        policies_k = [policies[int(i)] for i in pol_ids]

        # repo way: recompute policy_means on the localized D_k/y_k
        pm_k = _loss.compute_policy_means(D_k, y_k, len(policies_k))

        # sigma: expand ProfilePart to full, else treat None as all-inf
        part_k = state[k]
        Sigma_k_full = AIS.assemble_sigma_full_for_profile(part_k, M, R)  # your function
        if Sigma_k_full is None:
            Sigma_k_full = np.full((M, C), np.inf, dtype=float)

        # lattice edges: repo computes per-profile edges from policies_k; do that if possible
        edges_k = None
        if lattice_edges is None:
            edges_k = None
        elif callable(lattice_edges):
            # if you passed a builder function
            edges_k = lattice_edges(policies_k)
        else:
            # if you passed a precomputed GLOBAL edge list, we can't safely subset without a map;
            # best repo-faithful fallback is to rebuild if hasse is available.
            if _hasse is not None:
                try:
                    policies_sorted = _hasse.is_policies_sorted(policies_k)
                    # note: repo uses R-1 for lattice_edges
                    edges_k = _hasse.lattice_edges(policies_k, sorted=policies_sorted, M=M, R=R-1)
                except Exception:
                    edges_k = None
            else:
                edges_k = None

        Q_k = _loss.compute_Q(
            D=D_k,
            y=y_k,
            sigma=Sigma_k_full,
            policies=policies_k,
            policy_means=pm_k,
            reg=reg,
            normalize=normalize,
            lattice_edges=edges_k
        )
        print(Q_k)
        total += float(Q_k)

    return float(total)

# ---------- 3) AIS score: exp(-beta * global_loss_raw(state)) ----------
def make_score_s_expneg_raw(
    *,
    D,
    y,
    M,
    R,
    prof_idx_of_policy,
    policies,
    policy_means,
    reg: float = 1.0,
    normalize: int = 0,
    lattice_edges=None,
    beta: float = 1.0,                 # set 1.0 for exp(-loss)
    prior_logprob=lambda state: 0.0,   # optional log-prior
):
    """Return score = exp(-loss)"""
    def score_s(state):
        Q = global_loss_raw(
            state=state,
            D=D, y=y, M=M, R=R,
            prof_idx_of_policy = prof_idx_of_policy,
            policies=policies,
            policy_means=policy_means,
            reg=reg, normalize=normalize,
            lattice_edges=lattice_edges,
        )
        return float(np.exp(prior_logprob(state) - beta * Q))
    return score_s

###
# 4) AIS step
###

@dataclass
class AISConfig:
    n_paths:int=600; n_levels:int=40; moves_per_level:int=12; min_len:int=1; seed:Optional[int]=2

@dataclass
class AISOutput:
    terminals: List[State]; logw: np.ndarray; normw: np.ndarray; ladder: np.ndarray

def make_ladder(K:int, gamma:float=4.0) -> np.ndarray:
    if K<2: return np.array([0.0,1.0], float)
    g=np.linspace(0,1,K); return g**gamma # if gamma > 0, the ladder is denser near 0 and sparser near 1

def mh_step_state_uniform_neighbors(x: State, t: float,
                                    log_p0: Callable[[State], float],
                                    score_s: Callable[[State], float],
                                    min_len:int=1) -> State:
    """One step of MH used in each temperature step of the AIS, with the uniform neighborhood proposal"""
    N_cur = [n for n in state_neighbors_ubs(x, min_len=min_len) if not states_equal(n, x)]
    if not N_cur: return x
    prop = random.choice(N_cur)
    N_prop = [n for n in state_neighbors_ubs(prop, min_len=min_len) if not states_equal(n, prop)]
    def logpi_t(z: State) -> float:
        lq = log_p0(z)
        if lq==float("-inf"): return float("-inf")
        return (1.0 - t)*lq + t*math.log(max(1e-300, score_s(z))) # target dist determined by t
    lcur, lprop = logpi_t(x), logpi_t(prop)
    if lprop == float("-inf"): return x
    logr = (lprop-lcur) + math.log(max(1,len(N_cur))) - math.log(max(1,len(N_prop))) # matropolis hasting
    if math.log(random.random()+1e-300) < min(0.0, logr): return prop
    return x

def mh_step_state_uniform_neighbors_parallel(x: State, t: float,
                                    D, y, M, R, 
                                    prof_idx_of_policy, 
                                    all_policies, policy_means, reg,
                                    log_p0: Callable[[State], float],
                                    min_len:int=1
                                    ) -> State:
    """One step of MH used in each temperature step of the AIS, with the uniform neighborhood proposal"""
    N_cur = [n for n in state_neighbors_ubs(x, min_len=min_len) if not states_equal(n, x)]
    if not N_cur: return x
    prop = random.choice(N_cur)
    N_prop = [n for n in state_neighbors_ubs(prop, min_len=min_len) if not states_equal(n, prop)]
    # def logpi_t(z: State) -> float:
    #     lq = log_p0(z)
    #     if lq==float("-inf"): return float("-inf")
    #     return (1.0 - t)*lq + t*math.log(max(1e-300, score_s(z))) # target dist determined by t
    lcur = (1.0 - t)*log_p0(x) + t*math.log(max(1e-300, score_s_expneg_raw(x, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg)))
    lprop = (1.0 - t)*log_p0(prop) + t*math.log(max(1e-300, score_s_expneg_raw(prop, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg)))
    if lprop == float("-inf"): return x
    logr = (lprop-lcur) + math.log(max(1,len(N_cur))) - math.log(max(1,len(N_prop))) # matropolis hasting
    if math.log(random.random()+1e-300) < min(0.0, logr): return prop
    return x

def run_ais_state(anchors: List[State],
                  score_s: Callable[[State], float],
                  cfg: AISConfig = AISConfig(),
                  RPS: Optional[List[State]] = None,
                  R_per: Optional[np.ndarray] = None,
                  eps1: float = 0.05, eps2: float = 0.25,
                  tau_init: float = 1.0,
                  ladder: Optional[np.ndarray] = None) -> AISOutput:
    if cfg.seed is not None:
        np.random.seed(cfg.seed); random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # already compute this earlier in the code:
    log_alpha = [math.log(max(1e-300, score_s(A))) for A in anchors]                       # existing line in your code

    # NEW p0 built from RPS using existing log_alpha (weighted S0)
    if RPS is None or R_per is None:
        raise ValueError("Provide RPS and R_per for the distance-bucket p0.")
    buckets = make_p0_buckets_weighted_S0(RPS, np.asarray(R_per,int), log_alpha, eps1=eps1, eps2=eps2, min_len=cfg.min_len)
    log_p0 = lambda z: log_p0_distance_weighted_S0(z, buckets)     

    ladder = ladder

    terminals=[]; logw=np.zeros(cfg.n_paths,float)
    for p in range(cfg.n_paths):
        # initial x from RPS by loss weights
        x = sample_p0(buckets,RPS,R_per)
        lw = 0.0                                                     # keep simple; add init-correction if you want

        beta_prev = ladder[0]
        for beta_cur in ladder[1:]:
            # AIS weight increment
            lq = log_p0(x)
            lp = math.log(max(1e-300, score_s(x)))
            lw += (beta_cur - beta_prev) * (lp - lq) # move to next ladder, update weight
            # MH moves at level beta_cur
            for _ in range(cfg.moves_per_level):
                x = mh_step_state_uniform_neighbors(x, beta_cur, log_p0, score_s, min_len=cfg.min_len)
                # proposal step, moves_per_level steps of MH movements
            beta_prev = beta_cur

        terminals.append(_copy_state(x))
        logw[p]=lw

    m=float(np.max(logw)); w=np.exp(logw-m)
    return AISOutput(terminals, logw, w/w.sum(), ladder)


@dataclass
class AISResult:
    states: List[Any]
    logw: List[float]

import os
import json
from rashomon import MCMC

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_ais_state_streaming(
    buckets,
    anchors: List[State],
    score_s: Callable[[State], float],
    cfg: AISConfig = AISConfig(),
    RPS: Optional[List[State]] = None,
    R_per: Optional[np.ndarray] = None,
    eps1: float = 0.05, eps2: float = 0.25,
    out_jsonl: str = "AIS_samples.jsonl",
    ladder: Optional[List[State]] = None,
    tau_init: float = 1.0,
    keep_in_memory: bool = True
) -> Dict[str, List[Any]]:

    if RPS is None or R_per is None:
        raise ValueError("Provide RPS and R_per for the distance-bucket p0.")
    buckets = buckets
    log_p0 = lambda z: log_p0_distance_weighted_S0(z, buckets)     

    ladder = ladder

    terminals=[]; logw=np.zeros(cfg.n_paths,float)

    if out_jsonl and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for p in range(cfg.n_paths):
            # initial x from RPS by loss weights
            x = sample_p0(buckets,RPS,R_per)
            lw = 0.0                                                     # keep simple; add init-correction if you want

            beta_prev = ladder[0]
            for beta_cur in ladder[1:]:
                # AIS weight increment
                lq = log_p0(x)
                lp = math.log(max(1e-300, score_s(x)))
                lw += (beta_cur - beta_prev) * (lp - lq) # move to next ladder, update weight
                # MH moves at level beta_cur
                for _ in range(cfg.moves_per_level):
                    x = mh_step_state_uniform_neighbors(x, beta_cur, log_p0, score_s, min_len=cfg.min_len)
                    # proposal step, moves_per_level steps of MH movements
                beta_prev = beta_cur

            rec = {
                    "iter": p,
                    "state": MCMC._state_to_jsonable(x),
                    "unnormalized_log_weight": lw
                }
            f.write(json.dumps(rec) + "\n")
            f.flush()  # ensures it’s on disk right away

            terminals.append(_copy_state(x))
            logw[p]=lw

    m=float(np.max(logw)); w=np.exp(logw-m)
    return AISOutput(terminals, logw, w/w.sum(), ladder)

def run_ais_state_streaming_from_data(
    M, 
    R, 
    H, 
    D, 
    y, 
    theta, 
    reg,
    eps1,
    eps2,
    score_s,
    all_policies,
    policy_means,
    out_dir,
    cfg,
    tau_init: float = 1.0,
    keep_in_memory: bool = True
) -> Dict[str, List[Any]]:
    ## redo the RPS steps
    R_set, R_profiles = RAggregate(M, R, H, D, y, theta, reg, verbose=True)
    anchors = build_anchor_states(R_set, R_profiles, M, R)
    prof_idx_of_policy, profiles = build_profile_index_of_policy(all_policies, hasse.policy_to_profile)
    RPS_states = raggregate_to_states((R_set, R_profiles), profiles)
    log_alpha = [np.log(max(1e-300, score_s_expneg_raw(A, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg))) for A in anchors]
    buckets = make_p0_buckets_weighted_S0(RPS_states, np.asarray(R,int), log_alpha, eps1=eps1, eps2=eps2, min_len=cfg.min_len)
    log_p0 = partial(log_p0_distance_weighted_S0_wrapper, buckets=buckets)
    init_sampler = partial(
        init_from_RPS_batch_wrapper,
        RPS_states=RPS_states,
        log_alpha=log_alpha,
        rng_seed=777,
    )

    print("Running pilot adaptive ladder")
    ladder, ess_ratios = pilot_adaptive_ladder(
        init_sampler=init_sampler,
        log_p0=log_p0,
        score_s=score_s,
        N=512,
        ess_target=0.80,
        beta0=0.0, beta1=1.0,
        initial_delta=0.05,
        min_delta=1e-3,
        moves_per_probe=3,   # small mixing at each accepted β (optional)
        min_len=1,
        rng_seed=777
    )

    terminals=[]; logw=np.zeros(cfg.n_paths,float)
    print("Adaptive ladder has", len(ladder), "levels; first 10:", ladder[:10])
    print("Per-step ESS ratios (len =", len(ess_ratios), "):", ess_ratios[:10])

    out_jsonl = os.path.join(out_dir, f"AIS_samples_{theta}.jsonl")
    if out_jsonl and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for p in range(cfg.n_paths):
            # initial x from RPS by loss weights
            x = sample_p0(buckets,RPS_states,R)
            lw = 0.0                                                     # keep simple; add init-correction if you want

            beta_prev = ladder[0]
            for beta_cur in ladder[1:]:
                # AIS weight increment
                lq = log_p0(x)
                lp = math.log(max(1e-300, score_s(x)))
                lw += (beta_cur - beta_prev) * (lp - lq) # move to next ladder, update weight
                # MH moves at level beta_cur
                for _ in range(cfg.moves_per_level):
                    x = mh_step_state_uniform_neighbors(x, beta_cur, log_p0, score_s, min_len=cfg.min_len)
                    # proposal step, moves_per_level steps of MH movements
                beta_prev = beta_cur

            rec = {
                    "iter": p,
                    "state": MCMC._state_to_jsonable(x),
                    "unnormalized_log_weight": lw
                }
            f.write(json.dumps(rec) + "\n")
            f.flush()  # ensures it’s on disk right away

            terminals.append(_copy_state(x))
            logw[p]=lw

    m=float(np.max(logw)); w=np.exp(logw-m)
    return AISOutput(terminals, logw, w/w.sum(), ladder)


def run_ais_state_streaming_from_custom_states(
    ladder,
    init_states,        # List[State]: warm-start states (e.g. SSS-visited states)
    init_log_alpha,     # List[float]: log-scores for each init_state
    R,
    eps1,
    eps2,
    score_s,
    cfg,
    out_dir,
    label: str = "custom",  # used in output filename
) -> Dict[str, List[Any]]:
    """
    AIS seeded from an arbitrary set of states with associated log-scores.
    Identical annealing logic as run_ais_state_streaming_from_data but
    bypasses RAggregate — use for SSS-seeded AIS or any custom warm start.
    """
    buckets = make_p0_buckets_weighted_S0(
        init_states, np.asarray(R, int), init_log_alpha,
        eps1=eps1, eps2=eps2, min_len=cfg.min_len,
    )
    log_p0 = partial(log_p0_distance_weighted_S0_wrapper, buckets=buckets)
 
    terminals = []; logw = np.zeros(cfg.n_paths, float)

    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"AIS_samples_{label}.jsonl")
    if out_jsonl and os.path.exists(out_jsonl):
        try:
            os.remove(out_jsonl)
        except FileNotFoundError:
            pass

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for p in range(cfg.n_paths):
            x  = sample_p0(buckets, init_states, R)
            lw = 0.0
            beta_prev = ladder[0]
            for beta_cur in ladder[1:]:
                lq  = log_p0(x)
                lp  = math.log(max(1e-300, score_s(x)))
                lw += (beta_cur - beta_prev) * (lp - lq)
                for _ in range(cfg.moves_per_level):
                    x = mh_step_state_uniform_neighbors(
                        x, beta_cur, log_p0, score_s, min_len=cfg.min_len)
                beta_prev = beta_cur

            rec = {"iter": p, "state": MCMC._state_to_jsonable(x),
                   "unnormalized_log_weight": lw}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            terminals.append(_copy_state(x))
            logw[p] = lw

    m = float(np.max(logw)); w = np.exp(logw - m)
    return AISOutput(terminals, logw, w / w.sum(), ladder)


### Parallel processing!!

from multiprocessing import Pool, current_process

def score_s_expneg_raw(state, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg):
    Q = global_loss_raw(
        state=state,
        D=D, y=y, M=M, R=R,
        prof_idx_of_policy=prof_idx_of_policy,
        policies=all_policies,
        policy_means=policy_means,
        reg=reg,
        normalize=0,
        lattice_edges=None,
    )
    return float(np.exp(-Q))

def parallel_ais(
    out_jsonl: str,
    cfg,
    buckets,
    RPS_states,
    R,
    ladder,
    log_p0,
    x,D,y,M,
    prof_idx_of_policy, 
    all_policies, 
    policy_means, reg,
    seed
):
    
    with open(out_jsonl, "a", encoding="utf-8") as f:
        # initial x from RPS by loss weights
        random.seed(seed)
        x = sample_p0(buckets,RPS_states,R)
        lw = 0.0                                                     # keep simple; add init-correction if you want

        beta_prev = ladder[0]
        for beta_cur in ladder[1:]:
            # AIS weight increment
            lq = log_p0(x)
            lp = np.log(max(1e-300, score_s_expneg_raw(x, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg)))
            lw += (beta_cur - beta_prev) * (lp - lq) # move to next ladder, update weight
            # MH moves at level beta_cur
            for _ in range(cfg.moves_per_level):
                x = mh_step_state_uniform_neighbors_parallel(x, beta_cur,
                    D, y, M, R, 
                    prof_idx_of_policy, 
                    all_policies, policy_means, reg,
                    log_p0,
                    min_len=cfg.min_len)
                # proposal step, moves_per_level steps of MH movements
            beta_prev = beta_cur

        rec = {
                "state": MCMC._state_to_jsonable(x),
                "unnormalized_log_weight": lw
            }
        f.write(json.dumps(rec) + "\n")
        f.flush()  # ensures it’s on disk right away

    return(_copy_state(x), lw)

def parallel_ais_chunk(path_ids, out_jsonl, cfg, buckets, RPS_states, R, ladder,
                       log_p0, x, D, y, M, prof_idx_of_policy,
                       all_policies, policy_means, reg):
    terminals = []
    logw = []

    for i in path_ids:
        term_i, logw_i = parallel_ais(
            out_jsonl, cfg, buckets, RPS_states, R, ladder, log_p0, x, D, y, M,
            prof_idx_of_policy, all_policies, policy_means, reg,
            seed=i   # if useful
        )
        terminals.append(term_i)
        logw.append(logw_i)

    return terminals, logw

from functools import partial

def log_p0_distance_weighted_S0_wrapper(z, buckets):
    return log_p0_distance_weighted_S0(z, buckets)

def init_from_RPS_batch_wrapper(N, RPS_states, log_alpha, rng_seed):
    return MCMC.init_from_RPS_batch(RPS_states, log_alpha, N, rng_seed=rng_seed)

def run_ais_streaming_from_data_parallel(
    epoch,
    M, R, H, x, D, y, 
    theta, reg, eps1, eps2,
    score_s,
    all_policies,
    policy_means,
    out_dir,
    num_workers,
    cfg,
    tau_init: float = 1.0,
    keep_in_memory: bool = True
):
    
    ## redo the RPS steps
    R_set, R_profiles = RAggregate(M, R, H, D, y, theta, reg, verbose=True)
    anchors = build_anchor_states(R_set, R_profiles, M, R)
    prof_idx_of_policy, profiles = build_profile_index_of_policy(all_policies, hasse.policy_to_profile)
    RPS_states = raggregate_to_states((R_set, R_profiles), profiles)
    log_alpha = [np.log(max(1e-300, score_s_expneg_raw(A, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg))) for A in anchors]
    buckets = make_p0_buckets_weighted_S0(RPS_states, np.asarray(R,int), log_alpha, eps1=eps1, eps2=eps2, min_len=cfg.min_len)
    log_p0 = partial(log_p0_distance_weighted_S0_wrapper, buckets=buckets)
    init_sampler = partial(
        init_from_RPS_batch_wrapper,
        RPS_states=RPS_states,
        log_alpha=log_alpha,
        rng_seed=777,
    )

    print("Running pilot adaptive ladder")
    ladder, ess_ratios = pilot_adaptive_ladder(
        init_sampler=init_sampler,
        log_p0=log_p0,
        score_s=score_s,
        N=512,
        ess_target=0.80,
        beta0=0.0, beta1=1.0,
        initial_delta=0.05,
        min_delta=1e-3,
        moves_per_probe=3,   # small mixing at each accepted β (optional)
        min_len=1,
        rng_seed=777
    )

    terminals=[]; logw=np.zeros(cfg.n_paths,float)
    print("Adaptive ladder has", len(ladder), "levels; first 10:", ladder[:10])
    print("Per-step ESS ratios (len =", len(ess_ratios), "):", ess_ratios[:10])

    out_jsonl = os.path.join(out_dir, f"AIS_samples_{theta}_{epoch}.jsonl")
    if out_jsonl and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    path_ids = list(range(cfg.n_paths))
    chunks = np.array_split(path_ids, num_workers)

    args = [
        (chunk.tolist(), out_jsonl, cfg, buckets, RPS_states, R, ladder,
        log_p0, x, D, y, M, prof_idx_of_policy, all_policies, policy_means, reg)
        for chunk in chunks
    ]

    with Pool(num_workers) as p:
        chunk_results = p.starmap(parallel_ais_chunk, args)

    terminals = []
    logw = []
    for t_chunk, w_chunk in chunk_results:
        terminals.extend(t_chunk)
        logw.extend(w_chunk)

    logw = np.array(logw)

    m=float(np.max(logw)); w=np.exp(logw-m)
    return AISOutput(terminals, logw, w/w.sum(), ladder)

def run_ais_streaming_from_data_parallel_fixed_ladder(
    epoch,
    M, R, H, x, D, y, 
    theta, reg, eps1, eps2,
    score_s,
    all_policies,
    policy_means,
    out_dir,
    num_workers,
    cfg,
    ladder,
    tau_init: float = 1.0,
    keep_in_memory: bool = True
):
    
    ## redo the RPS steps
    R_set, R_profiles = RAggregate(M, R, H, D, y, theta, reg, verbose=True)
    anchors = build_anchor_states(R_set, R_profiles, M, R)
    prof_idx_of_policy, profiles = build_profile_index_of_policy(all_policies, hasse.policy_to_profile)
    RPS_states = raggregate_to_states((R_set, R_profiles), profiles)
    log_alpha = [np.log(max(1e-300, score_s_expneg_raw(A, D, y, M, R, prof_idx_of_policy, all_policies, policy_means, reg))) for A in anchors]
    buckets = make_p0_buckets_weighted_S0(RPS_states, np.asarray(R,int), log_alpha, eps1=eps1, eps2=eps2, min_len=cfg.min_len)
    log_p0 = partial(log_p0_distance_weighted_S0_wrapper, buckets=buckets)
    
    out_jsonl = os.path.join(out_dir, f"AIS_samples_{theta}_{epoch}.jsonl")
    if out_jsonl and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    path_ids = list(range(cfg.n_paths))
    chunks = np.array_split(path_ids, num_workers)

    args = [
        (chunk.tolist(), out_jsonl, cfg, buckets, RPS_states, R, ladder,
        log_p0, x, D, y, M, prof_idx_of_policy, all_policies, policy_means, reg)
        for chunk in chunks
    ]

    with Pool(num_workers) as p:
        chunk_results = p.starmap(parallel_ais_chunk, args)

    terminals = []
    logw = []
    for t_chunk, w_chunk in chunk_results:
        terminals.extend(t_chunk)
        logw.extend(w_chunk)

    logw = np.array(logw)

    m=float(np.max(logw)); w=np.exp(logw-m)
    return AISOutput(terminals, logw, w/w.sum(), ladder)


def load_ais_from_jsonl(jsonl_path: str):
    """
    Read an AIS JSONL file and return an AISOutput-like dict:
      {
        "terminals": [State, ...],
        "logw": np.ndarray,
        "normw": np.ndarray,
        "iters": np.ndarray (optional)
      }

    Assumes each JSON line looks like either:
      {"state": [...], "logw": <float>, ...}
    or
      {"terminal": [...], "logw": <float>, ...}
    and that `_jsonable_to_state(...)` + ProfilePart are in scope.
    """
    terminals = []
    logw = []
    iters = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            st_obj = rec.get("state", None)
            if st_obj is None:
                st_obj = rec.get("terminal", None)
            if st_obj is None:
                raise ValueError("JSONL record missing 'state'/'terminal' field.")

            terminals.append(MCMC._jsonable_to_state(st_obj))
            logw.append(float(rec.get("unnormalized_log_weight", rec.get("log_weight", np.nan))))
            iters.append(int(rec.get("iter", rec.get("path", -1))))

    logw = np.asarray(logw, dtype=float)

    return {
        "terminals": terminals,
        "logw": logw,
        "iters": np.asarray(iters, dtype=int),
    }

###
# 5) Formatting of rashomon output
###

# The output from RAggregate is R_set and R_profile, convert to the list of states (each state a list of profiles) as we defined

def _as_compact_B(sigma):
    """Normalize per-profile sigma from RAggregate into B (or None)."""
    if sigma is None:
        return None
    A = np.asarray(sigma)
    # Some profiles return a single column of +inf when R=2 (no interior cuts).
    # Treat any-all-inf row/array as 'no editable bits' -> None.
    if np.isinf(A).all():
        return None
    return A

def raggregate_to_states(
    RPS_output,
    profiles: List[Tuple[int, ...]],   # cov_ids per profile (binary tuple of length M)
) -> List[State]:
    """
    Convert RAggregate output into a list[State].
    - RPS_output can be a tuple (R_set, R_profiles) from RAggregate,
      or just R_set if you've already dereferenced sigmas.
    - profiles: list of binary tuples (cov_ids) aligned with per-profile Rashomon sets.
    """
    # Unpack
    if isinstance(RPS_output, tuple) and len(RPS_output) == 2:
        R_set, R_profiles = RPS_output
        # R_profiles is a list (one entry per profile), where each entry
        # holds a collection (list/array) of candidate sigma matrices for that profile.
        def get_sigma(profile_idx: int, idx_in_profile: int):
            rp = R_profiles[profile_idx]
            # RashomonSet object or plain list
            if hasattr(rp, "sigma"):
                return rp.sigma[idx_in_profile]
            return rp[idx_in_profile]
    else:
        # If caller already resolved to matrices per profile, assume R_set has concrete sigmas.
        R_set = RPS_output
        R_profiles = None
        def get_sigma(profile_idx: int, idx_in_profile: int):
            # Here R_set[i][k] is expected to be the actual sigma for profile k
            return idx_in_profile  # pass-through

    states: List[State] = []
    n_profiles = len(profiles)

    for g in R_set:
        # Each g is a list of indices, one per profile: g[k] indexes the chosen sigma for profile k
        if len(g) != n_profiles:
            raise ValueError(f"Global choice length {len(g)} != number of profiles {n_profiles}")
        state: State = []
        for k, choice in enumerate(g):
            cov_ids = tuple(profiles[k]) if profiles[k] is not None else None
            sigma_k = get_sigma(k, choice)
            B_k = _as_compact_B(sigma_k)
            state.append(ProfilePart(cov_ids=cov_ids, B=B_k))
        states.append(state)

    return states

###
# 6) Calculate beta from AIS output
###

# ---------- 1) posterior-weighted prediction per policy from AIS terminals ----------

def estimate_policy_means_from_ais(
    ais_out,                        # AISOutput: terminals (list[State]), normw (np.ndarray)
    all_policies,                   # list of all policy ids / feature combinations
    policy_means,                   # np.ndarray [P,2] == [sum_y, count] per global policy
    R_per,
    M,
    prof_idx_of_policy,             # np.ndarray or list of length P; maps policy_id -> profile_k
    lattice_edges=None,             # optional lattice for extract_pools (pass None if unused)
):
    """
    Returns: np.ndarray [P] of posterior-weighted mean outcome per policy.
    """
    P = len(all_policies)
    mu_hat = np.zeros(P, dtype=float)

    # Build per-profile index lists (global indices of policies belonging to each profile)
    K = int(np.max(prof_idx_of_policy)) + 1 # number of profiles
    prof_to_global = [ [] for _ in range(K) ]
    for pid in range(P):
        prof_to_global[prof_idx_of_policy[pid]].append(pid) # ith entry contains the list of policies in ith profile

    # For speed: precompute per-profile slices of (policies, policy_means)
    prof_policies   = [ [all_policies[i] for i in idxs] for idxs in prof_to_global ] # from policies index to tuple e.g. (0,0,1)
    prof_means_arr  = [ policy_means[np.array(idxs, int), :] if len(idxs)>0 else np.zeros((0,2))
                        for idxs in prof_to_global ]

    # Accumulate weighted predictions over AIS terminals
    for x_state, w in zip(ais_out.terminals, ais_out.normw):
        # x_state is a State: list[ProfilePart], aligned with profiles 0..K-1
        for k in range(K):
            idxs = prof_to_global[k]
            if not idxs: 
                continue

            # Partition matrix for this profile from the state
            Bk = x_state[k].B  # compact sigma for profile k (or None)
            # If None: no interior cuts -> one pool; still works with extract_pools
            # Build pools & policy→pool mapping for this profile's policy list
            sigma_full_k = assemble_sigma_full_for_profile(x_state[k], M, R_per)   # <-- NEW

            pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)

            # Compute pool means using the *per-profile* policy_means slice
            mu_pools_k = loss.compute_pool_means(prof_means_arr[k], pi_pools_k)

            # Write predictions back to global positions, weighted by particle weight
            for local_idx, pid in enumerate(idxs):
                pool_id = pi_policies_k[local_idx ]  # map policy→pool
                mu_hat[pid] += w * mu_pools_k[pool_id]

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
    w = _softmax_logalpha(log_alpha)

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
            sigma_full_k = assemble_sigma_full_for_profile(x_state[k], M, R_per)

            # pools + policy→pool mapping on the *local* list for profile k
            pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)
            mu_pools_k = loss.compute_pool_means(prof_means_arr[k], pi_pools_k)

            # IMPORTANT: pi_policies_k is keyed by LOCAL INDEX, not policy tuple
            for local_idx, pid in enumerate(idxs):
                pool_id = pi_policies_k[local_idx]
                mu_hat[pid] += ww * mu_pools_k[pool_id]

    return mu_hat

# ---------- 2) quick comparison against data-generation truth ----------

def compare_estimates_to_truth(mu_hat, true_mu_per_policy, top_n=10):
    """
    Returns a dict with MAE/RMSE and a small table of the largest absolute errors.
    """
    diff = mu_hat - true_mu_per_policy
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # top errors
    order = np.argsort(-np.abs(diff))[:top_n]
    table = [
        {"policy_id": int(i), "mu_hat": float(mu_hat[i]),
         "mu_true": float(true_mu_per_policy[i]), "abs_err": float(abs(diff[i]))}
        for i in order
    ]
    return {"MAE": mae, "RMSE": rmse, "top_abs_errors": table}

def mu_qqplot(mu_hat, mu_true):

    qq_plot = sm.qqplot_2samples(mu_hat, mu_true, line='45')
    plt.title("Q-Q Plot of Two Samples")
    plt.xlabel("Quantiles of estimated betas")
    plt.ylabel("Quantiles of true betas")

    # 4. Display the plot
    plt.show()


###
# 7) Pilot run to decide temperature ladder
###

def _logp(state, score_s: Callable) -> float:
    return math.log(max(1e-300, float(score_s(state))))

def _ess_from_logw(logw: np.ndarray) -> float:
    """ESS for *incremental* weights; returns ESS count (not ratio)."""
    m = float(np.max(logw))
    w = np.exp(logw - m)
    Z = w.sum()
    if Z <= 0: return 0.0
    w /= Z
    return 1.0 / float(np.sum(w*w))

def _normalize_logw(logw: np.ndarray) -> np.ndarray:
    """Get normalzied_weight from logw"""
    m = float(np.max(logw)); w = np.exp(logw - m); s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))

def pilot_adaptive_ladder(
    init_sampler: Callable[[int], List],         # returns a list of N initial States ~ q0 (or your RPS-init)
    log_p0: Callable[[List], float],             # log q0(x)
    score_s: Callable[[List], float],            # unnormalized target (positive), log taken inside
    N: int = 512,                                # probe particle count
    ess_target: float = 0.80,                    # target ESS ratio per step (0.7–0.9 typical)
    beta0: float = 0.0, beta1: float = 1.0,      # bridge endpoints
    initial_delta: float = 0.10,                 # starting step size Δβ
    min_delta: float = 1e-3,                     # do not split below this Δβ
    max_levels: int = 1000,                      # safety cap
    moves_per_probe: int = 0,                    # optional light mixing at each accepted β
    min_len: int = 1,                            # passed to your MH move
    rng_seed: Optional[int] = 1234
) -> Tuple[np.ndarray, List[float]]:
    """
    Returns:
      ladder: np.array of β's including endpoints (monotone, starts at beta0, ends at beta1)
      ess_ratios: list of ESS/N achieved at each accepted step (len = len(ladder)-1)
    Behavior:
      - Inserts rungs by halving Δβ whenever ESS/N < ess_target.
      - After accepting a step, does multinomial resample and (optional) a few MH moves at that β.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed); random.seed(rng_seed)

    # 1) initialize probe particles ~ q0 (or your RPS init) with equal weights
    X = init_sampler(N)                 # list of N States
    beta = beta0
    ladder = [beta0]
    ess_ratios = []
    step = 0
    delta = initial_delta

    # Precompute logp−logq0 for efficiency (updated only after moves)
    logp_minus_logq = np.array([_logp(x, score_s) - float(log_p0(x)) for x in X], dtype=float)

    while beta < beta1 - 1e-12 and len(ladder) < max_levels:
        beta_try = min(beta + delta, beta1)
        d = beta_try - beta
        # 2) incremental log-weights for proposed step
        inc_logw = d * logp_minus_logq
        ess = _ess_from_logw(inc_logw)            # ESS count
        ess_ratio = ess / float(N)

        if ess_ratio < ess_target and d > min_delta:
            # too steep → split step
            delta *= 0.5
            continue

        # 3) accept this rung
        ladder.append(beta_try)
        ess_ratios.append(float(ess_ratio))

        # 4) resample to equal weights (multinomial); this stabilizes chaining of probe steps
        nw = _normalize_logw(inc_logw)
        idx = np.random.choice(N, size=N, p=nw)
        X = [X[i] for i in idx]
        logp_minus_logq = logp_minus_logq[idx]    # keep aligned

        # 5) optional light mixing at β_try so the next probe is representative
        if moves_per_probe > 0:
            for i in range(N):
                xi = X[i]
                for _ in range(moves_per_probe):
                    xi = mh_step_state_uniform_neighbors(
                        xi, t=beta_try, log_p0=log_p0, score_s=score_s, min_len=min_len
                    )
                X[i] = xi
            # refresh logp−logq0 after moves
            logp_minus_logq = np.array([_logp(x, score_s) - float(log_p0(x)) for x in X], dtype=float)

        # 6) advance
        beta = beta_try
        # adjust next delta mildly (optional heuristic): grow if easy, shrink if tight
        if ess_ratio > max(ess_target, 0.9):
            delta = min(2*delta, beta1 - beta)    # speed up a bit on easy segments
        else:
            # keep current delta unless we barely passed (no change)
            delta = min(delta, beta1 - beta)
        step += 1

        if beta >= beta1 - 1e-12:
            break

    # ensure endpoint
    if ladder[-1] < beta1:
        ladder.append(beta1)

    return np.asarray(ladder, float), ess_ratios


###
# 7) Pilot run to decide temperature ladder
###

import itertools
# ---- Assumed available in your codebase -------------------------------------
# class ProfilePart:   # ProfilePart(cov_ids: tuple|None, B: np.ndarray|list|None)
#     ...
# def assemble_sigma_full_for_profile(pp, R_per: np.ndarray) -> np.ndarray: ...
# from rashomon.loss import compute_Q   # the repo's loss.compute_Q

State = List["ProfilePart"]

# ---------- 1) enumerate all compact B for a single profile ------------------

def _all_bitrows_for_arm(c: int):
    """All 0/1 bit rows of length c as float arrays (empty if c==0)."""
    if c <= 0:
        return [np.zeros((0,), float)]
    for bits in itertools.product((0.0, 1.0), repeat=c):
        yield np.asarray(bits, float)

def enumerate_compact_B_for_profile(cov_ids: Tuple[int, ...], R_per: np.ndarray):
    """
    Generate every compact partition matrix B for a given profile:
      - one row per active arm (in cov_ids order),
      - each row is a 0/1 vector for that arm's interior cut positions (length R_m-2),
      - returned as a 2D ndarray padded with +inf to equal width,
        or None if the profile has no interior bits.
    """
    cov = np.asarray(cov_ids, int)
    C_per = np.maximum(np.asarray(R_per, int) - 2, 0)
    active = np.flatnonzero(cov == 1)

    # For each active arm m, list all possible bit-rows length C_per[m]
    per_arm_rows = []
    for m in active:
        c = int(C_per[m])
        rows_m = list(_all_bitrows_for_arm(c))  # each is 1D float array
        # if c==0, rows_m == [array([],float)] meaning "no row content" for this arm
        per_arm_rows.append(rows_m)

    if not per_arm_rows:
        # No active arms -> no interior bits -> only partition is None
        yield None
        return

    # Cartesian product across active arms; pad to rectangular with +inf
    for choice in itertools.product(*per_arm_rows):
        # Filter out empty rows (from arms with c==0)
        rows = [r for r in choice if r.size > 0]
        if not rows:
            # All active arms had c==0 → no interior bits
            yield None
            continue
        W = max(r.size for r in rows)
        padded = [
            (r if r.size == W else np.pad(r, (0, W - r.size), constant_values=np.inf))
            for r in rows
        ]
        yield np.vstack(padded).astype(float)

# ---------- 2) enumerate all global states over all profiles -----------------

def enumerate_all_states(profiles: List[Tuple[int, ...]], R_per: np.ndarray):
    """
    Yield every global State (list[ProfilePart]), taking the Cartesian product
    over all profiles’ compact B choices.
    """
    # For each profile, precompute the list of all compact B (including None)
    per_profile_B_lists = [
        list(enumerate_compact_B_for_profile(cov, R_per))
        for cov in profiles
    ]

    for combo in itertools.product(*per_profile_B_lists):
        state = [ProfilePart(cov_ids=tuple(profiles[k]), B=combo[k]) for k in range(len(profiles))]
        yield state

# ---------- 3) precompute per-profile slices (D_k, y_k, policies_k, pm_k) ----

def prepare_profile_slices(
    policies,                   # global policies list (length P)
    policy_means,               # np.ndarray [P,2] = [sum_y, count]
    prof_idx_of_policy,         # length-P array: policy_id -> profile index k
    D, y                        # global D, y (D's first col = policy id for lookups)
):
    """
    Returns per-profile slices dicts keyed by k:
      - 'idxs': global policy indices belonging to profile k
      - 'policies': local list of policy objects
      - 'pm': local policy_means slice [n_k,2]
      - 'D': rows of D with those policies
      - 'y': matching y rows
    """
    P = len(policies)
    K = int(np.max(prof_idx_of_policy)) + 1
    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[int(prof_idx_of_policy[pid])].append(pid)

    slices = []
    for k in range(K):
        idxs = prof_to_global[k]
        pol_k = [policies[i] for i in idxs]
        pm_k = policy_means[np.array(idxs, int), :] if idxs else np.zeros((0,2))
        # mask rows in D/y for policies in this profile
        # (Assumes D[:,0] are integer policy IDs in 0..P-1)
        mask = np.isin(D[:, 0].astype(int), np.array(idxs, int))
        D_k = D[mask]
        y_k = y[mask]
        slices.append({"idxs": idxs, "policies": pol_k, "pm": pm_k, "D": D_k, "y": y_k})
    return slices

# ---------- 4) compute global loss for a given state -------------------------

# def compute_global_loss_for_state(
#     state: State,
#     slices,                     # output of prepare_profile_slices()
#     R_per: np.ndarray,
#     reg: float = 1.0,
#     normalize: int = 0,
#     lattice_edges=None          # pass None to let loss/predict compute internally
# ) -> float:
#     """
#     Sum of per-profile losses using repo's loss.compute_Q.
#     """
#     total = 0.0
#     R_per = np.asarray(R_per, int)
#     for k, pp in enumerate(state):
#         sk = assemble_sigma_full_for_profile(pp, R_per)  # full σ for profile k
#         sl = slices[k]
#         if sl["pm"].shape[0] == 0:
#             continue  # no data/policies in this profile; contributes 0
#         Qk = compute_Q(
#             D=sl["D"], y=sl["y"], sigma=sk,
#             policies=sl["policies"], policy_means=sl["pm"],
#             reg=reg, normalize=normalize, lattice_edges=lattice_edges
#         )
#         total += float(Qk)
#     return total

# ---------- 5) enumerate *and* score everything (WARNING: exponential) -------

def enumerate_all_states_and_losses(
    profiles: List[Tuple[int, ...]],
    M,
    R,
    policies,
    policy_means,
    prof_idx_of_policy,
    D, y,
    reg: float = 1.0,
    normalize: int = 0,
    lattice_edges=None,
    max_states: int | None = None   # optional cap to avoid explosion
):
    """
    Enumerate every global partition (state) and compute its total loss.
    Returns a list of tuples: (state, loss).
    WARNING: the count grows as ∏_k 2^{C_k}; use `max_states` to cap if needed.
    """
    slices = prepare_profile_slices(policies, policy_means, prof_idx_of_policy, D, y)
    out = []
    ctr = 0
    states = []
    for state in enumerate_all_states(profiles, R):
        states.append(state)
        L = global_loss_raw(state, D, y, policies, policy_means, prof_idx_of_policy, M, R, reg=reg, normalize=normalize, lattice_edges=lattice_edges)
        out.append((state, L))
        ctr += 1
        if (max_states is not None) and (ctr >= max_states):
            break
    return (states, out)


###
# 8) Making inference about the quantile of beta posterior dist
###

def extract_policy_mu_sigma_nig(
    state,                    # State: list[ProfilePart], length = num_profiles
    D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
    M,
    policies,                 # global policies list (len P)
    prof_idx_of_policy,       # length-P array: global policy id -> profile k
    R_per,                    # per-arm levels (len M)
    lattice_edges=None,
    mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
    seed=None
):
    """
    For a *single* global partition state, extract one (mu_k, sigma_k) for each policy k
    under a Normal-Inverse-Gamma model

    Returns:
      mu   : np.ndarray (K,)
      sigma: np.ndarray (K,)   # std dev samples (sqrt(sigma^2))
    """
    rng = np.random.default_rng(seed)
    P = len(policies)

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

    # For each profile, build pools & also pool sufficient stats
    per_prof_poolmap = [None] * K
    per_prof_poolstats = [None] * K
    per_prof_npools = [0] * K

    mu = np.zeros(P, float)
    t_scale = np.zeros(P, float)
    df = np.zeros(P, float)

    for k in range(K):
        idxs = prof_to_global[k]
        if not idxs:
            continue

        sigma_full_k = assemble_sigma_full_for_profile(state[k], M, np.asarray(R_per, int))
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

        idxs = prof_to_global[k]
        if not idxs:
            continue
        pi_policies_k = per_prof_poolmap[k]
        n, sy, sy2 = per_prof_poolstats[k]
        n_pools = per_prof_npools[k]

        # sample a mean for each pool
        pool_mu = np.zeros(n_pools, float)
        pool_t_scale = np.zeros(n_pools, float)
        pool_df = np.zeros(n_pools, float)
        for j in range(n_pools):
            mu_n, k_n, a_n, b_n = MCMC.nig_posterior_params(
                n[j], sy[j], sy2[j],
                mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0
            )

            # Marginal posterior:
            # mu | data ~ t_{2a_n}(loc=mu_n, scale=sqrt(b_n / (a_n * k_n)))
            df_n = 2.0 * a_n
            t_scale_n = np.sqrt(b_n / (a_n * k_n))
        
            pool_mu[j] = float(mu_n)
            pool_t_scale[j] = float(np.sqrt(t_scale_n))
            pool_df[j] = float(df_n)

        for local_idx, pid in enumerate(idxs):
            pool_id = pi_policies_k[local_idx]
            mu[pid] = pool_mu[pool_id]
            t_scale[pid] = pool_t_scale[pool_id]
            df[pid] = pool_df[pool_id]

    return mu, t_scale, df


def extract_policy_mu_sigma_flat(
    state,
    D, y,
    M,
    policies,
    prof_idx_of_policy,
    R_per,
    lattice_edges=None,
):
    """
    Flat-prior version of extract_policy_mu_sigma_nig.

    Implements eq. (A6): for each profile with H pools and n total observations,

        gamma_h | Pi, Z  ~  t_{n-H}( gamma_hat_h,  MSE_Pi / n_h )

    where
        gamma_hat_h = sy_h / n_h          (pool sample mean)
        SSE_Pi      = sum_h (sy2_h - n_h * gamma_hat_h^2)
        MSE_Pi      = SSE_Pi / (n - H)    (shared across all pools in profile)

    Returns
    -------
    mu      : np.ndarray (P,)  -- posterior location per policy
    t_scale : np.ndarray (P,)  -- posterior scale  per policy = sqrt(MSE / n_h)
    df      : np.ndarray (P,)  -- degrees of freedom per policy = n_total - H
    """
    P = len(policies)
    y1 = y[:, 0] if (isinstance(y, np.ndarray) and y.ndim == 2) else np.asarray(y).ravel()
    pid_all = D[:, 0].astype(int)

    K = int(np.max(prof_idx_of_policy)) + 1

    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[int(prof_idx_of_policy[pid])].append(pid)

    prof_policies = [[policies[i] for i in idxs] for idxs in prof_to_global]
    prof_pid_to_local = []
    for k in range(K):
        idxs = prof_to_global[k]
        prof_pid_to_local.append({pid: j for j, pid in enumerate(idxs)})

    prof_data_idx = []
    for k in range(K):
        idxs = np.array(prof_to_global[k], dtype=int)
        if idxs.size == 0:
            prof_data_idx.append(np.array([], dtype=int))
            continue
        mask = np.isin(pid_all, idxs)
        prof_data_idx.append(np.flatnonzero(mask))

    mu = np.zeros(P, float)
    t_scale = np.zeros(P, float)
    df = np.zeros(P, float)

    for k in range(K):
        idxs = prof_to_global[k]
        if not idxs:
            continue

        sigma_full_k = assemble_sigma_full_for_profile(state[k], M, np.asarray(R_per, int))
        pi_pools_k, pi_policies_k = extract_pools.extract_pools(prof_policies[k], sigma_full_k, lattice_edges)
        n_pools = len(pi_pools_k)

        n_h = np.zeros(n_pools, int)
        sy = np.zeros(n_pools, float)
        sy2 = np.zeros(n_pools, float)

        didx = prof_data_idx[k]
        pid_k = pid_all[didx]
        y_k = y1[didx]
        pid2loc = prof_pid_to_local[k]

        for pid_obs, y_obs in zip(pid_k, y_k):
            loc = pid2loc[int(pid_obs)]
            pool = pi_policies_k[loc]
            n_h[pool] += 1
            sy[pool] += float(y_obs)
            sy2[pool] += float(y_obs) ** 2

        # Profile-level sufficient stats for (A6)
        gamma_hat = np.where(n_h > 0, sy / np.maximum(n_h, 1), 0.0)
        sse_per_pool = sy2 - n_h * gamma_hat ** 2        # sum (y_i - ybar_h)^2 per pool
        SSE = float(np.sum(sse_per_pool))
        n_total = int(np.sum(n_h))
        H = n_pools
        df_k = float(n_total - H)                         # eq. (A6): nu = n - H

        if df_k <= 0:
            # Degenerate: fewer obs than pools; leave zeros (uninformative)
            continue

        MSE = SSE / df_k

        pool_mu = gamma_hat                               # location  = pool sample mean
        pool_scale = np.sqrt(MSE / np.maximum(n_h, 1))   # scale     = sqrt(MSE / n_h)

        for local_idx, pid in enumerate(idxs):
            pool_id = pi_policies_k[local_idx]
            mu[pid] = pool_mu[pool_id]
            t_scale[pid] = pool_scale[pool_id]
            df[pid] = df_k

    return mu, t_scale, df


# ---------------------------------------------------------------------------
# Normal posterior (sigma^2 known, fixed from saturated model)
# Paper eq. 3: gamma_h | Pi, Z, sigma^2 ~ N(g/(g+1)*gamma_hat_h, g/(g+1)*sigma^2/n_h)
# ---------------------------------------------------------------------------

def compute_sigma2_saturated(D, y, policies):
    """Estimate sigma^2 from the saturated model (each policy its own pool)."""
    y1 = y[:, 0] if (isinstance(y, np.ndarray) and y.ndim == 2) else np.asarray(y).ravel()
    pid_all = D[:, 0].astype(int)
    P = len(policies)
    SSE = 0.0
    for k in range(P):
        y_k = y1[pid_all == k]
        if len(y_k) > 1:
            SSE += float(np.sum((y_k - np.mean(y_k)) ** 2))
    dof = len(y1) - P
    return SSE / dof if dof > 0 else 1.0


def extract_policy_mu_sigma_normal(
    state,
    D, y,
    M,
    policies,
    prof_idx_of_policy,
    R_per,
    g=1.0,
    sigma2=None,
    lattice_edges=None,
):
    """
    Normal posterior for pool means given fixed sigma^2 (paper eq. 3).

        gamma_h | Pi, Z, sigma^2 ~ N(g/(g+1) * gamma_hat_h,  g/(g+1) * sigma^2 / n_h)

    Parameters
    ----------
    sigma2 : float or None
        Known noise variance. If None, estimated from the saturated model via
        compute_sigma2_saturated.

    Returns
    -------
    mu  : np.ndarray (P,)  -- posterior mean per policy
    std : np.ndarray (P,)  -- posterior std  per policy
    """
    if sigma2 is None:
        sigma2 = compute_sigma2_saturated(D, y, policies)

    P = len(policies)
    y1 = y[:, 0] if (isinstance(y, np.ndarray) and y.ndim == 2) else np.asarray(y).ravel()
    pid_all = D[:, 0].astype(int)
    K = int(np.max(prof_idx_of_policy)) + 1
    shrink = g / (g + 1.0)

    prof_to_global = [[] for _ in range(K)]
    for pid in range(P):
        prof_to_global[int(prof_idx_of_policy[pid])].append(pid)

    prof_policies = [[policies[i] for i in idxs] for idxs in prof_to_global]
    prof_pid_to_local = [{pid: j for j, pid in enumerate(idxs)} for idxs in prof_to_global]

    prof_data_idx = []
    for k in range(K):
        idxs = np.array(prof_to_global[k], dtype=int)
        if idxs.size == 0:
            prof_data_idx.append(np.array([], dtype=int))
            continue
        prof_data_idx.append(np.flatnonzero(np.isin(pid_all, idxs)))

    mu = np.zeros(P, float)
    std = np.zeros(P, float)

    for k in range(K):
        idxs = prof_to_global[k]
        if not idxs:
            continue

        sigma_full_k = assemble_sigma_full_for_profile(state[k], M, np.asarray(R_per, int))
        pi_pools_k, pi_policies_k = extract_pools.extract_pools(
            prof_policies[k], sigma_full_k, lattice_edges
        )
        n_pools = len(pi_pools_k)

        n_h = np.zeros(n_pools, int)
        sy = np.zeros(n_pools, float)

        didx = prof_data_idx[k]
        pid2loc = prof_pid_to_local[k]
        for pid_obs, y_obs in zip(pid_all[didx], y1[didx]):
            pool = pi_policies_k[pid2loc[int(pid_obs)]]
            n_h[pool] += 1
            sy[pool] += float(y_obs)

        gamma_hat = np.where(n_h > 0, sy / np.maximum(n_h, 1), 0.0)
        pool_mu = shrink * gamma_hat
        pool_std = np.sqrt(shrink * sigma2 / np.maximum(n_h, 1))

        for local_idx, pid in enumerate(idxs):
            pool_id = pi_policies_k[local_idx]
            mu[pid] = pool_mu[pool_id]
            std[pid] = pool_std[pool_id]

    return mu, std


def extract_policy_posteriors_normal(
    ais_sample,
    D, y,
    M,
    policies,
    prof_idx_of_policy,
    R,
    g=1.0,
    sigma2=None,
    lattice_edges=None,
):
    """Like extract_policy_posteriors_from_ais_sample but returns Normal parameters.

    Returns
    -------
    logw : ndarray (n_particles,)
    mu   : ndarray (n_particles, n_policies)
    std  : ndarray (n_particles, n_policies)
    """
    if sigma2 is None:
        sigma2 = compute_sigma2_saturated(D, y, policies)

    logw, mu_list, std_list = [], [], []
    for state, lw in zip(ais_sample.terminals, ais_sample.logw):
        mu_i, std_i = extract_policy_mu_sigma_normal(
            state=state, D=D, y=y, M=M,
            policies=policies, prof_idx_of_policy=prof_idx_of_policy,
            R_per=R, g=g, sigma2=sigma2, lattice_edges=lattice_edges,
        )
        logw.append(lw)
        mu_list.append(mu_i)
        std_list.append(std_i)

    return np.asarray(logw), np.asarray(mu_list), np.asarray(std_list)


def extract_policy_posteriors_normal_from_states(
    states,
    logw,
    D, y,
    M,
    policies,
    prof_idx_of_policy,
    R,
    g=1.0,
    sigma2=None,
    lattice_edges=None,
):
    """Same as extract_policy_posteriors_normal but for a plain list of states + logw.
    Used for MCMC, RPS, and exact enumeration.
    """
    if sigma2 is None:
        sigma2 = compute_sigma2_saturated(D, y, policies)

    mu_list, std_list = [], []
    for state in states:
        mu_i, std_i = extract_policy_mu_sigma_normal(
            state=state, D=D, y=y, M=M,
            policies=policies, prof_idx_of_policy=prof_idx_of_policy,
            R_per=R, g=g, sigma2=sigma2, lattice_edges=lattice_edges,
        )
        mu_list.append(mu_i)
        std_list.append(std_i)

    return np.asarray(logw), np.asarray(mu_list), np.asarray(std_list)


from scipy.stats import norm as _scipy_norm

def _normal_mixture_cdf(u, logw, mu_k, std_k):
    """Weighted Normal mixture CDF at u."""
    w = _normalize_logw(logw)
    vals = _scipy_norm.cdf((u - mu_k) / np.maximum(std_k, 1e-12))
    return float(np.sum(w * vals))


def _normal_mixture_quantile(logw, mu_k, std_k, alpha, tol=1e-8, maxiter=100):
    """Alpha-quantile of a weighted Normal mixture via Brent's method."""
    w = _normalize_logw(logw)
    result = quantiles.q_mixture_normal_root(
        k=len(w), w=w, mu=mu_k, s=std_k,
        prob=[alpha], xtol=tol, maxiter=maxiter,
    )
    return float(result[0])


def ais_quantiles_normal_for_all_policies(
    ais_sample,
    D, y,
    M,
    policies,
    prof_idx_of_policy,
    R,
    g=1.0,
    sigma2=None,
    lattice_edges=None,
    p=[0.025, 0.5, 0.975],
):
    """
    Normal-posterior quantiles for every policy from AIS output.

    Uses the paper's model (eq. 3) with sigma^2 fixed:
        gamma_h | Pi, Z, sigma^2 ~ N(g/(g+1)*gamma_hat_h, g/(g+1)*sigma^2/n_h)
    """
    if sigma2 is None:
        sigma2 = compute_sigma2_saturated(D, y, policies)

    logw, mu, std = extract_policy_posteriors_normal(
        ais_sample, D, y, M, policies, prof_idx_of_policy, R,
        g=g, sigma2=sigma2, lattice_edges=lattice_edges,
    )

    output = {}
    n_policies = mu.shape[1]
    for quantile in p:
        q = np.empty(n_policies, float)
        for k in range(n_policies):
            q[k] = _normal_mixture_quantile(logw, mu[:, k], std[:, k], alpha=quantile)
        output[str(quantile)] = q
    return output


def states_quantiles_normal_for_all_policies(
    states,
    logw,
    D, y,
    M,
    policies,
    prof_idx_of_policy,
    R,
    g=1.0,
    sigma2=None,
    lattice_edges=None,
    p=[0.025, 0.5, 0.975],
):
    """
    Normal-posterior quantiles for every policy from a list of states + log-weights.
    Drop-in replacement for MCMC.quantiles_for_all_policies_root / exact enumeration.
    """
    if sigma2 is None:
        sigma2 = compute_sigma2_saturated(D, y, policies)

    logw_arr, mu, std = extract_policy_posteriors_normal_from_states(
        states, logw, D, y, M, policies, prof_idx_of_policy, R,
        g=g, sigma2=sigma2, lattice_edges=lattice_edges,
    )

    output = {}
    n_policies = mu.shape[1]
    for quantile in p:
        q = np.empty(n_policies, float)
        for k in range(n_policies):
            q[k] = _normal_mixture_quantile(logw_arr, mu[:, k], std[:, k], alpha=quantile)
        output[str(quantile)] = q
    return output


def extract_policy_posteriors_from_ais_sample(
    ais_sample,
    D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
    M,
    policies,                 # global policies list (len P)
    prof_idx_of_policy,       # length-P array: global policy id -> profile k
    R,                    # per-arm levels (len M)
    lattice_edges=None,
    mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
    seed=None):
    """
    Parameters
    ----------
    ais_sample : AISOutput

    Returns
    -------
    logw : ndarray, shape (n_particles,)
    mu   : ndarray, shape (n_particles, n_policies)
    sd   : ndarray, shape (n_particles, n_policies)
    """
    logw = []
    mu_list = []
    scale_list = []
    df_list = []

    state_list = ais_sample.terminals
    lw_list = ais_sample.logw

    for state, lw in zip(state_list, lw_list):
        mu_i, scale_i, df_i = extract_policy_mu_sigma_flat(
            state=state,
            D=D, y=y,
            M=M,
            policies=policies,
            prof_idx_of_policy=prof_idx_of_policy,
            R_per=R,
            lattice_edges=None,
        )

        logw.append(lw)
        mu_list.append(mu_i)
        scale_list.append(scale_i)
        df_list.append(df_i)

    logw = np.asarray(logw, dtype=float)
    mu = np.asarray(mu_list, dtype=float)
    scale = np.asarray(scale_list, dtype=float)
    df = np.asarray(df_list, dtype=float)

    return logw, mu, scale, df

from scipy.stats import t as student_t

def ais_policy_cdf(u, logw, mu_k, scale_k, df_k):
    """
    AIS mixture CDF for one policy/profile k under a mixture of Student-t posteriors:
        F_k(u) = sum_i w_i * T_df_i((u - mu_ik) / scale_ik)
    This is achieved by viewing the F_k(u) as a function of the state x, g(x), then use the 
    theory of important sampling to estimate E[g(x)]

    Parameters
    ----------
    u : float
        Value at which to evaluate the CDF.
    logw : array-like, shape (n_particles,)
        Unnormalized AIS log-weights.
    mu_k : array-like, shape (n_particles,)
        Location parameter for policy/profile k under each AIS particle.
    scale_k : array-like, shape (n_particles,)
        Scale parameter of the Student-t posterior for policy/profile k under each AIS particle.
        This is the t-scale, not the posterior standard deviation.
    df_k : array-like or float
        Degrees of freedom for each AIS particle. Can be scalar if shared.

    Returns
    -------
    float
        Weighted AIS mixture CDF at u.
    """
    w = _normalize_logw(logw)
    mu_k = np.asarray(mu_k, dtype=float)
    scale_k = np.asarray(scale_k, dtype=float)
    df_k = np.asarray(df_k, dtype=float)

    if df_k.ndim == 0:
        df_k = np.full_like(mu_k, float(df_k), dtype=float)

    vals = np.empty_like(mu_k, dtype=float)

    for i, (m, s, df) in enumerate(zip(mu_k, scale_k, df_k)):
        if s <= 0:
            vals[i] = 1.0 if u >= m else 0.0
        else:
            vals[i] = student_t.cdf((u - m) / s, df=df)

    return float(np.sum(w * vals))

def ais_policy_quantile(logw, mu_k, scale_k, df_k, alpha=0.95, tol=1e-8, maxiter=200):
    """Reverse engineer, using binary dissection to find the quantile value"""
    mu_k = np.asarray(mu_k, dtype=float)
    scale_k = np.asarray(scale_k, dtype=float)
    df_k = np.asarray(df_k, dtype=float)

    if df_k.ndim == 0:
        df_k = np.full_like(mu_k, float(df_k), dtype=float)

    eps = 1e-12
    lo = float(np.min(mu_k - 20.0 * np.maximum(scale_k, eps)))
    hi = float(np.max(mu_k + 20.0 * np.maximum(scale_k, eps)))

    while ais_policy_cdf(lo, logw, mu_k, scale_k, df_k) >= alpha:
        lo -= max(1.0, 0.5 * max(abs(lo), 1.0))

    while ais_policy_cdf(hi, logw, mu_k, scale_k, df_k) < alpha:
        hi += max(1.0, 0.5 * max(abs(hi), 1.0))

    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = ais_policy_cdf(mid, logw, mu_k, scale_k, df_k)
        if fmid < alpha:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < tol:
            break

    return 0.5 * (lo + hi)

def ais_quantiles_for_all_policies(
    ais_sample,
    D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
    M,
    policies,                 # global policies list (len P)
    prof_idx_of_policy,       # length-P array: global policy id -> profile k
    R,                    # per-arm levels (len M)
    lattice_edges=None,
    mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
    p=[0.025, 0.5, 0.975],
    seed=None):
    """
    Compute AIS posterior alpha-quantile for every policy.

    Returns
    -------
    dict with:
      - alpha
      - quantiles : shape (n_policies,)
      - logw, mu, sd
    """
    logw, mu, scale, df = extract_policy_posteriors_from_ais_sample(
        ais_sample,
        D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
        M,
        policies,                 # global policies list (len P)
        prof_idx_of_policy,       # length-P array: global policy id -> profile k
        R,                    # per-arm levels (len M)
        lattice_edges=None,
        mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0,
        seed=None
    )

    output = dict()

    for quantile in p:
        n_policies = mu.shape[1]
        q = np.empty(n_policies, dtype=float)

        for k in range(n_policies):
            q[k] = ais_policy_quantile(
                logw=logw,
                mu_k=mu[:, k],
                scale_k=scale[:, k],
                df_k=df[:, k],
                alpha=quantile,
            )

        output[f"{quantile}"] = q
    
    return output

def ais_quantiles_for_all_policies_root_original(
    ais_sample,
    D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
    M,
    policies,                 # global policies list (len P)
    prof_idx_of_policy,       # length-P array: global policy id -> profile k
    R,                    # per-arm levels (len M)
    lattice_edges=None,
    mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
    p=[0.025, 0.5, 0.975],
    seed=None):
    """
    Compute AIS posterior alpha-quantile for every policy.

    Returns
    -------
    dict with:
      - alpha
      - quantiles : shape (n_policies,)
      - logw, mu, sd
    """
    logw, mu, scale, df = extract_policy_posteriors_from_ais_sample(
        ais_sample,
        D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
        M,
        policies,                 # global policies list (len P)
        prof_idx_of_policy,       # length-P array: global policy id -> profile k
        R,                    # per-arm levels (len M)
        lattice_edges=None,
        mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0,
        seed=None
    )

    output = dict()

    for quantile in p:
        n_policies = mu.shape[1]
        q = np.empty(n_policies, dtype=float)

        # for k in range(n_policies):
        #     w=[np.exp(i) for i in logw],
        #     mu_k=mu[:, k],
        #     scale_k=scale[:, k],
        #     df_k=df[:, k]
        #     q[k] = quantiles.q_mixture_t_root(
        #         k=len(df), w=w, mu=mu_k, s=scale_k, df=df_k, prob=quantile
        #     )[0]

        for k in range(n_policies):
            w=[np.exp(i) for i in logw]
            mu_k=mu[:, k]
            scale_k=scale[:, k]
            df_k=df[:, k]
            q_sum = 0
            for i in range(len(w)):
                w_i = w[i]
                mu_i = mu_k[i]
                scale_i = scale_k[i]
                df_i = df_k[i]
                
                q_i = quantiles.q_mixture_t_root(
                    k=1, w=[1], mu=[mu_i], s=[scale_i], df=[df_i], prob=quantile
                    )[0]
                q_sum = q_sum + w_i*q_i
            q[k] = q_sum/np.sum(w)

        output[f"{quantile}"] = q
    
    return output

def ais_quantiles_for_all_policies_root(
    ais_sample,
    D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
    M,
    policies,                 # global policies list (len P)
    prof_idx_of_policy,       # length-P array: global policy id -> profile k
    R,                    # per-arm levels (len M)
    lattice_edges=None,
    mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
    p=[0.025, 0.5, 0.975],
    seed=None):
    """
    Compute AIS posterior alpha-quantile for every policy.

    Returns
    -------
    dict with:
      - alpha
      - quantiles : shape (n_policies,)
      - logw, mu, sd
    """
    logw, mu, scale, df = extract_policy_posteriors_from_ais_sample(
        ais_sample,
        D, y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
        M,
        policies,                 # global policies list (len P)
        prof_idx_of_policy,       # length-P array: global policy id -> profile k
        R,                    # per-arm levels (len M)
        lattice_edges=None,
        mu0=mu0, kappa0=kappa0, alpha0=alpha0, beta0=beta0,
        seed=None
    )

    output = dict()

    for quantile in p:
        n_policies = mu.shape[1]
        q = np.empty(n_policies, dtype=float)

        for k in range(n_policies):
            w = np.exp(logw)
            mu_k = mu[:, k]
            scale_k = scale[:, k]
            df_k = df[:, k]
            q[k] = quantiles.q_mixture_t_root(
                k=len(df_k), w=w, mu=mu_k, s=scale_k, df=df_k, prob=quantile
            )[0]

        output[f"{quantile}"] = q
    
    return output

def random_partition_state_uniform_bits(
    M: int,
    R: Sequence[int],
    seed: Optional[int] = None,
    profiles: Optional[List[Tuple[int, ...]]] = None
) -> State:
    """
    Randomly generate a global partition State with each editable position (finite entry)
    having equal probability to be 0 or 1.

    - Constructs profiles as all 2^M binary cov_id tuples unless `profiles` is provided.
    - For each profile:
        * active arms are where cov_ids[m]==1
        * for each active arm m, generate a row of length (R_m - 2) with iid Bernoulli(p_one)
        * rows are padded with +inf to rectangular ndarray
        * if no active arms or all active arms have R_m<=2 -> B=None
    """
    p_one = 0.5
    rng = np.random.default_rng(seed)
    R = np.asarray(R, dtype=int)
    if R.size != M:
        raise ValueError("R must be length M")
    C_per = np.maximum(R - 2, 0)

    out: State = []
    for cov_ids in profiles:
        cov = np.asarray(cov_ids, dtype=int)
        if cov.size != M:
            raise ValueError("each cov_ids must be length M")
        active = np.flatnonzero(cov == 1)

        rows = []
        for m in active:
            c = int(C_per[m])
            if c <= 0:
                continue
            row = (rng.random(c) < p_one).astype(float)  # 0/1 with equal prob when p_one=0.5
            rows.append(row)

        if not rows:
            B = None
        else:
            W = max(r.size for r in rows)
            padded = [
                r if r.size == W else np.pad(r, (0, W - r.size), constant_values=np.inf)
                for r in rows
            ]
            B = np.vstack(padded).astype(float)

        out.append(ProfilePart(cov_ids=tuple(cov.tolist()), B=B))

    return out


# ---------------------------------------------------------------------------
# PAC-Bayes helpers
# ---------------------------------------------------------------------------

def pac_bayes_bound(log_scores, n_obs, n_prior, delta=0.05):
    """Rockova-style PAC-Bayes bound for a Gibbs posterior over a visited set."""
    ls = np.asarray(log_scores, dtype=float)
    ls = ls[np.isfinite(ls)]
    if len(ls) == 0:
        return {"bound": np.inf, "E_Q_risk": np.inf, "kl": np.inf, "H_Q": 0.0}
    lw = ls - ls.max()
    q  = np.exp(lw); q /= q.sum()
    e_q_loss  = -ls.max() - float(np.dot(q, lw))
    e_q_risk  = e_q_loss / float(n_obs)
    h_q       = float(-np.dot(q, np.log(np.maximum(q, 1e-300))))
    kl        = max(float(np.log(n_prior) - h_q), 0.0)
    penalty   = np.log(2.0 * np.sqrt(float(n_obs)) / delta)
    bound     = e_q_risk + float(np.sqrt((kl + penalty) / (2.0 * n_obs)))
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
    Greedy PAC-Bayes state-space explorer seeded from an initial set of states.
    At each step, adds the frontier neighbor that minimises the PAC-Bayes bound.

    Returns:
        states (list): All visited states.
        logs (list): Corresponding log-scores.
        trace (list[dict]): PAC-Bayes bound dict at each step.
    """
    rng = np.random.default_rng(seed)
    visited    = {}
    log_scores = []

    def add_visited(state, log_score):
        sig = state_signature(state)
        if sig in visited:
            return False
        visited[sig] = (state, float(log_score))
        log_scores.append(float(log_score))
        return True

    def score_state(state):
        import math
        return math.log(max(1e-300, score_s(state)))

    for state, log_score in zip(init_states, init_log_scores):
        add_visited(state, log_score)

    frontier = {}

    def add_neighbors(state):
        for nb in state_neighbors_ubs(state, min_len=min_len):
            sig = state_signature(nb)
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
            idx   = rng.choice(len(items), size=frontier_cap, replace=False)
            items = [items[i] for i in idx]

        chosen_sig, chosen_state, chosen_log_score, chosen_bound = None, None, None, None
        for sig, state in items:
            ls = score_state(state)
            cb = pac_bayes_bound(log_scores + [ls], n_obs, n_prior, delta=delta)
            if chosen_bound is None or cb["bound"] < chosen_bound["bound"]:
                chosen_sig, chosen_state, chosen_log_score, chosen_bound = sig, state, ls, cb

        frontier.pop(chosen_sig, None)
        add_visited(chosen_state, chosen_log_score)
        add_neighbors(chosen_state)
        trace.append(chosen_bound)

    states = [state for state, _ in visited.values()]
    logs   = [ls    for _,     ls in visited.values()]
    return states, logs, trace
