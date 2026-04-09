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
from rashomon import hasse, extract_pools, loss

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
        beyond = True
        while beyond == False:
            s = _copy_state(random_state_jitter_from_RPS(RPS, R_per, 3))
            sig = state_signature(s)
            if sig in buckets.S0_set or sig in buckets.S1_sigs: 
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
    for state in enumerate_all_states(profiles, R):
        L = global_loss_raw(state, D, y, policies, policy_means, prof_idx_of_policy, M, R, reg=reg, normalize=normalize, lattice_edges=lattice_edges)
        out.append((state, L))
        ctr += 1
        if (max_states is not None) and (ctr >= max_states):
            break
    return out


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
    ais_sample : list of dicts
        Each element must have:
          - rec["state"]
          - rec["logw"]

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

    state_list = ais_sample["terminals"]
    lw_list = ais_sample["logw"]

    for state, lw in zip(state_list, lw_list):
        mu_i, scale_i, df_i = extract_policy_mu_sigma_nig(
            state=state,                    # State: list[ProfilePart], length = num_profiles
            D=D, y=y,                     # D[:,0] = global policy id; y is (N,) or (N,1)
            M=M,
            policies=policies,                 # global policies list (len P)
            prof_idx_of_policy=prof_idx_of_policy,       # length-P array: global policy id -> profile k
            R_per=R,                    # per-arm levels (len M)
            lattice_edges=None,
            mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0,
            seed=None
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