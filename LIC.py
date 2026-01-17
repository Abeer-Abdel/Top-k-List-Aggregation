# Imports
import time
start = time.time()
import os, re
import numpy as np
import pandas as pd
from itertools import combinations


# Config
df = pd.read_csv(CSV)
RANKINGS = "Testt.csv"
ROW_FRACTION = 1          # keep top fraction of rows
p = 0.5                   
TOL = 1e-9                # tolerance for tie-breaking in LIC
LOCAL_OPT_METHOD = "bubble_back"  # "bubble_back" or "repeat_passes"
np.random.seed(42)


# Helper to parse cell with ties
def _parse_cell_amp(cell):
    if pd.isna(cell): return []
    s = str(cell).strip()
    if s == "": return []
    if re.fullmatch(r"[-+]?\d+", s):
        return [int(s)]
    parts = [p.strip() for p in s.split("&") if p.strip() != ""]
    return [int(p) for p in parts]


# Build lists
def _build_lists_and_positions_tieaware(df):
    lists, pos_per_list = [], []
    for c in df.columns:
        col = df[c]
        seen, cleaned = set(), []
        pos_map, pos = {}, 0
        for r in range(len(col)):
            items_here = _parse_cell_amp(col.iloc[r])
            if not items_here:
                continue
            new_any = False
            for it in items_here:
                if it not in seen:
                    cleaned.append(it)
                    seen.add(it)
                    pos_map[it] = pos
                    new_any = True
            if new_any:
                pos += 1
        lists.append(cleaned)
        pos_per_list.append(pos_map)
    return lists, pos_per_list


# Kendall tau computation
def kendall_tau_p_cases_from_lists(lists, pos_per_list, tau_star, p):
    m = len(lists)
    tau_star_unique, seen = [], set()
    for x in tau_star:
        x = int(x)
        if x not in seen:
            tau_star_unique.append(x)
            seen.add(x)
    universe = set(tau_star_unique)
    for L in lists:
        universe.update(L)
    items = sorted(universe)
    z = {i: 0 for i in items}
    pos_star = {}
    for r, it in enumerate(tau_star_unique):
        z[it] = 1
        pos_star[it] = r
    present_per_list = [{item: (item in pos_map) for item in items} for pos_map in pos_per_list]

    total = 0.0
    for i, j in combinations(items, 2):
        zi, zj = z[i], z[j]
        for l in range(m):
            pos = pos_per_list[l]
            pres = present_per_list[l]
            mu_i, mu_j = pres[i], pres[j]

            # Case 1: both in tau* and present
            if zi and zj and mu_i and mu_j:
                pi_star_lt_pj_star = (pos_star[i] < pos_star[j])
                pi_l, pj_l = pos[i], pos[j]
                if pi_l == pj_l:
                    total += 0.5
                else:
                    total += 1.0 if (pi_star_lt_pj_star != (pi_l < pj_l)) else 0.0
                continue

            # Case 2A: both in tau*, exactly one present
            if zi and zj and (mu_i ^ mu_j):
                appearing = i if mu_i else j
                missing  = j if mu_i else i
                pen = 0.0 if pos_star[appearing] < pos_star[missing] else 1.0
                total += pen
                continue

            # Case 2B: exactly one in tau*, both present
            if (zi ^ zj) and mu_i and mu_j:
                if pos[i] == pos[j]:
                    total += 0.5
                else:
                    if zi == 1 and zj == 0:
                        total += 1.0 if pos[j] < pos[i] else 0.0
                    else:
                        total += 1.0 if pos[i] < pos[j] else 0.0
                continue

            # Case 3: exactly one in tau* and exactly one present
            if (zi ^ zj) and (mu_i ^ mu_j):
                triggers = (zi == 1 and zj == 0 and mu_i == 0 and mu_j == 1) or \
                           (zi == 0 and zj == 1 and mu_i == 1 and mu_j == 0)
                if triggers: total += 1.0
                continue

            # Case 4: both in tau*, neither present OR neither in tau*, both present
            if (zi and zj and not mu_i and not mu_j) or ((not zi) and (not zj) and mu_i and mu_j):
                total += p
                continue

    return total


# Compute S matrix and mu
def compute_s_matrix_from_df(df):
    """Return objects, S (tie-aware pairwise), μ (presence)."""
    lists, pos_per_list = _build_lists_and_positions_tieaware(df)
    objects = sorted(set().union(*map(set, lists)))
    n = len(objects)
    m = len(lists)
    obj_to_idx = {obj: idx for idx, obj in enumerate(objects)}

    # S: pairwise costs
    S = np.zeros((n, n), dtype=float)
    for l in range(m):
        pos = pos_per_list[l]
        present = {obj: (obj in pos) for obj in objects}
        for i in objects:
            i_idx = obj_to_idx[i]
            i_in = present[i]
            for j in objects:
                if i == j: continue
                j_idx = obj_to_idx[j]
                j_in = present[j]
                if i_in and j_in:
                    if pos[i] < pos[j]:
                        S[i_idx, j_idx] += 1.0
                    elif pos[i] == pos[j]:
                        S[i_idx, j_idx] += 0.5
                elif i_in and (not j_in):
                    S[i_idx, j_idx] += 1.0

    S_df = pd.DataFrame(S, index=objects, columns=objects)

    mu = np.zeros((n, m), dtype=int)
    for l in range(m):
        for it in lists[l]:
            mu[obj_to_idx[it], l] = 1

    return objects, S_df, mu


# Load & truncate data
df_raw = pd.read_csv(RANKINGS, header=None, dtype=str)
orig_rows = df_raw.shape[0]
df_raw = df_raw.iloc[:max(1, int(orig_rows * ROW_FRACTION)), :]
m = df_raw.shape[1]
k = df_raw.shape[0]
_lists_for_swaps, _pos_per_list_for_swaps = _build_lists_and_positions_tieaware(df_raw)
objects, S_df, mu = compute_s_matrix_from_df(df_raw)
S = S_df.to_numpy(dtype=float)
n = S.shape[0]
print(f"m = {m}, k = {k}, n = {n}")


# LIC base costs and pair penalties
base_cost = S.sum(axis=0) - np.diag(S)
M = 1 - mu
pair_penalty = p * (M @ M.T)
np.fill_diagonal(pair_penalty, 0.0)


# Greedy LIC selection
remaining = np.ones(n, dtype=bool)
cost = base_cost.copy()
tau_idx = []

for t in range(k):
    cand = np.where(remaining)[0]
    cvals = cost[cand]
    minc = cvals.min()
    tied = cand[np.flatnonzero(np.abs(cvals - minc) <= TOL)]
    i_star = int(np.random.choice(tied))
    tau_idx.append(i_star)
    remaining[i_star] = False

    rem = np.where(remaining)[0]
    if rem.size > 0:
        cost[rem] = cost[rem] - S[i_star, rem] + pair_penalty[rem, i_star]

tau_objs = [objects[i] for i in tau_idx]

val = kendall_tau_p_cases_from_lists(_lists_for_swaps, _pos_per_list_for_swaps, tau_objs, p)

print(f"Objective value: {val:.1f}")
print("Top k:")
print(f"[{', '.join(str(x) for x in tau_objs)}]")


# Local swap
def _pair_cost_in_one_list_cases(a, b, pa, pb, pos_l, p):
    i2 = (a in pos_l)
    j2 = (b in pos_l)
    if i2 and j2:
        if pos_l[a] == pos_l[b]:
            return 0.5
        return 1.0 if ((pa < pb) != (pos_l[a] < pos_l[b])) else 0.0
    if i2 ^ j2:
        if i2: return 0.0 if pa < pb else 1.0
        else: return 0.0 if pb < pa else 1.0
    return p

def _delta_adjacent_swap_cases(a, b, pos_live, pos_lists, p):
    pa, pb = pos_live[a], pos_live[b]
    before = after = 0.0
    for pos_l in pos_lists:
        before += _pair_cost_in_one_list_cases(a, b, pa, pb, pos_l, p)
        after  += _pair_cost_in_one_list_cases(a, b, pb, pa, pos_l, p)
    return after - before

def _total_obj(order):
    return kendall_tau_p_cases_from_lists(_lists_for_swaps, _pos_per_list_for_swaps, order, p)

#timing
time_before_swaps = time.time()
work = tau_objs[:]
curr = _total_obj(work)
if LOCAL_OPT_METHOD == "repeat_passes":
    while True:
        accepted = 0
        for t in range(len(work)-1):
            cand = work[:]
            cand[t], cand[t+1] = cand[t+1], cand[t]
            cand_val = _total_obj(cand)
            if cand_val + TOL < curr:
                work, curr = cand, cand_val
                accepted += 1
        if accepted == 0: break
else:  # bubble_back
    t = 0
    pos_live = {it: r for r, it in enumerate(work)}
    while t < len(work)-1:
        a, b = work[t], work[t+1]
        delta = _delta_adjacent_swap_cases(a, b, pos_live, _pos_per_list_for_swaps, p)
        if delta + TOL < 0.0:
            work[t], work[t+1] = b, a
            curr += delta
            pa, pb = pos_live[a], pos_live[b]
            pos_live[a], pos_live[b] = pb, pa
            t = 0
        else:
            t += 1

# Final report
print("Final strict ranking after swaps:", work)
print(f"Final total distance = {curr:.1f}")
end = time.time()
time_until_swaps = time_before_swaps - start
swaps_time = end - time_before_swaps
print(f"\nTime until swaps: {time_until_swaps:.3f} seconds")
print(f"Time for swaps: {swaps_time:.3f} seconds")


