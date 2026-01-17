# Imports
import time
start = time.time()
import os, re
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import linear_sum_assignment


# Config
CSV  = "dataset.csv"
P    = 0.5               
TOL  = 1e-9               
ROW_FRACTION = 1.0        
LOCAL_OPT_METHOD = "bubble_back"
np.random.seed(42)


# Parse cells with ties
def _parse_cell_amp(cell):
    if pd.isna(cell): return []
    s = str(cell).strip()
    if s == "": return []
    if re.fullmatch(r"[-+]?\d+", s):
        return [int(s)]
    return [int(p.strip()) for p in s.split("&") if p.strip() != ""]


# Build lists and position maps
def _build_lists_and_positions_tieaware(df):
    lists, pos_per_list = [], []
    for c in df.columns:
        col = df[c]
        seen, cleaned = set(), []
        pos_map, pos = {}, 0
        for r in range(len(col)):
            items_here = _parse_cell_amp(col.iloc[r])
            if not items_here: continue
            new_any = False
            for it in items_here:
                if it not in seen:
                    cleaned.append(it)
                    seen.add(it)
                    pos_map[it] = pos
                    new_any = True
            if new_any: pos += 1
        lists.append(cleaned)
        pos_per_list.append(pos_map)
    return lists, pos_per_list


# Kendall tau distance
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

            # Case 1: both in tau* and both present
            if zi and zj and mu_i and mu_j:
                pi_l, pj_l = pos[i], pos[j]
                if pi_l == pj_l: total += 0.5
                else: total += 1.0 if ((pos_star[i] < pos_star[j]) != (pi_l < pj_l)) else 0.0
                continue

            # Case 2A: both in tau*, exactly one present
            if zi and zj and (mu_i ^ mu_j):
                appearing = i if mu_i else j
                missing = j if mu_i else i
                total += 0.0 if pos_star[appearing] < pos_star[missing] else 1.0
                continue

            # Case 2B: exactly one in tau*, both present
            if (zi ^ zj) and mu_i and mu_j:
                if pos[i] == pos[j]: total += 0.5
                else:
                    if zi == 1 and zj == 0:
                        total += 1.0 if pos[j] < pos[i] else 0.0
                    else:
                        total += 1.0 if pos[i] < pos[j] else 0.0
                continue

            # Case 3: exactly one in tau*, exactly one present (other)
            if (zi ^ zj) and (mu_i ^ mu_j):
                triggers = (zi == 1 and zj == 0 and (not mu_i) and mu_j) or \
                           (zi == 0 and zj == 1 and mu_i and (not mu_j))
                if triggers: total += 1.0
                continue

            # Case 4: both in tau*, neither present OR neither in tau*, both present
            if (zi and zj and (not mu_i) and (not mu_j)) or ((not zi) and (not zj) and mu_i and mu_j):
                total += p
                continue

    return total


# Load and truncate data
df = pd.read_csv(CSV, header=None, dtype=str)
orig_rows = df.shape[0]
df = df.iloc[:max(1, int(orig_rows * ROW_FRACTION)), :]
lists, pos_per_list = _build_lists_and_positions_tieaware(df)
m = len(lists)
k = df.shape[0]
items = sorted(set().union(*map(set, lists))) if lists else []
n = len(items)
idx = {it: i for i, it in enumerate(items)}

print(f"n: {n}, m: {m}, k: {k}  (orig_rows={orig_rows}, kept≈{int(orig_rows*ROW_FRACTION)})")


# Footrule cost matrix
ranks = np.full((m, n), float(k+1))
for ell in range(m):
    pos_map = pos_per_list[ell]
    for it, pos in pos_map.items():
        if it in idx:
            ranks[ell, idx[it]] = pos + 1

r_items = ranks.T[:, :, None]                 
positions = np.arange(1, n+1, dtype=float)[None, None, :]  
W = np.abs(r_items - positions).sum(axis=1)        
row_ind, col_ind = linear_sum_assignment(W)
def _topk_from_assignment(row_ind, col_ind, k):
    assigned_pos = col_ind + 1
    perm = sorted(zip(row_ind, assigned_pos), key=lambda t: t[1])
    return [items[i] for i, p in perm if p <= k]
topk = _topk_from_assignment(row_ind, col_ind, k)
print("Top-k list (assignment):")
print(f"[{', '.join(str(x) for x in topk)}]")
val = kendall_tau_p_cases_from_lists(lists, pos_per_list, topk, P)
print(f"Kendall tau: {val:.1f}")


#local search (LS)
def _total_obj(order):
    return kendall_tau_p_cases_from_lists(lists, pos_per_list, order, P)
def _pair_cost_in_one_list_cases(a, b, pa, pb, pos_l, p):
    i2 = (a in pos_l)
    j2 = (b in pos_l)
    if i2 and j2:
        if pos_l[a] == pos_l[b]: return 0.5
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
time_before_swaps = time.time()
work = topk[:]
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
        delta = _delta_adjacent_swap_cases(a, b, pos_live, pos_per_list, P)
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
print(f"Final distance after swaps = {curr:.1f}")
end = time.time()
time_until_swaps = time_before_swaps - start
swaps_time = end - time_before_swaps
print(f"\nTime until swaps: {time_until_swaps:.3f} seconds")
print(f"Time for swaps: {swaps_time:.3f} seconds")
print(f"{val:.1f},{curr:.1f},{time_until_swaps:.3f},{swaps_time:.3f}")

