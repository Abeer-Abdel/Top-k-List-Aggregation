import time
start = time.time()
import os
import re
import itertools
import random
from collections import Counter
import pandas as pd


# Configuration
p = 0.5
CSV  = "dataset.csv"
ROW_FRACTION = 1.0
SEED         = 42
LOCAL_OPT_METHOD = "bubble_back"   # "bubble_back" or "repeat_passes"
TOL = 1e-9




# Load data
df = pd.read_csv(CSV, header=None, dtype=str)
orig_rows = df.shape[0]
df = df.iloc[:max(1, int(orig_rows * ROW_FRACTION)), :]
k = df.shape[0]   
m = df.shape[1]


# Cell parsing 
def parse_cell(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "":
        return []
    if re.fullmatch(r"[-+]?\d+", s):
        return [int(s)]
    return [int(x.strip()) for x in s.split("&") if x.strip()]


# Build per-list item orders and position maps 
lists_items = []
pos_per_list = []

for c in df.columns:
    col = df[c]
    seen = set()
    items_list = []
    pos_map = {}
    pos = 0

    for r in range(len(col)):
        grp = parse_cell(col.iloc[r])
        if not grp:
            continue

        new_any = False
        for it in grp:
            if it not in seen:
                items_list.append(it)
                seen.add(it)
                pos_map[it] = pos  
                new_any = True

        if new_any:
            pos += 1

    lists_items.append(items_list)
    pos_per_list.append(pos_map)


universe = sorted({it for mp in pos_per_list for it in mp})
print(f"m = {m}, k = {k}, n = {len(universe)}")


# Kendall tau distance calculation
def kendall_p_ties_between(pos1, pos2, p):
    total = 0.0

    for i, j in itertools.combinations(universe, 2):
        i1, j1 = (i in pos1), (j in pos1)
        i2, j2 = (i in pos2), (j in pos2)

        # Case 1
        if i1 and j1 and i2 and j2:
            tie1 = (pos1[i] == pos1[j])
            tie2 = (pos2[i] == pos2[j])
            if not tie1 and not tie2:
                if (pos1[i] - pos1[j]) * (pos2[i] - pos2[j]) < 0:
                    total += 1.0
            elif tie1 ^ tie2:
                total += 0.5

        # Case 2 - a 
        elif (i1 and j1) and (i2 ^ j2):
            total += (
                0.5 if (pos1[i] == pos1[j])
                else 0.0 if (i2 and pos1[i] < pos1[j]) or (j2 and pos1[j] < pos1[i])
                else 1.0
            )

        # Case 2 - b
        elif (i2 and j2) and (i1 ^ j1):
            total += (
                0.5 if (pos2[i] == pos2[j])
                else 0.0 if (i1 and pos2[i] < pos2[j]) or (j1 and pos2[j] < pos2[i])
                else 1.0
            )

        # Case 3
        elif (i1 and not j1 and j2 and not i2) or (j1 and not i1 and i2 and not j2):
            total += 1.0

        # Case 4
        elif (i1 and j1 and not i2 and not j2) or (i2 and j2 and not i1 and not j1):
            total += p

    return total

def tau_p_distance_for_order(order):
    pos_star = {it: r for r, it in enumerate(order)}
    return sum(
        kendall_p_ties_between(pos_star, pv, p=p)
        for pv in pos_per_list
    )


# Incremental swap cost
def _pair_cost_in_one_list(a, b, pos_order, pos_list, p):
    pa, pb = pos_order[a], pos_order[b]
    i2, j2 = (a in pos_list), (b in pos_list)

    # both in this list
    if i2 and j2:
        tie2 = (pos_list[a] == pos_list[b])
        if pa == pb:  
            return 0.5 if not tie2 else 0.0
        return (
            1.0 if ((pa - pb) * (pos_list[a] - pos_list[b]) < 0 and not tie2)
            else (0.5 if tie2 else 0.0)
        )

    # exactly one present
    elif i2 ^ j2:
        if (i2 and pa < pb) or (j2 and pb < pa):
            return 0.0
        return 1.0

    # neither present
    else:
        return p

class _SwappedView(dict):
    """View of pos_order with a and b swapped (no copy)."""
    __slots__ = ("_base", "_a", "_b", "_pa", "_pb")

    def __init__(self, base, a, b, pa, pb):
        self._base = base
        self._a, self._b = a, b
        self._pa, self._pb = pa, pb

    def __contains__(self, key):
        return key in self._base

    def __getitem__(self, key):
        if key == self._a:
            return self._pb
        if key == self._b:
            return self._pa
        return self._base[key]

def _delta_for_adjacent_swap(a, b, pos_order, pos_lists, p):
    pa, pb = pos_order[a], pos_order[b]
    before = after = 0.0
    pos_order_after = _SwappedView(pos_order, a, b, pa, pb)

    for L in pos_per_list:
        before += _pair_cost_in_one_list(a, b, pos_order, L, p)
        after  += _pair_cost_in_one_list(a, b, pos_order_after, L, p)

    return after - before


# Local search (LS)
def refine_adjacent(order, method=LOCAL_OPT_METHOD):
    work = order[:]
    pos_live = {it: r for r, it in enumerate(work)}
    curr = tau_p_distance_for_order(work)

    if method == "bubble_back":
        t = 0
        while t < len(work) - 1:
            a, b = work[t], work[t + 1]
            delta = _delta_for_adjacent_swap(a, b, pos_live, pos_per_list, p)
            if delta + TOL < 0.0:
                work[t], work[t + 1] = b, a
                pos_live[a], pos_live[b] = pos_live[b], pos_live[a]
                curr += delta
                t = 0
            else:
                t += 1
    else:
        while True:
            accepted = 0
            for t in range(len(work) - 1):
                cand = work[:]
                cand[t], cand[t + 1] = cand[t + 1], cand[t]
                cand_val = tau_p_distance_for_order(cand)
                if cand_val + TOL < curr:
                    work, curr = cand, cand_val
                    accepted += 1
            if accepted == 0:
                break

    return work, curr


# SNA: Sort by Number of Appearances
freq_counts = Counter()
for mp in pos_per_list:
    for it in mp:
        freq_counts[it] += 1

def random_tiebreak_sorted(items, keyfunc, seed=None):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    return sorted(items, key=keyfunc)

def tie_groups_by_key(items, keyfunc):
    decorated = [(keyfunc(x), x) for x in items]
    decorated.sort(key=lambda t: (t[0], t[1]))

    groups, curk, cur = [], None, []
    for kx, x in decorated:
        if curk is None or kx != curk:
            if cur:
                groups.append(cur)
            curk, cur = kx, [x]
        else:
            cur.append(x)

    if cur:
        groups.append(cur)
    return groups

def groups_to_string(groups):
    return "[" + ", ".join("&".join(str(x) for x in g) for g in groups) + "]"


# Run SNA + local refinement
print("\nSNA (Sort by Number of Appearances)")
key_SNA = lambda x: (-freq_counts[x],)
pre_groups = tie_groups_by_key(universe, key_SNA)
print("Full Ranking (With ties):", groups_to_string(pre_groups))
strict = random_tiebreak_sorted(universe, key_SNA, seed=SEED)[:k]
dist0 = tau_p_distance_for_order(strict)
print("Random strict order (k items):", strict)
print(f"Kendall Tau distance = {dist0:.1f}")

#Time Summary
time_before_swaps = time.time()
best_order, best_dist = refine_adjacent(strict)
print("Strict order after swaps:", best_order)
print(f"Kendall Tau distance after swaps = {best_dist:.1f}")
end = time.time()
time_until_swaps = time_before_swaps - start
swaps_time = end - time_before_swaps
print(f"\nTime until swaps: {time_until_swaps:.3f} seconds")
print(f"Time for swaps: {swaps_time:.3f} seconds")



