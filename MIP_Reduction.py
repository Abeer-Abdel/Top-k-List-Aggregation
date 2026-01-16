#Imports
import time
start = time.time()
import os
import re
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


REDUCTION_MODE = "explicit"   # "explicit" or "implicit"
base = r"C:/Users/abora/Desktop/Research/4. Top-K list/Github_Submission"
CSV  = "Testt.csv"
p    = 0.5
os.chdir(base)
df = pd.read_csv(CSV, header=None, dtype=str)



def _reduced_items_path(csv_name: str) -> str:
    root, ext = os.path.splitext(csv_name)
    return f"{root}_reduced_items{ext or '.csv'}"

def _read_reduced_items(path: str):
    if not os.path.exists(path):
        return set()
    rdf = pd.read_csv(path, header=None, dtype=str)
    red = set()
    for _, row in rdf.iterrows():
        for cell in row.values:
            if pd.isna(cell):
                continue
            for t in re.findall(r"[-+]?\d+", str(cell)):
                try:
                    red.add(int(t))
                except ValueError:
                    pass
    return red

reduced_items_file = _reduced_items_path(CSV)
reduced_items = _read_reduced_items(reduced_items_file)



def parse_cell(cell):
    if pd.isna(cell): return []
    s = str(cell).strip()
    if s == "": return []
    if re.fullmatch(r"[-+]?\d+", s):
        return [int(s)]
    return [int(x.strip()) for x in s.split("&") if x.strip() != ""]

lists = []
pos_per_list = []
for c in df.columns:
    col = df[c]
    seen, cleaned = set(), []
    pos_map, pos = {}, 0
    for r in range(len(col)):
        items_here = parse_cell(col.iloc[r])
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

m = len(lists)
all_items = sorted(set().union(*lists))
k = df.shape[0]




# Branching on whether explicit or implicit
if REDUCTION_MODE == "explicit":
    excluded = reduced_items
    items = [it for it in all_items if it not in excluded]
    n     = len(items)
    idx   = {it: t for t, it in enumerate(items)}
    reduced_idx = set()
    print(f"Distinct items before exclusion: {len(all_items)}")
    print(f"Remaining items (n): {n}")
    print(f"Lists (m): {m}")
    print(f"top k: {k}")
    if n < k:
        raise RuntimeError(f"After exclusion, n={n} < k={k}")
else:
    items = all_items
    n     = len(items)
    idx   = {it: t for t, it in enumerate(items)}
    reduced_idx = {idx[it] for it in reduced_items if it in idx}

    print(f"Distinct items (n): {n}")
    print(f"Lists (m): {m}")
    print(f"top k: {k}")
  
# Building S and Mu 
A = np.zeros((n, n), dtype=float)   
B = np.zeros((n, n), dtype=float)   
C = np.zeros((n, n), dtype=int)    
D_unord = np.zeros((n, n), dtype=int) 

mu = np.zeros((n, m), dtype=int)
for ell, pos in enumerate(pos_per_list):
    in_list = {it: (it in pos) for it in items}
    for it in items:
        if in_list[it]:
            mu[idx[it], ell] = 1

    # list logic
    for i_idx in range(n):
        i_it = items[i_idx]; mu_i = in_list[i_it]
        for j_idx in range(n):
            if i_idx == j_idx:
                continue
            j_it = items[j_idx]; mu_j = in_list[j_it]

            if mu_i and mu_j:
                if i_idx < j_idx:
                    D_unord[i_idx, j_idx] += 1
                if pos[j_it] < pos[i_it]:
                    A[j_idx, i_idx] += 1.0
                elif pos[j_it] == pos[i_it] and i_idx < j_idx:
                    A[j_idx, i_idx] += 0.5
                    A[i_idx, j_idx] += 0.5
            elif (not mu_i) and mu_j:
                B[j_idx, i_idx] += 1.0
            elif (not mu_i) and (not mu_j):
                C[i_idx, j_idx] += 1


Sji = A + B

both_absent  = C.copy()  
both_present = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(i+1, n):
        both_present[i, j] = int((mu[i, :] * mu[j, :]).sum())


# Building MIP
model = gp.Model("TOPk_GKT_linearized")
z   = model.addVars(n, vtype=GRB.BINARY,    name="z")          
w   = model.addVars(n, n, vtype=GRB.BINARY, name="w")           
xp  = model.addVars(n, n, lb=0.0, vtype=GRB.CONTINUOUS, name="xp")  
xpp = model.addVars([(i,j) for i in range(n) for j in range(i+1, n)],
                    lb=0.0, vtype=GRB.CONTINUOUS, name="xpp")       

for i in range(n):
    w[i, i].UB = 0
    xp[i, i].UB = 0.0

obj = gp.quicksum(
          (Sji[j, i] + p * both_absent[i, j]) * w[i, j]
          +  Sji[j, i] * xp[i, j]
          for i in range(n) for j in range(n) if i != j
      ) + gp.quicksum(
          p * both_present[i, j] * xpp[i, j]
          for (i, j) in xpp.keys()
      )
model.setObjective(obj, GRB.MINIMIZE)

# Constraints
if REDUCTION_MODE == "implicit":
    # Fix reduced items to zero
    for i in sorted(reduced_idx):
        model.addConstr(z[i] == 0, name=f"fix_zero[{i}]")

model.addConstr(gp.quicksum(z[i] for i in range(n)) == k, name="select_k")

for i in range(n):
    for j in range(i+1, n):
        model.addConstr(w[i, j] + w[j, i] <= z[i],            name=f"w_link_leZi[{i},{j}]")
        model.addConstr(w[i, j] + w[j, i] <= z[j],            name=f"w_link_leZj[{i},{j}]")
        model.addConstr(w[i, j] + w[j, i] >= z[i] + z[j] - 1, name=f"w_link_ge[{i},{j}]")

for i in range(n):
    for j in range(n):
        if i == j: continue
        model.addConstr(xp[i, j] >= z[i] - z[j], name=f"xp_lb[{i},{j}]")
model.addConstr(
    gp.quicksum(xp[i, j] for i in range(n) for j in range(n) if i != j) == k * (n - k),
    name="xp_sum"
)

for (i, j) in xpp.keys():
    model.addConstr(xpp[i, j] >= 1 - z[i] - z[j], name=f"xpp_lb[{i},{j}]")
model.addConstr(
    gp.quicksum(xpp[i, j] for (i, j) in xpp.keys()) == (n - k) * (n - k - 1) / 2,
    name="xpp_sum"
)


for h in range(n):
    for i in range(h+1, n):
        for j in range(h+1, n):
            if i == j:
                continue
            if REDUCTION_MODE == "implicit":
                if (h in reduced_idx) or (i in reduced_idx) or (j in reduced_idx):
                    continue
            model.addConstr(
                w[h, i] + w[i, j] + w[j, h] <= 2,
                name=f"tri[{h},{i},{j}]"
            )

model.update()

model.Params.PoolSearchMode = 2
model.Params.PoolSolutions  = 100
model.Params.PoolGap        = 0
model.Params.OutputFlag     = 1
model.optimize()



def order_selected_from_w(selected_idx, w_var, use_Xn=False):
    sel = list(selected_idx)
    adj   = {i: [] for i in sel}
    indeg = {i: 0  for i in sel}
    wins  = {i: 0  for i in sel}
    val = (lambda a,b: w_var[a,b].Xn) if use_Xn else (lambda a,b: w_var[a,b].X)
    for i in sel:
        for j in sel:
            if i == j: continue
            if val(i, j) > 0.5:
                adj[i].append(j); indeg[j] += 1; wins[i] += 1
    import heapq
    heap = [(-wins[i], i) for i in sel if indeg[i] == 0]
    heapq.heapify(heap)
    order, processed = [], 0
    indeg2 = indeg.copy()
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u); processed += 1
        for v in adj[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                heapq.heappush(heap, (-wins[v], v))
    if processed == len(sel): return order
    return sorted(sel, key=lambda t: (-wins[t], t))


def _build_lists_and_positions_tieaware(df_in):
    lists_, pos_per_list_ = [], []
    for c in df_in.columns:
        col = df_in[c]
        seen_, cleaned_, pos_map_, pos_ = set(), [], {}, 0
        for r in range(len(col)):
            items_here = parse_cell(col.iloc[r])
            if not items_here: continue
            new_any = False
            for it in items_here:
                if it not in seen_:
                    cleaned_.append(it); seen_.add(it)
                    pos_map_[it] = pos_
                    new_any = True
            if new_any: pos_ += 1
        lists_.append(cleaned_)
        pos_per_list_.append(pos_map_)
    return lists_, pos_per_list_

from itertools import combinations as _combinations
def kendall_tau_p_cases(csv_path, tau_star, p=0.5, excluded=None):
    df_eval = pd.read_csv(csv_path, header=None, dtype=str)
    lists_eval, pos_eval = _build_lists_and_positions_tieaware(df_eval)
    m_eval = len(lists_eval)
    excluded = excluded or set()

    tau_star_unique, seen_u = [], set()
    for x in tau_star:
        x = int(x)
        if x not in seen_u:
            tau_star_unique.append(x); seen_u.add(x)

    universe = set(tau_star_unique)
    for L in lists_eval: universe.update(L)
    universe.difference_update(excluded)
    items_eval = sorted(universe)

    z_map = {i: 0 for i in items_eval}
    pos_star = {}
    for r, it in enumerate([u for u in tau_star_unique if u in universe]):
        z_map[it] = 1; pos_star[it] = r

    total = 0.0
    for (i, j) in _combinations(items_eval, 2):
        zi, zj = z_map[i], z_map[j]
        for l in range(m_eval):
            pos = pos_eval[l]
            mu_i = (i in pos); mu_j = (j in pos)
            # Case 1
            if zi and zj and mu_i and mu_j:
                pi_star_lt_pj_star = (pos_star[i] < pos_star[j])
                pi_l = pos[i]; pj_l = pos[j]
                if pi_l == pj_l:
                    total += 0.5
                else:
                    if pi_star_lt_pj_star != (pi_l < pj_l): total += 1.0
                continue
            # Case 2A
            if zi and zj and (mu_i ^ mu_j):
                appearing = i if mu_i else j
                missing  = j if mu_i else i
                total += (0.0 if pos_star[appearing] < pos_star[missing] else 1.0)
                continue
            # Case 2B
            if (zi ^ zj) and mu_i and mu_j:
                if pos[i] == pos[j]:
                    total += 0.5
                else:
                    if zi == 1 and zj == 0:
                        total += 1.0 if (pos[j] < pos[i]) else 0.0
                    else:
                        total += 1.0 if (pos[i] < pos[j]) else 0.0
                continue
            # Case 3
            if (zi ^ zj) and (mu_i ^ mu_j):
                triggers = (zi == 1 and zj == 0 and mu_i == 0 and mu_j == 1) or \
                           (zi == 0 and zj == 1 and mu_i == 1 and mu_j == 0)
                if triggers: total += 1.0
                continue
            # Case 4a
            if zi and zj and (not mu_i) and (not mu_j):
                total += p; continue
            # Case 4b
            if (not zi) and (not zj) and mu_i and mu_j:
                total += p; continue
    return total


#Summary
if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    best_obj = model.ObjVal
    unique_orders = set()
    optimal_orderings = []

    for s in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, s)
        sol_obj = model.PoolObjVal
        if abs(sol_obj - best_obj) > 1e-9:
            continue  

        chosen = [i for i in range(n) if z[i].Xn > 0.5]
        ord_idx = order_selected_from_w(chosen, w, use_Xn=True)
        ord_items = tuple(items[i] for i in ord_idx)

        if ord_items not in unique_orders:
            unique_orders.add(ord_items)
            optimal_orderings.append(ord_items)

    if REDUCTION_MODE == "explicit":
        print(f"MIP Objective (reduced universe): {best_obj:.1f}")
    else:
        print(f"Objective Function Value: {best_obj:.1f}") 

    if len(optimal_orderings) == 0:   
        chosen = [i for i in range(n) if z[i].X > 0.5]
        ord_idx = order_selected_from_w(chosen, w, use_Xn=False)
        ord_items = [items[i] for i in ord_idx]
        print("Selected Objects (ordered): [" + ",".join(str(x) for x in ord_items) + "]")

        if REDUCTION_MODE == "explicit":
            excluded = set(all_items) - set(items)
            val_reduced = kendall_tau_p_cases(CSV, ord_items, p, excluded=excluded)
            val_full    = kendall_tau_p_cases(CSV, ord_items, p, excluded=None)
            print(f"Distance (reduced universe): {val_reduced:.1f}")
            print(f"Distance (full/original universe): {val_full:.1f}")

    elif len(optimal_orderings) == 1:
        print("Selected Objects (ordered): [" + ",".join(str(x) for x in optimal_orderings[0]) + "]")

        if REDUCTION_MODE == "explicit":
            excluded = set(all_items) - set(items)
            val_reduced = kendall_tau_p_cases(CSV, optimal_orderings[0], p, excluded=excluded)
            val_full    = kendall_tau_p_cases(CSV, optimal_orderings[0], p, excluded=None)
            print(f"Distance (reduced universe): {val_reduced:.1f}")
            print(f"Distance (full universe): {val_full:.1f}")

    else:
        print("Optimal solutions (ordered):")
        for sol in optimal_orderings:
            print("[" + ",".join(str(x) for x in sol) + "]")
            if REDUCTION_MODE == "explicit":
                excluded = set(all_items) - set(items)
                val_reduced = kendall_tau_p_cases(CSV, sol, p, excluded=excluded)
                val_full    = kendall_tau_p_cases(CSV, sol, p, excluded=None)
                print(f" Distance (reduced): {val_reduced:.1f} | Distance (full): {val_full:.1f}")
else:
    print(f"No feasible/optimal solution; status: {model.status}")
end = time.time()
print(f"\nExecution time: {end - start:.3f} seconds")