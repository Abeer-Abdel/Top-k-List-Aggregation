import time
import os
import re
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

#Setup
start = time.time()
base = r"C:/Users/abora/Desktop/Research/4. Top-K list/Github_Submission"
CSV  = "Testt.csv"
p    = 0.5
os.chdir(base)
df = pd.read_csv(CSV, header=None, dtype=str)

#Data parsing
def parse_cell(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "":
        return []
    if re.fullmatch(r"[-+]?\d+", s):
        return [int(s)]
    return [int(x.strip()) for x in s.split("&") if x.strip()]

lists = []
pos_per_list = []

for c in df.columns:
    col = df[c]
    seen = set()
    cleaned = []
    pos_map = {}
    pos = 0

    for r in range(len(col)):
        items_here = parse_cell(col.iloc[r])
        if not items_here:
            continue

        new_any = False
        for it in items_here:
            if it not in seen:
                seen.add(it)
                cleaned.append(it)
                pos_map[it] = pos   
                new_any = True

        if new_any:
            pos += 1

    lists.append(cleaned)
    pos_per_list.append(pos_map)


# Dimensions
m = len(lists)
items = sorted(set().union(*lists))
idx   = {it: t for t, it in enumerate(items)}
n     = len(items)
k     = df.shape[0]

print(f"Distinct items (n): {n}")
print(f"Lists (m): {m}")
print(f"top k: {k}")


# Preference matrices
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

    for i_idx in range(n):
        i_it = items[i_idx]
        mu_i = in_list[i_it]

        for j_idx in range(n):
            if i_idx == j_idx:
                continue

            j_it = items[j_idx]
            mu_j = in_list[j_it]

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


#Presence counts
both_absent = C.copy()
both_present = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(i + 1, n):
        both_present[i, j] = int((mu[i, :] * mu[j, :]).sum())


# Model
model = gp.Model("TOPk_GKT_linearized")
z   = model.addVars(n, vtype=GRB.BINARY, name="z")
w   = model.addVars(n, n, vtype=GRB.BINARY, name="w")
xp  = model.addVars(n, n, lb=0.0, vtype=GRB.CONTINUOUS, name="xp")
xpp = model.addVars([(i, j) for i in range(n) for j in range(i + 1, n)],
                    lb=0.0, vtype=GRB.CONTINUOUS, name="xpp")

for i in range(n):
    w[i, i].UB = 0
    xp[i, i].UB = 0.0


# Objective
obj = gp.quicksum(
          (Sji[j, i] + p * both_absent[i, j]) * w[i, j]
          + Sji[j, i] * xp[i, j]
          for i in range(n) for j in range(n) if i != j
      ) + gp.quicksum(
          p * both_present[i, j] * xpp[i, j]
          for (i, j) in xpp.keys()
      )

model.setObjective(obj, GRB.MINIMIZE)


# Constraints
model.addConstr(gp.quicksum(z[i] for i in range(n)) == k, name="select_k")

for i in range(n):
    for j in range(i + 1, n):
        model.addConstr(w[i, j] + w[j, i] <= z[i])
        model.addConstr(w[i, j] + w[j, i] <= z[j])
        model.addConstr(w[i, j] + w[j, i] >= z[i] + z[j] - 1)

for i in range(n):
    for j in range(n):
        if i != j:
            model.addConstr(xp[i, j] >= z[i] - z[j])

model.addConstr(
    gp.quicksum(xp[i, j] for i in range(n) for j in range(n) if i != j)
    == k * (n - k)
)

for (i, j) in xpp.keys():
    model.addConstr(xpp[i, j] >= 1 - z[i] - z[j])

model.addConstr(
    gp.quicksum(xpp[i, j] for (i, j) in xpp.keys())
    == (n - k) * (n - k - 1) / 2
)

for h in range(n):
    for i in range(h + 1, n):
        for j in range(h + 1, n):
            if i != j:
                model.addConstr(w[h, i] + w[i, j] + w[j, h] <= 2)


# Solve
model.Params.PoolSearchMode = 2
model.Params.PoolSolutions  = 100
model.Params.PoolGap        = 0
model.Params.OutputFlag     = 1
model.optimize()


def order_selected_from_w(selected_idx, w_var, use_Xn=False):
    sel = list(selected_idx)
    wins = {i: 0 for i in sel}

    val = (lambda a, b: w_var[a, b].Xn) if use_Xn else (lambda a, b: w_var[a, b].X)

    for i in sel:
        for j in sel:
            if i != j and val(i, j) > 0.5:
                wins[i] += 1

    return sorted(sel, key=lambda t: (-wins[t], t))

# Reporting
if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
    best_obj = model.ObjVal
    unique_orders = set()

    for s in range(model.SolCount):
        model.setParam(GRB.Param.SolutionNumber, s)
        if abs(model.PoolObjVal - best_obj) > 1e-9:
            continue

        chosen = [i for i in range(n) if z[i].Xn > 0.5]
        ord_idx = order_selected_from_w(chosen, w, use_Xn=True)
        unique_orders.add(tuple(items[i] for i in ord_idx))

    print(f"\nObjective Function Value: {best_obj:.1f}")
    for sol in unique_orders:
        print("[" + ",".join(str(x) for x in sol) + "]")
else:
    print(f"No feasible/optimal solution; status: {model.status}")
print(f"\nExecution time: {time.time() - start:.3f} seconds")
