import pandas as pd
import os
from collections import Counter, defaultdict
from pathlib import Path
os.chdir("C:/Users/abora/Desktop/Research/4. Top-K list/Github_Submission")
csv_file = "Testt.csv"
df = pd.read_csv(csv_file, header=None, dtype=str)


def _parse_cell(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "":
        return []
    if "&" in s:
        return [int(x.strip()) for x in s.split("&") if x.strip() != ""]
    return [int(s)]

# Dimensions
k, m = df.shape  

# Counting frequency of appearances
all_items = []
for cell in df.values.ravel().tolist():
    all_items.extend(_parse_cell(cell))
freq_counts = Counter(all_items)

# Computing average rank
rank_sums = defaultdict(float)
rank_counts = defaultdict(int)
for col in range(m):
    col_vals = df.iloc[:, col].tolist()
    for rank, cell in enumerate(col_vals, start=1):
        items_here = _parse_cell(cell)
        for item in items_here:
            rank_sums[item] += rank
            rank_counts[item] += 1
avg_rank = {item: rank_sums[item] / rank_counts[item] for item in rank_sums}


# Thresholds
rank_threshold = k / 3
appearance_threshold = m / 3


# Items to remove
weak_items = [
    item for item in avg_rank
    if avg_rank[item] >= rank_threshold and freq_counts[item] <= appearance_threshold
]


# Summary
initial = len(avg_rank)
final = initial - len(weak_items)
reduction = 100 * (initial - final) / initial if initial > 0 else 0
print(f"Initial distinct items: {initial}")
print(f"Final distinct items:   {final}")
print(f"Percentage reduction:   {reduction:.1f}%")
print(f"Rank Threshold: {rank_threshold}")
print(f"Appearance Threshold: {appearance_threshold:.1f}")
print("Removed Objects")
for item in sorted(weak_items):
    print(f"{item} -- Avg Rank: {avg_rank[item]:.1f} -- Appearances: {freq_counts[item]}")

# Export removed items
in_path = Path(csv_file)
out_path = in_path.with_name(f"{in_path.stem}_reduced_items.csv")
pd.DataFrame(sorted(weak_items)).to_csv(out_path, index=False, header=False)


