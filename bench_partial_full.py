"""FULL verification: SURGE-Partial vs NEXUS.
5 inst \u00d7 128 seeds = 640 pairs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import stats
from landscapes import NKLandscape
from nexus import NEXUS
from surge_partial import SURGEPartial

N, K, O = 14, 7, 5
N_INSTANCES = 5
N_SEEDS = 128
BUDGETS = [300, 500, 750]

results = {b: {"nexus": [], "partial": []} for b in BUDGETS}

for inst in range(N_INSTANCES):
    nk = NKLandscape(N, K, O, seed=inst)
    for seed in range(N_SEEDS):
        for b in BUDGETS:
            r_n = NEXUS(O, N, seed=seed).search(nk, budget=b)
            r_p = SURGEPartial(O, N, nk_landscape=nk, seed=seed).search(nk, budget=b)
            results[b]["nexus"].append(r_n.best_fitness)
            results[b]["partial"].append(r_p.best_fitness)
    print(f"Instance {inst} done")

print(f"\n{'='*70}")
print(f"FINAL VERIFICATION: SURGE-Partial vs NEXUS on NK({N},{K},{O})")
print(f"  {N_INSTANCES} instances \u00d7 {N_SEEDS} seeds = {N_INSTANCES*N_SEEDS} pairs")
print(f"  Fair comparison: same total computation (B\u00d7N subfunction evals)")
print(f"{'='*70}\n")

for b in BUDGETS:
    nx = np.array(results[b]["nexus"])
    sg = np.array(results[b]["partial"])
    delta = sg.mean() - nx.mean()
    wins = int((sg > nx + 0.01).sum())
    losses = int((sg < nx - 0.01).sum())
    ties = len(nx) - wins - losses
    diff = sg - nx
    d_eff = diff.mean() / (diff.std() + 1e-10)
    nonzero = diff[np.abs(diff) > 0.001]
    if len(nonzero) > 5:
        stat, p = stats.wilcoxon(nonzero)
    else:
        p = 1.0
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""

    print(f"  B={b:>3}:")
    print(f"    NEXUS:   {nx.mean():.3f} \u00b1 {nx.std():.3f}")
    print(f"    Partial: {sg.mean():.3f} \u00b1 {sg.std():.3f}")
    print(f"    Delta:   {delta:>+.3f}")
    print(f"    W/L/T:   {wins}/{losses}/{ties}")
    print(f"    Cohen's d: {d_eff:.3f}")
    print(f"    p-value:   {p:.6f}{sig}")
    print()

# Per-instance breakdown
print("Per-instance breakdown (B=750):")
for inst in range(N_INSTANCES):
    start = inst * N_SEEDS
    end = start + N_SEEDS
    nx = np.array(results[750]["nexus"][start:end])
    sg = np.array(results[750]["partial"][start:end])
    delta = sg.mean() - nx.mean()
    diff = sg - nx
    d = diff.mean() / (diff.std() + 1e-10)
    print(f"  Instance {inst}: NEXUS={nx.mean():.2f}  Partial={sg.mean():.2f}  delta={delta:+.3f}  d={d:.3f}")
