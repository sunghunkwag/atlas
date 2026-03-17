"""Benchmark: SURGE-Partial (gray-box partial eval) vs NEXUS.
Same TOTAL COMPUTATION budget (B * N subfunction evals).
3 inst \u00d7 80 seeds = 240 pairs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import stats
from landscapes import NKLandscape
from nexus import NEXUS
from surge_partial import SURGEPartial, PartialConfig

N, K, O = 14, 7, 5
N_INSTANCES = 3
N_SEEDS = 80
BUDGETS = [300, 500, 750]

results = {b: {"nexus": [], "partial": []} for b in BUDGETS}
diag_samples = {b: [] for b in BUDGETS}

for inst in range(N_INSTANCES):
    nk = NKLandscape(N, K, O, seed=inst)
    for seed in range(N_SEEDS):
        for b in BUDGETS:
            r_n = NEXUS(O, N, seed=seed).search(nk, budget=b)
            r_p = SURGEPartial(O, N, nk_landscape=nk, seed=seed).search(nk, budget=b)
            results[b]["nexus"].append(r_n.best_fitness)
            results[b]["partial"].append(r_p.best_fitness)
            if inst == 0 and seed < 3:
                diag_samples[b].append(r_p.diagnostics)
    print(f"Instance {inst} done")

# Diagnostics
print("\n=== Diagnostics (inst=0, seeds 0-2) ===")
for b in BUDGETS:
    for d in diag_samples[b]:
        print(f"  B={b}: ILS_iters={d.get('ils_iterations', '?')}, "
              f"full_evals={d.get('full_evals', '?')}, "
              f"effective_full_evals={d.get('effective_full_evals', '?'):.0f}")

print(f"\n=== SURGE-Partial (gray-box) vs NEXUS (black-box) on NK({N},{K},{O}) ===")
print(f"  {N_INSTANCES}\u00d7{N_SEEDS}={N_INSTANCES*N_SEEDS} pairs")
print(f"  Same total computation budget: B \u00d7 N subfunction evals\n")

for b in BUDGETS:
    nx = np.array(results[b]["nexus"])
    sg = np.array(results[b]["partial"])
    delta = sg.mean() - nx.mean()
    wins = int((sg > nx + 0.01).sum())
    losses = int((sg < nx - 0.01).sum())
    diff = sg - nx
    d_eff = diff.mean() / (diff.std() + 1e-10)
    nonzero = diff[np.abs(diff) > 0.001]
    p = stats.wilcoxon(nonzero)[1] if len(nonzero) > 5 else 1.0
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    print(f"  B={b:>3}: NEXUS={nx.mean():.2f}  Partial={sg.mean():.2f}  "
          f"delta={delta:>+.3f}  W/L={wins}/{losses}  d={d_eff:.3f}  p={p:.4f}{sig}")
