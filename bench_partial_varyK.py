"""SURGE-Partial advantage across K values.
2 inst \u00d7 60 seeds = 120 pairs per K. B=750.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import stats
from landscapes import NKLandscape
from nexus import NEXUS
from surge_partial import SURGEPartial

N, O = 14, 5
N_INSTANCES = 2
N_SEEDS = 60

for K in [2, 4, 7, 10]:
    results = {"nexus": [], "partial": []}
    for inst in range(N_INSTANCES):
        nk = NKLandscape(N, K, O, seed=inst)
        for seed in range(N_SEEDS):
            r_n = NEXUS(O, N, seed=seed).search(nk, budget=750)
            r_p = SURGEPartial(O, N, nk_landscape=nk, seed=seed).search(nk, budget=750)
            results["nexus"].append(r_n.best_fitness)
            results["partial"].append(r_p.best_fitness)

    nx = np.array(results["nexus"])
    sg = np.array(results["partial"])
    diff = sg - nx
    delta = diff.mean()
    d = delta / (diff.std() + 1e-10)
    wins = int((sg > nx + 0.01).sum())
    losses = int((sg < nx - 0.01).sum())
    nonzero = diff[np.abs(diff) > 0.001]
    p = stats.wilcoxon(nonzero)[1] if len(nonzero) > 5 else 1.0
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""

    # Compute savings ratio: avg subfuncs per variable / N
    avg_affected = np.mean([len(set(sf_idx for sf_idx in range(N)
                                    for d_dep in [nk.dep_sets[sf_idx]]
                                    if var in d_dep))
                            for var in range(N)])
    savings = 1 - avg_affected / N

    print(f"K={K:>2}: NEXUS={nx.mean():.2f}  Partial={sg.mean():.2f}  "
          f"delta={delta:>+.3f}  d={d:.3f}  p={p:.4f}{sig}  "
          f"W/L={wins}/{losses}  savings\u2248{savings:.0%}")
