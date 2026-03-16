"""
NEXUS vs ATLAS Benchmark Suite

Head-to-head comparison of NEXUS (topological-spectral-IDS) against
all existing methods (REA, PAR, FLAS, ATLAS) on multiple landscape types.

Experiments:
  1. Budget sweep — phase transition behavior
  2. Dimensionality scaling — how methods scale with N
  3. Ruggedness sweep — how methods handle increasing K
  4. Topology diagnostics — what NEXUS sees in the landscape
"""

import numpy as np
from scipy import stats
import sys
import time
from typing import List, Dict, Callable

from landscapes import NKLandscape
from searchers import REA, PAR, FLAS, ATLAS
from nexus import NEXUS, NEXUSConfig


def cohen_d(a: List[float], b: List[float]) -> float:
    a_, b_ = np.array(a), np.array(b)
    na, nb = len(a_), len(b_)
    sp = np.sqrt(((na - 1) * a_.std(ddof=1)**2 + (nb - 1) * b_.std(ddof=1)**2)
                 / max(na + nb - 2, 1))
    return (a_.mean() - b_.mean()) / sp if sp > 0 else 0.0


def wilcoxon_greater(a: List[float], b: List[float]) -> float:
    diff = [x - y for x, y in zip(a, b)]
    try:
        _, p = stats.wilcoxon(diff, alternative="greater")
    except Exception:
        p = 1.0
    return p


def sig_str(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "ns "


def run_all_methods(landscape_fn, num_ops, num_edges, budget,
                    n_instances=2, n_seeds=10):
    """Run REA, PAR, FLAS, ATLAS, NEXUS on landscape."""
    methods = {
        "REA":   lambda O, E, s: REA(O, E, seed=s),
        "PAR":   lambda O, E, s: PAR(O, E, seed=s),
        "FLAS":  lambda O, E, s: FLAS(O, E, seed=s),
        "ATLAS": lambda O, E, s: ATLAS(O, E, seed=s),
        "NEXUS": lambda O, E, s: NEXUS(O, E, seed=s),
    }
    results = {name: [] for name in methods}
    diag_collection = []

    for inst in range(n_instances):
        landscape = landscape_fn(inst)
        for seed in range(n_seeds):
            for name, make in methods.items():
                searcher = make(num_ops, num_edges, seed)
                result = searcher.search(landscape, budget)
                results[name].append(result.best_fitness)
                if name == "NEXUS":
                    diag_collection.append(result.diagnostics)

    return results, diag_collection


# ═════════════════════════════════════════════════════════════════════
# Experiment 1: Budget Sweep — NEXUS vs All
# ═════════════════════════════════════════════════════════════════════

def exp_budget_sweep(N=14, K=7, O=5, n_inst=1, n_seeds=8):
    """Budget sweep comparing all 5 methods."""
    print("=" * 100)
    print(f"NEXUS BENCHMARK 1: Budget Sweep  (N={N}, K={K}, O={O})")
    print("=" * 100)

    header = (f"  {'B':>5s}  {'REA':>6s}  {'PAR':>6s}  {'FLAS':>6s}  "
              f"{'ATLAS':>6s}  {'NEXUS':>6s}  |  "
              f"{'NEX>REA':>10s}  {'NEX>PAR':>10s}  {'NEX>ATLAS':>10s}")
    print(header)
    print("-" * len(header))

    for B in [100, 200, 300, 500, 750]:
        t0 = time.time()
        results, diags = run_all_methods(
            landscape_fn=lambda inst, _N=N, _K=K, _O=O: NKLandscape(_N, _K, _O, inst),
            num_ops=O, num_edges=N, budget=B,
            n_instances=n_inst, n_seeds=n_seeds,
        )

        line = f"  {B:5d}"
        for name in ["REA", "PAR", "FLAS", "ATLAS", "NEXUS"]:
            line += f"  {np.mean(results[name]):6.1f}"
        line += "  |"

        rea = results["REA"]
        nexus = results["NEXUS"]
        for ref_name in ["REA", "PAR", "ATLAS"]:
            ref = results[ref_name]
            d = cohen_d(nexus, ref)
            p = wilcoxon_greater(nexus, ref)
            short = ref_name[:3] if ref_name != "ATLAS" else "ATL"
            line += f"  {d:+.2f}{sig_str(p)}"

        # Show NEXUS strategy distribution
        strategies = [d.get("strategy", "?") for d in diags]
        strat_counts = {}
        for s in strategies:
            strat_counts[s] = strat_counts.get(s, 0) + 1
        strat_str = " ".join(f"{k}={v}" for k, v in sorted(strat_counts.items()))

        line += f"  [{strat_str}]  ({time.time()-t0:.0f}s)"
        print(line)
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════
# Experiment 2: Dimensionality Scaling
# ═════════════════════════════════════════════════════════════════════

def exp_n_scaling(O=5, B=300, n_inst=1, n_seeds=8):
    """How does NEXUS scale with search space size?"""
    print("\n" + "=" * 100)
    print(f"NEXUS BENCHMARK 2: N Scaling  (O={O}, B={B})")
    print("=" * 100)

    for N in [8, 10, 12, 14, 16]:
        K = N // 2
        t0 = time.time()
        results, diags = run_all_methods(
            landscape_fn=lambda inst, _N=N, _K=K: NKLandscape(_N, _K, O, inst),
            num_ops=O, num_edges=N, budget=B,
            n_instances=n_inst, n_seeds=n_seeds,
        )

        nexus = results["NEXUS"]
        line = f"  N={N:2d} K={K:2d}:"
        for name in ["REA", "PAR", "FLAS", "ATLAS", "NEXUS"]:
            v = results[name]
            d_vs_rea = cohen_d(v, results["REA"])
            line += f"  {name}={np.mean(v):.1f}(d={d_vs_rea:+.2f})"

        # NEXUS topology diagnostics
        avg_basins = np.mean([d.get("topology", {}).get("n_basins", 0) for d in diags])
        avg_entropy = np.mean([d.get("topology", {}).get("persistence_entropy", 0) for d in diags])
        line += f"  topo=[basins={avg_basins:.1f} ent={avg_entropy:.2f}]"
        line += f"  ({time.time()-t0:.0f}s)"
        print(line)
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════
# Experiment 3: Ruggedness Sweep (K)
# ═════════════════════════════════════════════════════════════════════

def exp_k_sweep(N=14, O=5, B=300, n_inst=1, n_seeds=8):
    """How does NEXUS adapt to landscape ruggedness?"""
    print("\n" + "=" * 100)
    print(f"NEXUS BENCHMARK 3: K Sweep  (N={N}, O={O}, B={B})")
    print("=" * 100)

    for K in [1, 3, 5, 7, 10, 13]:
        t0 = time.time()
        results, diags = run_all_methods(
            landscape_fn=lambda inst, _K=K: NKLandscape(N, _K, O, inst),
            num_ops=O, num_edges=N, budget=B,
            n_instances=n_inst, n_seeds=n_seeds,
        )

        line = f"  K={K:2d}:"
        for name in ["REA", "PAR", "ATLAS", "NEXUS"]:
            v = results[name]
            d_vs_rea = cohen_d(v, results["REA"])
            line += f"  {name}={np.mean(v):.1f}(d={d_vs_rea:+.2f})"

        # Show topology adaptation
        strategies = [d.get("strategy", "?") for d in diags]
        complexity = np.mean([d.get("topology", {}).get("complexity_score", 0) for d in diags])
        line += f"  complexity={complexity:.2f} strats={set(strategies)}"
        line += f"  ({time.time()-t0:.0f}s)"
        print(line)
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════
# Experiment 4: Topology Diagnostics Deep Dive
# ═════════════════════════════════════════════════════════════════════

def exp_topology_diagnostics(N=14, K=7, O=5, B=300, n_seeds=5):
    """Show what NEXUS's topological probe discovers about landscapes."""
    print("\n" + "=" * 100)
    print(f"NEXUS TOPOLOGY DIAGNOSTICS  (N={N}, K={K}, O={O}, B={B})")
    print("=" * 100)

    for K in [1, 4, 7, 10, 13]:
        landscape = NKLandscape(N, K, O, seed=0)
        print(f"\n  K={K} (interaction order):")
        print(f"  {'seed':>4s}  {'basins':>6s}  {'entropy':>7s}  {'fractal':>7s}  "
              f"{'max_pers':>8s}  {'spectral':>8s}  {'strategy':>15s}  {'fitness':>7s}")

        for seed in range(n_seeds):
            searcher = NEXUS(O, N, seed=seed)
            result = searcher.search(landscape, budget=B)
            d = result.diagnostics
            t = d.get("topology", {})
            print(f"  {seed:4d}  {t.get('n_basins', 0):6d}  "
                  f"{t.get('persistence_entropy', 0):7.3f}  "
                  f"{t.get('fractal_estimate', 0):7.3f}  "
                  f"{t.get('max_persistence', 0):8.3f}  "
                  f"{d.get('spectral_gap', 0):8.3f}  "
                  f"{d.get('strategy', '?'):>15s}  "
                  f"{result.best_fitness:7.2f}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEXUS benchmark suite")
    parser.add_argument("--exp", type=int, default=0,
                        help="Experiment (0=all, 1=budget, 2=N-scale, 3=K-sweep, 4=topology)")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced seeds for quick testing")
    args = parser.parse_args()

    ni = 1
    ns = 5 if args.fast else 10

    if args.exp in (0, 1):
        exp_budget_sweep(n_inst=ni, n_seeds=ns)
    if args.exp in (0, 2):
        exp_n_scaling(n_inst=ni, n_seeds=ns)
    if args.exp in (0, 3):
        exp_k_sweep(n_inst=ni, n_seeds=ns)
    if args.exp in (0, 4):
        exp_topology_diagnostics(n_seeds=ns)
