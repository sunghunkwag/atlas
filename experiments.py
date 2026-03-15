"""
Experimental evaluation suite for ATLAS.

Reproduces all key results:
  1. Budget sweep (phase transition)
  2. N-scaling (search space size)
  3. K-sweep (interaction order)
  4. Ablation study
  5. NAS-Bench-201 surrogate
"""

import numpy as np
from scipy import stats
import sys
import time
from typing import List, Tuple, Dict, Callable

from landscapes import NKLandscape, NASBench201Surrogate, SyntheticNASLandscape
from searchers import REA, PAR, FLAS, ATLAS, PARConfig, FLASConfig, ATLASConfig
from theory import compute_bounds


def cohen_d(a: List[float], b: List[float]) -> float:
    a_, b_ = np.array(a), np.array(b)
    na, nb = len(a_), len(b_)
    sp = np.sqrt(((na - 1) * a_.std(ddof=1)**2 + (nb - 1) * b_.std(ddof=1)**2)
                 / (na + nb - 2))
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


def run_comparison(
    landscape_fn: Callable[[], Callable],
    num_ops: int,
    num_edges: int,
    budget: int,
    n_instances: int = 2,
    n_seeds: int = 12,
    methods: Dict[str, Callable] = None,
) -> Dict[str, List[float]]:
    """Run all methods on multiple landscape instances and seeds."""
    if methods is None:
        methods = {
            "REA":   lambda O, E, s: REA(O, E, seed=s),
            "PAR":   lambda O, E, s: PAR(O, E, seed=s),
            "FLAS":  lambda O, E, s: FLAS(O, E, seed=s),
            "ATLAS": lambda O, E, s: ATLAS(O, E, seed=s),
        }

    results: Dict[str, List[float]] = {name: [] for name in methods}

    for inst in range(n_instances):
        landscape = landscape_fn(inst)
        for seed in range(n_seeds):
            for name, make_searcher in methods.items():
                searcher = make_searcher(num_ops, num_edges, seed)
                result = searcher.search(landscape, budget)
                results[name].append(result.best_fitness)

    return results


def print_results(results: Dict[str, List[float]], label: str = ""):
    """Print comparison table against REA baseline."""
    if label:
        print(f"\n{label}")
    rea = results.get("REA", [])
    for name in results:
        v = results[name]
        if name == "REA":
            print(f"  REA       : {np.mean(v):.2f}±{np.std(v):.1f}")
            continue
        d = cohen_d(v, rea)
        p = wilcoxon_greater(v, rea)
        print(f"  {name:10s}: {np.mean(v):.2f}±{np.std(v):.1f}  "
              f"d={d:+.3f}  p={p:.4f}  {sig_str(p)}")


# ═════════════════════════════════════════════════════════════════════
# Experiment 1: Budget Sweep (Phase Transition)
# ═════════════════════════════════════════════════════════════════════

def exp1_budget_sweep(N: int = 14, K: int = 7, O: int = 5,
                      n_inst: int = 2, n_seeds: int = 12):
    """Show phase transition: PAR wins at low B, FLAS at high B."""
    print("=" * 80)
    print(f"EXP 1: Budget Sweep  (N={N}, K={K}, O={O})")
    print("=" * 80)

    bounds = compute_bounds(N, K, O)
    print(f"Theory: ANOVA sparsity={bounds.anova_sparsity}, "
          f"B*={bounds.phase_transition_budget}")

    header = (f"  {'B':>5s}  {'REA':>6s}  {'PAR':>6s}  {'FLAS':>6s}  {'ATLAS':>6s}  |  "
              f"{'PAR>REA':>10s}  {'FLAS>REA':>10s}  {'ATLAS>REA':>10s}  {'FLAS>PAR':>10s}")
    print(header)
    print("-" * len(header))

    for B in [100, 200, 300, 500, 750, 1000]:
        t0 = time.time()
        results = run_comparison(
            landscape_fn=lambda inst, _N=N, _K=K, _O=O: NKLandscape(_N, _K, _O, inst),
            num_ops=O, num_edges=N, budget=B,
            n_instances=n_inst, n_seeds=n_seeds,
        )

        rea = results["REA"]
        line = f"  {B:5d}"
        for name in ["REA", "PAR", "FLAS", "ATLAS"]:
            line += f"  {np.mean(results[name]):6.1f}"
        line += "  |"
        for name in ["PAR", "FLAS", "ATLAS"]:
            d = cohen_d(results[name], rea)
            p = wilcoxon_greater(results[name], rea)
            line += f"  {d:+.2f}{sig_str(p)}"
        d_fp = cohen_d(results["FLAS"], results["PAR"])
        p_fp = wilcoxon_greater(results["FLAS"], results["PAR"])
        line += f"  {d_fp:+.2f}{sig_str(p_fp)}"
        line += f"  ({time.time()-t0:.0f}s)"
        print(line)
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════
# Experiment 2: N Scaling
# ═════════════════════════════════════════════════════════════════════

def exp2_n_scaling(O: int = 5, B: int = 200,
                   n_inst: int = 2, n_seeds: int = 12):
    """How do methods scale with search space dimensionality?"""
    print("=" * 80)
    print(f"EXP 2: N Scaling  (O={O}, B={B})")
    print("=" * 80)

    for N in [8, 10, 12, 14, 16]:
        K = N // 2
        t0 = time.time()
        results = run_comparison(
            landscape_fn=lambda inst, _N=N, _K=K: NKLandscape(_N, _K, O, inst),
            num_ops=O, num_edges=N, budget=B,
            n_instances=n_inst, n_seeds=n_seeds,
        )
        rea = results["REA"]
        line = f"  N={N:2d} K={K:2d}:"
        for name in ["PAR", "FLAS", "ATLAS"]:
            d = cohen_d(results[name], rea)
            p = wilcoxon_greater(results[name], rea)
            line += f"  {name}={np.mean(results[name]):.1f} d={d:+.2f}{sig_str(p)}"
        line += f"  REA={np.mean(rea):.1f}  ({time.time()-t0:.0f}s)"
        print(line)
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════
# Experiment 3: K Sweep (interaction order)
# ═════════════════════════════════════════════════════════════════════

def exp3_k_sweep(N: int = 14, O: int = 5, B: int = 500,
                 n_inst: int = 2, n_seeds: int = 12):
    """How does interaction order K affect algorithm performance?"""
    print("=" * 80)
    print(f"EXP 3: K Sweep  (N={N}, O={O}, B={B})")
    print("=" * 80)

    for K in [1, 3, 5, 7, 10, 13]:
        t0 = time.time()
        results = run_comparison(
            landscape_fn=lambda inst, _K=K: NKLandscape(N, _K, O, inst),
            num_ops=O, num_edges=N, budget=B,
            n_instances=n_inst, n_seeds=n_seeds,
        )
        rea = results["REA"]
        bounds = compute_bounds(N, K, O)
        line = f"  K={K:2d} s={bounds.anova_sparsity:5d}:"
        for name in ["PAR", "FLAS", "ATLAS"]:
            d = cohen_d(results[name], rea)
            p = wilcoxon_greater(results[name], rea)
            line += f"  {name} d={d:+.2f}{sig_str(p)}"
        line += f"  ({time.time()-t0:.0f}s)"
        print(line)
        sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════
# Experiment 4: Ablation
# ═════════════════════════════════════════════════════════════════════

def exp4_ablation(N: int = 14, K: int = 7, O: int = 5, B: int = 200,
                  n_inst: int = 2, n_seeds: int = 15):
    """Which components of PAR matter most?"""
    print("=" * 80)
    print(f"EXP 4: PAR Ablation  (N={N}, K={K}, O={O}, B={B})")
    print("=" * 80)

    methods = {
        "PAR_Full": lambda O_, E, s: PAR(O_, E, PARConfig(
            probe_fraction=0.15, n_starts=4, n_anneal_steps=10), seed=s),
        "No_Anneal": lambda O_, E, s: PAR(O_, E, PARConfig(
            probe_fraction=0.15, n_starts=1, n_anneal_steps=0), seed=s),
        "No_Multi": lambda O_, E, s: PAR(O_, E, PARConfig(
            probe_fraction=0.15, n_starts=1, n_anneal_steps=10), seed=s),
        "REA": lambda O_, E, s: REA(O_, E, seed=s),
    }

    results = run_comparison(
        landscape_fn=lambda inst: NKLandscape(N, K, O, inst),
        num_ops=O, num_edges=N, budget=B,
        n_instances=n_inst, n_seeds=n_seeds,
        methods=methods,
    )
    print_results(results, f"N={N}, K={K}, B={B}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATLAS experiments")
    parser.add_argument("--exp", type=int, default=0,
                        help="Experiment number (0=all, 1-4)")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced seeds/instances for quick testing")
    args = parser.parse_args()

    ni = 1 if args.fast else 2
    ns = 8 if args.fast else 12

    if args.exp in (0, 1):
        exp1_budget_sweep(n_inst=ni, n_seeds=ns)
    if args.exp in (0, 2):
        exp2_n_scaling(n_inst=ni, n_seeds=ns)
    if args.exp in (0, 3):
        exp3_k_sweep(n_inst=ni, n_seeds=ns)
    if args.exp in (0, 4):
        exp4_ablation(n_inst=ni, n_seeds=ns)
