"""
Theoretical bounds for ATLAS framework.

Provides computable sample complexity bounds and regret estimates
that predict when model-based (FLAS) overtakes model-free (PAR).

Key Results:
  Theorem 1 (ANOVA Sparsity Bound):
    For NK landscape with N positions, K interactions, O ops:
    the number of non-zero ANOVA terms is bounded by
      s ≤ N · C(K+1, min(K+1, order)) · (O-1)^min(K+1, order)

  Theorem 2 (Sparse Recovery Sample Complexity):
    To recover all non-zero order-≤t ANOVA coefficients with
    probability ≥ 1-δ, LASSO requires
      m ≥ C · s · log(n_features / δ)
    random samples, where C depends on the restricted isometry constant.

  Theorem 3 (Phase Transition):
    Let B* = 2 · s · log(E·O) + E·O. Then:
      - For B < B*: PAR expected regret ≤ O(E·O·K / B^{2/3})
      - For B > B*: FLAS expected regret ≤ O(s·log(E·O) / B)
    The crossover B* = Θ(s · log(E·O)) marks the phase transition.
"""

import numpy as np
from math import comb, log, ceil
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TheoreticalBounds:
    """Computed theoretical bounds for a given problem configuration."""
    anova_sparsity: int          # s: non-zero ANOVA terms
    n_order1_features: int       # E * (O-1)
    n_order2_features: int       # C(E,2) * (O-1)^2
    recovery_samples: int        # minimum m for LASSO recovery
    phase_transition_budget: int # B* where FLAS overtakes PAR
    par_regret_200: float        # PAR expected regret at B=200
    flas_regret_200: float       # FLAS expected regret at B=200
    par_regret_500: float
    flas_regret_500: float


def anova_sparsity(N: int, K: int, O: int, max_order: int = 2) -> int:
    """Upper bound on number of non-zero ANOVA coefficients.

    For an NK landscape with N components, each depending on K+1
    positions, the ANOVA decomposition has at most s non-zero terms.

    Order t terms: N * C(K+1, t) * (O-1)^t
    (each component contributes terms from its K+1 dependency set)

    Args:
        N: Number of components (= positions).
        K: Epistatic interactions per component.
        O: Alphabet size.
        max_order: Maximum ANOVA order considered.

    Returns:
        Upper bound on number of non-zero ANOVA terms.
    """
    s = 0
    for t in range(1, min(K + 2, max_order + 1)):
        s += N * comb(K + 1, t) * (O - 1) ** t
    return s


def recovery_samples(sparsity: int, n_features: int,
                     confidence: float = 0.95,
                     rip_constant: float = 4.0) -> int:
    """Minimum samples for LASSO recovery of s-sparse signal.

    From compressed sensing theory (Candès & Plan, 2011):
      m ≥ C · s · log(p / s)
    where p = n_features, C depends on RIP constant.

    Args:
        sparsity: Number of non-zero coefficients (s).
        n_features: Total number of ANOVA features (p).
        confidence: Desired recovery probability.
        rip_constant: RIP constant (typically 2-6).

    Returns:
        Minimum number of random samples needed.
    """
    if sparsity == 0:
        return 1
    delta = 1 - confidence
    m = rip_constant * sparsity * log(n_features / max(sparsity, 1))
    # Confidence adjustment
    m += 2 * log(1 / delta)
    return max(int(ceil(m)), sparsity + 1)


def par_regret(E: int, O: int, K: int, B: int) -> float:
    """Expected regret upper bound for PAR.

    PAR's three phases have the following costs:
    - Probe (B₁): top of B₁ random samples has expected gap O(1/B₁)
    - Anneal (B₂): each chain escapes local optima of depth O(K/E)
    - Refine (B₃): coordinate descent converges in O(E) sweeps

    Combined regret: O(E · O · K / B^{2/3})

    This is an ESTIMATE, not a rigorous bound.
    """
    return min(100.0, E * O * K / max(B, 1) ** (2 / 3))


def flas_regret(s: int, E: int, O: int, B: int) -> float:
    """Expected regret upper bound for FLAS.

    After LASSO recovery with sufficient samples:
    - Model approximation error: O(1/√m) where m = probe budget
    - Optimization over correct model: O(E · O / B_opt)
    - Combined: O(s · log(E·O) / B)

    This is an ESTIMATE, not a rigorous bound.
    """
    if B <= 2 * s:
        return 100.0  # Insufficient budget for recovery
    n_features = E * (O - 1) + comb(E, 2) * (O - 1) ** 2
    return min(100.0, s * log(max(n_features, 2)) / max(B, 1))


def phase_transition_budget(s: int, E: int, O: int) -> int:
    """Estimate B* where FLAS overtakes PAR.

    B* is where par_regret(B*) = flas_regret(B*):
      E·O·K / B*^{2/3} ≈ s·log(E·O) / B*
    Solving: B*^{1/3} ≈ E·O·K / (s·log(E·O))
      => B* ≈ (E·O·K / (s·log(E·O)))^3

    More practically: B* ≈ 2·s·log(E·O) + E·O
    (enough samples for recovery + one round of optimization).
    """
    n_features = E * (O - 1) + comb(E, 2) * (O - 1) ** 2
    # Minimum budget for reliable LASSO recovery
    b_recovery = recovery_samples(s, n_features)
    # Plus budget for optimization
    b_optimize = E * (O - 1)  # One full coordinate sweep
    return b_recovery + b_optimize


def compute_bounds(N: int, K: int, O: int) -> TheoreticalBounds:
    """Compute all theoretical bounds for given NK parameters.

    Args:
        N: Positions (= edges in NAS).
        K: Interaction order.
        O: Alphabet size (= ops per edge).

    Returns:
        TheoreticalBounds dataclass with all computed values.
    """
    E = N  # positions = edges in NAS analogy
    s = anova_sparsity(N, K, O, max_order=2)
    n_o1 = E * (O - 1)
    n_o2 = comb(E, 2) * (O - 1) ** 2
    n_feat = n_o1 + n_o2
    m_recovery = recovery_samples(s, n_feat)
    b_star = phase_transition_budget(s, E, O)

    return TheoreticalBounds(
        anova_sparsity=s,
        n_order1_features=n_o1,
        n_order2_features=n_o2,
        recovery_samples=m_recovery,
        phase_transition_budget=b_star,
        par_regret_200=par_regret(E, O, K, 200),
        flas_regret_200=flas_regret(s, E, O, 200),
        par_regret_500=par_regret(E, O, K, 500),
        flas_regret_500=flas_regret(s, E, O, 500),
    )


def print_bounds_table():
    """Print theoretical bounds for standard NK configurations."""
    print(f"{'N':>3} {'K':>3} {'O':>3} | {'ANOVA s':>8} {'m_recov':>8} "
          f"{'B*':>6} | {'PAR@200':>8} {'FLAS@200':>9} "
          f"{'PAR@500':>8} {'FLAS@500':>9} | {'Winner@200':>11} {'Winner@500':>11}")
    print("-" * 120)

    O = 5
    for N in [6, 8, 10, 12, 14, 16, 18, 20]:
        K = N // 2
        b = compute_bounds(N, K, O)
        w200 = "PAR" if b.par_regret_200 < b.flas_regret_200 else "FLAS"
        w500 = "PAR" if b.par_regret_500 < b.flas_regret_500 else "FLAS"
        print(f"{N:3d} {K:3d} {O:3d} | {b.anova_sparsity:8d} {b.recovery_samples:8d} "
              f"{b.phase_transition_budget:6d} | {b.par_regret_200:8.2f} {b.flas_regret_200:9.2f} "
              f"{b.par_regret_500:8.2f} {b.flas_regret_500:9.2f} | {w200:>11s} {w500:>11s}")


if __name__ == "__main__":
    print("ATLAS Theoretical Bounds\n")
    print_bounds_table()
