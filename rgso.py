"""
RGSO: Renormalization Group Search Optimization

Built on top of the NEXUS search engine with one key addition:
UNCERTAINTY-GUIDED PERTURBATION (UGP).

During coordinate descent, each variable settles on a value with some
"margin" — how much better the chosen value was vs. the next-best
alternative. Variables with THIN margins sit at the BOUNDARY of the
current basin of attraction.

Standard ILS perturbs RANDOM variables (uniform probability).
RGSO perturbs UNCERTAIN variables (inverse-margin probability).

This targets the weakest points in the local optimum's basin boundary,
maximizing the probability of crossing into a new (potentially better)
basin with each perturbation.

The engine matches NEXUS exactly (probe, anneal, importance-ordered CD,
ILS, warm-started evolution) with the single change of using margin-based
perturbation targeting instead of uniform random selection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from searchers import (
    BaseSearcher, SearchResult, EvalTracker, EvalFn, Architecture
)
from apex import WalshFeatureEngine, VariableInteractionGraph


# ═══════════════════════════════════════════════════════════════════════
# 1. COARSENING ENGINE (retained for API compatibility + future use)
# ═══════════════════════════════════════════════════════════════════════

class CoarsenedVariable:
    """A super-variable representing a merged pair of original variables."""
    __slots__ = ("var_i", "var_j", "mapping", "n_values")

    def __init__(self, var_i: int, var_j: int,
                 mapping: List[Tuple[int, int]], n_values: int):
        self.var_i = var_i
        self.var_j = var_j
        self.mapping = mapping
        self.n_values = n_values

    def decode(self, super_val: int) -> Tuple[int, int]:
        return self.mapping[super_val]


class CoarseningLevel:
    """One level of the RG hierarchy."""

    def __init__(self, super_vars: List[CoarsenedVariable],
                 remainder: List[int], n_ops_original: int):
        self.super_vars = super_vars
        self.remainder = remainder
        self.O_original = n_ops_original
        self.n_super = len(super_vars)
        self.n_total = self.n_super + len(remainder)

    def decode_solution(self, super_sol: list,
                        base_solution: list) -> list:
        result = base_solution[:]
        for idx, sv in enumerate(self.super_vars):
            vi, vj = sv.decode(super_sol[idx])
            result[sv.var_i] = vi
            result[sv.var_j] = vj
        for idx, var_idx in enumerate(self.remainder):
            result[var_idx] = super_sol[self.n_super + idx]
        return result

    def encode_solution(self, original_sol: list) -> list:
        super_sol = []
        for sv in self.super_vars:
            vi_val = original_sol[sv.var_i]
            vj_val = original_sol[sv.var_j]
            best_k = 0
            for k, (mi, mj) in enumerate(sv.mapping):
                if mi == vi_val and mj == vj_val:
                    best_k = k
                    break
            super_sol.append(best_k)
        for var_idx in self.remainder:
            super_sol.append(original_sol[var_idx])
        return super_sol


def build_coarsening(walsh: WalshFeatureEngine,
                     vig: VariableInteractionGraph,
                     E: int, O: int) -> CoarseningLevel:
    """Build one coarsening level from Walsh/VIG analysis."""
    W = vig.W
    edges = []
    for i in range(E):
        for j in range(i + 1, E):
            if W[i, j] > 0:
                edges.append((W[i, j], i, j))
    edges.sort(reverse=True)

    matched = set()
    pairs = []
    for _, i, j in edges:
        if i not in matched and j not in matched:
            pairs.append((i, j))
            matched.add(i)
            matched.add(j)

    remainder = [i for i in range(E) if i not in matched]

    super_vars = []
    for var_i, var_j in pairs:
        configs = []
        for vi in range(O):
            for vj in range(O):
                score = 0.0
                for idx, c in walsh._sparse_idx:
                    order, edges_w, values = walsh.feature_map[idx]
                    if order == 1:
                        if edges_w[0] == var_i and values[0] == vi:
                            score += c
                        elif edges_w[0] == var_j and values[0] == vj:
                            score += c
                    elif order == 2:
                        if (edges_w[0] == var_i and edges_w[1] == var_j
                                and values[0] == vi and values[1] == vj):
                            score += c
                        elif (edges_w[0] == var_j and edges_w[1] == var_i
                              and values[0] == vj and values[1] == vi):
                            score += c
                configs.append((score, vi, vj))

        configs.sort(reverse=True)
        mapping = [(vi, vj) for _, vi, vj in configs[:O]]
        super_vars.append(CoarsenedVariable(var_i, var_j, mapping, O))

    return CoarseningLevel(super_vars, remainder, O)


# ═══════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RGSOConfig:
    # Probe
    probe_fraction: float = 0.15
    walsh_max_order: int = 2
    vig_threshold: float = 0.001

    # Anneal
    n_anneal_starts: int = 5
    anneal_steps: int = 10
    max_perturb_frac: float = 0.3

    # ILS
    ils_max_stagnant: int = 3

    # UGP
    ugp_enabled: bool = True
    ugp_uncertainty_weight: float = 1.0

    # Evolution
    evolution_pop_size: int = 25
    fallback_r2: float = 0.05


# ═══════════════════════════════════════════════════════════════════════
# 3. RGSO SEARCH — NEXUS engine + UGP perturbation
# ═══════════════════════════════════════════════════════════════════════

class RGSO(BaseSearcher):
    """Renormalization Group Search Optimization.

    NEXUS engine with uncertainty-guided perturbation (UGP).
    CD margins identify basin-boundary variables; perturbation
    targets these weak points to maximize basin escape probability.
    """

    def __init__(self, num_ops: int, num_edges: int,
                 config: RGSOConfig = None, seed: int = 0):
        super().__init__(num_ops, num_edges, seed)
        self.cfg = config or RGSOConfig()

    def search(self, eval_fn: EvalFn, budget: int) -> SearchResult:
        ev = EvalTracker(eval_fn)
        cfg = self.cfg
        E, O = self.E, self.O
        rng = self.rng

        diag: Dict[str, Any] = {"method": "RGSO"}

        # ══════════════════════════════════════════════════════════
        # PHASE 1: PROBE (identical to NEXUS)
        # ══════════════════════════════════════════════════════════
        probe_n = int(budget * cfg.probe_fraction)
        X_list, y_list = [], []
        for _ in range(probe_n):
            if ev.budget_used >= budget:
                break
            a = tuple(rng.randint(O, size=E))
            f = ev.evaluate(a)
            X_list.append(list(a))
            y_list.append(f)

        if not X_list:
            return self._result(ev, diag)

        X = np.array(X_list)
        y = np.array(y_list)

        # Edge importance (identical to NEXUS)
        edge_importance = np.zeros(E)
        for i in range(E):
            group_means = []
            for o in range(O):
                mask = X[:, i] == o
                if mask.sum() >= 2:
                    group_means.append(y[mask].mean())
            if len(group_means) >= 2:
                edge_importance[i] = np.std(group_means)
        imp_order = np.argsort(-edge_importance)

        diag["walsh_r2"] = 0.0
        diag["n_vig_components"] = 0

        # ══════════════════════════════════════════════════════════
        # PHASE 2: MULTI-START ANNEAL (identical to NEXUS)
        # ══════════════════════════════════════════════════════════
        sorted_idx = np.argsort(-y)
        starts: List[list] = []
        for idx in sorted_idx:
            if len(starts) >= cfg.n_anneal_starts - 1:
                break
            cand = list(X[idx])
            if all(sum(c[i] != cand[i] for i in range(E)) > max(2, E // 4)
                   for c in starts) or not starts:
                starts.append(cand)
        starts.append(list(rng.randint(O, size=E)))

        best_cur, best_cf = None, -np.inf
        for start in starts:
            if ev.budget_used >= budget:
                break
            cur = start[:]
            cf = ev.evaluate(tuple(cur))
            for step in range(cfg.anneal_steps):
                if ev.budget_used >= budget:
                    break
                noise = 1.0 - step / cfg.anneal_steps
                n_flip = max(1, int(E * noise * cfg.max_perturb_frac))
                cand = cur[:]
                for ed in rng.choice(E, min(n_flip, E), replace=False):
                    cand[ed] = rng.randint(O)
                f = ev.evaluate(tuple(cand))
                if f > cf:
                    cf = f
                    cur = cand
            if cf > best_cf:
                best_cf = cf
                best_cur = cur[:]

        if best_cur is None:
            return self._result(ev, diag)

        # ══════════════════════════════════════════════════════════
        # PHASE 3: ILS WITH UGP (NEXUS engine + margin tracking)
        # ══════════════════════════════════════════════════════════
        n_stagnant = 0
        ils_iterations = 0
        n_ugp_used = 0
        elite: List[Tuple[list, float]] = []
        if ev.best_arch is not None:
            elite.append((list(ev.best_arch), ev.best_fitness))

        cur = list(ev.best_arch) if ev.best_arch else best_cur
        if ev.budget_used < budget:
            cf = ev.evaluate(tuple(cur))
        else:
            cf = ev.best_fitness

        # Margin tracker: margin[i] = fitness gap (chosen vs next-best)
        margins = np.full(E, np.inf)
        # Runner-up tracker: second_best_val[i] = the value CD almost chose
        runner_up = np.zeros(E, dtype=int)

        while ev.budget_used < budget and n_stagnant < 3:
            pre_refine_best = ev.best_fitness

            # ── Importance-ordered CD (identical to NEXUS) ──
            # WITH margin tracking: record how "sure" CD is about each var
            improved_in_sweep = True
            while improved_in_sweep and ev.budget_used < budget:
                improved_in_sweep = False
                for ed in imp_order:
                    if ev.budget_used >= budget:
                        break
                    best_op, best_f = cur[ed], cf
                    second_best_f = -np.inf
                    second_best_op = -1
                    for o in range(O):
                        if o == cur[ed] or ev.budget_used >= budget:
                            continue
                        cand = cur[:]
                        cand[ed] = o
                        f = ev.evaluate(tuple(cand))
                        if f > best_f:
                            second_best_f = best_f
                            second_best_op = best_op
                            best_f = f
                            best_op = o
                        elif f > second_best_f:
                            second_best_f = f
                            second_best_op = o
                    if best_op != cur[ed]:
                        cur[ed] = best_op
                        cf = best_f
                        improved_in_sweep = True
                    # Track margin + runner-up (zero extra cost)
                    if second_best_f > -np.inf:
                        margins[ed] = best_f - second_best_f
                        runner_up[ed] = second_best_op

            elite.append((cur[:], cf))
            ils_iterations += 1

            if ev.best_fitness > pre_refine_best:
                n_stagnant = 0
            else:
                n_stagnant += 1

            if ev.budget_used >= budget:
                break

            # ── PERTURBATION: UGP (adaptive) or random ──
            # First ILS iteration: random (matches NEXUS trajectory)
            # After stagnation: switch to UGP targeting
            perturb_n = max(2, min(E // 3, 2 + n_stagnant))

            use_ugp = (cfg.ugp_enabled
                       and n_stagnant >= 1
                       and np.isfinite(margins).any())

            if use_ugp:
                # UGP: select variables to perturb weighted by uncertainty
                # Small margin → high uncertainty → more likely to be perturbed
                finite_mask = np.isfinite(margins)
                safe_margins = np.where(finite_mask, margins, margins[finite_mask].max() if finite_mask.any() else 1.0)
                uncertainty = 1.0 / (safe_margins + 1e-8)
                pert_weights = uncertainty ** cfg.ugp_uncertainty_weight
                pert_probs = pert_weights / pert_weights.sum()

                edges_to_flip = rng.choice(
                    E, min(perturb_n, E), replace=False, p=pert_probs)
                n_ugp_used += 1

                for ed in edges_to_flip:
                    cur[ed] = rng.randint(O)
            else:
                edges_to_flip = rng.choice(
                    E, min(perturb_n, E), replace=False)
                for ed in edges_to_flip:
                    cur[ed] = rng.randint(O)
            cf = ev.evaluate(tuple(cur))

        diag["ils_iterations"] = ils_iterations
        diag["n_ugp_perturbations"] = n_ugp_used
        diag["fitness_after_ils"] = ev.best_fitness

        # ══════════════════════════════════════════════════════════
        # PHASE 4: WARM-STARTED EVOLUTION (identical to NEXUS)
        # ══════════════════════════════════════════════════════════
        if ev.budget_used < budget:
            pop_size = cfg.evolution_pop_size
            pop: List[Tuple[list, float]] = []
            seen = set()
            for arch, fit in sorted(elite, key=lambda x: -x[1]):
                key = tuple(arch)
                if key not in seen:
                    pop.append((arch, fit))
                    seen.add(key)
                if len(pop) >= pop_size // 2:
                    break

            while len(pop) < pop_size and ev.budget_used < budget:
                base = list(ev.best_arch) if ev.best_arch \
                    else list(rng.randint(O, size=E))
                mutant = base[:]
                n_mut = rng.randint(1, max(2, E // 3))
                for ed in rng.choice(E, n_mut, replace=False):
                    mutant[ed] = rng.randint(O)
                f = ev.evaluate(tuple(mutant))
                pop.append((mutant, f))

            tourn_size = 5
            gen = 0
            while ev.budget_used < budget:
                gen += 1
                idx = rng.choice(len(pop),
                                 size=min(tourn_size, len(pop)),
                                 replace=False)
                parent = pop[max(idx, key=lambda i: pop[i][1])][0]
                child = parent[:]
                n_mut = 1 if rng.random() < 0.6 else 2
                for _ in range(n_mut):
                    ed = rng.randint(E)
                    child[ed] = rng.randint(O)
                f = ev.evaluate(tuple(child))
                pop.append((child, f))
                if len(pop) > pop_size:
                    pop.pop(0)

            diag["evo_generations"] = gen

        return self._result(ev, diag)

    def _result(self, ev: EvalTracker, diag: Dict) -> SearchResult:
        return SearchResult(
            best_arch=ev.best_arch,
            best_fitness=ev.best_fitness,
            total_evals=ev.total_calls,
            unique_evals=ev.unique_evals,
            history=ev.history,
            diagnostics=diag,
        )
