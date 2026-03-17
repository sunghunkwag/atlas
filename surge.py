"""
SURGE: Structured Unified Recombinative Guided Evolution
========================================================

A search algorithm for discrete black-box optimization that extends NEXUS
with cross-domain-inspired mechanisms:

1. GUARANTEED PERTURBATION (Information Theory):
   Each perturbed variable is always changed to a DIFFERENT value.
   NEXUS has a 1/O (20% for O=5) chance of no-op per variable flip;
   SURGE eliminates this wasted evaluation budget.

2. MARGINAL-BEST ANNEAL START (Statistics):
   One annealing start is seeded with the per-variable marginal-best
   from probe data. Replaces one random start — zero extra cost.

3. ONLINE LINKAGE LEARNING (Genetics: epistasis detection):
   Track variable interactions as a side-effect of ILS iterations.
   After perturbation + CD refinement, variables that changed value
   WITHOUT being directly perturbed must interact with perturbed
   variables. These interactions guide future perturbation to flip
   linked variable groups together.

4. EPIGENETIC LOCKING (Epigenetics: gene methylation):
   Variables that hold the SAME value across ALL discovered local
   optima are "epigenetically locked" — excluded from perturbation.
   These consensus variables are likely globally optimal. This focuses
   perturbation budget on truly variable dimensions.

5. ANTIGENIC SHIFT (Virology: viral reassortment):
   When stagnation pressure exceeds a threshold, perform a massive
   "antigenic shift": restart from the global best, then for each
   unlocked variable, randomly swap in values from an alternate elite
   solution. This combines structural elements from different basins,
   analogous to how influenza viruses reassort genome segments between
   strains during co-infection, producing novel variants that escape
   existing immune pressure.

6. ADAPTIVE MUTATION PRESSURE (Virology: mutation rate):
   Normal perturbation (point mutation) at low stagnation; escalating
   to hypermutation at moderate stagnation; culminating in antigenic
   shift at high stagnation. Mirrors how viral mutation rate increases
   under selective pressure.

All mechanisms are zero or near-zero extra evaluation cost.

References:
  - ILS-LL2: Tinós et al., "Iterated Local Search with Linkage Learning"
    (ACM TELO, 2024; arxiv 2410.01583)
  - Extremal Optimization: Boettcher & Percus (2000), Bak-Sneppen model
  - Antigenic Shift: Webster & Laver (1971), influenza reassortment
  - Phase-Transition-RSI: sunghunkwag/Phase-Transition-RSI
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any

from searchers import (
    BaseSearcher, SearchResult, EvalTracker, EvalFn, Architecture
)


@dataclass
class SURGEConfig:
    """Configuration for SURGE algorithm."""

    # Phase 1: Probe (match NEXUS)
    probe_fraction: float = 0.15

    # Phase 2: Annealing (match NEXUS, +1 marginal-best start)
    n_anneal_starts: int = 5
    anneal_steps: int = 10
    max_perturb_frac: float = 0.3

    # Phase 3: ILS
    ils_stagnation_limit: int = 4    # +1 vs NEXUS for antigenic shift

    # Linkage learning
    linkage_perturb_prob: float = 0.5

    # Virology-inspired
    shift_stagnation_threshold: int = 3  # antigenic shift after this many stagnant iters
    shift_swap_prob: float = 0.4         # prob of swapping each unlocked var during shift

    # Phase 4: Evolution (match NEXUS)
    evolution_pop_size: int = 25


class SURGE(BaseSearcher):
    """SURGE: extends NEXUS with cross-domain-inspired mechanisms."""

    def __init__(self, num_ops: int, num_edges: int,
                 config: SURGEConfig | None = None, seed: int = 0):
        super().__init__(num_ops, num_edges, seed)
        self.cfg = config or SURGEConfig()

    def search(self, eval_fn: EvalFn, budget: int) -> SearchResult:
        ev = EvalTracker(eval_fn)
        cfg = self.cfg
        E, O = self.E, self.O
        rng = self.rng

        diagnostics: Dict[str, Any] = {"method": "SURGE"}

        # ══════════════════════════════════════════════════════════
        # PHASE 1: PROBE — random sampling (identical to NEXUS)
        # ══════════════════════════════════════════════════════════
        probe_n = int(budget * cfg.probe_fraction)
        X_list: List[list] = []
        y_list: List[float] = []

        for _ in range(probe_n):
            if ev.budget_used >= budget:
                break
            arch = tuple(rng.randint(O, size=E))
            fitness = ev.evaluate(arch)
            X_list.append(list(arch))
            y_list.append(fitness)

        if not X_list:
            return self._result(ev, diagnostics)

        X = np.array(X_list)
        y = np.array(y_list)

        # Compute per-variable importance via ANOVA-like variance
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

        # Compute marginal-best: per-variable value with highest avg fitness
        marginal_best = []
        for i in range(E):
            best_val, best_mean = 0, -np.inf
            for v in range(O):
                mask = X[:, i] == v
                if mask.sum() >= 1:
                    mean_f = y[mask].mean()
                    if mean_f > best_mean:
                        best_mean = mean_f
                        best_val = v
            marginal_best.append(best_val)

        # ══════════════════════════════════════════════════════════
        # PHASE 2: MULTI-START ANNEAL
        # Same as NEXUS but last start replaced with marginal-best
        # ══════════════════════════════════════════════════════════
        sorted_idx = np.argsort(-y)
        starts: List[list] = []
        for idx in sorted_idx:
            if len(starts) >= cfg.n_anneal_starts - 1:
                break
            cand = list(X[idx])
            # Ensure diversity among starts
            if all(sum(s[i] != cand[i] for i in range(E)) > max(2, E // 4)
                   for s in starts) or not starts:
                starts.append(cand)

        starts.append(marginal_best)

        best_cur: list | None = None
        best_cf = -np.inf

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
            return self._result(ev, diagnostics)

        # ══════════════════════════════════════════════════════════
        # PHASE 3: ILS with viral mutation mechanisms
        # ══════════════════════════════════════════════════════════
        n_stagnant = 0
        ils_iterations = 0
        stagnation_pressure = 0
        elite: List[Tuple[list, float]] = []

        if ev.best_arch is not None:
            elite.append((list(ev.best_arch), ev.best_fitness))

        cur = list(ev.best_arch) if ev.best_arch else best_cur
        if ev.budget_used < budget:
            cf = ev.evaluate(tuple(cur))
        else:
            cf = ev.best_fitness

        # Linkage: variable interaction graph
        interaction_count: Dict[Tuple[int, int], int] = defaultdict(int)
        prev_optimal_values: tuple | None = None
        last_perturbed_vars: Set[int] = set()

        # Epigenetics: track local optima for consensus detection
        local_optima: List[list] = []

        while (ev.budget_used < budget
               and n_stagnant < cfg.ils_stagnation_limit):

            pre_refine_best = ev.best_fitness

            # ── Coordinate Descent (importance-ordered, identical to NEXUS) ──
            improved_in_sweep = True
            while improved_in_sweep and ev.budget_used < budget:
                improved_in_sweep = False
                for ed in imp_order:
                    if ev.budget_used >= budget:
                        break
                    best_op = cur[ed]
                    best_f = cf
                    for o in range(O):
                        if o == cur[ed] or ev.budget_used >= budget:
                            continue
                        cand = cur[:]
                        cand[ed] = o
                        f = ev.evaluate(tuple(cand))
                        if f > best_f:
                            best_f = f
                            best_op = o
                    if best_op != cur[ed]:
                        cur[ed] = best_op
                        cf = best_f
                        improved_in_sweep = True

            # Record this local optimum
            local_optima.append(cur[:])

            # ── Linkage learning: detect variable interactions ──
            current_opt = tuple(cur)
            if (prev_optimal_values is not None and last_perturbed_vars):
                for i in range(E):
                    if i in last_perturbed_vars:
                        continue
                    if current_opt[i] != prev_optimal_values[i]:
                        # Variable i changed without being perturbed →
                        # it interacts with the perturbed variables
                        for j in last_perturbed_vars:
                            key = (min(i, j), max(i, j))
                            interaction_count[key] += 1
            prev_optimal_values = current_opt

            # Archive this local optimum
            elite.append((cur[:], cf))
            ils_iterations += 1

            if ev.best_fitness > pre_refine_best:
                n_stagnant = 0
                stagnation_pressure = 0
            else:
                n_stagnant += 1
                stagnation_pressure += 1

            if ev.budget_used >= budget:
                break

            # ── Epigenetic analysis: find locked (consensus) variables ──
            locked_vars: Set[int] = set()
            if len(local_optima) >= 2:
                for i in range(E):
                    values = set(opt[i] for opt in local_optima)
                    if len(values) == 1:
                        locked_vars.add(i)

            unlocked_vars = [i for i in range(E) if i not in locked_vars]
            if not unlocked_vars:
                unlocked_vars = list(range(E))  # safety: unlock all

            # ══════════════════════════════════════════════════════
            # PERTURBATION — three-tier viral mutation strategy
            # ══════════════════════════════════════════════════════

            if (stagnation_pressure >= cfg.shift_stagnation_threshold
                    and len(elite) >= 2):
                # ── ANTIGENIC SHIFT: massive viral reassortment ──
                # Restart from global best, swap unlocked vars from alternate elite
                global_best = (list(ev.best_arch) if ev.best_arch
                               else elite[0][0])
                elite_sorted = sorted(elite, key=lambda x: -x[1])

                # Find an alternate elite (different from global best)
                alternate = None
                for arch, fit in elite_sorted:
                    if tuple(arch) != tuple(global_best):
                        alternate = arch
                        break
                if alternate is None:
                    alternate = global_best

                # Reassort: start from global best, swap some unlocked vars
                cur = global_best[:]
                last_perturbed_vars = set()
                for i in unlocked_vars:
                    if rng.random() < cfg.shift_swap_prob:
                        cur[i] = alternate[i]
                        if cur[i] != global_best[i]:
                            last_perturbed_vars.add(i)

                # Add random mutations to a few unlocked variables
                n_mutations = max(1, rng.randint(1, min(4, len(unlocked_vars))))
                for _ in range(n_mutations):
                    ed = rng.choice(unlocked_vars)
                    old_val = cur[ed]
                    cur[ed] = (old_val + rng.randint(1, O)) % O
                    last_perturbed_vars.add(ed)

                cf = ev.evaluate(tuple(cur))
                stagnation_pressure = 0  # reset after shift

            else:
                # ── POINT MUTATION / HYPERMUTATION ──
                perturb_n = max(2, min(E // 3, 2 + n_stagnant))

                # Variable selection: linkage-guided or random
                if (interaction_count
                        and rng.random() < cfg.linkage_perturb_prob):
                    # Linkage-guided: seed from important variable, follow interactions
                    seed_var = int(rng.choice(imp_order[:max(1, E // 2)]))
                    neighbors = []
                    for (a, b), count in interaction_count.items():
                        if a == seed_var:
                            neighbors.append((b, count))
                        elif b == seed_var:
                            neighbors.append((a, count))

                    if neighbors:
                        neighbors.sort(key=lambda x: -x[1])
                        group = [seed_var]
                        for nb, _ in neighbors[:perturb_n - 1]:
                            if nb not in group:
                                group.append(nb)
                        while len(group) < perturb_n:
                            r = rng.randint(E)
                            if r not in group:
                                group.append(r)
                        edges_to_flip = group[:perturb_n]
                    else:
                        edges_to_flip = list(
                            rng.choice(E, min(perturb_n, E), replace=False))
                else:
                    edges_to_flip = list(
                        rng.choice(E, min(perturb_n, E), replace=False))

                # GUARANTEED CHANGE: always flip to a different value
                last_perturbed_vars = set()
                for ed in edges_to_flip:
                    old_val = cur[ed]
                    cur[ed] = (old_val + rng.randint(1, O)) % O
                    last_perturbed_vars.add(ed)
                cf = ev.evaluate(tuple(cur))

        diagnostics["ils_iterations"] = ils_iterations
        diagnostics["n_local_optima"] = len(local_optima)

        # ══════════════════════════════════════════════════════════
        # PHASE 4: WARM-STARTED EVOLUTION (identical to NEXUS)
        # ══════════════════════════════════════════════════════════
        if ev.budget_used < budget:
            pop_size = cfg.evolution_pop_size
            pop: List[Tuple[list, float]] = []
            seen: set = set()

            for arch, fit in sorted(elite, key=lambda x: -x[1]):
                key = tuple(arch)
                if key not in seen:
                    pop.append((arch, fit))
                    seen.add(key)
                if len(pop) >= pop_size // 2:
                    break

            while len(pop) < pop_size and ev.budget_used < budget:
                base = (list(ev.best_arch) if ev.best_arch
                        else list(rng.randint(O, size=E)))
                mutant = base[:]
                n_mut = rng.randint(1, max(2, E // 3))
                for ed in rng.choice(E, n_mut, replace=False):
                    mutant[ed] = rng.randint(O)
                f = ev.evaluate(tuple(mutant))
                pop.append((mutant, f))

            tourn_size = 5
            evo_gen = 0
            while ev.budget_used < budget:
                evo_gen += 1
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

            diagnostics["evo_generations"] = evo_gen

        return self._result(ev, diagnostics)

    def _result(self, ev: EvalTracker, diag: Dict) -> SearchResult:
        return SearchResult(
            best_arch=ev.best_arch,
            best_fitness=ev.best_fitness,
            total_evals=ev.total_calls,
            unique_evals=ev.unique_evals,
            history=ev.history,
            diagnostics=diag,
        )
