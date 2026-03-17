"""
CODA: CO-improvement Driven Adaptive Search

Novel mechanism: discovers higher-order variable interactions from
the TEMPORAL PATTERN of coordinate descent improvements.

Key insight: when single-variable CD improves variable i, it changes
the local landscape for all other variables. If variable j becomes
improvable AFTER i was improved (but wasn't before), then i and j
participate in a shared higher-order interaction.

This "co-improvement signal" is invisible to:
  - Walsh models (which decompose the static fitness function)
  - Pairwise interaction graphs (which measure marginal correlation)
  - Any surrogate model (which predicts f(x), not search dynamics)

The co-improvement signal captures the landscape's DYNAMIC structure:
how the optimizability of one variable depends on the state of others.
This is a fundamentally different information source than the landscape
itself — it's information about HOW SEARCH BEHAVES on the landscape.

From accumulated co-improvement data, CODA discovers variable groups
and executes GROUP-LEVEL moves that break through single-variable CD
barriers, directly addressing the order-2 model ceiling on higher-order
NK landscapes.

Cost: zero additional evaluations. Co-improvement data is extracted
from CD evaluations that are already being performed.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from searchers import (
    BaseSearcher, SearchResult, EvalTracker, EvalFn, Architecture
)
from apex import WalshFeatureEngine, VariableInteractionGraph


# ═══════════════════════════════════════════════════════════════════════
# 1. CO-IMPROVEMENT TRACKER
# ═══════════════════════════════════════════════════════════════════════

class CoImprovementTracker:
    """Tracks temporal co-improvement patterns across CD sweeps.

    During a CD sweep, records which variables improved. Across multiple
    sweeps, builds a co-improvement matrix:

      C[i][j] = number of times improving variable i in sweep t
                caused variable j to become improvable in sweep t+1
                (where j was NOT improvable in sweep t)

    This captures dynamic interaction: i's improvement unlocks j.
    """

    def __init__(self, num_edges: int):
        self.E = num_edges
        # C[i][j] = count of times i's improvement enabled j's improvement
        self.co_improve = np.zeros((num_edges, num_edges), dtype=np.float64)
        # Track which variables improved in each sweep
        self.sweep_history: List[set] = []
        # Track which variables were "stuck" (CD tried but couldn't improve)
        self.stuck_history: List[set] = []

    def record_sweep(self, improved: set, stuck: set):
        """Record one CD sweep's results.

        Args:
            improved: set of variable indices that improved this sweep
            stuck: set of variable indices that CD tried but couldn't improve
        """
        if self.sweep_history:
            prev_stuck = self.stuck_history[-1]
            prev_improved = self.sweep_history[-1]

            # Signal 1: variables stuck before, improved now → unlocked
            newly_improvable = improved & prev_stuck
            for enabler in prev_improved:
                for enabled in newly_improvable:
                    if enabler != enabled:
                        self.co_improve[enabler, enabled] += 1.0

            # Signal 2: variables that CONSISTENTLY co-improve across
            # sweeps — if i and j both improve in the same sweep,
            # they're responding to the same landscape region.
            # Weaker signal (0.3) but fires much more often.
            improved_list = sorted(improved)
            for idx_a in range(len(improved_list)):
                for idx_b in range(idx_a + 1, len(improved_list)):
                    a, b = improved_list[idx_a], improved_list[idx_b]
                    self.co_improve[a, b] += 0.3
                    self.co_improve[b, a] += 0.3

        self.sweep_history.append(improved.copy())
        self.stuck_history.append(stuck.copy())

    def get_groups(self, min_strength: float = 1.0,
                   max_group_size: int = 5) -> List[List[int]]:
        """Extract variable groups from co-improvement matrix.

        Uses a greedy clustering: start from the strongest co-improvement
        pair, expand the group by adding variables with strong mutual
        co-improvement, up to max_group_size.

        Returns list of variable groups (each group = list of indices).
        """
        # Symmetrize: if i enables j OR j enables i, they interact
        C = self.co_improve + self.co_improve.T
        groups = []
        used = set()

        # Find pairs sorted by co-improvement strength
        pairs = []
        for i in range(self.E):
            for j in range(i + 1, self.E):
                if C[i, j] >= min_strength:
                    pairs.append((C[i, j], i, j))
        pairs.sort(reverse=True)

        for strength, i, j in pairs:
            if i in used or j in used:
                continue
            group = [i, j]
            used.add(i)
            used.add(j)

            # Try to expand group
            while len(group) < max_group_size:
                best_k = -1
                best_score = min_strength
                for k in range(self.E):
                    if k in used:
                        continue
                    # Average co-improvement with existing group members
                    score = np.mean([C[k, g] + C[g, k] for g in group])
                    if score > best_score:
                        best_score = score
                        best_k = k
                if best_k >= 0:
                    group.append(best_k)
                    used.add(best_k)
                else:
                    break

            groups.append(sorted(group))

        return groups

    @property
    def n_observations(self) -> int:
        return len(self.sweep_history)

    def total_co_improvements(self) -> float:
        return float(self.co_improve.sum())


# ═══════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CODAConfig:
    # Probe phase
    probe_fraction: float = 0.15
    walsh_max_order: int = 2

    # Anneal phase
    n_anneal_starts: int = 5
    anneal_steps: int = 10
    max_perturb_frac: float = 0.3

    # ILS phase
    ils_max_stagnant: int = 3

    # Co-improvement tracking
    co_improve_min_strength: float = 0.5
    co_improve_max_group_size: int = 4
    group_cd_min_sweeps: int = 2       # min CD sweeps before group CD
    group_cd_max_evals: int = 40       # max evals per group CD attempt
    group_perturb_after: int = 1       # ILS iterations before group perturbation

    # Model screening (from APEX v8i, for late ILS)
    n_screen_candidates: int = 30
    fallback_r2: float = 0.05

    # Evolution phase
    evolution_pop_size: int = 25


# ═══════════════════════════════════════════════════════════════════════
# 3. CODA SEARCH
# ═══════════════════════════════════════════════════════════════════════

class CODA(BaseSearcher):
    """CO-improvement Driven Adaptive Search.

    Discovers higher-order variable interactions from the temporal
    pattern of CD improvements. Uses discovered groups for group-level
    CD and group-guided perturbation.
    """

    def __init__(self, num_ops: int, num_edges: int,
                 config: CODAConfig = None, seed: int = 0):
        super().__init__(num_ops, num_edges, seed)
        self.cfg = config or CODAConfig()

    def search(self, eval_fn: EvalFn, budget: int) -> SearchResult:
        ev = EvalTracker(eval_fn)
        cfg = self.cfg
        E, O = self.E, self.O
        rng = self.rng

        # ══════════════════════════════════════════════════════════
        # PHASE 1: PROBE
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
            return self._result(ev, {"method": "CODA"})

        X = np.array(X_list)
        y = np.array(y_list)

        # Edge importance (variance-based)
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

        # Walsh model
        walsh = WalshFeatureEngine(O, E, max_order=cfg.walsh_max_order)
        r2 = walsh.fit(X, y)
        use_model = (r2 > cfg.fallback_r2)

        # Co-improvement tracker
        cotracker = CoImprovementTracker(E)
        total_cd_sweeps = 0

        diag: Dict[str, Any] = {
            "method": "CODA",
            "walsh_r2": float(r2),
            "use_model": use_model,
        }

        # ══════════════════════════════════════════════════════════
        # PHASE 2: MULTI-START ANNEAL
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
        # PHASE 3: ILS WITH CO-IMPROVEMENT TRACKING + GROUP CD
        # ══════════════════════════════════════════════════════════
        n_stagnant = 0
        ils_iterations = 0
        n_group_cd_improvements = 0
        n_group_perturbs = 0
        elite: List[Tuple[list, float]] = []
        if ev.best_arch is not None:
            elite.append((list(ev.best_arch), ev.best_fitness))

        cur = list(ev.best_arch) if ev.best_arch else best_cur
        if ev.budget_used < budget:
            cf = ev.evaluate(tuple(cur))
        else:
            cf = ev.best_fitness

        # Inverse importance for perturbation weighting
        inv_imp = 1.0 / (edge_importance + 1e-8)
        inv_probs = inv_imp / inv_imp.sum()

        while ev.budget_used < budget and n_stagnant < cfg.ils_max_stagnant:
            pre_refine_best = ev.best_fitness

            # ── TRACKED CD: record which variables improve ──────
            improved_in_sweep = True
            while improved_in_sweep and ev.budget_used < budget:
                improved_in_sweep = False
                sweep_improved = set()
                sweep_stuck = set()

                for ed in imp_order:
                    if ev.budget_used >= budget:
                        break
                    best_op, best_f = cur[ed], cf
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
                        sweep_improved.add(ed)
                    else:
                        sweep_stuck.add(ed)

                # Record this sweep's pattern
                cotracker.record_sweep(sweep_improved, sweep_stuck)
                total_cd_sweeps += 1

            # ── GROUP CD: try multi-variable moves on discovered groups ──
            # Uses three candidate sources:
            #   (a) Elite transplant: copy group values from previous optima
            #   (b) Walsh-screened random: random group values, pick best pred
            #   (c) Neighborhood: slight variations of current group values
            if total_cd_sweeps >= cfg.group_cd_min_sweeps:
                groups = cotracker.get_groups(
                    min_strength=cfg.co_improve_min_strength,
                    max_group_size=cfg.co_improve_max_group_size)

                if groups:
                    group_evals_used = 0
                    for group in groups:
                        if (ev.budget_used >= budget or
                                group_evals_used >= cfg.group_cd_max_evals):
                            break

                        candidates = []

                        # (a) Elite transplant: use group values from
                        # previous local optima in the elite archive
                        for arch, _ in elite:
                            cand = cur[:]
                            for g in group:
                                cand[g] = arch[g]
                            if tuple(cand) != tuple(cur):
                                candidates.append(cand)

                        # (b) Neighborhood: flip 1-2 variables in group
                        for _ in range(min(10, O * len(group))):
                            cand = cur[:]
                            n_flip = rng.randint(1, len(group) + 1)
                            for g in rng.choice(group, min(n_flip, len(group)),
                                                replace=False):
                                cand[g] = rng.randint(O)
                            candidates.append(cand)

                        # (c) Walsh-screened random
                        if use_model:
                            raw = []
                            for _ in range(30):
                                cand = cur[:]
                                for g in group:
                                    cand[g] = rng.randint(O)
                                raw.append((walsh.predict_single(cand), cand))
                            raw.sort(key=lambda x: -x[0])
                            candidates.extend([c for _, c in raw[:5]])

                        # Deduplicate and evaluate
                        seen_cands = set()
                        best_group_cand = None
                        best_group_f = cf
                        for cand in candidates:
                            key = tuple(cand)
                            if key in seen_cands:
                                continue
                            seen_cands.add(key)
                            if ev.budget_used >= budget:
                                break
                            if group_evals_used >= cfg.group_cd_max_evals:
                                break
                            f = ev.evaluate(key)
                            group_evals_used += 1
                            if f > best_group_f:
                                best_group_f = f
                                best_group_cand = list(cand)

                        if best_group_cand is not None:
                            cur = best_group_cand
                            cf = best_group_f
                            n_group_cd_improvements += 1

            # Archive this local optimum
            elite.append((cur[:], cf))
            ils_iterations += 1

            if ev.best_fitness > pre_refine_best:
                n_stagnant = 0
            else:
                n_stagnant += 1

            if ev.budget_used >= budget:
                break

            # ── Online Walsh refit ──────────────────────────────
            if ils_iterations >= 2:
                cache_items = list(ev.cache.items())
                if len(cache_items) > len(X_list) + 20:
                    X_all = np.array([list(a) for a, _ in cache_items])
                    y_all = np.array([f for _, f in cache_items])
                    r2_new = walsh.fit(X_all, y_all)
                    use_model = (r2_new > cfg.fallback_r2)

            # ── PERTURBATION: co-improvement group guided ───────
            perturb_n = max(2, min(E // 3, 2 + n_stagnant))
            if n_stagnant >= 2:
                perturb_n = max(E // 2, perturb_n)

            groups = cotracker.get_groups(
                min_strength=cfg.co_improve_min_strength,
                max_group_size=cfg.co_improve_max_group_size)

            if (groups and ils_iterations >= cfg.group_perturb_after
                    and n_stagnant >= 1):
                # GROUP-GUIDED PERTURBATION: flip entire co-improvement
                # groups together, then fill remaining perturbation slots
                # with random variables. Screen by Walsh if available.
                n_cand = cfg.n_screen_candidates
                best_pert = None
                best_pred = -np.inf

                # Flatten groups into a priority list
                group_vars = []
                for g in groups:
                    group_vars.extend(g)
                group_vars = list(dict.fromkeys(group_vars))  # deduplicate

                for _ in range(n_cand):
                    pert = cur[:]
                    # Always flip at least one full group
                    if group_vars:
                        n_from_groups = min(len(group_vars), perturb_n)
                        edges_to_flip = list(rng.choice(
                            group_vars,
                            min(n_from_groups, len(group_vars)),
                            replace=False))
                        # Fill remaining with random
                        remaining = perturb_n - len(edges_to_flip)
                        if remaining > 0:
                            other = [e for e in range(E) if e not in edges_to_flip]
                            if other:
                                extra = rng.choice(
                                    other,
                                    min(remaining, len(other)),
                                    replace=False)
                                edges_to_flip.extend(extra)
                    else:
                        edges_to_flip = rng.choice(
                            E, min(perturb_n, E), replace=False)

                    for ed in edges_to_flip:
                        pert[ed] = rng.randint(O)

                    if use_model:
                        pred = walsh.predict_single(pert)
                    else:
                        pred = rng.random()  # random tiebreak
                    if pred > best_pred:
                        best_pred = pred
                        best_pert = pert

                cur = best_pert if best_pert is not None else cur
                n_group_perturbs += 1
            else:
                # Early ILS or no groups: random perturbation (diversity)
                edges_to_flip = rng.choice(
                    E, min(perturb_n, E), replace=False)
                for ed in edges_to_flip:
                    cur[ed] = rng.randint(O)

            cf = ev.evaluate(tuple(cur))

        diag["ils_iterations"] = ils_iterations
        diag["total_cd_sweeps"] = total_cd_sweeps
        diag["n_co_improve_groups"] = len(cotracker.get_groups())
        diag["total_co_improvements"] = cotracker.total_co_improvements()
        diag["n_group_cd_improvements"] = n_group_cd_improvements
        diag["n_group_perturbs"] = n_group_perturbs

        # ══════════════════════════════════════════════════════════
        # PHASE 4: WARM-STARTED EVOLUTION
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
