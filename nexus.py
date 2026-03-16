"""
NEXUS: Neural EXploration via Unified Spectral-topological Search

A next-generation architecture search framework that combines three ideas
never before unified in NAS:

  1. PERSISTENT HOMOLOGY for online landscape topology characterization.
     Computes Betti numbers (β₀ = basins, β₁ = saddle loops) from sampled
     fitness values via sublevel-set filtration on the Hamming graph.
     This replaces crude heuristic complexity ratios with rigorous
     topological invariants.

  2. SPECTRAL GRAPH SURROGATE on the sample Hamming graph.
     Instead of ad-hoc ANOVA indicator features, uses eigenvectors of the
     graph Laplacian as basis functions — the mathematically correct
     Fourier analysis on discrete metric spaces. This gives multi-resolution
     landscape modeling with automatic bandwidth selection via spectral gap.

  3. INFORMATION-DIRECTED SEARCH (IDS) for budget allocation.
     Models each search phase's value as a distribution and allocates
     budget to maximize expected information gain about the global optimum,
     replacing fixed-fraction heuristics with a principled explore-exploit
     tradeoff.

Additionally introduces:
  4. DISCRETE CURVATURE TENSOR estimation for interaction-aware local search.
     Estimates the discrete Hessian H[i,j] around the current best to
     identify strongly-interacting edge pairs for joint optimization.

References:
  - Edelsbrunner & Harer (2010) Computational Topology.
  - Chung (1997) Spectral Graph Theory.
  - Russo & Van Roy (2018) Learning to Optimize via Information-Directed
    Sampling. NeurIPS.
  - Kauffman (1993) Origins of Order. NK Landscapes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from searchers import (
    BaseSearcher, SearchResult, EvalTracker, EvalFn, Architecture
)


# ═══════════════════════════════════════════════════════════════════════
# 1. PERSISTENT HOMOLOGY — Topological Landscape Fingerprinting
# ═══════════════════════════════════════════════════════════════════════

class UnionFind:
    """Weighted quick-union with path compression for β₀ tracking."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True


@dataclass
class PersistenceDiagram:
    """Birth-death pairs from sublevel set filtration."""
    pairs: List[Tuple[float, float]]  # (birth, death) — birth > death for superlevel
    n_points: int = 0

    @property
    def lifetimes(self) -> np.ndarray:
        if not self.pairs:
            return np.array([0.0])
        return np.array([b - d for b, d in self.pairs])

    @property
    def n_significant(self) -> int:
        """Count of persistent features (lifetime > median)."""
        lt = self.lifetimes
        if len(lt) <= 1:
            return len(lt)
        return int(np.sum(lt > np.median(lt)))


@dataclass
class TopologicalFingerprint:
    """Complete topological characterization of a fitness landscape."""
    n_basins: int                    # β₀: connected components (significant)
    total_persistence: float         # Σ lifetimes — overall ruggedness
    max_persistence: float           # Deepest basin depth
    persistence_entropy: float       # Shannon entropy of normalized lifetimes
    mean_basin_depth: float          # Average basin depth
    fractal_estimate: float          # Estimated box-counting dimension
    persistence_diagram: PersistenceDiagram = field(repr=False, default=None)

    @property
    def is_multimodal(self) -> bool:
        return self.n_basins >= 3

    @property
    def is_rugged(self) -> bool:
        return self.persistence_entropy > 0.7

    @property
    def complexity_score(self) -> float:
        """Single scalar summarizing topological complexity."""
        return (
            0.3 * min(self.n_basins / 10.0, 1.0)
            + 0.3 * min(self.persistence_entropy, 1.0)
            + 0.2 * min(self.fractal_estimate / 2.0, 1.0)
            + 0.2 * min(self.total_persistence / (self.max_persistence + 1e-10), 1.0)
        )


class PersistentHomologyProbe:
    """Computes persistent homology (β₀) from fitness samples.

    Algorithm (superlevel-set filtration on sample Hamming graph):
      1. Sort sampled architectures by DECREASING fitness.
      2. Process in order: for each new point, create a component (birth).
      3. For each already-processed neighbor (Hamming dist ≤ radius),
         if in different component → merge. Younger component dies.
      4. Record (birth, death) for each dying component.
      5. Surviving components: death = min fitness.

    The persistence diagram captures the "birth and death of basins" as
    we lower a fitness threshold from max to min.

    Complexity: O(n² · E) for n samples, E edges.
    """

    def __init__(self, hamming_radius: int = 0):
        # radius=0 means auto-select based on dimensionality
        self.radius = hamming_radius

    def compute(self, architectures: np.ndarray,
                fitnesses: np.ndarray) -> TopologicalFingerprint:
        n = len(fitnesses)
        if n < 3:
            return self._trivial_fingerprint()

        E = architectures.shape[1]

        # Auto-select radius: use median Hamming distance * 0.6
        # This ensures a connected-enough graph for meaningful topology
        if self.radius <= 0:
            dists = []
            sample_n = min(n, 50)
            for i in range(sample_n):
                for j in range(i + 1, sample_n):
                    dists.append(np.sum(architectures[i] != architectures[j]))
            median_dist = np.median(dists) if dists else E // 2
            radius = max(2, int(median_dist * 0.6))
        else:
            radius = self.radius

        # Sort by decreasing fitness (superlevel filtration)
        order = np.argsort(-fitnesses)
        sorted_arch = architectures[order]
        sorted_fit = fitnesses[order]

        uf = UnionFind(n)
        birth = np.zeros(n)
        processed = set()
        pairs: List[Tuple[float, float]] = []

        # Component birth times (indexed by representative)
        comp_birth: Dict[int, float] = {}

        for idx in range(n):
            i = idx  # position in sorted order
            birth_val = sorted_fit[i]
            comp_birth[i] = birth_val
            processed.add(i)

            # Find neighbors among already-processed points
            for j in processed:
                if j == i:
                    continue
                hdist = np.sum(sorted_arch[i] != sorted_arch[j])
                if hdist <= radius:
                    ri, rj = uf.find(i), uf.find(j)
                    if ri != rj:
                        # Merge: younger (lower birth) dies
                        bi, bj = comp_birth.get(ri, birth_val), comp_birth.get(rj, birth_val)
                        if bi < bj:
                            # ri is younger → ri dies
                            pairs.append((bi, birth_val))
                            uf.union(j, i)
                            new_rep = uf.find(i)
                            comp_birth[new_rep] = bj
                        else:
                            # rj is younger → rj dies
                            pairs.append((bj, birth_val))
                            uf.union(i, j)
                            new_rep = uf.find(i)
                            comp_birth[new_rep] = bi

        # Surviving components: death = min fitness
        min_fit = sorted_fit[-1]
        survivors = set()
        for i in range(n):
            r = uf.find(i)
            if r not in survivors:
                survivors.add(r)
                pairs.append((comp_birth.get(r, sorted_fit[0]), min_fit))

        pd = PersistenceDiagram(pairs=pairs, n_points=n)
        return self._fingerprint_from_diagram(pd, fitnesses)

    def _fingerprint_from_diagram(self, pd: PersistenceDiagram,
                                   fitnesses: np.ndarray) -> TopologicalFingerprint:
        lt = pd.lifetimes
        lt = lt[lt > 0]  # remove zero-lifetime pairs

        if len(lt) == 0:
            return self._trivial_fingerprint()

        # Persistence entropy
        lt_norm = lt / (lt.sum() + 1e-15)
        entropy = -np.sum(lt_norm * np.log(lt_norm + 1e-15))
        max_entropy = np.log(len(lt) + 1e-15)
        norm_entropy = entropy / (max_entropy + 1e-15)

        # Significant basins (lifetime > 25th percentile)
        threshold = np.percentile(lt, 25) if len(lt) > 2 else 0
        n_basins = int(np.sum(lt > threshold))

        # Fractal dimension estimate from persistence scaling
        if len(lt) >= 4:
            sorted_lt = np.sort(lt)[::-1]
            ranks = np.arange(1, len(sorted_lt) + 1).astype(float)
            mask = sorted_lt > 0
            if mask.sum() >= 2:
                log_r = np.log(ranks[mask])
                log_l = np.log(sorted_lt[mask])
                # Linear regression: log(lifetime) ~ -α * log(rank)
                A = np.vstack([log_r, np.ones_like(log_r)]).T
                slope, _ = np.linalg.lstsq(A, log_l, rcond=None)[0]
                fractal_est = max(0.0, -slope)
            else:
                fractal_est = 0.0
        else:
            fractal_est = 0.0

        return TopologicalFingerprint(
            n_basins=max(1, n_basins),
            total_persistence=float(lt.sum()),
            max_persistence=float(lt.max()),
            persistence_entropy=float(np.clip(norm_entropy, 0, 1)),
            mean_basin_depth=float(lt.mean()),
            fractal_estimate=fractal_est,
            persistence_diagram=pd,
        )

    def _trivial_fingerprint(self) -> TopologicalFingerprint:
        return TopologicalFingerprint(
            n_basins=1,
            total_persistence=0.0,
            max_persistence=0.0,
            persistence_entropy=0.0,
            mean_basin_depth=0.0,
            fractal_estimate=0.0,
            persistence_diagram=PersistenceDiagram(pairs=[], n_points=0),
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. SPECTRAL GRAPH SURROGATE — Laplacian Eigenvector Basis
# ═══════════════════════════════════════════════════════════════════════

class SpectralSurrogate:
    """Surrogate model using graph Laplacian eigenvectors as basis.

    Given n sampled (architecture, fitness) pairs:
      1. Build a weighted graph: nodes = samples, edges = Hamming neighbors.
         Weight w(i,j) = exp(-d_H(i,j)² / (2σ²)).
      2. Compute normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}.
      3. Eigendecompose: L = U Λ U^T.
      4. Truncate to k eigenvectors (low-frequency components).
      5. Surrogate f̂ = U_k (U_k^T U_k)^{-1} U_k^T y  (spectral regression).

    This is the mathematically correct Fourier analysis on the sample graph:
    low-frequency eigenvectors capture smooth landscape trends,
    high-frequency capture noise.

    The spectral gap λ₂ - λ₁ indicates landscape smoothness:
    large gap → smooth → good surrogate; small gap → rugged → poor surrogate.
    """

    def __init__(self, bandwidth: float = 2.0, n_components: int = 0):
        self.sigma = bandwidth
        self.k = n_components  # 0 = auto-select via spectral gap
        self.U_k: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.spectral_gap: float = 0.0
        self.eigenvalues: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralSurrogate":
        n = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy()

        if n < 5:
            return self

        # Build weighted adjacency matrix
        # Hamming distance matrix
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                H[i, j] = H[j, i] = np.sum(X[i] != X[j])

        # Gaussian kernel weights
        W = np.exp(-H ** 2 / (2 * self.sigma ** 2))
        np.fill_diagonal(W, 0)

        # Normalized graph Laplacian
        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(W.sum(axis=1)) + 1e-10))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(L)
        self.eigenvalues = eigvals

        # Spectral gap (smoothness indicator)
        self.spectral_gap = float(eigvals[1] - eigvals[0]) if n > 1 else 0.0

        # Auto-select k via spectral gap analysis
        if self.k == 0:
            # Find the largest gap in eigenvalues (elbow method)
            gaps = np.diff(eigvals)
            if len(gaps) > 1:
                # Use at least 3 components, at most n//2
                # Find largest gap after position 2
                search_end = min(len(gaps), max(n // 2, 5))
                if search_end > 2:
                    best_k = np.argmax(gaps[2:search_end]) + 3
                else:
                    best_k = min(3, n - 1)
                k = int(np.clip(best_k, 3, n - 1))
            else:
                k = min(3, n - 1)
        else:
            k = min(self.k, n - 1)

        # Truncated spectral basis
        self.U_k = eigvecs[:, :k]

        # Spectral regression: y ≈ U_k α
        # α = (U_k^T U_k)^{-1} U_k^T y
        UtU = self.U_k.T @ self.U_k
        UtU += 1e-6 * np.eye(k)  # regularization
        self.alpha = np.linalg.solve(UtU, self.U_k.T @ y)

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict fitness for new architectures via Nyström extension."""
        if self.alpha is None or self.X_train is None:
            return np.zeros(len(X_new))

        n_train = self.X_train.shape[0]
        n_new = X_new.shape[0]

        # Compute kernel between new and training points
        K_new = np.zeros((n_new, n_train))
        for i in range(n_new):
            for j in range(n_train):
                h = np.sum(X_new[i] != self.X_train[j])
                K_new[i, j] = np.exp(-h ** 2 / (2 * self.sigma ** 2))

        # Nyström extension: map new points to spectral coordinates
        D_train_inv = 1.0 / (K_new.sum(axis=1, keepdims=True) + 1e-10)
        # Approximate spectral coordinates
        U_new = D_train_inv * (K_new @ self.U_k)

        return U_new @ self.alpha

    def predict_single(self, arch: list) -> float:
        """Fast single-point prediction."""
        return float(self.predict(np.array([arch]))[0])

    @property
    def smoothness_score(self) -> float:
        """How well this surrogate can model the landscape (0-1)."""
        if self.eigenvalues is None:
            return 0.0
        gap = self.spectral_gap
        return float(np.clip(gap / (gap + 0.5), 0, 1))


# ═══════════════════════════════════════════════════════════════════════
# 3. DISCRETE CURVATURE TENSOR — Local Interaction Structure
# ═══════════════════════════════════════════════════════════════════════

class DiscreteCurvatureTensor:
    """Estimates local discrete Hessian around a point.

    For a discrete function f: {0,...,O-1}^E → ℝ, the discrete curvature
    tensor at point x is:

      H[i,j] = E_{o_i,o_j}[f(x ⊕ e_i(o_i) ⊕ e_j(o_j))]
               - E_{o_i}[f(x ⊕ e_i(o_i))] - E_{o_j}[f(x ⊕ e_j(o_j))]
               + f(x)

    where ⊕ e_i(o) means "set position i to value o".

    Large |H[i,j]| → edges i,j interact strongly → sweep together.
    This is estimated from cached evaluations near the current best.
    """

    def __init__(self, E: int, O: int):
        self.E = E
        self.O = O

    def estimate(self, center: list, ev: EvalTracker,
                 budget_limit: int, rng: np.random.RandomState
                 ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Estimate curvature tensor and return interacting pairs.

        Returns:
            H: (E, E) curvature matrix
            pairs: sorted list of (i, j) strongly-interacting pairs
        """
        E, O = self.E, self.O
        f_center = ev.evaluate(tuple(center))
        H = np.zeros((E, E))

        # Estimate marginal effects
        marginal = np.zeros(E)
        for i in rng.permutation(E)[:min(E, budget_limit // (2 * O))]:
            if ev.budget_used >= budget_limit:
                break
            effects = []
            for o in range(O):
                if o == center[i]:
                    effects.append(f_center)
                    continue
                cand = center[:]
                cand[i] = o
                effects.append(ev.evaluate(tuple(cand)))
            marginal[i] = np.mean(effects) - f_center

        # Estimate pairwise interactions (top pairs only)
        n_pairs = min(E * (E - 1) // 2, max(3, budget_limit // (O * O)))
        pair_candidates = []
        for i in range(E):
            for j in range(i + 1, E):
                # Priority: both have large marginal effects
                score = abs(marginal[i]) + abs(marginal[j])
                pair_candidates.append((score, i, j))
        pair_candidates.sort(reverse=True)

        for _, i, j in pair_candidates[:n_pairs]:
            if ev.budget_used >= budget_limit:
                break
            # Sample a few (o_i, o_j) combinations
            samples = []
            for _ in range(min(3, O)):
                o_i = rng.randint(O)
                o_j = rng.randint(O)
                cand = center[:]
                cand[i], cand[j] = o_i, o_j
                f_ij = ev.evaluate(tuple(cand))

                cand_i = center[:]
                cand_i[i] = o_i
                f_i = ev.evaluate(tuple(cand_i))

                cand_j = center[:]
                cand_j[j] = o_j
                f_j = ev.evaluate(tuple(cand_j))

                interaction = f_ij - f_i - f_j + f_center
                samples.append(interaction)

            H[i, j] = H[j, i] = np.mean(samples) if samples else 0.0

        # Identify strongly interacting pairs
        H_abs = np.abs(H)
        threshold = np.percentile(H_abs[H_abs > 0], 70) if np.any(H_abs > 0) else 0
        pairs = []
        for i in range(E):
            for j in range(i + 1, E):
                if H_abs[i, j] > threshold:
                    pairs.append((i, j))

        return H, pairs


# ═══════════════════════════════════════════════════════════════════════
# 4. INFORMATION-DIRECTED SEARCH — Principled Budget Allocation
# ═══════════════════════════════════════════════════════════════════════

class InformationDirectedAllocator:
    """Allocates budget across search phases using Thompson Sampling.

    Models each phase's improvement potential as a Beta distribution.
    At each allocation step, samples from each phase's posterior and
    allocates budget to the phase with highest expected improvement.

    This replaces fixed-fraction allocation (e.g., 15% probe, 35% exploit)
    with an adaptive scheme that reallocates to whichever phase is
    currently most productive.
    """

    def __init__(self, n_phases: int, prior_alpha: float = 1.0,
                 prior_beta: float = 1.0):
        self.n_phases = n_phases
        self.alpha = np.full(n_phases, prior_alpha)
        self.beta = np.full(n_phases, prior_beta)
        self.allocations = np.zeros(n_phases, dtype=int)

    def select_phase(self, rng: np.random.RandomState) -> int:
        """Thompson sampling: select phase with highest sampled value."""
        samples = rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, phase: int, improved: bool):
        """Update posterior after observing outcome."""
        if improved:
            self.alpha[phase] += 1
        else:
            self.beta[phase] += 1
        self.allocations[phase] += 1

    def allocate_budget(self, total: int, rng: np.random.RandomState,
                        min_fractions: Optional[List[float]] = None
                        ) -> List[int]:
        """Pre-allocate budget across phases with minimums."""
        if min_fractions is None:
            min_fractions = [0.05] * self.n_phases

        budgets = [int(total * f) for f in min_fractions]
        remaining = total - sum(budgets)

        # Allocate remaining via Thompson sampling
        for _ in range(remaining):
            phase = self.select_phase(rng)
            budgets[phase] += 1

        return budgets


# ═══════════════════════════════════════════════════════════════════════
# 5. NEXUS — The Unified Algorithm
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NEXUSConfig:
    """NEXUS hyperparameters."""
    probe_fraction: float = 0.15
    hamming_radius: int = 0
    spectral_bandwidth: float = 2.0
    n_starts: int = 5
    anneal_steps: int = 10
    max_perturb_frac: float = 0.3


class NEXUS(BaseSearcher):
    """NEXUS v2: Topology-Informed Iterated Local Search.

    Three innovations that work in practice (not just in theory):

      1. EDGE IMPORTANCE ORDERING: Estimate each edge's contribution to
         fitness variance from probe data. Sweep important edges FIRST
         in coordinate descent. If budget runs out mid-sweep, we've
         already optimized the most impactful edges.

      2. ITERATED LOCAL SEARCH (ILS): After greedy refine converges,
         DON'T STOP. Perturb the solution and refine again. PAR wastes
         all remaining budget after convergence — NEXUS uses it.
         Perturbation size escalates: small → medium → large restart.

      3. IMPORTANCE-WEIGHTED PERTURBATION: When perturbing, preserve
         high-importance edges and flip low-importance ones. This keeps
         the good structure while escaping the local basin.

    Additionally computes (at zero extra cost) topological fingerprint
    and spectral analysis from probe data for diagnostics.
    """

    def __init__(self, num_ops: int, num_edges: int,
                 config: Optional[NEXUSConfig] = None, seed: int = 0):
        super().__init__(num_ops, num_edges, seed)
        self.cfg = config or NEXUSConfig()

    def search(self, eval_fn: EvalFn, budget: int) -> SearchResult:
        ev = EvalTracker(eval_fn)
        E, O = self.E, self.O
        rng = self.rng
        cfg = self.cfg
        diagnostics: Dict[str, Any] = {"method": "NEXUS"}

        # ════════════════════════════════════════════════════════════
        # PHASE 1: PROBE + FREE ANALYSIS
        # Same cost as PAR's probe. Topology & edge importance = free.
        # ════════════════════════════════════════════════════════════
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
            return self._result(ev, diagnostics)

        X = np.array(X_list)
        y = np.array(y_list)

        # ── Edge importance (free: computed from probe data) ──
        # For each edge, measure how much the mean fitness varies
        # across different operation choices. High variance = important.
        edge_importance = np.zeros(E)
        for i in range(E):
            group_means = []
            for o in range(O):
                mask = X[:, i] == o
                if mask.sum() >= 2:
                    group_means.append(y[mask].mean())
            if len(group_means) >= 2:
                edge_importance[i] = np.std(group_means)
        # Normalized importance for edge ordering (descending)
        imp_order = np.argsort(-edge_importance)
        diagnostics["edge_importance_std"] = round(float(edge_importance.std()), 4)

        # ── Topology fingerprint (free: zero extra evals) ──
        topo = PersistentHomologyProbe(hamming_radius=cfg.hamming_radius)
        fp = topo.compute(X, y)
        diagnostics["topology"] = {
            "n_basins": fp.n_basins,
            "persistence_entropy": round(fp.persistence_entropy, 4),
            "complexity_score": round(fp.complexity_score, 4),
        }

        # ── Spectral analysis (free) ──
        surrogate = SpectralSurrogate(bandwidth=cfg.spectral_bandwidth)
        surrogate.fit(X, y)
        diagnostics["spectral_gap"] = round(surrogate.spectral_gap, 4)

        # ════════════════════════════════════════════════════════════
        # PHASE 2: MULTI-START ANNEAL (like PAR, with diverse starts)
        # ════════════════════════════════════════════════════════════
        sorted_idx = np.argsort(-y)
        starts: List[list] = []
        for idx in sorted_idx:
            if len(starts) >= cfg.n_starts - 1:
                break
            cand = list(X[idx])
            if all(sum(c[i] != cand[i] for i in range(E)) > max(2, E // 4)
                   for c in starts) or not starts:
                starts.append(cand)
        starts.append(list(rng.randint(O, size=E)))  # diversity

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
            return self._result(ev, diagnostics)

        # ════════════════════════════════════════════════════════════
        # PHASE 3: ITERATED LOCAL SEARCH + WARM-STARTED EVOLUTION
        #
        # Two sub-phases:
        # 3A: ILS — refine → perturb (from PERTURBED, not global best)
        #     → refine → ... Uses importance-ordered sweeps.
        # 3B: Evolution — when ILS stagnates, switch to population-based
        #     search warm-started from the best solutions found so far.
        #     This is what makes NEXUS competitive at high budgets.
        # ════════════════════════════════════════════════════════════
        n_stagnant = 0
        ils_iterations = 0
        # Archive: keep top diverse solutions for evolution phase
        elite: List[Tuple[list, float]] = []
        if ev.best_arch is not None:
            elite.append((list(ev.best_arch), ev.best_fitness))

        # Start ILS from best anneal result
        cur = list(ev.best_arch) if ev.best_arch else best_cur
        if ev.budget_used < budget:
            cf = ev.evaluate(tuple(cur))
        else:
            cf = ev.best_fitness

        while ev.budget_used < budget and n_stagnant < 3:
            # ── Importance-ordered greedy refine from cur ──
            pre_refine_best = ev.best_fitness

            improved_in_sweep = True
            while improved_in_sweep and ev.budget_used < budget:
                improved_in_sweep = False
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

            # Archive this local optimum
            elite.append((cur[:], cf))
            ils_iterations += 1

            if ev.best_fitness > pre_refine_best:
                n_stagnant = 0
            else:
                n_stagnant += 1

            if ev.budget_used >= budget:
                break

            # ── Perturbation: from CURRENT solution, not global best ──
            perturb_n = max(2, min(E // 3, 2 + n_stagnant))
            edges_to_flip = rng.choice(E, min(perturb_n, E), replace=False)
            for ed in edges_to_flip:
                cur[ed] = rng.randint(O)
            cf = ev.evaluate(tuple(cur))

        # ════════════════════════════════════════════════════════════
        # PHASE 3B: WARM-STARTED EVOLUTION (for remaining budget)
        #
        # ILS found good solutions but stagnated. Now switch to
        # population-based search warm-started from the elite archive.
        # This gives REA's diversity advantage at high budgets.
        # ════════════════════════════════════════════════════════════
        if ev.budget_used < budget:
            # Build population from elite + random
            pop_size = 25
            pop: List[Tuple[list, float]] = []
            # Add elite (deduplicated)
            seen = set()
            for arch, fit in sorted(elite, key=lambda x: -x[1]):
                key = tuple(arch)
                if key not in seen:
                    pop.append((arch, fit))
                    seen.add(key)
                if len(pop) >= pop_size // 2:
                    break
            # Fill rest with mutations of best
            while len(pop) < pop_size and ev.budget_used < budget:
                base = list(ev.best_arch) if ev.best_arch else list(rng.randint(O, size=E))
                mutant = base[:]
                n_mut = rng.randint(1, max(2, E // 3))
                for ed in rng.choice(E, n_mut, replace=False):
                    mutant[ed] = rng.randint(O)
                f = ev.evaluate(tuple(mutant))
                pop.append((mutant, f))

            # Evolution loop (REA-style but with 2-edge mutations)
            tourn_size = 5
            while ev.budget_used < budget:
                # Tournament selection
                idx = rng.choice(len(pop), size=min(tourn_size, len(pop)),
                                 replace=False)
                parent = pop[max(idx, key=lambda i: pop[i][1])][0]

                # Mutation: 1-2 edges (more than REA's 1)
                child = parent[:]
                n_mut = 1 if rng.random() < 0.6 else 2
                for _ in range(n_mut):
                    ed = rng.randint(E)
                    child[ed] = rng.randint(O)

                f = ev.evaluate(tuple(child))
                pop.append((child, f))

                # Aging: remove oldest
                if len(pop) > pop_size:
                    pop.pop(0)

        diagnostics["ils_iterations"] = ils_iterations
        diagnostics["total_evals"] = ev.total_calls
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

    # ── Strategy implementations ──────────────────────────────────

    def _multi_basin_explore(self, ev: EvalTracker,
                              X: np.ndarray, y: np.ndarray,
                              surrogate: SpectralSurrogate,
                              fingerprint: TopologicalFingerprint,
                              budget_limit: int,
                              rng: np.random.RandomState):
        """Multi-modal strategy: explore each basin, then cross-basin jumps."""
        E, O = self.E, self.O
        target_budget = ev.budget_used + budget_limit

        # Identify basin centers from top-k diverse samples
        n_starts = min(fingerprint.n_basins, 6)
        top_idx = np.argsort(-y)
        centers = []
        for idx in top_idx:
            if len(centers) >= n_starts:
                break
            cand = list(X[idx])
            # Check diversity: Hamming dist > E//3 from existing centers
            if all(sum(c[i] != cand[i] for i in range(E)) > E // 3
                   for c in centers):
                centers.append(cand)
            elif len(centers) == 0:
                centers.append(cand)

        # Explore each basin with decreasing perturbation
        for center in centers:
            if ev.budget_used >= target_budget:
                break
            cur = center[:]
            cf = ev.evaluate(tuple(cur))
            for step in range(8):
                if ev.budget_used >= target_budget:
                    break
                noise = 1.0 - step / 8.0
                n_flip = max(1, int(E * noise * 0.25))
                cand = cur[:]
                for ed in rng.choice(E, min(n_flip, E), replace=False):
                    # Use spectral surrogate to guide mutations
                    pred_scores = []
                    for o in range(O):
                        test = cand[:]
                        test[ed] = o
                        pred_scores.append(surrogate.predict_single(test))
                    # Softmax selection weighted by surrogate
                    scores = np.array(pred_scores)
                    temp = max(0.1, noise)
                    logp = (scores - scores.max()) / temp
                    p = np.exp(logp)
                    p /= p.sum()
                    cand[ed] = rng.choice(O, p=p)
                f = ev.evaluate(tuple(cand))
                if f > cf:
                    cf = f
                    cur = cand

        # Cross-basin jumps: interpolate between top basin solutions
        if len(centers) >= 2:
            for _ in range(min(5, target_budget - ev.budget_used)):
                if ev.budget_used >= target_budget:
                    break
                c1, c2 = centers[rng.randint(len(centers))], centers[rng.randint(len(centers))]
                child = c1[:]
                for ed in range(E):
                    if rng.random() < 0.5:
                        child[ed] = c2[ed]
                ev.evaluate(tuple(child))

    def _spectral_exploit(self, ev: EvalTracker,
                           X: np.ndarray, y: np.ndarray,
                           surrogate: SpectralSurrogate,
                           budget_limit: int,
                           rng: np.random.RandomState):
        """Smooth landscape: aggressive spectral surrogate exploitation."""
        E, O = self.E, self.O
        target_budget = ev.budget_used + budget_limit

        # Generate candidates by optimizing spectral surrogate
        best = list(ev.best_arch) if ev.best_arch else list(X[np.argmax(y)])

        for _ in range(3):
            if ev.budget_used >= target_budget:
                break
            cur = best[:]
            # Greedy coordinate descent on surrogate
            for sweep in range(2):
                for ed in rng.permutation(E):
                    if ev.budget_used >= target_budget:
                        break
                    best_o, best_p = cur[ed], surrogate.predict_single(cur)
                    for o in range(O):
                        cur[ed] = o
                        p = surrogate.predict_single(cur)
                        if p > best_p:
                            best_p = p
                            best_o = o
                        cur[ed] = best_o
                    cur[ed] = best_o

            f = ev.evaluate(tuple(cur))
            if f > ev.best_fitness:
                best = cur[:]

        # Surrogate-guided sampling with uncertainty
        while ev.budget_used < target_budget:
            arch = best[:]
            n_flip = rng.randint(1, min(4, E))
            for ed in rng.choice(E, n_flip, replace=False):
                scores = []
                for o in range(O):
                    test = arch[:]
                    test[ed] = o
                    scores.append(surrogate.predict_single(test))
                scores = np.array(scores)
                logp = (scores - scores.max()) / 0.3
                p = np.exp(logp)
                p /= p.sum()
                arch[ed] = rng.choice(O, p=p)
            f = ev.evaluate(tuple(arch))
            if f > ev.best_fitness:
                best = list(ev.best_arch)

    def _rugged_spectral_search(self, ev: EvalTracker,
                                 X: np.ndarray, y: np.ndarray,
                                 surrogate: SpectralSurrogate,
                                 fingerprint: TopologicalFingerprint,
                                 budget_limit: int,
                                 rng: np.random.RandomState):
        """Rugged landscape: spectral smoothing + diverse exploration."""
        E, O = self.E, self.O
        target_budget = ev.budget_used + budget_limit

        # Start from multiple good points
        top_idx = np.argsort(-y)[:min(4, len(y))]
        best = list(ev.best_arch) if ev.best_arch else list(X[top_idx[0]])

        while ev.budget_used < target_budget:
            # Alternate: spectral-guided (60%) vs random jump (40%)
            if rng.random() < 0.6:
                arch = list(ev.best_arch) if ev.best_arch else best[:]
                n_flip = rng.randint(1, max(2, E // 3))
                for ed in rng.choice(E, n_flip, replace=False):
                    scores = []
                    for o in range(O):
                        test = arch[:]
                        test[ed] = o
                        scores.append(surrogate.predict_single(test))
                    scores = np.array(scores)
                    logp = (scores - scores.max()) / 0.5
                    p = np.exp(logp)
                    p /= p.sum()
                    arch[ed] = rng.choice(O, p=p)
            else:
                # Random perturbation from a random top sample
                idx = top_idx[rng.randint(len(top_idx))]
                arch = list(X[idx])
                n_flip = rng.randint(1, max(2, E // 2))
                for ed in rng.choice(E, n_flip, replace=False):
                    arch[ed] = rng.randint(O)

            ev.evaluate(tuple(arch))

    def _balanced_search(self, ev: EvalTracker,
                          X: np.ndarray, y: np.ndarray,
                          surrogate: SpectralSurrogate,
                          budget_limit: int,
                          rng: np.random.RandomState):
        """Default balanced strategy."""
        E, O = self.E, self.O
        target_budget = ev.budget_used + budget_limit

        best = list(ev.best_arch) if ev.best_arch else list(X[np.argmax(y)])

        # Anneal-style search guided by spectral surrogate
        for step in range(20):
            if ev.budget_used >= target_budget:
                break
            noise = 1.0 - step / 20.0
            cur = list(ev.best_arch) if ev.best_arch else best[:]
            n_flip = max(1, int(E * noise * 0.3))
            for ed in rng.choice(E, min(n_flip, E), replace=False):
                if rng.random() < 0.7:  # spectral-guided
                    scores = []
                    for o in range(O):
                        test = cur[:]
                        test[ed] = o
                        scores.append(surrogate.predict_single(test))
                    scores = np.array(scores)
                    temp = max(0.1, noise * 0.5)
                    logp = (scores - scores.max()) / temp
                    p = np.exp(logp)
                    p /= p.sum()
                    cur[ed] = rng.choice(O, p=p)
                else:
                    cur[ed] = rng.randint(O)
            f = ev.evaluate(tuple(cur))

    def _curvature_refine(self, ev: EvalTracker,
                           interacting: List[Tuple[int, int]],
                           budget: int,
                           rng: np.random.RandomState):
        """Refine using curvature-detected interactions."""
        E, O = self.E, self.O
        if ev.best_arch is None:
            return

        cur = list(ev.best_arch)
        cf = ev.best_fitness

        # Joint sweeps for interacting pairs
        for i, j in interacting[:5]:
            if ev.budget_used >= budget:
                break
            best_ij, best_f = (cur[i], cur[j]), cf
            for o1 in range(O):
                for o2 in range(O):
                    if (o1, o2) == (cur[i], cur[j]):
                        continue
                    if ev.budget_used >= budget:
                        break
                    cand = cur[:]
                    cand[i], cand[j] = o1, o2
                    f = ev.evaluate(tuple(cand))
                    if f > best_f:
                        best_f = f
                        best_ij = (o1, o2)
            if best_ij != (cur[i], cur[j]):
                cur[i], cur[j] = best_ij
                cf = best_f

        # Single-edge sweeps for non-interacting edges
        for ed in rng.permutation(E):
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

    def _greedy_local_search(self, ev: EvalTracker,
                              budget: int,
                              rng: np.random.RandomState):
        """Greedy coordinate-wise local search — strongest at high B.

        This is the key phase that makes NEXUS competitive at large budgets.
        Repeatedly sweeps all edges, setting each to its best value.
        Terminates when no improvement is found in a full sweep.
        """
        E, O = self.E, self.O
        if ev.best_arch is None:
            return

        cur = list(ev.best_arch)
        cf = ev.best_fitness

        # Allocate up to 40% of remaining budget
        local_budget = ev.budget_used + int((budget - ev.budget_used) * 0.6)
        local_budget = min(local_budget, budget)

        while ev.budget_used < local_budget:
            improved = False
            for ed in rng.permutation(E):
                if ev.budget_used >= local_budget:
                    break
                best_op, best_f = cur[ed], cf
                for o in range(O):
                    if o == cur[ed] or ev.budget_used >= local_budget:
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
                    improved = True
            if not improved:
                # No improvement in full sweep — restart from best with noise
                if ev.best_fitness > cf:
                    cur = list(ev.best_arch)
                    cf = ev.best_fitness
                else:
                    break

    def _ids_polish(self, ev: EvalTracker,
                     surrogate: SpectralSurrogate,
                     budget: int,
                     rng: np.random.RandomState):
        """Information-directed polishing with Thompson sampling."""
        E, O = self.E, self.O
        if ev.best_arch is None:
            return

        allocator = InformationDirectedAllocator(
            n_phases=4, prior_alpha=self.cfg.ids_prior
        )
        prev_best = ev.best_fitness

        while ev.budget_used < budget:
            phase = allocator.select_phase(rng)
            cur = list(ev.best_arch)

            if phase == 0:  # Spectral-guided mutation
                ed = rng.randint(E)
                scores = []
                for o in range(O):
                    test = cur[:]
                    test[ed] = o
                    scores.append(surrogate.predict_single(test))
                scores = np.array(scores)
                logp = (scores - scores.max()) / 0.2
                p = np.exp(logp)
                p /= p.sum()
                cur[ed] = rng.choice(O, p=p)

            elif phase == 1:  # Local greedy step
                ed = rng.randint(E)
                best_o = cur[ed]
                best_f = ev.best_fitness
                for o in range(O):
                    if o == cur[ed] or ev.budget_used >= budget:
                        continue
                    cand = cur[:]
                    cand[ed] = o
                    f = ev.evaluate(tuple(cand))
                    if f > best_f:
                        best_f = f
                        best_o = o
                cur[ed] = best_o
                improved = best_f > prev_best
                allocator.update(phase, improved)
                prev_best = max(prev_best, ev.best_fitness)
                continue  # already evaluated

            elif phase == 2:  # Multi-flip perturbation
                n_flip = rng.randint(2, min(5, E))
                for ed in rng.choice(E, n_flip, replace=False):
                    cur[ed] = rng.randint(O)

            else:  # Random restart from best with noise
                for ed in range(E):
                    if rng.random() < 0.3:
                        cur[ed] = rng.randint(O)

            f = ev.evaluate(tuple(cur))
            improved = ev.best_fitness > prev_best
            allocator.update(phase, improved)
            prev_best = max(prev_best, ev.best_fitness)
