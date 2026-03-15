"""
Tunable combinatorial fitness landscapes for benchmarking discrete search.

Provides NK landscapes with controllable epistasis (K), alphabet size (O),
and dimensionality (N). Includes both standard NK and NAS-specific surrogates.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LandscapeStats:
    """Statistics of a fitness landscape computed from sampled evaluations."""
    global_opt: float
    mean_fitness: float
    std_fitness: float
    autocorrelation: float  # 1-step Hamming neighbor correlation
    effective_dim: float    # estimated effective dimensionality


class NKLandscape:
    """NK fitness landscape with tunable epistasis.

    The fitness f(x) for architecture x in Z_O^N is:
        f(x) = (1/N) * sum_{i=1}^{N} f_i(x_{S_i})
    where S_i is the dependency set for component i, |S_i| = K+1.

    The ANOVA decomposition has at most N * C(N,K) * (O-1)^(K+1)
    non-zero coefficients, making it Fourier-sparse for small K.

    Args:
        N: Number of positions (edges in NAS analogy).
        K: Epistatic interactions per position (0 = fully separable).
        O: Alphabet size (number of operations per position).
        seed: Random seed for reproducible landscape generation.
    """

    def __init__(self, N: int, K: int, O: int = 5, seed: int = 0):
        self.N = N
        self.K = min(K, N - 1)
        self.O = O
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.tables = []
        self.dep_sets = []
        self._cache: Dict[tuple, float] = {}

        for i in range(N):
            # Build dependency set: position i + K random others
            others = list(range(N))
            others.remove(i)
            deps = sorted([i] + list(rng.choice(others, min(self.K, len(others)),
                                                  replace=False)))
            self.dep_sets.append(deps)
            # Random fitness contribution table
            shape = tuple([O] * len(deps))
            self.tables.append((deps, rng.random(shape)))

    def __call__(self, arch) -> float:
        arch = tuple(arch)
        if arch in self._cache:
            return self._cache[arch]
        f = sum(table[tuple(arch[d] for d in deps)]
                for deps, table in self.tables) * 100.0 / self.N
        self._cache[arch] = f
        return f

    def get_dependency_graph(self) -> np.ndarray:
        """Return N x N adjacency matrix of variable dependencies."""
        adj = np.zeros((self.N, self.N), dtype=int)
        for i, deps in enumerate(self.dep_sets):
            for d in deps:
                adj[i, d] = 1
                adj[d, i] = 1
        return adj

    def estimate_stats(self, n_samples: int = 5000,
                       seed: int = 99) -> LandscapeStats:
        """Estimate landscape statistics from random sampling."""
        rng = np.random.RandomState(seed)
        archs = [tuple(rng.randint(self.O, size=self.N))
                 for _ in range(n_samples)]
        fits = np.array([self(a) for a in archs])

        # Autocorrelation: average correlation with 1-step Hamming neighbors
        n_corr = min(500, n_samples)
        neighbor_fits = []
        for a in archs[:n_corr]:
            a_list = list(a)
            pos = rng.randint(self.N)
            old = a_list[pos]
            a_list[pos] = (old + 1) % self.O
            neighbor_fits.append(self(tuple(a_list)))
        autocorr = np.corrcoef(fits[:n_corr], neighbor_fits)[0, 1]

        # Effective dimensionality: number of positions with high variance
        # in marginal fitness distribution
        pos_vars = []
        for i in range(self.N):
            means_per_op = []
            for o in range(self.O):
                mask = np.array([a[i] == o for a in archs])
                if mask.sum() > 5:
                    means_per_op.append(fits[mask].mean())
            pos_vars.append(np.var(means_per_op) if len(means_per_op) > 1 else 0)
        threshold = np.mean(pos_vars) * 0.5
        eff_dim = sum(1 for v in pos_vars if v > threshold)

        return LandscapeStats(
            global_opt=float(fits.max()),
            mean_fitness=float(fits.mean()),
            std_fitness=float(fits.std()),
            autocorrelation=float(autocorr),
            effective_dim=float(eff_dim),
        )


class NASBench201Surrogate:
    """Surrogate for NAS-Bench-201 search space.

    6 edges, 5 operations each. Fitness landscape is generated to mimic
    the statistical properties of real NAS-Bench-201 (high baseline ~90%,
    narrow range, sparse high-performers).

    Args:
        seed: Random seed.
        noise_std: Evaluation noise (0 = deterministic).
    """

    def __init__(self, seed: int = 42, noise_std: float = 0.0):
        self.E = 6
        self.O = 5
        self.noise_std = noise_std
        self._cache: Dict[tuple, float] = {}

        rng = np.random.RandomState(seed)
        n = self.O ** self.E  # 15625 total architectures

        # Base fitness: normal around 90 with std 5
        self.lut = rng.normal(90, 5, n)
        self.lut = np.clip(self.lut, 60, 97)

        # Sparse high performers (top 1%)
        top_idx = rng.choice(n, n // 100, replace=False)
        self.lut[top_idx] += rng.uniform(2, 7, len(top_idx))
        self.lut = np.clip(self.lut, 60, 97)

        self.rng = np.random.RandomState(seed + 1000)

    def __call__(self, arch) -> float:
        arch = tuple(arch)
        if arch in self._cache and self.noise_std == 0:
            return self._cache[arch]
        idx = sum(int(arch[i]) * (self.O ** i) for i in range(self.E))
        f = float(self.lut[idx])
        if self.noise_std > 0:
            f += self.rng.normal(0, self.noise_std)
        self._cache[arch] = f
        return f


class SyntheticNASLandscape:
    """Synthetic NAS landscape with controllable structure.

    Generates a landscape that mimics real NAS properties:
    - Modular structure (groups of interacting edges)
    - Diminishing returns (marginal gains decrease)
    - Skip-connection effects (some ops dominate others)

    Args:
        E: Number of edges.
        O: Number of operations per edge.
        n_modules: Number of interacting modules.
        module_size: Edges per module (overlapping allowed).
        seed: Random seed.
    """

    def __init__(self, E: int = 14, O: int = 5,
                 n_modules: int = 4, module_size: int = 5,
                 seed: int = 0):
        self.E = E
        self.O = O
        self.seed = seed
        self._cache: Dict[tuple, float] = {}

        rng = np.random.RandomState(seed)

        # Module structure
        self.modules = []
        for _ in range(n_modules):
            members = sorted(rng.choice(E, min(module_size, E),
                                        replace=False).tolist())
            table_shape = tuple([O] * len(members))
            table = rng.normal(0, 1, table_shape)
            self.modules.append((members, table))

        # Marginal (additive) effects
        self.marginals = rng.normal(0, 0.5, (E, O))
        # Make one op slightly dominant per edge
        for i in range(E):
            best_op = rng.randint(O)
            self.marginals[i, best_op] += rng.uniform(0.5, 1.5)

        # Base accuracy
        self.base = 75.0

    def __call__(self, arch) -> float:
        arch = tuple(arch)
        if arch in self._cache:
            return self._cache[arch]

        f = self.base
        # Additive marginal effects
        for i in range(self.E):
            f += self.marginals[i, arch[i]]
        # Module interaction effects
        for members, table in self.modules:
            key = tuple(arch[m] for m in members)
            f += table[key] * 0.5
        # Diminishing returns (softcap)
        f = 60 + 37 * (1 - np.exp(-(f - 60) / 20))

        self._cache[arch] = f
        return f
