"""Unit tests for NEXUS and its components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from nexus import (
    NEXUS, NEXUSConfig,
    UnionFind,
    PersistentHomologyProbe, TopologicalFingerprint,
    SpectralSurrogate,
    DiscreteCurvatureTensor,
    InformationDirectedAllocator,
)
from landscapes import NKLandscape
from searchers import REA


# ═════════════════════════════════════════════════════════════════════
# UnionFind
# ═════════════════════════════════════════════════════════════════════

class TestUnionFind:
    def test_basic(self):
        uf = UnionFind(5)
        assert uf.n_components == 5
        uf.union(0, 1)
        assert uf.n_components == 4
        assert uf.find(0) == uf.find(1)

    def test_transitive(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)
        assert uf.n_components == 2

    def test_no_duplicate_merge(self):
        uf = UnionFind(3)
        assert uf.union(0, 1) == True
        assert uf.union(0, 1) == False  # already same
        assert uf.n_components == 2


# ═════════════════════════════════════════════════════════════════════
# PersistentHomologyProbe
# ═════════════════════════════════════════════════════════════════════

class TestPersistentHomology:
    def test_single_basin(self):
        """Unimodal landscape should have ~1 dominant basin."""
        rng = np.random.RandomState(0)
        E, O = 6, 3
        # Create smooth unimodal landscape: fitness = -hamming_dist(x, target)
        target = rng.randint(O, size=E)
        X = np.array([rng.randint(O, size=E) for _ in range(50)])
        y = np.array([-np.sum(x != target) for x in X], dtype=float)

        probe = PersistentHomologyProbe(hamming_radius=2)
        fp = probe.compute(X, y)
        assert fp.n_basins >= 1
        assert fp.persistence_entropy >= 0

    def test_multimodal(self):
        """Multimodal landscape should detect multiple basins."""
        rng = np.random.RandomState(42)
        E, O = 8, 4
        # 3 distinct peaks
        peaks = [rng.randint(O, size=E) for _ in range(3)]
        X = np.array([rng.randint(O, size=E) for _ in range(80)])
        y = np.array([max(-np.sum(x != p) for p in peaks) for x in X], dtype=float)

        probe = PersistentHomologyProbe(hamming_radius=2)
        fp = probe.compute(X, y)
        # Should detect some structure
        assert isinstance(fp, TopologicalFingerprint)
        assert fp.total_persistence >= 0

    def test_trivial_input(self):
        """Should handle very small inputs gracefully."""
        X = np.array([[0, 1, 2]])
        y = np.array([1.0])
        probe = PersistentHomologyProbe()
        fp = probe.compute(X, y)
        assert fp.n_basins == 1

    def test_complexity_score_bounded(self):
        """Complexity score should be in [0, 1]."""
        rng = np.random.RandomState(0)
        X = rng.randint(5, size=(40, 10))
        y = rng.randn(40)
        probe = PersistentHomologyProbe()
        fp = probe.compute(X, y)
        assert 0 <= fp.complexity_score <= 1


# ═════════════════════════════════════════════════════════════════════
# SpectralSurrogate
# ═════════════════════════════════════════════════════════════════════

class TestSpectralSurrogate:
    def test_fit_predict(self):
        """Surrogate should fit training data reasonably."""
        nk = NKLandscape(8, 4, 5, seed=0)
        rng = np.random.RandomState(0)
        X = np.array([rng.randint(5, size=8) for _ in range(50)])
        y = np.array([nk(tuple(x)) for x in X])

        surr = SpectralSurrogate(bandwidth=2.0)
        surr.fit(X, y)

        # Predictions on training data should correlate with truth
        y_pred = surr.predict(X)
        corr = np.corrcoef(y, y_pred)[0, 1]
        assert corr > 0.1  # at least weak positive correlation on rugged NK

    def test_spectral_gap(self):
        """Spectral gap should be computed."""
        rng = np.random.RandomState(0)
        X = rng.randint(3, size=(30, 6))
        y = rng.randn(30)
        surr = SpectralSurrogate()
        surr.fit(X, y)
        assert surr.spectral_gap >= 0

    def test_smoothness_score_bounded(self):
        rng = np.random.RandomState(0)
        X = rng.randint(3, size=(30, 6))
        y = rng.randn(30)
        surr = SpectralSurrogate()
        surr.fit(X, y)
        assert 0 <= surr.smoothness_score <= 1

    def test_small_input(self):
        """Should handle very small datasets."""
        X = np.array([[0, 1], [1, 0], [0, 0]])
        y = np.array([1.0, 2.0, 1.5])
        surr = SpectralSurrogate()
        surr.fit(X, y)
        # Should not crash


# ═════════════════════════════════════════════════════════════════════
# InformationDirectedAllocator
# ═════════════════════════════════════════════════════════════════════

class TestIDS:
    def test_selection(self):
        alloc = InformationDirectedAllocator(n_phases=3)
        rng = np.random.RandomState(0)
        phase = alloc.select_phase(rng)
        assert 0 <= phase < 3

    def test_update(self):
        alloc = InformationDirectedAllocator(n_phases=2)
        alloc.update(0, True)
        alloc.update(1, False)
        # Phase 0 should now be preferred
        assert alloc.alpha[0] > alloc.alpha[1]

    def test_budget_allocation(self):
        alloc = InformationDirectedAllocator(n_phases=4)
        rng = np.random.RandomState(42)
        budgets = alloc.allocate_budget(100, rng, min_fractions=[0.1, 0.1, 0.1, 0.1])
        assert sum(budgets) == 100
        assert all(b >= 10 for b in budgets)


# ═════════════════════════════════════════════════════════════════════
# NEXUS (end-to-end)
# ═════════════════════════════════════════════════════════════════════

class TestNEXUS:
    def test_basic(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        s = NEXUS(5, 10, seed=0)
        r = s.search(nk, budget=200)
        assert r.best_fitness > 0
        assert r.total_evals <= 200
        assert "topology" in r.diagnostics
        assert "ils_iterations" in r.diagnostics

    def test_deterministic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        r1 = NEXUS(5, 8, seed=42).search(nk, budget=150)
        r2 = NEXUS(5, 8, seed=42).search(nk, budget=150)
        assert r1.best_fitness == r2.best_fitness

    def test_budget_respected(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        for b in [50, 100, 300]:
            r = NEXUS(5, 8, seed=0).search(nk, budget=b)
            assert r.total_evals <= b

    def test_diagnostics_complete(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        r = NEXUS(5, 10, seed=0).search(nk, budget=200)
        d = r.diagnostics
        assert d["method"] == "NEXUS"
        assert "topology" in d
        assert "n_basins" in d["topology"]
        assert "persistence_entropy" in d["topology"]
        assert "spectral_gap" in d
        assert "ils_iterations" in d

    def test_beats_random_search(self):
        """NEXUS should beat pure random search on average."""
        nk = NKLandscape(12, 6, 5, seed=0)
        nexus_fits, rand_fits = [], []
        for s in range(8):
            nr = NEXUS(5, 12, seed=s).search(nk, budget=200)
            nexus_fits.append(nr.best_fitness)
            rng = np.random.RandomState(s)
            best = -np.inf
            for _ in range(200):
                a = tuple(rng.randint(5, size=12))
                f = nk(a)
                if f > best:
                    best = f
            rand_fits.append(best)
        assert np.mean(nexus_fits) > np.mean(rand_fits)

    def test_various_landscapes(self):
        """NEXUS should work on different NK configurations."""
        for N, K in [(6, 2), (10, 5), (14, 7)]:
            nk = NKLandscape(N, K, 5, seed=0)
            r = NEXUS(5, N, seed=0).search(nk, budget=200)
            assert r.best_fitness > 0
            assert r.diagnostics["ils_iterations"] >= 1

    def test_config(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        cfg = NEXUSConfig(
            probe_fraction=0.20,
            hamming_radius=3,
            spectral_bandwidth=1.5,
        )
        r = NEXUS(5, 8, config=cfg, seed=0).search(nk, budget=150)
        assert r.best_fitness > 0


if __name__ == "__main__":
    for cls_name in sorted(dir()):
        cls = eval(cls_name)
        if isinstance(cls, type) and cls_name.startswith("Test"):
            instance = cls()
            for method_name in sorted(dir(instance)):
                if method_name.startswith("test_"):
                    print(f"  {cls_name}.{method_name}...", end=" ", flush=True)
                    try:
                        getattr(instance, method_name)()
                        print("OK")
                    except Exception as e:
                        print(f"FAIL: {e}")
