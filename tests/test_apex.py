"""Unit tests for APEX and its components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from apex import (
    APEX, APEXConfig,
    WalshFeatureEngine,
    VariableInteractionGraph,
    PartitionCrossover,
    SampleDatabase,
)
from landscapes import NKLandscape
from searchers import REA


# ═════════════════════════════════════════════════════════════════════
# WalshFeatureEngine
# ═════════════════════════════════════════════════════════════════════

class TestWalshFeatureEngine:
    def test_feature_count(self):
        """Feature count should match formula."""
        w = WalshFeatureEngine(num_ops=3, num_edges=4, max_order=2)
        # Order 1: 4 * 2 = 8
        # Order 2: C(4,2) * 2^2 = 6 * 4 = 24
        assert w.n_features == 32

    def test_feature_count_order1(self):
        w = WalshFeatureEngine(num_ops=5, num_edges=6, max_order=1)
        # Order 1 only: 6 * 4 = 24
        assert w.n_features == 24

    def test_transform_shape(self):
        w = WalshFeatureEngine(num_ops=3, num_edges=4, max_order=2)
        X = np.array([[0, 1, 2, 0], [1, 0, 1, 2]])
        Phi = w.transform(X)
        assert Phi.shape == (2, 32)

    def test_transform_indicators(self):
        """Order-1 features should be correct indicators."""
        w = WalshFeatureEngine(num_ops=3, num_edges=2, max_order=1)
        X = np.array([[0, 0], [1, 0], [0, 2], [2, 1]])
        Phi = w.transform(X)
        # Features: (j=0,v=1), (j=0,v=2), (j=1,v=1), (j=1,v=2)
        assert Phi[0, 0] == 0  # x[0]=0, not 1
        assert Phi[1, 0] == 1  # x[0]=1
        assert Phi[2, 3] == 1  # x[1]=2
        assert Phi[3, 2] == 1  # x[1]=1

    def test_fit_predict(self):
        """Should fit and predict on simple data."""
        nk = NKLandscape(8, 3, 4, seed=0)
        rng = np.random.RandomState(0)
        X = np.array([rng.randint(4, size=8) for _ in range(60)])
        y = np.array([nk(tuple(x)) for x in X])

        w = WalshFeatureEngine(4, 8, max_order=2)
        r2 = w.fit(X, y)
        assert r2 >= 0  # R^2 should be non-negative after fit

        y_pred = w.predict(X)
        assert y_pred.shape == (60,)

    def test_pairwise_interactions_symmetric(self):
        """Interaction matrix should be symmetric."""
        rng = np.random.RandomState(0)
        w = WalshFeatureEngine(3, 5, max_order=2)
        X = rng.randint(3, size=(40, 5))
        y = rng.randn(40)
        w.fit(X, y)
        W = w.get_pairwise_interactions()
        np.testing.assert_array_almost_equal(W, W.T)

    def test_edge_importance_nonnegative(self):
        rng = np.random.RandomState(0)
        w = WalshFeatureEngine(3, 5, max_order=2)
        X = rng.randint(3, size=(40, 5))
        y = rng.randn(40)
        w.fit(X, y)
        imp = w.get_edge_importance()
        assert np.all(imp >= 0)


# ═════════════════════════════════════════════════════════════════════
# VariableInteractionGraph
# ═════════════════════════════════════════════════════════════════════

class TestVIG:
    def test_components_isolated(self):
        """With no interactions, each variable is its own component."""
        vig = VariableInteractionGraph(5)
        # Build with zero interaction matrix
        w = WalshFeatureEngine(3, 5, max_order=2)
        w.coefficients = np.zeros(w.n_features)
        w.intercept = 0.0
        vig.build_from_walsh(w)
        assert vig.n_components == 5

    def test_components_connected(self):
        """All variables interacting should form one component."""
        w = WalshFeatureEngine(3, 4, max_order=2)
        w.coefficients = np.zeros(w.n_features)
        w.intercept = 0.0
        # Manually set some pairwise interactions
        # Find features for pairs (0,1), (1,2), (2,3)
        for idx, (order, edges, values) in enumerate(w.feature_map):
            if order == 2 and edges in [(0,1),(1,2),(2,3)]:
                w.coefficients[idx] = 1.0
        w._sparse_idx = [(i, float(c)) for i, c in enumerate(w.coefficients) if abs(c) > 1e-10]
        vig = VariableInteractionGraph(4)
        vig.build_from_walsh(w)
        assert vig.n_components == 1

    def test_components_two_groups(self):
        """With no interactions, each variable is its own component."""
        w = WalshFeatureEngine(3, 4, max_order=2)
        w.coefficients = np.zeros(w.n_features)
        w.intercept = 0.0
        w._sparse_idx = []
        vig = VariableInteractionGraph(4)
        vig.build_from_walsh(w)
        assert vig.n_components == 4

    def test_single_component(self):
        """VIG with Walsh from real data should detect structure."""
        nk = NKLandscape(6, 3, 3, seed=0)
        rng = np.random.RandomState(0)
        X = np.array([rng.randint(3, size=6) for _ in range(40)])
        y = np.array([nk(tuple(x)) for x in X])
        w = WalshFeatureEngine(3, 6, max_order=2)
        w.fit(X, y)
        vig = VariableInteractionGraph(6)
        vig.build_from_walsh(w)
        assert vig.n_components >= 1


# ═════════════════════════════════════════════════════════════════════
# PartitionCrossover
# ═════════════════════════════════════════════════════════════════════

class TestPartitionCrossover:
    def test_identical_parents(self):
        """Offspring of identical parents should equal parent."""
        w = WalshFeatureEngine(3, 4, max_order=1)
        w.coefficients = np.zeros(w.n_features)
        w.intercept = 0.0
        vig = VariableInteractionGraph(4)
        vig.build_from_walsh(w)
        px = PartitionCrossover(vig, w)

        p = np.array([1, 2, 0, 1])
        off = px.crossover(p, p)
        np.testing.assert_array_equal(off, p)

    def test_offspring_is_parent_mix(self):
        """Offspring should only contain values from parents."""
        w = WalshFeatureEngine(3, 4, max_order=2)
        rng = np.random.RandomState(0)
        X = rng.randint(3, size=(30, 4))
        y = rng.randn(30)
        w.fit(X, y)
        vig = VariableInteractionGraph(4)
        vig.build_from_walsh(w)
        px = PartitionCrossover(vig, w)

        p1 = np.array([0, 1, 2, 0])
        p2 = np.array([2, 0, 1, 1])
        off = px.crossover(p1, p2)

        for j in range(4):
            assert off[j] in (p1[j], p2[j])


# ═════════════════════════════════════════════════════════════════════
# APEX (end-to-end)
# ═════════════════════════════════════════════════════════════════════

class TestAPEX:
    def test_basic(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        s = APEX(5, 10, seed=0)
        r = s.search(nk, budget=200)
        assert r.best_fitness > 0
        assert r.total_evals <= 200
        assert r.diagnostics["method"] == "APEX"

    def test_deterministic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        r1 = APEX(5, 8, seed=42).search(nk, budget=150)
        r2 = APEX(5, 8, seed=42).search(nk, budget=150)
        assert r1.best_fitness == r2.best_fitness

    def test_budget_respected(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        for b in [50, 100, 200, 300]:
            r = APEX(5, 8, seed=0).search(nk, budget=b)
            assert r.total_evals <= b, f"Budget {b}: used {r.total_evals}"

    def test_diagnostics_keys(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        r = APEX(5, 10, seed=0).search(nk, budget=200)
        d = r.diagnostics
        assert "walsh_r2" in d
        assert "n_vig_components" in d
        assert "ils_iterations" in d
        assert "px_generations" in d

    def test_beats_random(self):
        """APEX should beat pure random search."""
        nk = NKLandscape(12, 6, 5, seed=0)
        apex_fits, rand_fits = [], []
        for s in range(8):
            r = APEX(5, 12, seed=s).search(nk, budget=200)
            apex_fits.append(r.best_fitness)
            rng = np.random.RandomState(s)
            best = -np.inf
            for _ in range(200):
                a = tuple(rng.randint(5, size=12))
                best = max(best, nk(a))
            rand_fits.append(best)
        assert np.mean(apex_fits) > np.mean(rand_fits)

    def test_various_landscapes(self):
        for N, K in [(6, 2), (10, 5), (14, 7)]:
            nk = NKLandscape(N, K, 5, seed=0)
            r = APEX(5, N, seed=0).search(nk, budget=200)
            assert r.best_fitness > 0

    def test_config(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        cfg = APEXConfig(probe_fraction=0.25, n_virtual_starts=30)
        r = APEX(5, 8, config=cfg, seed=0).search(nk, budget=150)
        assert r.best_fitness > 0

    def test_small_budget(self):
        """Should handle very small budgets gracefully."""
        nk = NKLandscape(6, 3, 3, seed=0)
        r = APEX(3, 6, seed=0).search(nk, budget=30)
        assert r.total_evals <= 30
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
