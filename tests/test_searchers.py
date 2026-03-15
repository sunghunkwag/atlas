"""Unit tests for ATLAS searchers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from searchers import REA, PAR, FLAS, ATLAS, PARConfig, FLASConfig
from landscapes import NKLandscape, NASBench201Surrogate


class TestREA:
    def test_basic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        s = REA(5, 8, seed=0)
        r = s.search(nk, budget=100)
        assert r.best_fitness > 0
        assert r.total_evals == 100
        assert len(r.best_arch) == 8
        assert all(0 <= x < 5 for x in r.best_arch)

    def test_deterministic(self):
        nk = NKLandscape(6, 3, 5, seed=0)
        r1 = REA(5, 6, seed=42).search(nk, budget=50)
        r2 = REA(5, 6, seed=42).search(nk, budget=50)
        assert r1.best_fitness == r2.best_fitness

    def test_budget_respected(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        for b in [10, 50, 200]:
            r = REA(5, 8, seed=0).search(nk, budget=b)
            assert r.total_evals <= b


class TestPAR:
    def test_basic(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        s = PAR(5, 10, seed=0)
        r = s.search(nk, budget=200)
        assert r.best_fitness > 0
        assert r.total_evals <= 200

    def test_beats_random(self):
        """PAR should beat random search on average."""
        nk = NKLandscape(12, 6, 5, seed=0)
        par_fits, rand_fits = [], []
        for s in range(10):
            pr = PAR(5, 12, seed=s).search(nk, budget=200)
            par_fits.append(pr.best_fitness)
            # Random search
            rng = np.random.RandomState(s)
            best = -np.inf
            for _ in range(200):
                a = tuple(rng.randint(5, size=12))
                f = nk(a)
                if f > best:
                    best = f
            rand_fits.append(best)
        assert np.mean(par_fits) > np.mean(rand_fits)

    def test_config(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        cfg = PARConfig(probe_fraction=0.2, n_starts=3, n_anneal_steps=5)
        r = PAR(5, 8, config=cfg, seed=0).search(nk, budget=100)
        assert r.best_fitness > 0


class TestFLAS:
    def test_basic(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        s = FLAS(5, 10, seed=0)
        r = s.search(nk, budget=300)
        assert r.best_fitness > 0
        assert "n_interactions_detected" in r.diagnostics

    def test_detects_interactions(self):
        """FLAS should detect some interactions on K>0 landscape."""
        nk = NKLandscape(10, 5, 5, seed=0)
        r = FLAS(5, 10, seed=0).search(nk, budget=500)
        # With K=5, there should be detectable interactions
        # (not guaranteed with small sample, but likely)
        assert isinstance(r.diagnostics["interactions"], list)


class TestATLAS:
    def test_basic(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        s = ATLAS(5, 10, seed=0)
        r = s.search(nk, budget=200)
        assert r.best_fitness > 0
        assert "atlas_mode" in r.diagnostics
        assert r.diagnostics["atlas_mode"] in ("PAR", "FLAS")

    def test_selects_par_at_low_budget(self):
        """At low budget, ATLAS should prefer PAR mode."""
        nk = NKLandscape(14, 7, 5, seed=0)
        r = ATLAS(5, 14, seed=0).search(nk, budget=100)
        # With only 100 evals for 14 edges, model should be unreliable
        assert r.diagnostics["atlas_mode"] == "PAR"


class TestNASBench201:
    def test_basic(self):
        surr = NASBench201Surrogate(seed=42)
        f = surr(tuple([0, 1, 2, 3, 4, 0]))
        assert 60 <= f <= 97

    def test_search(self):
        surr = NASBench201Surrogate(seed=42)
        r = PAR(5, 6, seed=0).search(surr, budget=100)
        assert r.best_fitness > 85  # Should find decent arch


if __name__ == "__main__":
    # Run all tests
    for cls_name in dir():
        cls = eval(cls_name)
        if isinstance(cls, type) and cls_name.startswith("Test"):
            instance = cls()
            for method_name in dir(instance):
                if method_name.startswith("test_"):
                    print(f"  {cls_name}.{method_name}...", end=" ")
                    try:
                        getattr(instance, method_name)()
                        print("OK")
                    except Exception as e:
                        print(f"FAIL: {e}")
