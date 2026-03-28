"""Unit tests for CODA search."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from coda import CODA, CODAConfig, CoImprovementTracker
from landscapes import NKLandscape


class TestCoImprovementTracker:
    def test_basic_tracking(self):
        ct = CoImprovementTracker(5)
        ct.record_sweep({0, 1}, {2, 3, 4})
        ct.record_sweep({2, 3}, {0, 1, 4})  # 2,3 were stuck, now improved
        # Variables 0,1 (improved in sweep 1) should get credit for enabling 2,3
        assert ct.co_improve[0, 2] > 0
        assert ct.co_improve[0, 3] > 0
        assert ct.co_improve[1, 2] > 0
        assert ct.co_improve[1, 3] > 0

    def test_no_self_credit(self):
        ct = CoImprovementTracker(3)
        ct.record_sweep({0}, {1, 2})
        ct.record_sweep({0, 1}, {2})
        assert ct.co_improve[0, 0] == 0

    def test_groups_empty_when_no_data(self):
        ct = CoImprovementTracker(5)
        groups = ct.get_groups()
        assert groups == []

    def test_groups_from_strong_signal(self):
        ct = CoImprovementTracker(5)
        # Simulate: improving 0 always enables 1
        for _ in range(5):
            ct.record_sweep({0}, {1, 2, 3, 4})
            ct.record_sweep({1}, {0, 2, 3, 4})
        groups = ct.get_groups(min_strength=1.0)
        # Should find a group containing 0 and 1
        found = any(0 in g and 1 in g for g in groups)
        assert found

    def test_n_observations(self):
        ct = CoImprovementTracker(3)
        ct.record_sweep({0}, {1, 2})
        ct.record_sweep({1}, {0, 2})
        assert ct.n_observations == 2


class TestCODA:
    def test_basic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        s = CODA(5, 8, seed=0)
        r = s.search(nk, budget=100)
        assert r.best_fitness > 0
        assert r.total_evals <= 100
        assert r.diagnostics["method"] == "CODA"

    def test_deterministic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        r1 = CODA(5, 8, seed=42).search(nk, budget=100)
        r2 = CODA(5, 8, seed=42).search(nk, budget=100)
        assert r1.best_fitness == r2.best_fitness

    def test_budget_respected(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        for b in [30, 50, 100, 200, 500]:
            r = CODA(5, 8, seed=0).search(nk, budget=b)
            assert r.total_evals <= b

    def test_diagnostics(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        r = CODA(5, 10, seed=0).search(nk, budget=300)
        assert "total_cd_sweeps" in r.diagnostics
        assert "n_co_improve_groups" in r.diagnostics
        assert "total_co_improvements" in r.diagnostics
        assert "n_group_cd_improvements" in r.diagnostics

    def test_beats_random(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        rng = np.random.RandomState(0)
        random_best = max(nk(tuple(rng.randint(5, size=10))) for _ in range(200))
        r = CODA(5, 10, seed=0).search(nk, budget=200)
        assert r.best_fitness >= random_best * 0.95

    def test_large_budget(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        r = CODA(5, 10, seed=0).search(nk, budget=500)
        assert r.best_fitness > 70
        assert r.diagnostics["total_cd_sweeps"] >= 2

    def test_config(self):
        cfg = CODAConfig(
            probe_fraction=0.10,
            ils_max_stagnant=5,
            group_cd_min_sweeps=2,
        )
        nk = NKLandscape(8, 4, 5, seed=0)
        r = CODA(5, 8, config=cfg, seed=0).search(nk, budget=200)
        assert r.best_fitness > 0


if __name__ == "__main__":
    for cls_name in sorted(dir()):
        cls = eval(cls_name)
        if isinstance(cls, type) and cls_name.startswith("Test"):
            inst = cls()
            for m in sorted(dir(inst)):
                if m.startswith("test_"):
                    print(f"  {cls_name}.{m}...", end=" ", flush=True)
                    try:
                        getattr(inst, m)()
                        print("OK")
                    except Exception as e:
                        print(f"FAIL: {e}")
