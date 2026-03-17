"""Unit tests for RGSO (Renormalization Group Search Optimization)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from rgso import RGSO, RGSOConfig, CoarsenedVariable, CoarseningLevel, build_coarsening
from apex import WalshFeatureEngine, VariableInteractionGraph
from landscapes import NKLandscape


class TestCoarsenedVariable:
    def test_basic(self):
        mapping = [(0, 0), (1, 1), (0, 1), (1, 0), (2, 2)]
        sv = CoarsenedVariable(var_i=0, var_j=1, mapping=mapping, n_values=5)
        assert sv.decode(0) == (0, 0)
        assert sv.decode(1) == (1, 1)
        assert sv.decode(4) == (2, 2)
        assert sv.n_values == 5

    def test_single_value(self):
        sv = CoarsenedVariable(0, 1, [(3, 4)], 1)
        assert sv.decode(0) == (3, 4)


class TestCoarseningLevel:
    def _make_level(self):
        sv0 = CoarsenedVariable(0, 1, [(0, 0), (1, 1), (0, 1)], 3)
        sv1 = CoarsenedVariable(2, 3, [(0, 0), (1, 1), (2, 2)], 3)
        return CoarseningLevel([sv0, sv1], remainder=[4], n_ops_original=3)

    def test_decode(self):
        level = self._make_level()
        base = [0, 0, 0, 0, 0]
        # super_sol: sv0=1 -> (1,1), sv1=2 -> (2,2), rem[4]=2
        result = level.decode_solution([1, 2, 2], base)
        assert result == [1, 1, 2, 2, 2]

    def test_encode(self):
        level = self._make_level()
        # Original [1, 1, 0, 0, 2] -> sv0 maps (1,1)=val 1, sv1 maps (0,0)=val 0
        encoded = level.encode_solution([1, 1, 0, 0, 2])
        assert encoded == [1, 0, 2]

    def test_roundtrip(self):
        level = self._make_level()
        base = [0, 0, 0, 0, 0]
        super_sol = [1, 2, 1]
        decoded = level.decode_solution(super_sol, base)
        re_encoded = level.encode_solution(decoded)
        assert re_encoded == super_sol

    def test_n_total(self):
        level = self._make_level()
        assert level.n_super == 2
        assert level.n_total == 3  # 2 super + 1 remainder

    def test_decode_preserves_unmatched(self):
        """Decode should only modify paired variables + remainder."""
        level = self._make_level()
        base = [9, 9, 9, 9, 9]  # non-zero base
        result = level.decode_solution([0, 0, 0], base)
        # sv0=0 -> (0,0) at vars 0,1; sv1=0 -> (0,0) at vars 2,3; rem=0 at var 4
        assert result == [0, 0, 0, 0, 0]


class TestBuildCoarsening:
    def test_builds_from_walsh(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        rng = np.random.RandomState(0)
        X = np.array([rng.randint(5, size=8) for _ in range(50)])
        y = np.array([nk(tuple(x)) for x in X])

        walsh = WalshFeatureEngine(5, 8, max_order=2)
        r2 = walsh.fit(X, y)

        vig = VariableInteractionGraph(8)
        vig.build_from_walsh(walsh, threshold=0.001)

        level = build_coarsening(walsh, vig, 8, 5)
        assert level.n_super >= 1
        assert level.n_super + len(level.remainder) <= 8
        # Each super-var should have exactly O=5 mappings
        for sv in level.super_vars:
            assert len(sv.mapping) == 5
            assert sv.n_values == 5

    def test_all_vars_accounted(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        rng = np.random.RandomState(0)
        X = np.array([rng.randint(5, size=8) for _ in range(50)])
        y = np.array([nk(tuple(x)) for x in X])

        walsh = WalshFeatureEngine(5, 8, max_order=2)
        walsh.fit(X, y)
        vig = VariableInteractionGraph(8)
        vig.build_from_walsh(walsh, threshold=0.001)

        level = build_coarsening(walsh, vig, 8, 5)
        paired_vars = set()
        for sv in level.super_vars:
            paired_vars.add(sv.var_i)
            paired_vars.add(sv.var_j)
        all_vars = paired_vars | set(level.remainder)
        assert all_vars == set(range(8))


class TestRGSO:
    def test_basic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        s = RGSO(5, 8, seed=0)
        r = s.search(nk, budget=100)
        assert r.best_fitness > 0
        assert r.total_evals <= 100
        assert r.diagnostics["method"] == "RGSO"

    def test_deterministic(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        r1 = RGSO(5, 8, seed=42).search(nk, budget=100)
        r2 = RGSO(5, 8, seed=42).search(nk, budget=100)
        assert r1.best_fitness == r2.best_fitness

    def test_budget_respected(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        for b in [30, 50, 100, 200, 500]:
            r = RGSO(5, 8, seed=0).search(nk, budget=b)
            assert r.total_evals <= b, f"Budget {b} exceeded: {r.total_evals}"

    def test_diagnostics(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        r = RGSO(5, 10, seed=0).search(nk, budget=300)
        assert "method" in r.diagnostics
        assert r.diagnostics["method"] == "RGSO"
        assert "n_ugp_perturbations" in r.diagnostics
        assert "ils_iterations" in r.diagnostics

    def test_beats_random(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        rng = np.random.RandomState(0)
        random_best = max(nk(tuple(rng.randint(5, size=10))) for _ in range(200))
        r = RGSO(5, 10, seed=0).search(nk, budget=200)
        assert r.best_fitness >= random_best * 0.95

    def test_large_budget(self):
        nk = NKLandscape(10, 5, 5, seed=0)
        r = RGSO(5, 10, seed=0).search(nk, budget=500)
        assert r.best_fitness > 70
        assert r.diagnostics.get("ils_iterations", 0) >= 1

    def test_config(self):
        cfg = RGSOConfig(
            probe_fraction=0.10,
            ugp_uncertainty_weight=3.0,
            ugp_enabled=True,
            ils_max_stagnant=5,
        )
        nk = NKLandscape(8, 4, 5, seed=0)
        r = RGSO(5, 8, config=cfg, seed=0).search(nk, budget=200)
        assert r.best_fitness > 0

    def test_ugp_tracked(self):
        """UGP perturbation count should be reported in diagnostics."""
        nk = NKLandscape(10, 5, 5, seed=0)
        r = RGSO(5, 10, seed=0).search(nk, budget=300)
        assert "n_ugp_perturbations" in r.diagnostics
        assert isinstance(r.diagnostics["n_ugp_perturbations"], int)
        assert r.diagnostics["n_ugp_perturbations"] >= 0

    def test_small_problem(self):
        """RGSO should handle very small problems gracefully."""
        nk = NKLandscape(4, 2, 3, seed=0)
        r = RGSO(3, 4, seed=0).search(nk, budget=50)
        assert r.best_fitness > 0
        assert r.total_evals <= 50

    def test_minimal_budget(self):
        """With very small budget, should still return a result."""
        nk = NKLandscape(8, 4, 5, seed=0)
        r = RGSO(5, 8, seed=0).search(nk, budget=10)
        assert r.best_fitness > 0
        assert r.total_evals <= 10


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
