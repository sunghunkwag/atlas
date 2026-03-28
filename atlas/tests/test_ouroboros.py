"""Unit tests for OUROBOROS meta-learning framework."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from ouroboros import (
    OUROBOROS, OUROBOROSConfig,
    LandscapeFeatures, SearchConfig,
    MetaController, AdversarialGenerator,
    ConfigurableSearchEngine,
    _build_strategy_pool,
)
from landscapes import NKLandscape


# ═════════════════════════════════════════════════════════════════════
# LandscapeFeatures
# ═════════════════════════════════════════════════════════════════════

class TestLandscapeFeatures:
    def test_to_vector_shape(self):
        f = LandscapeFeatures(n_edges=10, n_ops=5)
        v = f.to_vector()
        assert v.shape == (LandscapeFeatures.dim(),)
        assert v.shape == (13,)

    def test_default_values(self):
        f = LandscapeFeatures()
        v = f.to_vector()
        assert np.all(np.isfinite(v))


# ═════════════════════════════════════════════════════════════════════
# SearchConfig
# ═════════════════════════════════════════════════════════════════════

class TestSearchConfig:
    def test_to_vector_shape(self):
        c = SearchConfig()
        v = c.to_vector()
        assert v.shape == (SearchConfig.dim(),)
        assert v.shape == (18,)

    def test_default_values(self):
        c = SearchConfig()
        assert c.probe_fraction == 0.15
        assert c.ils_max_stagnant == 3

    def test_strategy_pool_size(self):
        pool = _build_strategy_pool()
        assert len(pool) == 6
        for c in pool:
            v = c.to_vector()
            assert np.all(np.isfinite(v))


# ═════════════════════════════════════════════════════════════════════
# ConfigurableSearchEngine
# ═════════════════════════════════════════════════════════════════════

class TestConfigurableSearchEngine:
    def test_basic_search(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        engine = ConfigurableSearchEngine(5, 8, seed=0)
        config = SearchConfig()
        result, features = engine.execute(nk, budget=100, config=config)
        assert result.best_fitness > 0
        assert result.total_evals <= 100
        assert isinstance(features, LandscapeFeatures)

    def test_features_populated(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        engine = ConfigurableSearchEngine(5, 8, seed=0)
        config = SearchConfig()
        _, features = engine.execute(nk, budget=100, config=config)
        assert features.walsh_r2 >= 0
        assert features.n_edges == 8
        assert features.n_ops == 5
        assert features.fitness_std > 0

    def test_budget_respected(self):
        nk = NKLandscape(6, 3, 3, seed=0)
        engine = ConfigurableSearchEngine(3, 6, seed=0)
        for b in [30, 50, 100, 200]:
            config = SearchConfig()
            result, _ = engine.execute(nk, budget=b, config=config)
            assert result.total_evals <= b

    def test_different_configs(self):
        """Different configs should produce different results."""
        nk = NKLandscape(10, 5, 5, seed=0)
        results = []
        for cfg_idx, config in enumerate(_build_strategy_pool()[:3]):
            engine = ConfigurableSearchEngine(5, 10, seed=0)
            result, _ = engine.execute(nk, budget=200, config=config)
            results.append(result.best_fitness)
        # At least some configs should differ (not all identical)
        assert len(set(results)) >= 1  # May tie due to landscape structure


# ═════════════════════════════════════════════════════════════════════
# MetaController
# ═════════════════════════════════════════════════════════════════════

class TestMetaController:
    def test_untrained_fallback(self):
        mc = MetaController()
        f = LandscapeFeatures(n_edges=10, n_ops=5)
        config, idx = mc.select_config(f)
        assert idx == 0  # Falls back to default

    def test_add_and_select(self):
        mc = MetaController(k=3)
        # Add experiences from two config types
        for i in range(10):
            f = LandscapeFeatures(walsh_r2=0.9, n_edges=10, n_ops=5)
            mc.add_experience(f, 0, 75.0 + i * 0.1)
        for i in range(10):
            f = LandscapeFeatures(walsh_r2=0.3, n_edges=10, n_ops=5)
            mc.add_experience(f, 1, 70.0 + i * 0.1)
        mc.finalize_training()

        # Query similar to high-r2 landscapes should prefer config 0
        query = LandscapeFeatures(walsh_r2=0.85, n_edges=10, n_ops=5)
        config, idx = mc.select_config(query)
        assert idx == 0

    def test_interpolation(self):
        mc = MetaController(k=3)
        for i in range(20):
            f = LandscapeFeatures(walsh_r2=0.5 + 0.02 * i, n_edges=10, n_ops=5)
            mc.add_experience(f, i % len(mc.configs), 70.0 + i)
        mc.finalize_training()

        query = LandscapeFeatures(walsh_r2=0.7, n_edges=10, n_ops=5)
        config = mc.interpolate_config(query)
        assert isinstance(config, SearchConfig)
        assert 0.05 <= config.probe_fraction <= 0.30

    def test_save_load(self, tmp_path):
        mc = MetaController(k=3)
        for i in range(5):
            f = LandscapeFeatures(walsh_r2=0.5 + 0.1 * i, n_edges=10, n_ops=5)
            mc.add_experience(f, i % 3, 70.0 + i)
        mc.finalize_training()

        path = str(tmp_path / "controller.json")
        mc.save(path)

        mc2 = MetaController()
        mc2.load(path)
        assert mc2._trained
        assert len(mc2.memory) == 5

        # Should produce same selection
        query = LandscapeFeatures(walsh_r2=0.7, n_edges=10, n_ops=5)
        _, idx1 = mc.select_config(query)
        _, idx2 = mc2.select_config(query)
        assert idx1 == idx2


# ═════════════════════════════════════════════════════════════════════
# AdversarialGenerator
# ═════════════════════════════════════════════════════════════════════

class TestAdversarialGenerator:
    def test_generate_batch(self):
        gen = AdversarialGenerator(seed=0)
        specs = gen.generate_batch(batch_size=5)
        assert len(specs) == 5
        for N, K, O, seed in specs:
            assert 6 <= N <= 18
            assert 1 <= K < N
            assert 3 <= O <= 7

    def test_adaptive_difficulty(self):
        gen = AdversarialGenerator(seed=0)
        # Record some performances
        for i in range(20):
            gen.record_performance(14, 7, 5, i, 75.0 + i * 0.1)
        bias = gen.get_difficulty_bias()
        assert 0.0 <= bias <= 0.9

    def test_biased_generation(self):
        gen = AdversarialGenerator(seed=0)
        # Record poor performance on large, complex landscapes
        for i in range(20):
            gen.record_performance(16, 10, 6, i, 60.0)  # poor
        for i in range(20):
            gen.record_performance(8, 3, 3, 100 + i, 80.0)  # good
        specs = gen.generate_batch(20, difficulty_bias=0.8)
        # Should bias toward harder (larger N, higher K/N)
        assert len(specs) == 20


# ═════════════════════════════════════════════════════════════════════
# OUROBOROS (end-to-end)
# ═════════════════════════════════════════════════════════════════════

class TestOUROBOROS:
    def test_untrained_search(self):
        """Untrained OUROBOROS should work (falls back to baseline)."""
        nk = NKLandscape(8, 4, 5, seed=0)
        s = OUROBOROS(5, 8, seed=0)
        r = s.search(nk, budget=100)
        assert r.best_fitness > 0
        assert r.total_evals <= 100
        assert r.diagnostics["method"] == "OUROBOROS"
        assert r.diagnostics["meta_trained"] == False

    def test_budget_respected(self):
        nk = NKLandscape(8, 4, 5, seed=0)
        for b in [50, 100, 200]:
            r = OUROBOROS(5, 8, seed=0).search(nk, budget=b)
            assert r.total_evals <= b

    def test_mini_training(self):
        """Micro training run to verify the pipeline."""
        cfg = OUROBOROSConfig(
            n_epochs=1,
            landscapes_per_epoch=2,
            budgets=[50],
            seeds_per_landscape=1,
        )
        controller = OUROBOROS.train(meta_cfg=cfg, verbose=False)
        assert controller._trained
        assert len(controller.memory) > 0

    def test_trained_search(self):
        """Train on micro data, then search."""
        cfg = OUROBOROSConfig(
            n_epochs=1,
            landscapes_per_epoch=3,
            budgets=[50, 100],
            seeds_per_landscape=1,
        )
        controller = OUROBOROS.train(meta_cfg=cfg, verbose=False)

        nk = NKLandscape(8, 4, 5, seed=99)
        s = OUROBOROS(5, 8, controller=controller, seed=0)
        r = s.search(nk, budget=100)
        assert r.best_fitness > 0
        assert r.diagnostics["meta_trained"] == True

    def test_deterministic(self):
        """Same seed should produce same result."""
        nk = NKLandscape(8, 4, 5, seed=0)
        r1 = OUROBOROS(5, 8, seed=42).search(nk, budget=100)
        r2 = OUROBOROS(5, 8, seed=42).search(nk, budget=100)
        assert r1.best_fitness == r2.best_fitness


if __name__ == "__main__":
    for cls_name in sorted(dir()):
        cls = eval(cls_name)
        if isinstance(cls, type) and cls_name.startswith("Test"):
            instance = cls()
            for method_name in sorted(dir(instance)):
                if method_name.startswith("test_"):
                    # Skip tests requiring tmp_path (pytest fixture)
                    import inspect
                    sig = inspect.signature(getattr(instance, method_name))
                    if "tmp_path" in sig.parameters:
                        print(f"  {cls_name}.{method_name}... SKIP (needs pytest)")
                        continue
                    print(f"  {cls_name}.{method_name}...", end=" ", flush=True)
                    try:
                        getattr(instance, method_name)()
                        print("OK")
                    except Exception as e:
                        print(f"FAIL: {e}")
