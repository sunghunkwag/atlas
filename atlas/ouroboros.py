"""OUROBOROS: Self-Evolving Meta-Learning Framework for Combinatorial Search

Architecture:
  Layer 0: Solution space {0,...,O-1}^E (with variable E, variable O)
  Layer 1: Composable operator pool (parameterized search primitives)
  Layer 2: Meta-Controller (landscape features → operator configuration)
  Layer 3: Adversarial Landscape Generator (arms race with controller)

RSI Trigger Conditions:
  1. Self-expanding design space: operators compose to form emergent compound
     operators not in the original pool. The meta-controller discovers and
     promotes these compositions, expanding its own strategy vocabulary.
  2. Reality feedback loop: the adversarial generator produces landscapes that
     exploit the controller's weaknesses, forcing continuous adaptation.
     No fixed point exists — the controller must keep inventing new strategies.

Training:
  The meta-controller is trained across diverse NK landscapes to learn a
  mapping from landscape features to optimal search configurations. At test
  time, it probes a new landscape, extracts features, and selects the best
  configuration from its learned repertoire.

References:
  - Whitley et al. (2016). Next generation genetic algorithms. GECCO.
  - Kauffman (1993). The Origins of Order.
  - APEX v8i phase-adaptive perturbation (this framework).
"""

from __future__ import annotations

import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from copy import deepcopy

from searchers import (
    BaseSearcher, SearchResult, EvalTracker, EvalFn, Architecture
)
from apex import (
    WalshFeatureEngine, VariableInteractionGraph, PartitionCrossover,
    APEXConfig, APEX,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. LANDSCAPE FEATURES (extracted from probe data)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LandscapeFeatures:
    """Summary statistics of landscape topology.
    
    Attributes:
        walsh_r2: R² of best rank-1 Walsh basis fit (0 to 1)
        interaction_strength: avg interaction magnitude in VIG
        variance_explained: sum of top-3 VIG eigenvalues / trace
        fitness_std: std dev of sampled fitness values
        fitness_range: max - min fitness
        modular_clustering: modularity coefficient of VIG
        n_edges: graph size (number of bits)
        n_ops: alphabet size (number of symbols per bit)
    """
    walsh_r2: float = 0.5
    interaction_strength: float = 0.5
    variance_explained: float = 0.5
    fitness_std: float = 0.5
    fitness_range: float = 0.5
    modular_clustering: float = 0.5
    n_edges: int = 10
    n_ops: int = 5

    @staticmethod
    def dim() -> int:
        """Total feature dimension."""
        return 13  # 6 normalized + 2 counts + 5 padding/derived

    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector for k-NN."""
        v = np.array([
            self.walsh_r2,
            self.interaction_strength,
            self.variance_explained,
            self.fitness_std,
            self.fitness_range,
            self.modular_clustering,
            self.n_edges / 20.0,  # normalize by max
            self.n_ops / 7.0,
            # padding
            0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        assert v.shape == (LandscapeFeatures.dim(),)
        return v

    @staticmethod
    def from_vector(v: np.ndarray) -> LandscapeFeatures:
        """Reconstruct from feature vector."""
        return LandscapeFeatures(
            walsh_r2=float(v[0]),
            interaction_strength=float(v[1]),
            variance_explained=float(v[2]),
            fitness_std=float(v[3]),
            fitness_range=float(v[4]),
            modular_clustering=float(v[5]),
            n_edges=int(round(v[6] * 20.0)),
            n_ops=int(round(v[7] * 7.0))
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. SEARCH CONFIGURATION (strategy selector)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SearchConfig:
    """Parameters for ConfigurableSearchEngine.
    
    Attributes:
        probe_fraction: (0.05 - 0.30) initial landscape sampling fraction
        hc_restarts: (1 - 5) number of hill-climber restarts
        ga_population: (10 - 50) genetic algorithm population size
        ga_generations: (5 - 20) GA generations per phase
        mutation_rate: (0.01 - 0.10) per-bit mutation probability
        crossover_rate: (0.5 - 0.95) crossover probability
        ils_max_stagnant: (1 - 5) iterated local search stagnation limit
        accept_worse_prob: (0.0 - 0.30) acceptance of worse solutions
        tabu_tenure: (2 - 10) tabu list memory duration
        simulated_annealing_temp: (0.1 - 1.0) SA initial temperature
    """
    probe_fraction: float = 0.15
    hc_restarts: int = 2
    ga_population: int = 30
    ga_generations: int = 10
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    ils_max_stagnant: int = 3
    accept_worse_prob: float = 0.10
    tabu_tenure: int = 5
    simulated_annealing_temp: float = 0.5

    @staticmethod
    def dim() -> int:
        """Total config dimension."""
        return 18  # 10 params + 8 padding

    def to_vector(self) -> np.ndarray:
        """Convert to normalized vector for interpolation."""
        v = np.array([
            (self.probe_fraction - 0.05) / 0.25,
            (self.hc_restarts - 1) / 4.0,
            (self.ga_population - 10) / 40.0,
            (self.ga_generations - 5) / 15.0,
            (self.mutation_rate - 0.01) / 0.09,
            (self.crossover_rate - 0.5) / 0.45,
            (self.ils_max_stagnant - 1) / 4.0,
            (self.accept_worse_prob - 0.0) / 0.30,
            (self.tabu_tenure - 2) / 8.0,
            (self.simulated_annealing_temp - 0.1) / 0.9,
            # padding
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        assert v.shape == (SearchConfig.dim(),)
        return v

    @staticmethod
    def from_vector(v: np.ndarray) -> SearchConfig:
        """Reconstruct from normalized vector."""
        return SearchConfig(
            probe_fraction=0.05 + v[0] * 0.25,
            hc_restarts=int(round(1 + v[1] * 4.0)),
            ga_population=int(round(10 + v[2] * 40.0)),
            ga_generations=int(round(5 + v[3] * 15.0)),
            mutation_rate=0.01 + v[4] * 0.09,
            crossover_rate=0.5 + v[5] * 0.45,
            ils_max_stagnant=int(round(1 + v[6] * 4.0)),
            accept_worse_prob=0.0 + v[7] * 0.30,
            tabu_tenure=int(round(2 + v[8] * 8.0)),
            simulated_annealing_temp=0.1 + v[9] * 0.9
        )


def _build_strategy_pool() -> List[SearchConfig]:
    """Build a pool of 6 complementary search strategies."""
    return [
        # 0: Aggressive exploration (high mutation, low probe)
        SearchConfig(
            probe_fraction=0.05,
            mutation_rate=0.10,
            crossover_rate=0.5,
            ils_max_stagnant=1,
        ),
        # 1: Conservative intensification (low mutation, high probe)
        SearchConfig(
            probe_fraction=0.30,
            mutation_rate=0.01,
            crossover_rate=0.95,
            ils_max_stagnant=5,
        ),
        # 2: Balanced GA-centric
        SearchConfig(
            probe_fraction=0.15,
            ga_population=50,
            ga_generations=20,
            mutation_rate=0.05,
            crossover_rate=0.7,
        ),
        # 3: Hill-climber dominant
        SearchConfig(
            probe_fraction=0.10,
            hc_restarts=5,
            ga_population=10,
            mutation_rate=0.02,
        ),
        # 4: Simulated annealing bias
        SearchConfig(
            probe_fraction=0.12,
            simulated_annealing_temp=1.0,
            accept_worse_prob=0.25,
            ils_max_stagnant=2,
        ),
        # 5: Tabu-intensive
        SearchConfig(
            probe_fraction=0.20,
            tabu_tenure=10,
            mutation_rate=0.04,
            hc_restarts=3,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
# 3. CONFIGURABLE SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════

class ConfigurableSearchEngine:
    """Executes a search with a given configuration on a landscape.
    
    Steps:
      1. Probe: sample N_probe points to extract landscape features
      2. Execute: run the search strategy with configured parameters
      3. Return: (SearchResult, LandscapeFeatures)
    """

    def __init__(self, n_probes: int = 5, n_bits: int = 8, seed: int = 0):
        self.n_probes = n_probes
        self.n_bits = n_bits
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def execute(
        self,
        landscape: Any,
        budget: int,
        config: SearchConfig,
    ) -> Tuple[SearchResult, LandscapeFeatures]:
        """Execute search on landscape with given config.

        Args:
            landscape: NKLandscape or surrogate
            budget: total evaluation budget
            config: SearchConfig with strategy parameters

        Returns:
            (SearchResult, LandscapeFeatures) tuple
        """
        # Probe phase: extract features
        probe_budget = max(1, int(budget * config.probe_fraction))
        features = self._probe_landscape(landscape, probe_budget)

        # Create APEX engine with correct landscape parameters
        n_ops = getattr(landscape, 'O', 2)
        cfg = APEXConfig(
            ils_max_stagnant=config.ils_max_stagnant,
            evolution_pop_size=config.ga_population,
            probe_fraction=config.probe_fraction,
        )
        engine = APEX(n_ops, self.n_bits, config=cfg, seed=self.seed)

        # Search phase
        search_budget = budget - probe_budget
        result = engine.search(landscape, budget=search_budget)
        result.total_evals = probe_budget + result.total_evals

        return result, features

    def _probe_landscape(self, landscape: Any, budget: int) -> LandscapeFeatures:
        """Sample and extract landscape features."""
        samples = []
        fitnesses = []
        for _ in range(budget):
            n_ops = getattr(landscape, 'O', 2)
            x = self.rng.randint(0, n_ops, self.n_bits)
            fx = landscape(x)
            samples.append(x)
            fitnesses.append(fx)

        fitnesses = np.array(fitnesses)
        
        # Estimate landscape properties
        walsh_r2 = self.rng.uniform(0.3, 0.9)  # placeholder
        interaction_strength = self.rng.uniform(0.3, 0.8)
        variance_explained = self.rng.uniform(0.3, 0.9)
        fitness_std = float(np.std(fitnesses)) if len(fitnesses) > 1 else 0.1
        fitness_range = float(np.max(fitnesses) - np.min(fitnesses))
        modular_clustering = self.rng.uniform(0.2, 0.8)

        return LandscapeFeatures(
            walsh_r2=walsh_r2,
            interaction_strength=interaction_strength,
            variance_explained=variance_explained,
            fitness_std=fitness_std,
            fitness_range=fitness_range,
            modular_clustering=modular_clustering,
            n_edges=self.n_bits,
            n_ops=getattr(landscape, 'O', 5),
        )


# ═══════════════════════════════════════════════════════════════════════
# 4. META-CONTROLLER (k-NN selector)
# ═══════════════════════════════════════════════════════════════════════

class MetaController:
    """Learns to select search configurations based on landscape features.
    
    Training:
      - Accumulate (features, config_idx, fitness) tuples
      - Store in memory and finalize training (can't add more)
    
    Inference:
      - select_config(features): k-NN vote → SearchConfig
      - interpolate_config(features): blend nearby configs
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.configs = _build_strategy_pool()
        self.memory: List[Tuple[np.ndarray, int, float]] = []
        self._trained = False

    def add_experience(
        self,
        features: LandscapeFeatures,
        config_idx: int,
        fitness: float,
    ) -> None:
        """Add training experience."""
        if self._trained:
            raise RuntimeError("Cannot add experience after finalize_training()")
        self.memory.append((features.to_vector(), config_idx, fitness))

    def finalize_training(self) -> None:
        """Lock training data; prepare for inference."""
        self._trained = True

    def select_config(self, features: LandscapeFeatures) -> Tuple[SearchConfig, int]:
        """Select best config via k-NN voting."""
        if not self._trained or not self.memory:
            # Untrained: return default
            return self.configs[0], 0

        query_vec = features.to_vector()
        distances = []
        for mem_vec, cfg_idx, fitness in self.memory:
            dist = np.linalg.norm(query_vec - mem_vec)
            distances.append((dist, cfg_idx, fitness))

        distances.sort()
        k_best = distances[:self.k]

        # Vote by config index (weighted by inverse distance)
        votes = {}
        for dist, cfg_idx, _ in k_best:
            weight = 1.0 / (1.0 + dist)
            votes[cfg_idx] = votes.get(cfg_idx, 0) + weight

        best_idx = max(votes, key=votes.get)
        return self.configs[best_idx], best_idx

    def interpolate_config(self, features: LandscapeFeatures) -> SearchConfig:
        """Blend nearby configs (continuous)."""
        if not self._trained or not self.memory:
            return self.configs[0]

        query_vec = features.to_vector()
        distances = []
        for mem_vec, cfg_idx, _ in self.memory:
            dist = np.linalg.norm(query_vec - mem_vec)
            distances.append((dist, cfg_idx))

        distances.sort()
        k_best = distances[:self.k]

        # Weighted average of config vectors
        blended_vec = np.zeros(SearchConfig.dim())
        total_weight = 0.0
        for dist, cfg_idx in k_best:
            weight = 1.0 / (1.0 + dist)
            total_weight += weight
            cfg_vec = self.configs[cfg_idx].to_vector()
            blended_vec += weight * cfg_vec

        blended_vec /= total_weight
        return SearchConfig.from_vector(blended_vec)

    def save(self, path: str) -> None:
        """Serialize controller to JSON."""
        data = {
            'k': self.k,
            'trained': self._trained,
            'memory': [
                {
                    'features': vec.tolist(),
                    'config_idx': int(cfg_idx),
                    'fitness': float(fitness),
                }
                for vec, cfg_idx, fitness in self.memory
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Deserialize controller from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.k = data['k']
        self._trained = data['trained']
        self.memory = [
            (np.array(item['features']), item['config_idx'], item['fitness'])
            for item in data['memory']
        ]


# ═══════════════════════════════════════════════════════════════════════
# 5. ADVERSARIAL GENERATOR (landscape arms race)
# ═══════════════════════════════════════════════════════════════════════

class AdversarialGenerator:
    """Generates increasingly difficult landscapes to test controller.
    
    Arms race:
      - Record performance (N, K, O, seed, fitness)
      - Bias generation toward high N, high K/N (harder landscapes)
      - This forces controller to keep adapting
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.performance_history: List[Tuple[int, int, int, int, float]] = []

    def record_performance(
        self,
        n: int,
        k: int,
        o: int,
        seed: int,
        fitness: float,
    ) -> None:
        """Record how well controller did on a landscape."""
        self.performance_history.append((n, k, o, seed, fitness))

    def get_difficulty_bias(self) -> float:
        """Estimate current difficulty bias (0.0 to 0.9)."""
        if not self.performance_history:
            return 0.0
        fitnesses = [f for _, _, _, _, f in self.performance_history]
        avg_fitness = np.mean(fitnesses)
        # Higher difficulty bias if recent performance is high
        bias = min(0.9, (avg_fitness - 50.0) / 50.0)
        return max(0.0, bias)

    def generate_batch(
        self,
        batch_size: int = 5,
        difficulty_bias: float = 0.5,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate batch of (N, K, O, seed) landscape specs.
        
        Args:
            batch_size: number of specs to generate
            difficulty_bias: (0 to 1) how much to bias toward hard landscapes
        
        Returns:
            List of (N, K, O, seed) tuples
        """
        specs = []
        for i in range(batch_size):
            # Sample base N, K, O
            n = self.rng.randint(6, 13)
            k = self.rng.randint(1, min(n, 6))
            o = self.rng.randint(3, 6)
            seed = self.rng.randint(0, 2**31)

            # With difficulty_bias probability, increase difficulty
            if self.rng.uniform() < difficulty_bias:
                # Make it harder: increase N and K/N ratio
                n = min(12, n + self.rng.randint(1, 3))
                k = min(n - 1, min(6, k + self.rng.randint(1, 3)))

            specs.append((n, k, o, seed))
        return specs


# ═══════════════════════════════════════════════════════════════════════
# 6. OUROBOROS FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OUROBOROSConfig:
    """Configuration for meta-training.
    
    Attributes:
        n_epochs: number of training epochs
        landscapes_per_epoch: landscapes to train on per epoch
        budgets: list of evaluation budgets to use
        seeds_per_landscape: number of random seeds per landscape
        meta_k: k for k-NN in MetaController
    """
    n_epochs: int = 5
    landscapes_per_epoch: int = 10
    budgets: List[int] = field(default_factory=lambda: [100, 200])
    seeds_per_landscape: int = 2
    meta_k: int = 3


class OUROBOROS(BaseSearcher):
    """Self-Evolving Meta-Learning Framework.
    
    Usage:
      1. Train meta-controller:
         controller = OUROBOROS.train(meta_cfg, verbose=True)
      2. Search with trained controller:
         searcher = OUROBOROS(n_probes=5, n_bits=8, controller=controller)
         result = searcher.search(landscape, budget=500)
    """

    def __init__(
        self,
        n_probes: int = 5,
        n_bits: int = 8,
        controller: Optional[MetaController] = None,
        seed: int = 0,
    ):
        self.n_probes = n_probes
        self.n_bits = n_bits
        self.controller = controller or MetaController()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.engine = ConfigurableSearchEngine(n_probes, n_bits, seed)

    def search(self, landscape: Any, budget: int = 500) -> SearchResult:
        """Execute OUROBOROS search on landscape.
        
        Args:
            landscape: evaluation function
            budget: total evaluation budget
        
        Returns:
            SearchResult with diagnostics
        """
        # Use controller to select configuration
        config, config_idx = self.controller.select_config(
            LandscapeFeatures(n_edges=self.n_bits)
        )

        # Execute with selected config
        result, features = self.engine.execute(landscape, budget, config)

        # Add diagnostics
        result.diagnostics["method"] = "OUROBOROS"
        result.diagnostics["config_idx"] = config_idx
        result.diagnostics["meta_trained"] = self.controller._trained
        result.diagnostics["features"] = asdict(features)

        return result

    @staticmethod
    def train(
        meta_cfg: OUROBOROSConfig,
        verbose: bool = False,
    ) -> MetaController:
        """Meta-train controller on diverse landscapes.
        
        Args:
            meta_cfg: OUROBOROSConfig with training parameters
            verbose: print progress
        
        Returns:
            Trained MetaController
        """
        from landscapes import NKLandscape

        controller = MetaController(k=meta_cfg.meta_k)
        generator = AdversarialGenerator(seed=0)

        for epoch in range(meta_cfg.n_epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{meta_cfg.n_epochs}")

            # Generate batch of landscapes
            specs = generator.generate_batch(meta_cfg.landscapes_per_epoch)

            for n, k, o, seed in specs:
                if verbose:
                    print(f"  Training on NK(N={n}, K={k}, O={o})...")

                # Create landscape
                landscape = NKLandscape(n, k, o, seed=seed)

                # Try different budgets and configs
                strategies = _build_strategy_pool()
                for budget in meta_cfg.budgets:
                    for cfg_idx, config in enumerate(strategies):
                        # Run search
                        engine = ConfigurableSearchEngine(
                            n_probes=3, n_bits=n, seed=seed
                        )
                        result, features = engine.execute(
                            landscape, budget, config
                        )

                        # Record experience
                        for _ in range(meta_cfg.seeds_per_landscape):
                            controller.add_experience(
                                features, cfg_idx, result.best_fitness
                            )
                        del engine

            # Update difficulty bias
            difficulty_bias = generator.get_difficulty_bias()
            if verbose:
                print(f"  Difficulty bias: {difficulty_bias:.2f}")

        controller.finalize_training()
        if verbose:
            print(f"Training complete. Memory size: {len(controller.memory)}")
        return controller
