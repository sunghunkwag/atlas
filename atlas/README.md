# ATLAS: Adaptive Three-phase Landscape-Aware Search

A framework for discrete combinatorial architecture search on NK landscapes. Provides model-free (PAR), model-based (FLAS), adaptive (ATLAS), topology-aware (NEXUS), structure-aware (APEX), and meta-learned (OUROBOROS) strategies.

## OUROBOROS Extension (v0.4.0)

OUROBOROS is a meta-learning layer that learns to select search configurations based on landscape features. It sits on top of APEX's search engine and parameterizes all search decisions (perturbation strategy, ILS depth, evolution parameters, etc.) into a configurable 18-dimensional strategy space.

Components:

- **LandscapeFeatures** — 13-dimensional feature vector extracted from probe data (Walsh R², VIG density, importance distribution, autocorrelation, etc.). Zero additional evaluation cost.
- **SearchConfig** — parameterized strategy controlling probe/anneal/ILS/perturbation/evolution phases. 6 predefined strategies in the pool, all sharing the same probe fraction to preserve RNG trajectory.
- **MetaController** — k-NN mapping from landscape features to best config. Supports both discrete selection and performance-weighted interpolation (interpolation produces configs not in the original pool).
- **AdversarialGenerator** — generates NK landscapes biased toward parameter regions where the controller underperforms. Prevents convergence to a fixed strategy.
- **Training pipeline** — `OUROBOROS.train()` runs all configs across diverse landscapes, records (features, config, performance) tuples, and fits the controller.

### Benchmark (NK Landscape, N=14, K=7, O=5)

5 instances × 16 seeds, interpolation mode:

| Budget | OUROBOROS | NEXUS | Δ | W/L/T |
|--------|----------|-------|---|-------|
| B=100 | 73.0 | 73.1 | −0.09 | 11/12/57 |
| B=300 | 77.1 | 76.8 | +0.23 | 17/6/57 |
| B=500 | 78.3 | 78.7 | −0.31 | 20/23/37 |
| B=750 | 78.9 | 78.7 | +0.20 | 26/22/32 |

High tie rates at low budgets (where the meta-controller correctly selects the baseline config) and differentiation at higher budgets. The B=500 regression reflects the known difficulty of importance-weighted perturbation at intermediate budgets — the meta-controller's interpolated config partially mitigates but does not fully resolve this.

## APEX Extension (v0.3.0)

APEX extends the framework with Walsh-basis landscape analysis and phase-adaptive local search:

- **Walsh/ANOVA feature engine** — sparse recovery of order-1 and order-2 effects via L1-regularized regression (Lasso).
- **Variable Interaction Graph (VIG)** — pairwise interaction strength from Walsh coefficients, connected component detection via BFS.
- **Partition crossover** — recombination using VIG components (Whitley et al., 2016; Tinos et al., 2015).
- **Phase-adaptive perturbation** — random perturbation for early ILS iterations; importance-weighted + model-screened for late iterations (≥ 3).
- **Online Walsh refit** — retrains Walsh model on all cached evaluations after each CD pass.

### Benchmark (NK Landscape, N=14, K=7, O=5)

5 instances × 48 seeds:

| Budget | REA | PAR | FLAS | ATLAS | NEXUS | APEX | APEX vs NEXUS |
|--------|-----|-----|------|-------|-------|------|---------------|
| B=100  | 73.8 | 76.2 | 73.3 | 76.0 | 72.8 | 72.8 | tied |
| B=200  | 75.6 | 76.3 | 74.8 | 76.3 | 75.8 | 75.8 | tied |
| B=300  | 76.3 | 75.7 | 75.3 | 75.0 | 77.0 | 77.0 | +0.06 (p=0.034) |
| B=500  | 76.3 | 75.8 | 76.5 | 74.4 | 78.1 | 78.1 | −0.04 (ns) |
| B=750  | 77.5 | 76.0 | 76.2 | 74.9 | 79.0 | 79.3 | +0.29 (p=0.060) |

APEX outperforms NEXUS at B=300 (p=0.034, W/L=4/0) with a positive trend at B=750 (p=0.060). At B=500, APEX and NEXUS are statistically equivalent.

## NEXUS Extension (v0.2.0)

NEXUS (Neural EXploration via Unified Spectral-topological search) extends ATLAS with:

- **Persistent homology** — superlevel-set filtration on Hamming graphs to estimate basin count and landscape complexity via topological invariants (Edelsbrunner & Harer, 2010).
- **Spectral graph surrogate** — graph Laplacian eigenvectors as basis functions for fitness prediction with Nyström extension (Chung, 1997).
- **Discrete curvature tensor** — Ollivier-Ricci style curvature estimation for edge importance ordering in coordinate descent.
- **Iterated local search (ILS)** with importance-ordered refinement and warm-started evolution fallback.

## Algorithms

**PAR** — Model-free probe → anneal → refine. Effective at low budgets (B < 300).

**FLAS** — ANOVA sparse recovery via LASSO → model-guided exploitation. Effective at higher budgets (B > 300).

**ATLAS** — Online complexity estimation for automatic PAR/FLAS selection.

**NEXUS** — Topology-aware ILS with spectral guidance and evolutionary fallback. Effective at B ≥ 300.

**APEX** — Walsh-basis structure analysis with phase-adaptive ILS. Random perturbation for early basin exploration, importance-weighted model-screened perturbation for late-stage targeted search.

**OUROBOROS** — Meta-learning layer over APEX. Trains a k-NN controller on diverse NK landscapes to map landscape features to optimal search configurations. Includes adversarial landscape generation for training diversity.

## Installation

```bash
pip install numpy scipy scikit-learn
```

## Quick Start

```python
from atlas import APEX, OUROBOROS, NKLandscape

landscape = NKLandscape(N=14, K=7, O=5, seed=0)

# APEX (no pre-training needed)
searcher = APEX(num_ops=5, num_edges=14, seed=42)
result = searcher.search(landscape, budget=300)
print(f"APEX fitness: {result.best_fitness:.2f}")

# OUROBOROS (with pre-trained controller)
from atlas import OUROBOROSConfig
controller = OUROBOROS.train(OUROBOROSConfig(n_epochs=2, landscapes_per_epoch=5, budgets=[300]))
searcher = OUROBOROS(num_ops=5, num_edges=14, controller=controller, seed=42)
result = searcher.search(landscape, budget=300)
print(f"OUROBOROS fitness: {result.best_fitness:.2f}")
```

## Experiments

```bash
# Original ATLAS experiments
python experiments.py --fast

# NEXUS benchmark suite
python experiments_nexus.py --fast
```

## Project Structure

```
atlas/
├── __init__.py            # Package exports (v0.4.0)
├── ouroboros.py            # OUROBOROS meta-learning framework
├── apex.py                # APEX algorithm + Walsh/VIG/PX
├── nexus.py               # NEXUS algorithm
├── searchers.py           # REA, PAR, FLAS, ATLAS
├── landscapes.py          # NK, NAS-Bench-201 surrogate, synthetic
├── theory.py              # Theoretical bounds
├── experiments.py         # ATLAS experiments
├── experiments_nexus.py   # NEXUS benchmark suite
└── tests/
    ├── test_searchers.py  # ATLAS tests (12 cases)
    ├── test_nexus.py      # NEXUS tests (21 cases)
    ├── test_apex.py       # APEX tests (21 cases)
    └── test_ouroboros.py  # OUROBOROS tests (21 cases)
```

## References

- Whitley, D. et al. (2016). Next generation genetic algorithms. *GECCO*.
- Tinos, R. et al. (2015). Efficient recombination via partition crossover. *FOGA*.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*.
- Chung, F. R. K. (1997). *Spectral Graph Theory*.
- Kauffman, S. A. (1993). *The Origins of Order*.
- Real, E. et al. (2019). Regularized evolution for image classifier architecture search. *AAAI*.

## License

MIT
