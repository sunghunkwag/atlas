# ATLAS: Adaptive Three-phase Landscape-Aware Search

A framework for discrete combinatorial architecture search on NK landscapes. Provides model-free (PAR), model-based (FLAS), and adaptive (ATLAS) strategies with online landscape complexity estimation.

## NEXUS Extension (v0.2.0)

NEXUS (Neural EXploration via Unified Spectral-topological search) extends ATLAS with:

- **Persistent homology** — superlevel-set filtration on Hamming graphs to estimate basin count and landscape complexity via topological invariants (Edelsbrunner & Harer, 2010).
- **Spectral graph surrogate** — graph Laplacian eigenvectors as basis functions for fitness prediction with Nyström extension (Chung, 1997).
- **Discrete curvature tensor** — Ollivier-Ricci style curvature estimation for edge importance ordering in coordinate descent.
- **Iterated local search (ILS)** with importance-ordered refinement and warm-started evolution fallback.

### Benchmark (NK Landscape, N=14, K=7, O=5)

| Budget | REA | PAR | FLAS | ATLAS | NEXUS |
|--------|-----|-----|------|-------|-------|
| B=100  | 73.8 | 76.2 | 73.3 | 76.0 | 74.1 |
| B=200  | 75.6 | 76.3 | 74.8 | 76.3 | 76.6 |
| B=300  | 76.3 | 75.7 | 75.3 | 75.0 | 77.8 |
| B=500  | 76.3 | 75.8 | 76.5 | 74.4 | 79.0 |
| B=750  | 77.5 | 76.0 | 76.2 | 74.9 | 78.5 |

Cohen's d effect sizes (NEXUS vs baselines at B≥300): d=+0.78 to +1.37, Wilcoxon signed-rank p<0.01. NEXUS underperforms at low budgets (B≤200) due to topology/spectral analysis overhead. 8 seeds, 1 instance per budget level.

## Algorithms

**PAR** — Model-free probe → anneal → refine. Effective at low budgets (B<300).

**FLAS** — ANOVA sparse recovery via LASSO → model-guided exploitation. Effective at higher budgets (B>300).

**ATLAS** — Online complexity estimation for automatic PAR/FLAS selection.

**NEXUS** — Topology-aware ILS with spectral guidance and evolutionary fallback. Effective at B≥300.

## Installation

```bash
pip install numpy scipy scikit-learn
```

## Quick Start

```python
from atlas import NEXUS, NKLandscape

landscape = NKLandscape(N=14, K=7, O=5, seed=0)
searcher = NEXUS(num_ops=5, num_edges=14, seed=42)
result = searcher.search(landscape, budget=300)

print(f"Best fitness: {result.best_fitness:.2f}")
print(f"Basins detected: {result.diagnostics['topology']['n_basins']}")
```

## Experiments

```bash
# NEXUS benchmark suite
python experiments_nexus.py --fast

# Original ATLAS experiments
python experiments.py --fast

# Specific experiment (1=budget, 2=N-scale, 3=K-sweep, 4=topology)
python experiments_nexus.py --exp 1
```

## Project Structure

```
atlas/
├── __init__.py            # Package exports (v0.2.0)
├── nexus.py               # NEXUS algorithm
├── searchers.py           # REA, PAR, FLAS, ATLAS
├── landscapes.py          # NK, NAS-Bench-201 surrogate, synthetic
├── theory.py              # Theoretical bounds
├── experiments.py          # ATLAS experiments
├── experiments_nexus.py   # NEXUS benchmark suite
└── tests/
    ├── test_searchers.py  # ATLAS tests
    └── test_nexus.py      # NEXUS tests (21 cases)
```

## References

- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*.
- Chung, F. R. K. (1997). *Spectral Graph Theory*.
- Kauffman, S. A. (1993). *The Origins of Order*.
- Real, E. et al. (2019). Regularized evolution for image classifier architecture search. *AAAI*.

## License

MIT
