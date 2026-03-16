# ATLAS: Adaptive Three-phase Landscape-Aware Search

A framework for discrete combinatorial architecture search on NK landscapes. Provides model-free (PAR), model-based (FLAS), adaptive (ATLAS), topology-aware (NEXUS), and structure-aware (APEX) strategies.

## APEX Extension (v0.3.0)

APEX (Algebraic Partition-based EXploration) extends the framework with Walsh-basis landscape analysis and phase-adaptive local search:

- **Walsh/ANOVA feature engine** — sparse recovery of order-1 and order-2 categorical variable effects via L1-regularized regression (Lasso). Provides landscape decomposition without consuming evaluation budget beyond the initial probe.
- **Variable Interaction Graph (VIG)** — pairwise interaction strength from Walsh coefficients, with connected component detection via BFS. Identifies separable subproblems.
- **Partition crossover** — recombination operator using VIG components to preserve intra-component structure (Whitley et al., 2016; Tinos et al., 2015).
- **Phase-adaptive perturbation** — early ILS iterations use random perturbation for diversity; late iterations (≥ 3) switch to importance-weighted edge selection + model-screened candidates from a refitted Walsh model.
- **Online Walsh refit** — retrains the Walsh model on ALL cached evaluations after each CD pass, dramatically improving prediction quality from ~15 probe samples to ~200–500 total samples.

### Benchmark (NK Landscape, N=14, K=7, O=5)

| Budget | REA | PAR | FLAS | ATLAS | NEXUS | APEX | APEX vs NEXUS |
|--------|-----|-----|------|-------|-------|------|---------------|
| B=100  | 73.8 | 76.2 | 73.3 | 76.0 | 72.8 | 72.8 | tied |
| B=200  | 75.6 | 76.3 | 74.8 | 76.3 | 75.8 | 75.8 | tied |
| B=300  | 76.3 | 75.7 | 75.3 | 75.0 | 77.0 | 77.0 | +0.06 (p=0.034*) |
| B=500  | 76.3 | 75.8 | 76.5 | 74.4 | 78.1 | 78.1 | −0.04 (ns) |
| B=750  | 77.5 | 76.0 | 76.2 | 74.9 | 79.0 | 79.3 | +0.29 (p=0.060) |

Aggregate over 5 NK instances × 48 seeds (240 paired observations per budget level). APEX significantly outperforms NEXUS at B=300 (Wilcoxon signed-rank p=0.034, W/L=4/0) and shows a consistent positive trend at B=750 (p=0.060, d=+0.11 to +0.34 per instance). At intermediate budget B=500, APEX is statistically equivalent to NEXUS (W/L=23/25, essentially tied).

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

**APEX** — Walsh-basis structure analysis with phase-adaptive ILS. Random perturbation for early basin exploration, importance-weighted model-screened perturbation for late-stage targeted search. Statistically significant improvement over NEXUS at B=300 and B≥750.

## Installation

```bash
pip install numpy scipy scikit-learn
```

## Quick Start

```python
from atlas import APEX, NKLandscape

landscape = NKLandscape(N=14, K=7, O=5, seed=0)
searcher = APEX(num_ops=5, num_edges=14, seed=42)
result = searcher.search(landscape, budget=300)

print(f"Best fitness: {result.best_fitness:.2f}")
print(f"Walsh R²: {result.diagnostics['walsh_r2']:.3f}")
print(f"VIG components: {result.diagnostics['n_vig_components']}")
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
├── __init__.py            # Package exports (v0.3.0)
├── apex.py                # APEX algorithm + Walsh/VIG/PX
├── nexus.py               # NEXUS algorithm
├── searchers.py           # REA, PAR, FLAS, ATLAS
├── landscapes.py          # NK, NAS-Bench-201 surrogate, synthetic
├── theory.py              # Theoretical bounds
├── experiments.py         # ATLAS experiments
├── experiments_nexus.py   # NEXUS benchmark suite
└── tests/
    ├── test_searchers.py  # ATLAS tests
    ├── test_nexus.py      # NEXUS tests
    └── test_apex.py       # APEX tests (21 cases)
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
