"""ATLAS: Adaptive Three-phase Landscape-Aware Search for NAS."""

from .searchers import REA, PAR, FLAS, ATLAS, PARConfig, FLASConfig, ATLASConfig
from .landscapes import NKLandscape, NASBench201Surrogate, SyntheticNASLandscape
from .theory import compute_bounds

__version__ = "0.1.0"
__all__ = [
    "REA", "PAR", "FLAS", "ATLAS",
    "PARConfig", "FLASConfig", "ATLASConfig",
    "NKLandscape", "NASBench201Surrogate", "SyntheticNASLandscape",
    "compute_bounds",
]
