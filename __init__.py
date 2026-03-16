"""ATLAS + NEXUS + APEX: Adaptive Architecture Search Framework."""

from .searchers import REA, PAR, FLAS, ATLAS, PARConfig, FLASConfig, ATLASConfig
from .landscapes import NKLandscape, NASBench201Surrogate, SyntheticNASLandscape
from .theory import compute_bounds
from .nexus import (
    NEXUS, NEXUSConfig,
    PersistentHomologyProbe, TopologicalFingerprint,
    SpectralSurrogate, DiscreteCurvatureTensor,
    InformationDirectedAllocator,
)
from .apex import (
    APEX, APEXConfig,
    WalshFeatureEngine, VariableInteractionGraph,
    PartitionCrossover, SampleDatabase,
)

__version__ = "0.3.0"
__all__ = [
    "REA", "PAR", "FLAS", "ATLAS", "NEXUS", "APEX",
    "PARConfig", "FLASConfig", "ATLASConfig", "NEXUSConfig", "APEXConfig",
    "NKLandscape", "NASBench201Surrogate", "SyntheticNASLandscape",
    "PersistentHomologyProbe", "TopologicalFingerprint",
    "SpectralSurrogate", "DiscreteCurvatureTensor",
    "InformationDirectedAllocator",
    "WalshFeatureEngine", "VariableInteractionGraph",
    "PartitionCrossover", "SampleDatabase",
    "compute_bounds",
]
