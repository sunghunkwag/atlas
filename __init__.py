"""ATLAS + NEXUS + APEX + OUROBOROS + RGSO: Adaptive Architecture Search Framework."""

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
from .rgso import (
    RGSO, RGSOConfig,
    CoarsenedVariable, CoarseningLevel,
)

__version__ = "0.5.0"
__all__ = [
    "REA", "PAR", "FLAS", "ATLAS", "NEXUS", "APEX", "OUROBOROS", "RGSO",
    "PARConfig", "FLASConfig", "ATLASConfig", "NEXUSConfig", "APEXConfig",
    "OUROBOROSConfig", "RGSOConfig",
    "NKLandscape", "NASBench201Surrogate", "SyntheticNASLandscape",
    "PersistentHomologyProbe", "TopologicalFingerprint",
    "SpectralSurrogate", "DiscreteCurvatureTensor",
    "InformationDirectedAllocator",
    "WalshFeatureEngine", "VariableInteractionGraph",
    "PartitionCrossover", "SampleDatabase",
    "CoarsenedVariable", "CoarseningLevel",
    "compute_bounds",
]
