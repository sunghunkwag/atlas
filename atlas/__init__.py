"""ATLAS + NEXUS + APEX + OUROBOROS + CODA: Adaptive Architecture Search Framework."""

import sys
import os

# Ensure root package directory is on the path so sub-modules can find
# top-level modules (searchers, landscapes, apex, etc.)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from .coda import CODA, CODAConfig, CoImprovementTracker
from .ouroboros import (
    OUROBOROS, OUROBOROSConfig,
    MetaController, AdversarialGenerator,
    LandscapeFeatures, SearchConfig,
    ConfigurableSearchEngine,
)

__version__ = "0.4.0"
__all__ = [
    "CODA", "CODAConfig", "CoImprovementTracker",
    "OUROBOROS", "OUROBOROSConfig",
    "MetaController", "AdversarialGenerator",
    "LandscapeFeatures", "SearchConfig",
    "ConfigurableSearchEngine",
]
