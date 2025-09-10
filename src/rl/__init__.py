"""
RL module for SO101 search optimization

This module contains reinforcement learning components for optimizing
the search behavior when the target object is lost from view.
"""

from .search_env import SearchRLEnv
from .metrics_tracker import ComprehensiveMetricsTracker

# Import SearchPPOAgent conditionally (requires stable-baselines3)
try:
    from .search_agent import SearchPPOAgent
    SEARCH_AGENT_AVAILABLE = True
except ImportError:
    SEARCH_AGENT_AVAILABLE = False

__all__ = [
    'SearchRLEnv',
    'ComprehensiveMetricsTracker'
]

if SEARCH_AGENT_AVAILABLE:
    __all__.append('SearchPPOAgent')