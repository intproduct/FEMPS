"""Reproducible baselines inherited from functional tensor networks."""

from .coupled_oscillators import exact_ground_energy, functional_mps_energy
from .functional_mps import random_functional_mps
from .training import BaselineConfig, run_baseline

__all__ = [
    "BaselineConfig",
    "exact_ground_energy",
    "functional_mps_energy",
    "random_functional_mps",
    "run_baseline",
]
