"""Reproducible, reduced-data runners for public point-cloud assembly models."""

from .registry import ModelSpec, load_registry

__all__ = ["ModelSpec", "load_registry"]
