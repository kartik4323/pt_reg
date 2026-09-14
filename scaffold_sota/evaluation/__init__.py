"""Shared evaluation independent of native model thresholds and failure modes."""
from .metrics import evaluate_prediction, aggregate, compare_conditions, failure_record
from .refinement import refine_prediction, rank_candidates

__all__ = ["evaluate_prediction", "aggregate", "compare_conditions", "failure_record", "refine_prediction", "rank_candidates"]
