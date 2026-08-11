"""Unified validation helpers for DR1 component-reference agreement."""

from .bbox_support import match_predictions_to_dr1_components
from .dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue
from .footprint import SkyFootprint, build_dr1_sky_footprint, filter_predictions_in_footprint
from .metrics import compute_support_rates, wilson_interval

__all__ = [
    "DEFAULT_DR1_COMPONENT_CSV",
    "SkyFootprint",
    "build_dr1_sky_footprint",
    "compute_support_rates",
    "filter_predictions_in_footprint",
    "load_dr1_component_catalogue",
    "match_predictions_to_dr1_components",
    "wilson_interval",
]
