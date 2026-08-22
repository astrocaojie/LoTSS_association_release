"""Shared ablation-switch helpers.

The switches default to the current production behaviour.  Callers can set
``config["ablation"]`` to disable one evidence term or rule at a time.
"""

from __future__ import annotations

from typing import Any


ABLATION_DEFAULTS: dict[str, bool] = {
    "use_multithreshold_contour": True,
    "use_ridge_continuity": True,
    "use_ellipse_overlap": True,
    "use_pa_alignment": True,
    "use_weak_edge_anti_chaining": True,
    "use_artifact_penalties_layer1": True,
    "use_artifact_penalties_layer2": True,
    "use_midpoint_host_support": True,
    "use_lobe_peak_host_contradiction": True,
    "use_stage2_relative_scale_constraints": True,
    "use_stage2_endpoint_filtering": True,
}


def ablation_config(config: dict[str, Any] | None) -> dict[str, bool]:
    """Return ablation flags with full-pipeline defaults."""

    raw = (config or {}).get("ablation", {}) or {}
    out = dict(ABLATION_DEFAULTS)
    for key in out:
        if key in raw:
            out[key] = bool(raw[key])
    return out


def ablation_enabled(config: dict[str, Any] | None, key: str) -> bool:
    """Return whether an ablation-controlled module is enabled."""

    return bool(ablation_config(config).get(key, True))
