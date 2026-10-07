"""Feature extraction and candidate-graph construction for HOCT."""

from importlib import import_module

from hoct.features.constants import EDGE_GT_KEY, REGIONPROPS

__all__ = [
    "EDGE_GT_KEY",
    "REGIONPROPS",
    "add_border_dist",
    "add_delta_t",
    "add_features",
    "add_is_div",
    "border_dist_2d",
    "border_dist_3d",
    "constants",
    "convert_to_2d",
    "convert_to_3d",
    "create_graph",
    "features",
    "graph",
    "normalize_image",
]


def __getattr__(name):
    if name in ("constants", "features", "graph"):
        return import_module(f"hoct.features.{name}")
    if name in ("add_features", "convert_to_2d", "convert_to_3d", "create_graph"):
        return getattr(import_module("hoct.features.graph"), name)
    if name in __all__:
        return getattr(import_module("hoct.features.features"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
