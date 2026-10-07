"""Inference API for Higher-Order Cell Tracking Transformer (HOCT) model."""

from hoct.__about__ import __version__

__all__ = ["__version__", "available_models", "load_model", "predict"]


def __getattr__(name):
    if name in ("available_models", "load_model"):
        from hoct import _models

        return getattr(_models, name)
    if name == "predict":
        from hoct._api import predict

        return predict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
