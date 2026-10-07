"""Scoring and correction without candidate construction or solver dependencies."""

import subprocess
import sys

import polars as pl
import pytest
import torch
import tracksdata as td
from torch import nn

from hoct.correction import fit_from_labels
from hoct.data import FrameDataset
from hoct.inference import ModelPrediction, predict_edge_scores


class ZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs, node_pos, edge_pos, edges, node_mask, edge_mask):
        features = torch.ones(*edges.shape[:2], 1, device=inputs.device)
        return ModelPrediction(
            self.head(features),
            inputs,
            features,
            torch.zeros(*inputs.shape[:2], 1, device=inputs.device),
        )


def make_dataset():
    graph = td.graph.IndexedRXGraph()
    for key in ["y", "x"]:
        graph.add_node_attr_key(key, pl.Float32, 0.0)
    graph.add_edge_attr_key("delta_t", pl.Float32, 1.0)
    graph.bulk_add_nodes(
        nodes=[{"t": 0, "y": 1.0, "x": 2.0}, {"t": 1, "y": 3.0, "x": 4.0}],
        indices=[11, 22],
    )
    graph.add_edge(11, 22, {"delta_t": 1.0})
    return FrameDataset(graph, properties=[], min_window_size=2)


def test_single_edge_scoring_and_correction():
    dataset = make_dataset()
    graph = dataset.graph
    model = ZeroModel()
    scores = predict_edge_scores(model, dataset, prefetch=False)
    assert scores["similarity"].to_list() == pytest.approx([0.5])
    graph.add_edge_attr_key("labeled", pl.Boolean, True)
    graph.add_edge_attr_key("correct", pl.Boolean, True)
    adapted = fit_from_labels(
        graph,
        model,
        "labeled",
        "correct",
        dataset=dataset,
        consistency_weight=0.0,
        n_steps=20,
    )
    assert adapted._edge_model is model
    assert predict_edge_scores(adapted, dataset)["similarity"][0] > 0.5
    assert model.head.weight.item() == 0.0


def test_empty_graph_scores():
    dataset = make_dataset()
    dataset.graph.remove_edge(11, 22)
    assert predict_edge_scores(ZeroModel(), dataset).is_empty()


def test_scoring_imports_do_not_require_tracking():
    code = """
import importlib.abc
import sys
class BlockTracking(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("hoct.tracking", "gurobipy", "hoct._api", "hoct.features.graph")):
            raise AssertionError("Scoring imported " + fullname)
sys.meta_path.insert(0, BlockTracking())
from hoct import load_model
from hoct.inference import predict_edge_scores
from hoct.correction import fit_from_labels
"""
    # Carry over the test environment's import paths (also supports source checkouts).
    code = "import sys\nsys.path[:] = " + repr(sys.path) + "\n" + code
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
