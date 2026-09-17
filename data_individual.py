"""
data_individual.py

Dataset and dataloader for individual strain contrastive learning.
Extends the original GEMTripletDataset to handle the new fields:
    - anchor_monoculture
    - positive_delta / negative_delta
    - positive_label_int / negative_label_int
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from data import GEMGraph

try:
    import cobra
    HAS_COBRA = True
except ImportError:
    HAS_COBRA = False


class IndividualTripletDataset(Dataset):
    """
    Dataset of (anchor, positive, negative) triplets for individual
    strain contrastive learning.

    Each triplet includes:
        - GEM graphs for anchor, positive partner, negative partner
        - anchor_monoculture: baseline growth rate of anchor alone
        - positive_delta: anchor co-culture growth - anchor monoculture
        - negative_delta: anchor co-culture growth - anchor monoculture
        - positive_label_int: categorical label int for positive pair
        - negative_label_int: categorical label int for negative pair
    """

    def __init__(
        self,
        triplets: list[dict],
        gem_builder: Optional[GEMGraph] = None,
        node_feature_dim: int = 64,
    ):
        self.triplets  = triplets
        self.gem_builder = gem_builder or GEMGraph(node_feature_dim=node_feature_dim)
        self._gem_cache: dict[str, Data] = {}

    @classmethod
    def from_json(cls, path: str,
                  node_feature_dim: int = 64) -> "IndividualTripletDataset":
        with open(path) as f:
            triplets = json.load(f)
        return cls(triplets=triplets, node_feature_dim=node_feature_dim)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        t = self.triplets[idx]

        anchor_graph   = self._load_gem(t["anchor"])
        positive_graph = self._load_gem(t["positive"])
        negative_graph = self._load_gem(t["negative"])

        item = {
            "anchor":   anchor_graph,
            "positive": positive_graph,
            "negative": negative_graph,
        }

        # Monoculture baseline
        if "anchor_monoculture" in t:
            item["anchor_monoculture"] = torch.tensor(
                [[t["anchor_monoculture"]]], dtype=torch.float32
            )

        # Continuous delta labels
        if "positive_delta" in t:
            item["positive_delta"] = torch.tensor(
                [[t["positive_delta"]]], dtype=torch.float32
            )
        if "negative_delta" in t:
            item["negative_delta"] = torch.tensor(
                [[t["negative_delta"]]], dtype=torch.float32
            )

        # Categorical label integers
        if "positive_label_int" in t:
            item["positive_label_int"] = torch.tensor(
                [t["positive_label_int"]], dtype=torch.long
            )
        if "negative_label_int" in t:
            item["negative_label_int"] = torch.tensor(
                [t["negative_label_int"]], dtype=torch.long
            )

        # Keep community growth for compatibility with visualize.py
        if "positive_growth" in t:
            item["growth_label"] = torch.tensor(
                [[t["positive_growth"]]], dtype=torch.float32
            )
        if "positive_growth" in t:
            item["positive_growth"] = torch.tensor(
                [[t["positive_growth"]]], dtype=torch.float32
            )
        if "negative_growth" in t:
            item["negative_growth"] = torch.tensor(
                [[t["negative_growth"]]], dtype=torch.float32
            )

        return item

    def _load_gem(self, path: str) -> Data:
        if path not in self._gem_cache:
            if not HAS_COBRA:
                raise RuntimeError("COBRApy required. pip install cobra")
            model = cobra.io.read_sbml_model(path)
            self._gem_cache[path] = self.gem_builder.from_cobra(model)
        return self._gem_cache[path]


def collate_individual_triplets(batch: list[dict]) -> dict:
    """Collate function for IndividualTripletDataset."""
    anchors   = Batch.from_data_list([item["anchor"]   for item in batch])
    positives = Batch.from_data_list([item["positive"] for item in batch])
    negatives = Batch.from_data_list([item["negative"] for item in batch])

    out = {"anchor": anchors, "positive": positives, "negative": negatives}

    for key in (
        "anchor_monoculture",
        "positive_delta", "negative_delta",
        "positive_label_int", "negative_label_int",
        "growth_label", "positive_growth", "negative_growth",
    ):
        if key in batch[0]:
            out[key] = torch.cat([item[key] for item in batch], dim=0)

    return out


def make_individual_dataloader(
    dataset: IndividualTripletDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_individual_triplets,
    )
