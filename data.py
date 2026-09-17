"""
Dataset utilities for GEM triplet contrastive learning.

A GEM (Genome-scale Metabolic Model) is represented as a bipartite graph:
    - Metabolite nodes  ←→  Reaction nodes
    - Edge weights      = stoichiometric coefficients

Triplet construction:
    Anchor   : a reference GEM (organism A in condition X)
    Positive : a GEM that should be CLOSE in embedding space
                 (e.g. same organism different condition, or high co-growth rate)
    Negative : a GEM that should be FAR in embedding space
                 (e.g. different phylogeny, or low / antagonistic co-growth rate)

This file provides:
    GEMGraph            — converts a COBRApy model to a PyG Data object
    GEMTripletDataset   — wraps a collection of GEMs + triplet index file
    collate_triplets    — DataLoader collate function
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

# Optional COBRApy import — falls back gracefully for unit tests / synthetic data
try:
    import cobra
    HAS_COBRA = True
except ImportError:
    HAS_COBRA = False


# ---------------------------------------------------------------------------
# Node / Edge feature extraction
# ---------------------------------------------------------------------------

COMPARTMENT_MAP = {"c": 0, "e": 1, "p": 2, "m": 3, "n": 4, "r": 5, "x": 6, "other": 7}
N_COMPARTMENTS = len(COMPARTMENT_MAP)


def _compartment_onehot(comp_id: str) -> list[float]:
    idx = COMPARTMENT_MAP.get(comp_id, COMPARTMENT_MAP["other"])
    oh = [0.0] * N_COMPARTMENTS
    oh[idx] = 1.0
    return oh


def metabolite_features(met, node_feature_dim: int = 64) -> list[float]:
    """
    Extract a fixed-length feature vector for a metabolite node.
    Uses: compartment (one-hot), molecular weight (normalised), charge,
          and zero-padding to node_feature_dim.
    """
    feats: list[float] = []

    # Compartment one-hot (8 dims)
    feats += _compartment_onehot(met.compartment if met.compartment else "other")

    # Molecular weight (normalised by 1000 Da)
    mw = 0.0
    if hasattr(met, "formula_weight") and met.formula_weight:
        try:
            mw = float(met.formula_weight) / 1000.0
        except (ValueError, TypeError):
            mw = 0.0
    feats.append(mw)

    # Charge (normalised by ±4)
    charge = 0.0
    if met.charge is not None:
        charge = float(met.charge) / 4.0
    feats.append(charge)

    # Pad / truncate to node_feature_dim
    feats = feats[:node_feature_dim]
    feats += [0.0] * (node_feature_dim - len(feats))
    return feats


def reaction_features(rxn, node_feature_dim: int = 64) -> list[float]:
    """
    Extract a fixed-length feature vector for a reaction node.
    Uses: lower/upper bounds (normalised), reversibility flag, # metabolites.
    """
    feats: list[float] = []

    lb = float(rxn.lower_bound) / 1000.0
    ub = float(rxn.upper_bound) / 1000.0
    reversible = 1.0 if rxn.reversibility else 0.0
    n_metabolites = len(rxn.metabolites) / 20.0  # normalise by typical max

    feats += [lb, ub, reversible, n_metabolites]

    # Pad / truncate
    feats = feats[:node_feature_dim]
    feats += [0.0] * (node_feature_dim - len(feats))
    return feats


# ---------------------------------------------------------------------------
# GEM → PyG Data
# ---------------------------------------------------------------------------

class GEMGraph:
    """Convert a COBRApy Model to a PyG Data object (bipartite graph)."""

    def __init__(self, node_feature_dim: int = 64):
        self.node_feature_dim = node_feature_dim

    def from_cobra(self, model) -> Data:
        """
        Build bipartite graph from a cobra.Model.

        Node layout: [metabolites (0..M-1), reactions (M..M+R-1)]
        Edges:       undirected metabolite–reaction edges
        Edge attrs:  stoichiometric coefficient (scalar)
        """
        mets = list(model.metabolites)
        rxns = list(model.reactions)
        M = len(mets)

        met_idx = {m.id: i for i, m in enumerate(mets)}
        rxn_idx = {r.id: M + i for i, r in enumerate(rxns)}

        # Node features
        met_feats = [metabolite_features(m, self.node_feature_dim) for m in mets]
        rxn_feats = [reaction_features(r, self.node_feature_dim) for r in rxns]
        x = torch.tensor(met_feats + rxn_feats, dtype=torch.float32)

        # Edges (both directions for undirected message passing)
        src, dst, edge_attrs = [], [], []
        for rxn in rxns:
            ri = rxn_idx[rxn.id]
            for met, coeff in rxn.metabolites.items():
                mi = met_idx[met.id]
                # met -> rxn  and  rxn -> met
                src += [mi, ri]
                dst += [ri, mi]
                edge_attrs += [float(coeff), float(coeff)]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).unsqueeze(-1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def synthetic(self, n_metabolites: int = 50, n_reactions: int = 80) -> Data:
        """
        Generate a synthetic GEM graph for testing / debugging.
        No COBRApy required.
        """
        M = n_metabolites
        R = n_reactions
        x = torch.randn(M + R, self.node_feature_dim)

        # Random bipartite edges
        src, dst = [], []
        for r in range(R):
            mets_connected = random.sample(range(M), k=min(5, M))
            for m in mets_connected:
                src += [m, M + r]
                dst += [M + r, m]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.randn(len(src), 1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Triplet Dataset
# ---------------------------------------------------------------------------

class GEMTripletDataset(Dataset):
    """
    Dataset of (anchor, positive, negative) GEM triplets.

    Expects either:
        A) A pre-built list of triplets passed directly.
        B) A triplet index JSON file with structure:
           [
             {
               "anchor":   "path/to/gem_a.json",
               "positive": "path/to/gem_b.json",
               "negative": "path/to/gem_c.json",
               "growth_label":            0.42,   // optional
               "metabolite_labels":       [...],  // optional
               "gene_expression_labels":  [...]   // optional
             },
             ...
           ]

    For quick experiments, use GEMTripletDataset.synthetic() to generate fake data.
    """

    def __init__(
        self,
        triplets: list[dict],
        gem_builder: Optional[GEMGraph] = None,
        node_feature_dim: int = 64,
    ):
        self.triplets = triplets
        self.gem_builder = gem_builder or GEMGraph(node_feature_dim=node_feature_dim)
        self._gem_cache: dict[str, Data] = {}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, triplet_index_path: str | Path, node_feature_dim: int = 64) -> "GEMTripletDataset":
        with open(triplet_index_path) as f:
            triplets = json.load(f)
        return cls(triplets=triplets, node_feature_dim=node_feature_dim)

    @classmethod
    def synthetic(
        cls,
        n_triplets: int = 200,
        n_metabolites: int = 50,
        n_reactions: int = 80,
        node_feature_dim: int = 64,
        metabolite_output_dim: int = 128,
        gene_expression_output_dim: int = 256,
        include_growth: bool = True,
        include_aux: bool = True,
    ) -> "GEMTripletDataset":
        """Generate fully synthetic triplets (no COBRApy needed)."""
        builder = GEMGraph(node_feature_dim=node_feature_dim)

        # Pre-generate a pool of synthetic GEM graphs (reused across triplets)
        pool_size = max(20, n_triplets // 5)
        gem_pool = [builder.synthetic(n_metabolites, n_reactions) for _ in range(pool_size)]

        triplets = []
        for i in range(n_triplets):
            anchor_idx, pos_idx, neg_idx = random.sample(range(pool_size), 3)
            entry: dict = {
                "_anchor_gem": gem_pool[anchor_idx],
                "_positive_gem": gem_pool[pos_idx],
                "_negative_gem": gem_pool[neg_idx],
            }
            if include_growth:
                entry["growth_label"] = float(np.random.exponential(0.5))
            if include_aux:
                entry["metabolite_labels"] = np.random.randn(metabolite_output_dim).tolist()
                entry["gene_expression_labels"] = np.random.randn(gene_expression_output_dim).tolist()
            triplets.append(entry)

        return cls(triplets=triplets, gem_builder=builder, node_feature_dim=node_feature_dim)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        t = self.triplets[idx]

        # Pre-built graphs (synthetic mode)
        if "_anchor_gem" in t:
            anchor_graph = t["_anchor_gem"]
            positive_graph = t["_positive_gem"]
            negative_graph = t["_negative_gem"]
        else:
            # Load from file paths (COBRApy mode)
            anchor_graph = self._load_gem(t["anchor"])
            positive_graph = self._load_gem(t["positive"])
            negative_graph = self._load_gem(t["negative"])

        item = {
            "anchor": anchor_graph,
            "positive": positive_graph,
            "negative": negative_graph,
        }

        if "growth_label" in t:
            item["growth_label"] = torch.tensor([[t["growth_label"]]], dtype=torch.float32)
        if "metabolite_labels" in t:
            item["metabolite_labels"] = torch.tensor(t["metabolite_labels"], dtype=torch.float32).unsqueeze(0)
        if "gene_expression_labels" in t:
            item["gene_expression_labels"] = torch.tensor(t["gene_expression_labels"], dtype=torch.float32).unsqueeze(0)

        return item

    def _load_gem(self, path: str) -> Data:
        if path not in self._gem_cache:
            if not HAS_COBRA:
                raise RuntimeError("COBRApy is required to load GEMs from file. Install it with: pip install cobra")
            model = cobra.io.read_sbml_model(path)
            self._gem_cache[path] = self.gem_builder.from_cobra(model)
        return self._gem_cache[path]


# ---------------------------------------------------------------------------
# DataLoader collation
# ---------------------------------------------------------------------------

def collate_triplets(batch: list[dict]) -> dict:
    """
    Custom collate function: batches the three PyG graphs separately,
    then stacks label tensors.
    """
    anchors = Batch.from_data_list([item["anchor"] for item in batch])
    positives = Batch.from_data_list([item["positive"] for item in batch])
    negatives = Batch.from_data_list([item["negative"] for item in batch])

    out = {"anchor": anchors, "positive": positives, "negative": negatives}

    for key in ("growth_label", "metabolite_labels", "gene_expression_labels"):
        if key in batch[0]:
            out[key] = torch.cat([item[key] for item in batch], dim=0)

    return out


def make_dataloader(
    dataset: GEMTripletDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_triplets,
    )