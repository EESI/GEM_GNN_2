"""
GNN-GLM: Graph Neural Network + Genome-scale Language Model
for microbial community growth prediction via contrastive (triplet) learning.

Architecture (mirrors the diagram):
  GEM (stoichiometric graph) -> GNN-GLM encoder -> embedding
  Triplet of (anchor, positive, negative) -> TripletLoss
  Optional auxiliary heads: metabolite profile, gene expression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GNNGLMConfig:
    # Graph encoder
    node_feature_dim: int = 64        # metabolite / reaction node features
    edge_feature_dim: int = 16        # stoichiometric coefficient features
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 4
    gnn_heads: int = 4                # GAT attention heads
    gnn_dropout: float = 0.1

    # Embedding / projection head
    embedding_dim: int = 128          # final contrastive embedding size
    projection_hidden_dim: int = 256

    # Auxiliary heads (optional)
    use_metabolite_head: bool = True
    use_gene_expression_head: bool = True
    metabolite_output_dim: int = 128  # number of metabolites to profile
    gene_expression_output_dim: int = 256  # number of genes

    # Growth prediction head (community or individual)
    use_growth_head: bool = True
    growth_output_dim: int = 1        # scalar growth rate

    # Loss weights  (λ in the diagram)
    lambda_aux: float = 0.1
    triplet_margin: float = 1.0


# ---------------------------------------------------------------------------
# GNN-GLM Encoder
# ---------------------------------------------------------------------------

class GNNLayer(nn.Module):
    """Single GAT layer with residual connection + LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.conv = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        # Residual projection if dims differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        out = self.conv(x, edge_index)
        out = F.gelu(out)
        out = self.drop(out)
        out = self.norm(out + self.residual(x))
        return out


class GNNGLMEncoder(nn.Module):
    """
    Encodes a genome-scale model (GEM) represented as a bipartite graph
    (metabolites <-> reactions) into a fixed-size embedding vector.

    Node types:
        - Metabolite nodes: chemical features (MW, charge, compartment…)
        - Reaction nodes:   stoichiometry, bounds, gene-reaction rules

    The 'GLM' suffix reflects that gene-reaction associations can be
    optionally encoded via a small embedding table (gene language model).
    """

    def __init__(self, cfg: GNNGLMConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.node_feature_dim, cfg.gnn_hidden_dim),
            nn.LayerNorm(cfg.gnn_hidden_dim),
            nn.GELU(),
        )

        # Stacked GAT layers
        self.layers = nn.ModuleList()
        for _ in range(cfg.gnn_num_layers):
            self.layers.append(GNNLayer(
                in_dim=cfg.gnn_hidden_dim,
                out_dim=cfg.gnn_hidden_dim,
                heads=cfg.gnn_heads,
                dropout=cfg.gnn_dropout,
            ))

        # Dual pooling: mean + sum -> captures both average state and total flux capacity
        pool_dim = cfg.gnn_hidden_dim * 2

        # Projection head (MLP) -> contrastive embedding space
        self.projection_head = nn.Sequential(
            nn.Linear(pool_dim, cfg.projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.gnn_dropout),
            nn.Linear(cfg.projection_hidden_dim, cfg.embedding_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object with:
                  .x           [N, node_feature_dim]  node features
                  .edge_index  [2, E]
                  .batch       [N]  (graph batch assignments)
        Returns:
            embedding: [B, embedding_dim]  L2-normalised
        """
        x = self.input_proj(data.x)

        for layer in self.layers:
            x = layer(x, data.edge_index)

        # Dual readout
        x_mean = global_mean_pool(x, data.batch)
        x_sum = global_add_pool(x, data.batch)
        x_pool = torch.cat([x_mean, x_sum], dim=-1)

        emb = self.projection_head(x_pool)
        return F.normalize(emb, p=2, dim=-1)  # unit-sphere for cosine / margin loss


# ---------------------------------------------------------------------------
# Auxiliary Prediction Heads
# ---------------------------------------------------------------------------

class AuxiliaryHead(nn.Module):
    """Generic MLP prediction head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GrowthHead(nn.Module):
    """
    Predicts community (or individual) growth rate from a *pair* of embeddings.
    Uses element-wise product + difference as interaction features (bilinear-style).
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Interaction: [emb_a, emb_b, emb_a * emb_b, |emb_a - emb_b|]
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # growth rate >= 0
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_a, emb_b: [B, emb_dim] embeddings of two community members
        Returns:
            growth: [B, 1]
        """
        interaction = torch.cat([
            emb_a,
            emb_b,
            emb_a * emb_b,
            (emb_a - emb_b).abs(),
        ], dim=-1)
        return self.net(interaction)


# ---------------------------------------------------------------------------
# Full GNN-GLM System
# ---------------------------------------------------------------------------

@dataclass
class ModelOutputs:
    # Embeddings for all three triplet members
    anchor_emb: torch.Tensor
    positive_emb: torch.Tensor
    negative_emb: torch.Tensor

    # Losses
    triplet_loss: torch.Tensor
    total_loss: torch.Tensor

    # Optional auxiliary outputs (anchor only, or can be extended)
    metabolite_profile: Optional[torch.Tensor] = None
    gene_expression: Optional[torch.Tensor] = None
    growth_rate: Optional[torch.Tensor] = None

    # Breakdown for logging
    loss_components: dict = field(default_factory=dict)


class GNNGLMSystem(nn.Module):
    """
    Full triplet-training system.

    Forward pass expects a triplet of PyG Batch objects:
        (anchor_batch, positive_batch, negative_batch)

    Returns ModelOutputs with all embeddings, predictions, and losses.
    """

    def __init__(self, cfg: GNNGLMConfig):
        super().__init__()
        self.cfg = cfg

        # Shared encoder (weights tied across anchor / positive / negative)
        self.encoder = GNNGLMEncoder(cfg)

        # Optional auxiliary heads (applied to anchor embedding)
        self.metabolite_head: Optional[nn.Module] = None
        self.gene_expression_head: Optional[nn.Module] = None
        if cfg.use_metabolite_head:
            self.metabolite_head = AuxiliaryHead(
                in_dim=cfg.embedding_dim,
                hidden_dim=cfg.embedding_dim * 2,
                out_dim=cfg.metabolite_output_dim,
            )
        if cfg.use_gene_expression_head:
            self.gene_expression_head = AuxiliaryHead(
                in_dim=cfg.embedding_dim,
                hidden_dim=cfg.embedding_dim * 2,
                out_dim=cfg.gene_expression_output_dim,
            )

        # Growth prediction head (anchor + positive pair)
        self.growth_head: Optional[nn.Module] = None
        if cfg.use_growth_head:
            self.growth_head = GrowthHead(
                emb_dim=cfg.embedding_dim,
                hidden_dim=cfg.embedding_dim * 2,
            )

        # Loss functions
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=cfg.triplet_margin, p=2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        anchor: Data,
        positive: Data,
        negative: Data,
        metabolite_labels: Optional[torch.Tensor] = None,
        gene_expression_labels: Optional[torch.Tensor] = None,
        growth_labels: Optional[torch.Tensor] = None,
    ) -> ModelOutputs:

        # 1. Encode all three GEMs with the shared encoder
        anchor_emb = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)

        # 2. Triplet loss: pull anchor↔positive together, push anchor↔negative apart
        triplet_loss = self.triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
        total_loss = triplet_loss
        loss_components = {"triplet": triplet_loss.item()}

        # 3. Auxiliary: metabolite profile (anchor)
        metabolite_profile = None
        if self.metabolite_head is not None:
            metabolite_profile = self.metabolite_head(anchor_emb)
            if metabolite_labels is not None:
                met_loss = F.mse_loss(metabolite_profile, metabolite_labels)
                total_loss = total_loss + self.cfg.lambda_aux * met_loss
                loss_components["metabolite_aux"] = met_loss.item()

        # 4. Auxiliary: gene expression (anchor)
        gene_expression = None
        if self.gene_expression_head is not None:
            gene_expression = self.gene_expression_head(anchor_emb)
            if gene_expression_labels is not None:
                gene_loss = F.mse_loss(gene_expression, gene_expression_labels)
                total_loss = total_loss + self.cfg.lambda_aux * gene_loss
                loss_components["gene_expression_aux"] = gene_loss.item()

        # 5. Community growth rate (anchor + positive pair)
        growth_rate = None
        if self.growth_head is not None:
            growth_rate = self.growth_head(anchor_emb, positive_emb)
            if growth_labels is not None:
                growth_loss = F.mse_loss(growth_rate, growth_labels)
                total_loss = total_loss + growth_loss
                loss_components["growth"] = growth_loss.item()

        loss_components["total"] = total_loss.item()

        return ModelOutputs(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            negative_emb=negative_emb,
            triplet_loss=triplet_loss,
            total_loss=total_loss,
            metabolite_profile=metabolite_profile,
            gene_expression=gene_expression,
            growth_rate=growth_rate,
            loss_components=loss_components,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, gem: Data) -> torch.Tensor:
        """Encode a single GEM (or batch) to its embedding."""
        self.eval()
        return self.encoder(gem)

    @torch.no_grad()
    def predict_growth(self, gem_a: Data, gem_b: Data) -> torch.Tensor:
        """Predict community growth rate for a pair of GEMs."""
        assert self.growth_head is not None, "Growth head not enabled in config."
        self.eval()
        emb_a = self.encoder(gem_a)
        emb_b = self.encoder(gem_b)
        return self.growth_head(emb_a, emb_b)
