"""
models_individual.py

Extended GNN-GLM system for individual strain contrastive learning.
Adds two new prediction heads on top of the shared encoder:

1. DeltaHead      — predicts the continuous delta (co-culture growth of
                    anchor minus its monoculture baseline)
2. LabelHead      — classifies the delta into one of 5 categorical labels:
                    very_bad / bad / neutral / good / very_good
                    (for implementation 2: training on labels directly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from dataclasses import dataclass, field
from typing import Optional

from models import GNNGLMEncoder, GNNGLMConfig, AuxiliaryHead, ModelOutputs


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
"""
LABEL_TO_INT = {
    "very_bad":  0,
    "bad":       1,
    "neutral":   2,
    "good":      3,
    "very_good": 4,
}
"""
LABEL_TO_INT = {
    "worse than -0.5": 0,
    "-0.4:-0.5": 1,
    "-0.3:-0.4": 2,
    "-0.2:-0.3": 3,
    "-0.1:-0.2": 4,
    "0:-0.1": 5,
    "positive": 6

}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
N_LABELS = len(LABEL_TO_INT)


# ---------------------------------------------------------------------------
# Extended config
# ---------------------------------------------------------------------------

@dataclass
class IndividualGNNConfig(GNNGLMConfig):
    # Delta regression head
    use_delta_head: bool = True

    # Label classification head (implementation 2)
    use_label_head: bool = True

    # Weight for delta regression loss
    lambda_delta: float = 1.0

    # Weight for label classification loss
    lambda_label: float = 1.0

    # Whether to train on labels instead of raw deltas (implementation 2)
    # When True: triplet loss uses label distance, label head is primary
    # When False: triplet loss uses delta values, delta head is primary
    label_mode: bool = False


# ---------------------------------------------------------------------------
# Delta prediction head
# Predicts continuous delta = anchor_coculture - anchor_monoculture
# Takes: [anchor_emb, partner_emb] as input
# ---------------------------------------------------------------------------

class DeltaHead(nn.Module):
    """
    Predicts how much better/worse an anchor grows in co-culture
    with a specific partner, relative to its monoculture baseline.
    Output is a scalar delta value (can be positive or negative).
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            # No activation: delta can be positive or negative
        )

    def forward(self, anchor_emb: torch.Tensor, partner_emb: torch.Tensor) -> torch.Tensor:
        interaction = torch.cat([
            anchor_emb,
            partner_emb,
            anchor_emb * partner_emb,
            (anchor_emb - partner_emb).abs(),
        ], dim=-1)
        return self.net(interaction)


# ---------------------------------------------------------------------------
# Label classification head
# Classifies delta into 5 categories: very_bad/bad/neutral/good/very_good
# ---------------------------------------------------------------------------

class LabelHead(nn.Module):
    """
    Classifies the growth interaction label between anchor and partner
    into one of 5 ordinal categories.
    Uses ordinal encoding to respect the natural ordering of labels.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256,
                 n_classes: int = N_LABELS, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_classes),
            # No softmax here — applied in loss function
        )

    def forward(self, anchor_emb: torch.Tensor, partner_emb: torch.Tensor) -> torch.Tensor:
        interaction = torch.cat([
            anchor_emb,
            partner_emb,
            anchor_emb * partner_emb,
            (anchor_emb - partner_emb).abs(),
        ], dim=-1)
        return self.net(interaction)   # logits [B, N_LABELS]

    def predict(self, anchor_emb: torch.Tensor, partner_emb: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices [B]."""
        logits = self.forward(anchor_emb, partner_emb)
        return logits.argmax(dim=-1)

    def predict_label(self, anchor_emb: torch.Tensor,
                      partner_emb: torch.Tensor) -> list[str]:
        """Returns predicted label strings."""
        indices = self.predict(anchor_emb, partner_emb)
        return [INT_TO_LABEL[i.item()] for i in indices]


# ---------------------------------------------------------------------------
# Label-aware triplet loss
# When label_mode=True, uses label distance as the similarity metric
# instead of embedding distance, so the model learns ordinal label structure
# ---------------------------------------------------------------------------

class LabelTripletLoss(nn.Module):
    """
    Triplet loss that incorporates label distance.
    Instead of using raw embedding distances, it weights the margin
    by the difference in label integers between positive and negative.
    A larger label gap → larger required margin → stronger push/pull.
    """

    def __init__(self, base_margin: float = 0.5):
        super().__init__()
        self.base_margin = base_margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_label_int: Optional[torch.Tensor] = None,
        negative_label_int: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)

        if positive_label_int is not None and negative_label_int is not None:
            # Scale margin by label gap: larger gap = stricter separation required
            label_gap = (negative_label_int - positive_label_int).float().abs()
            margin = self.base_margin * (1.0 + label_gap / (N_LABELS - 1))
        else:
            margin = self.base_margin

        loss = F.relu(d_pos - d_neg + margin)
        return loss.mean()


# ---------------------------------------------------------------------------
# Extended model outputs
# ---------------------------------------------------------------------------

@dataclass
class IndividualModelOutputs:
    anchor_emb:   torch.Tensor
    positive_emb: torch.Tensor
    negative_emb: torch.Tensor

    triplet_loss: torch.Tensor
    total_loss:   torch.Tensor

    # Delta predictions (continuous)
    positive_delta_pred: Optional[torch.Tensor] = None
    negative_delta_pred: Optional[torch.Tensor] = None

    # Label predictions (classification logits)
    positive_label_logits: Optional[torch.Tensor] = None
    negative_label_logits: Optional[torch.Tensor] = None

    loss_components: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Full individual strain GNN-GLM system
# ---------------------------------------------------------------------------

class IndividualGNNSystem(nn.Module):
    """
    Extended GNN-GLM for individual strain contrastive learning.

    Two training modes controlled by cfg.label_mode:

    Implementation 1 (label_mode=False):
        - Standard triplet loss on embeddings
        - DeltaHead supervised on continuous delta values
        - LabelHead supervised on categorical labels
        - L_total = L_triplet + λ_delta * L_delta + λ_label * L_label

    Implementation 2 (label_mode=True):
        - Label-aware triplet loss (margin scaled by label distance)
        - LabelHead is primary supervised signal
        - DeltaHead optional secondary signal
        - L_total = L_label_triplet + λ_label * L_label + λ_delta * L_delta
    """

    def __init__(self, cfg: IndividualGNNConfig):
        super().__init__()
        self.cfg = cfg

        # Shared encoder (same architecture as original system)
        self.encoder = GNNGLMEncoder(cfg)

        # Delta regression head
        self.delta_head: Optional[nn.Module] = None
        if cfg.use_delta_head:
            self.delta_head = DeltaHead(
                emb_dim=cfg.embedding_dim,
                hidden_dim=cfg.embedding_dim * 2,
                dropout=cfg.gnn_dropout,
            )

        # Label classification head
        self.label_head: Optional[nn.Module] = None
        if cfg.use_label_head:
            self.label_head = LabelHead(
                emb_dim=cfg.embedding_dim,
                hidden_dim=cfg.embedding_dim * 2,
                n_classes=N_LABELS,
                dropout=cfg.gnn_dropout,
            )

        # Loss functions
        if cfg.label_mode:
            self.triplet_loss_fn = LabelTripletLoss(base_margin=cfg.triplet_margin)
        else:
            self.triplet_loss_fn = nn.TripletMarginLoss(
                margin=cfg.triplet_margin, p=2
            )

        self.label_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        anchor:   Data,
        positive: Data,
        negative: Data,
        # Continuous delta labels
        positive_delta: Optional[torch.Tensor] = None,
        negative_delta: Optional[torch.Tensor] = None,
        # Categorical label integers
        positive_label_int: Optional[torch.Tensor] = None,
        negative_label_int: Optional[torch.Tensor] = None,
    ) -> IndividualModelOutputs:

        # 1. Encode all three GEMs
        anchor_emb   = self.encoder(anchor)
        positive_emb = self.encoder(positive)
        negative_emb = self.encoder(negative)

        # 2. Triplet loss
        if self.cfg.label_mode and positive_label_int is not None:
            triplet_loss = self.triplet_loss_fn(
                anchor_emb, positive_emb, negative_emb,
                positive_label_int, negative_label_int
            )
        else:
            triplet_loss = self.triplet_loss_fn(
                anchor_emb, positive_emb, negative_emb
            )

        total_loss = triplet_loss
        loss_components = {"triplet": triplet_loss.item()}

        # 3. Delta regression (anchor + positive, anchor + negative)
        pos_delta_pred, neg_delta_pred = None, None
        if self.delta_head is not None:
            pos_delta_pred = self.delta_head(anchor_emb, positive_emb)
            neg_delta_pred = self.delta_head(anchor_emb, negative_emb)

            if positive_delta is not None and negative_delta is not None:
                delta_loss = (
                    F.mse_loss(pos_delta_pred, positive_delta) +
                    F.mse_loss(neg_delta_pred, negative_delta)
                ) / 2
                total_loss = total_loss + self.cfg.lambda_delta * delta_loss
                loss_components["delta"] = delta_loss.item()

        # 4. Label classification (anchor + positive, anchor + negative)
        pos_label_logits, neg_label_logits = None, None
        if self.label_head is not None:
            pos_label_logits = self.label_head(anchor_emb, positive_emb)
            neg_label_logits = self.label_head(anchor_emb, negative_emb)

            if positive_label_int is not None and negative_label_int is not None:
                label_loss = (
                    self.label_loss_fn(pos_label_logits, positive_label_int) +
                    self.label_loss_fn(neg_label_logits, negative_label_int)
                ) / 2
                total_loss = total_loss + self.cfg.lambda_label * label_loss
                loss_components["label"] = label_loss.item()

        loss_components["total"] = total_loss.item()

        return IndividualModelOutputs(
            anchor_emb=anchor_emb,
            positive_emb=positive_emb,
            negative_emb=negative_emb,
            triplet_loss=triplet_loss,
            total_loss=total_loss,
            positive_delta_pred=pos_delta_pred,
            negative_delta_pred=neg_delta_pred,
            positive_label_logits=pos_label_logits,
            negative_label_logits=neg_label_logits,
            loss_components=loss_components,
        )

    @torch.no_grad()
    def encode(self, gem: Data) -> torch.Tensor:
        self.eval()
        return self.encoder(gem)

    @torch.no_grad()
    def predict_delta(self, anchor_gem: Data,
                      partner_gem: Data) -> torch.Tensor:
        """Predict continuous delta for anchor + partner."""
        assert self.delta_head is not None, "Delta head not enabled."
        self.eval()
        emb_a = self.encoder(anchor_gem)
        emb_b = self.encoder(partner_gem)
        return self.delta_head(emb_a, emb_b)

    @torch.no_grad()
    def predict_label(self, anchor_gem: Data,
                      partner_gem: Data) -> list[str]:
        """Predict categorical label for anchor + partner."""
        assert self.label_head is not None, "Label head not enabled."
        self.eval()
        emb_a = self.encoder(anchor_gem)
        emb_b = self.encoder(partner_gem)
        return self.label_head.predict_label(emb_a, emb_b)
