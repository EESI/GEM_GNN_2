"""
Training and evaluation loop for the GNN-GLM triplet system.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import GNNGLMConfig, GNNGLMSystem, ModelOutputs
from data import GEMTripletDataset, make_dataloader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Handles training, validation, checkpointing, and basic logging.

    Usage:
        cfg = GNNGLMConfig()
        model = GNNGLMSystem(cfg)
        trainer = Trainer(model, train_loader, val_loader)
        trainer.train(n_epochs=50)
    """

    def __init__(
        self,
        model: GNNGLMSystem,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints_new",
        log_every: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_triplet_loss": [], "val_triplet_loss": [],
        }
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: int = 50) -> dict:
        print(f"Training on {self.device} for {n_epochs} epochs")
        print(f"  Train batches / epoch: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val   batches / epoch: {len(self.val_loader)}")
        print()

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_stats = self._run_epoch(self.train_loader, train=True)
            val_stats = {}
            if self.val_loader:
                val_stats = self._run_epoch(self.val_loader, train=False)

            self.scheduler.step()

            # Logging
            self.history["train_loss"].append(train_stats["total"])
            self.history["train_triplet_loss"].append(train_stats["triplet"])
            if val_stats:
                self.history["val_loss"].append(val_stats["total"])
                self.history["val_triplet_loss"].append(val_stats["triplet"])

            if epoch % self.log_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                msg = (
                    f"Epoch {epoch:>4d}/{n_epochs} | "
                    f"train_loss={train_stats['total']:.4f}  "
                    f"triplet={train_stats['triplet']:.4f}"
                )
                if val_stats:
                    msg += f"  |  val_loss={val_stats['total']:.4f}"
                msg += f"  [{elapsed:.1f}s]"
                print(msg)

            # Checkpoint best model
            monitor = val_stats.get("total", train_stats["total"])
            if monitor < self.best_val_loss:
                self.best_val_loss = monitor
                self._save_checkpoint("best.pt")

        self._save_checkpoint("last.pt")
        print(f"\nTraining complete. Best loss: {self.best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        ctx = torch.enable_grad if train else torch.no_grad

        totals: dict[str, float] = {}
        n_batches = 0

        with ctx():
            for batch in loader:
                anchor = batch["anchor"].to(self.device)
                positive = batch["positive"].to(self.device)
                negative = batch["negative"].to(self.device)

                growth_labels = batch.get("growth_label")
                met_labels = batch.get("metabolite_labels")
                gene_labels = batch.get("gene_expression_labels")

                if growth_labels is not None:
                    growth_labels = growth_labels.to(self.device)
                if met_labels is not None:
                    met_labels = met_labels.to(self.device)
                if gene_labels is not None:
                    gene_labels = gene_labels.to(self.device)

                outputs: ModelOutputs = self.model(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    metabolite_labels=met_labels,
                    gene_expression_labels=gene_labels,
                    growth_labels=growth_labels,
                )

                if train:
                    self.optimizer.zero_grad()
                    outputs.total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                for k, v in outputs.loss_components.items():
                    totals[k] = totals.get(k, 0.0) + v
                n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.history = ckpt.get("history", self.history)
        print(f"Loaded checkpoint from {path}")


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_embeddings(
    model: GNNGLMSystem,
    dataset: GEMTripletDataset,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate embedding quality on a dataset.
    Returns:
        - mean cosine similarity for positive pairs
        - mean cosine similarity for negative pairs
        - triplet accuracy (anchor closer to positive than negative)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)

    pos_sims, neg_sims, correct = [], [], []

    for batch in loader:
        anchor_emb = model.encode(batch["anchor"].to(device))
        positive_emb = model.encode(batch["positive"].to(device))
        negative_emb = model.encode(batch["negative"].to(device))

        pos_sim = F_cosine_sim(anchor_emb, positive_emb)
        neg_sim = F_cosine_sim(anchor_emb, negative_emb)

        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)
        correct.append((pos_sim > neg_sim).float())

    pos_sims_t = torch.cat(pos_sims)
    neg_sims_t = torch.cat(neg_sims)
    correct_t = torch.cat(correct)

    return {
        "mean_pos_similarity": pos_sims_t.mean().item(),
        "mean_neg_similarity": neg_sims_t.mean().item(),
        "triplet_accuracy": correct_t.mean().item(),
        "n_triplets": len(correct_t),
    }


def F_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity between two [N, D] tensors."""
    return (a * b).sum(dim=-1)  # embeddings are already L2-normalised


# Re-export for convenience
import torch.nn.functional  # noqa
