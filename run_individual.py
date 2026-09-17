"""
run_individual.py

Training script for individual strain contrastive learning.
Supports two implementations:

    Implementation 1 (LABEL_MODE = False):
        Trains on continuous delta values (co-culture - monoculture).
        L_total = L_triplet + λ_delta * L_delta + λ_label * L_label

    Implementation 2 (LABEL_MODE = True):
        Trains on categorical labels directly.
        Uses label-aware triplet loss where the margin is scaled
        by the ordinal distance between positive and negative labels.
        L_total = L_label_triplet + λ_label * L_label + λ_delta * L_delta
"""

import json
import torch
import random
from torch_geometric.data import Batch

from data_individual import IndividualTripletDataset, make_individual_dataloader
from models_individual import IndividualGNNConfig, IndividualGNNSystem, INT_TO_LABEL
from train import Trainer
from config_loader import load_settings, build_individual_config, build_pairwise_config
S = load_settings()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TRIPLETS_JSON  = "triplets_individual.json"
CHECKPOINT_DIR = "checkpoints_individual"
HISTORY_OUT    = "training_history_individual.json"
VAL_PAIRS_OUT  = "val_pairs_individual.json"

# Switch between implementation 1 (False) and implementation 2 (True)
LABEL_MODE     =  S.model.label_mode

N_EPOCHS     = S.training.n_epochs
BATCH_SIZE   = S.training.batch_size
TRAIN_SPLIT    = S.training.train_split
RANDOM_SEED    = S.training.random_seed

cfg = IndividualGNNConfig(
    node_feature_dim=S.model.node_feature_dim,
    gnn_hidden_dim=S.model.gnn_hidden_dim,
    gnn_num_layers=S.model.gnn_num_layers,
    gnn_heads=S.model.gnn_heads,
    gnn_dropout=S.model.gnn_dropout,
    embedding_dim=S.model.embedding_dim,
    projection_hidden_dim=S.model.projection_hidden_dim,
    use_metabolite_head=S.model.use_metabolite_head,
    use_gene_expression_head=S.model.use_gene_expression_head,
    use_growth_head=S.model.use_growth_head,
    # Individual strain heads
    use_delta_head=S.model.use_delta_head,
    use_label_head=S.model.use_label_head,
    lambda_delta=S.model.lambda_delta,
    lambda_label=S.model.lambda_label,
    triplet_margin=S.model.triplet_margin,
    label_mode=LABEL_MODE,
)

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

TRAIN_JSON   = S.paths.triplets_individual_train
VAL_JSON     = S.paths.triplets_individual_val

print(f"Loading train dataset from {TRAIN_JSON}...")
train_dataset = IndividualTripletDataset.from_json(
    TRAIN_JSON, node_feature_dim=cfg.node_feature_dim
)
print(f"Loading val dataset from {VAL_JSON}...")
val_dataset = IndividualTripletDataset.from_json(
    VAL_JSON, node_feature_dim=cfg.node_feature_dim
)
print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

train_loader = make_individual_dataloader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = make_individual_dataloader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# ---------------------------------------------------------------------------
# Save val pairs
# ---------------------------------------------------------------------------

val_pairs = [
    {
        "anchor_id":          t["anchor_id"],
        "positive_id":        t["positive_id"],
        "negative_id":        t["negative_id"],
        "anchor_monoculture": t.get("anchor_monoculture"),
        "positive_delta":     t.get("positive_delta"),
        "negative_delta":     t.get("negative_delta"),
        "positive_label":     t.get("positive_label"),
        "negative_label":     t.get("negative_label"),
        "positive_label_int": t.get("positive_label_int"),
        "negative_label_int": t.get("negative_label_int"),
        "positive_growth":    t.get("positive_growth"),
        "negative_growth":    t.get("negative_growth"),
        "growth_label":       t.get("growth_label"),
    }
    for t in val_dataset.triplets
]
with open(VAL_PAIRS_OUT, "w") as f:
    json.dump(val_pairs, f, indent=2)
print(f"Saved {len(val_pairs)} val pairs to {VAL_PAIRS_OUT}")

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

model = IndividualGNNSystem(cfg)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
print(f"Training mode: {'Label mode (Implementation 2)' if LABEL_MODE else 'Delta mode (Implementation 1)'}")

# ---------------------------------------------------------------------------
# Custom training loop (extends Trainer to pass individual fields)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=S.training.lr,
    weight_decay=S.training.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-6
)

import time
history = {
    "train_loss": [], "val_loss": [],
    "train_triplet_loss": [], "val_triplet_loss": [],
    "train_delta_loss": [], "val_delta_loss": [],
    "train_label_loss": [], "val_label_loss": [],
}
best_val_loss = float("inf")

import torch.nn as nn


def run_epoch(loader, train: bool) -> dict:
    model.train(train)
    totals = {}
    n = 0

    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for batch in loader:
            anchor   = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            pos_delta     = batch.get("positive_delta")
            neg_delta     = batch.get("negative_delta")
            pos_label_int = batch.get("positive_label_int")
            neg_label_int = batch.get("negative_label_int")

            if pos_delta is not None:
                pos_delta = pos_delta.to(device)
            if neg_delta is not None:
                neg_delta = neg_delta.to(device)
            if pos_label_int is not None:
                pos_label_int = pos_label_int.squeeze(-1).to(device)
            if neg_label_int is not None:
                neg_label_int = neg_label_int.squeeze(-1).to(device)

            outputs = model(
                anchor=anchor,
                positive=positive,
                negative=negative,
                positive_delta=pos_delta,
                negative_delta=neg_delta,
                positive_label_int=pos_label_int,
                negative_label_int=neg_label_int,
            )

            if train:
                optimizer.zero_grad()
                outputs.total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k, v in outputs.loss_components.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

    return {k: v / n for k, v in totals.items()}


print(f"\nTraining on {device} for {N_EPOCHS} epochs")
import pathlib
pathlib.Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    train_stats = run_epoch(train_loader, train=True)
    val_stats   = run_epoch(val_loader,   train=False)
    scheduler.step()

    history["train_loss"].append(train_stats.get("total", 0))
    history["val_loss"].append(val_stats.get("total", 0))
    history["train_triplet_loss"].append(train_stats.get("triplet", 0))
    history["val_triplet_loss"].append(val_stats.get("triplet", 0))
    history["train_delta_loss"].append(train_stats.get("delta", 0))
    history["val_delta_loss"].append(val_stats.get("delta", 0))
    history["train_label_loss"].append(train_stats.get("label", 0))
    history["val_label_loss"].append(val_stats.get("label", 0))

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>4d}/{N_EPOCHS} | "
            f"train={train_stats.get('total',0):.4f}  "
            f"val={val_stats.get('total',0):.4f}  "
            f"triplet={train_stats.get('triplet',0):.4f}  "
            f"delta={train_stats.get('delta',0):.4f}  "
            f"label={train_stats.get('label',0):.4f}  "
            f"[{elapsed:.1f}s]"
        )

    val_loss = val_stats.get("total", float("inf"))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "cfg": cfg,
            "label_mode": LABEL_MODE,
        }, f"{CHECKPOINT_DIR}/best.pt")

torch.save({
    "model_state_dict": model.state_dict(),
    "best_val_loss": best_val_loss,
    "cfg": cfg,
}, f"{CHECKPOINT_DIR}/last.pt")

with open(HISTORY_OUT, "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
print(f"Saved checkpoint to {CHECKPOINT_DIR}/best.pt")
print(f"Saved history to {HISTORY_OUT}")

# ---------------------------------------------------------------------------
# Quick evaluation
# ---------------------------------------------------------------------------

print("\nEvaluating label classification accuracy on val set...")
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for batch in val_loader:
        anchor   = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        neg_gem  = batch["negative"].to(device)

        if "positive_label_int" not in batch:
            break

        pos_label = batch["positive_label_int"].squeeze(-1).to(device)
        neg_label = batch["negative_label_int"].squeeze(-1).to(device)

        anchor_emb   = model.encode(anchor)
        positive_emb = model.encode(positive)
        negative_emb = model.encode(neg_gem)

        pos_logits = model.label_head(anchor_emb, positive_emb)
        neg_logits = model.label_head(anchor_emb, negative_emb)

        correct += (pos_logits.argmax(dim=-1) == pos_label).sum().item()
        correct += (neg_logits.argmax(dim=-1) == neg_label).sum().item()
        total   += pos_label.size(0) * 2

if total > 0:
    print(f"  Label classification accuracy: {correct / total:.3f}  ({correct}/{total})")
