"""
ablation_study.py

Runs 5 sequential training runs where 20% of the training data is removed
at each step. For each run:
    1. Subsample the training data
    2. Train the model
    3. Evaluate on the same fixed holdout set
    4. Run visualize_individual.py to generate figures
    5. Save all results to a dedicated folder

Data fractions per run:
    Run 1: 100% of training data
    Run 2:  80% of training data
    Run 3:  60% of training data
    Run 4:  40% of training data
    Run 5:  20% of training data

The holdout (val) set is fixed across all runs so results are comparable.

Outputs:
    ablation/
        run_1_100pct/   -- checkpoints, figures, history, metrics
        run_2_80pct/
        run_3_60pct/
        run_4_40pct/
        run_5_20pct/
        ablation_summary.csv  -- metrics across all runs for comparison
"""

import json
import random
import time
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Batch

from config_loader import load_settings, build_individual_config, build_label_map
from data_individual import IndividualTripletDataset, make_individual_dataloader
from models_individual import IndividualGNNSystem
from data import GEMGraph

import cobra

# ---------------------------------------------------------------------------
# Load settings
# ---------------------------------------------------------------------------

S = load_settings()
cfg = build_individual_config(S)
LABEL_TO_INT, INT_TO_LABEL, N_LABELS = build_label_map(S)

# ---------------------------------------------------------------------------
# Ablation CONFIG
# ---------------------------------------------------------------------------

ABLATION_DIR    = "ablation"
N_RUNS          = 5
REDUCTION       = 0.20      # fraction removed per run
BASE_FRACTION   = 1.00      # start at 100%

BG   = "#FAFAF8"
GRID = "#E8E8E4"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "axes.labelweight":  "bold",
})

# ---------------------------------------------------------------------------
# Load full train and fixed val datasets once
# ---------------------------------------------------------------------------

print("Loading datasets...")
full_train = json.load(open(S.paths.triplets_individual_train))
val_data   = json.load(open(S.paths.triplets_individual_val))

print(f"  Full training triplets : {len(full_train)}")
print(f"  Fixed val triplets     : {len(val_data)}")

val_dataset = IndividualTripletDataset(
    val_data, node_feature_dim=cfg.node_feature_dim
)
val_loader = make_individual_dataloader(
    val_dataset, batch_size=S.training.batch_size, shuffle=False
)

# Fixed val pairs JSON for visualize
val_pairs = [
    {
        "anchor_id":          t.get("anchor_id"),
        "positive_id":        t.get("positive_id"),
        "negative_id":        t.get("negative_id"),
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
    for t in val_data
]

# ---------------------------------------------------------------------------
# GEM loader for evaluation
# ---------------------------------------------------------------------------

gem_dir   = Path(S.paths.gem_dir)
strains   = pd.read_csv(S.paths.strains_csv)
strain_lookup = dict(zip(strains["ID"], strains["Strain"]))
builder   = GEMGraph(node_feature_dim=cfg.node_feature_dim)
gem_cache = {}

def load_gem(strain_id):
    if strain_id not in gem_cache:
        path = gem_dir / strain_lookup[strain_id]
        m = cobra.io.read_sbml_model(str(path))
        gem_cache[strain_id] = builder.from_cobra(m)
    return gem_cache[strain_id]

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_model(train_triplets: list, run_dir: Path, run_label: str):
    """Train a model on the given triplets and return metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build fresh model for each run
    model = IndividualGNNSystem(cfg)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=S.training.lr,
        weight_decay=S.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=S.training.n_epochs, eta_min=1e-6
    )

    train_dataset = IndividualTripletDataset(
        train_triplets, node_feature_dim=cfg.node_feature_dim
    )
    train_loader = make_individual_dataloader(
        train_dataset, batch_size=S.training.batch_size, shuffle=True
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_triplet_loss": [], "val_triplet_loss": [],
        "train_delta_loss": [], "val_delta_loss": [],
        "train_label_loss": [], "val_label_loss": [],
    }
    best_val_loss = float("inf")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

                if pos_delta is not None:     pos_delta     = pos_delta.to(device)
                if neg_delta is not None:     neg_delta     = neg_delta.to(device)
                if pos_label_int is not None: pos_label_int = pos_label_int.squeeze(-1).to(device)
                if neg_label_int is not None: neg_label_int = neg_label_int.squeeze(-1).to(device)

                outputs = model(
                    anchor=anchor, positive=positive, negative=negative,
                    positive_delta=pos_delta, negative_delta=neg_delta,
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

    print(f"\n  Training on {device} | {len(train_triplets)} triplets | {S.training.n_epochs} epochs")

    for epoch in range(1, S.training.n_epochs + 1):
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

        if epoch % S.training.log_every == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:>4d}/{S.training.n_epochs} | "
                f"train={train_stats.get('total',0):.4f}  "
                f"val={val_stats.get('total',0):.4f}  "
                f"triplet={train_stats.get('triplet',0):.4f}  "
                f"label={train_stats.get('label',0):.4f}"
            )

        val_loss = val_stats.get("total", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss":    best_val_loss,
                "cfg":              cfg,
            }, ckpt_dir / "best.pt")

    # Save history
    with open(run_dir / "training_history_individual.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save val pairs for visualize
    with open(run_dir / "val_pairs_individual.json", "w") as f:
        json.dump(val_pairs, f, indent=2)

    print(f"  Best val loss: {best_val_loss:.4f}")
    return model, history, best_val_loss


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_run(model, device) -> dict:
    """Compute triplet accuracy and label classification accuracy on val set."""
    model.eval()

    triplet_correct, triplet_total = 0, 0
    label_correct,   label_total   = 0, 0

    for batch in val_loader:
        anchor   = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negative = batch["negative"].to(device)

        anchor_emb   = model.encode(anchor)
        positive_emb = model.encode(positive)
        negative_emb = model.encode(negative)

        # Triplet accuracy
        pos_sim = (anchor_emb * positive_emb).sum(dim=-1)
        neg_sim = (anchor_emb * negative_emb).sum(dim=-1)
        triplet_correct += (pos_sim > neg_sim).sum().item()
        triplet_total   += anchor_emb.size(0)

        # Label accuracy
        if model.label_head is not None and "positive_label_int" in batch:
            pos_label = batch["positive_label_int"].squeeze(-1).to(anchor.x.device)
            neg_label = batch["negative_label_int"].squeeze(-1).to(anchor.x.device)

            pos_logits = model.label_head(anchor_emb, positive_emb)
            neg_logits = model.label_head(anchor_emb, negative_emb)

            label_correct += (pos_logits.argmax(dim=-1) == pos_label).sum().item()
            label_correct += (neg_logits.argmax(dim=-1) == neg_label).sum().item()
            label_total   += pos_label.size(0) * 2

    return {
        "triplet_accuracy": triplet_correct / max(triplet_total, 1),
        "label_accuracy":   label_correct   / max(label_total,   1),
    }


# ---------------------------------------------------------------------------
# Plot: loss curves for a single run
# ---------------------------------------------------------------------------

def plot_run_losses(history: dict, run_dir: Path, run_label: str):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], color="#4A7C8E", linewidth=2,
            label="Train loss")
    ax.plot(epochs, history["val_loss"],   color="#E07B4F", linewidth=2,
            linestyle="--", label="Val loss")

    if any(v > 0 for v in history.get("train_label_loss", [])):
        ax.plot(epochs, history["train_label_loss"], color="#5B8C5A",
                linewidth=1.2, linestyle=":", label="Train label loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training & Validation Loss\n{run_label}",
                 fontweight="bold", pad=12)
    ax.legend(framealpha=0.6, fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "01_loss_curves.png",
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: ablation summary across all runs
# ---------------------------------------------------------------------------

def plot_ablation_summary(summary_df: pd.DataFrame, ablation_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)

    metrics = [
        ("best_val_loss",    "Best Validation Loss",       "#C94040"),
        ("triplet_accuracy", "Triplet Accuracy",           "#4A7C8E"),
        ("label_accuracy",   "Label Classification Accuracy", "#5B8C5A"),
    ]

    for ax, (metric, title, color) in zip(axes, metrics):
        ax.plot(
            summary_df["pct_data"], summary_df[metric],
            color=color, linewidth=2.5, marker="o",
            markersize=9, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=color,
            zorder=3,
        )
        for _, row in summary_df.iterrows():
            ax.annotate(
                f"{row[metric]:.3f}",
                (row["pct_data"], row[metric]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9,
            )
        ax.set_xlabel("Training data fraction (%)")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xticks(summary_df["pct_data"])

    fig.suptitle("Ablation Study — Effect of Training Data Size",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(ablation_dir / "ablation_summary.png",
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved ablation summary plot to {ablation_dir / 'ablation_summary.png'}")


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

Path(ABLATION_DIR).mkdir(exist_ok=True)
random.seed(S.training.random_seed)

# Shuffle training data once with fixed seed so subsampling is deterministic
shuffled_train = full_train.copy()
random.shuffle(shuffled_train)

summary_records = []

for run_idx in range(N_RUNS):
    fraction     = round(BASE_FRACTION - run_idx * REDUCTION, 2)
    pct_label    = int(fraction * 100)
    run_label    = f"Run {run_idx + 1} ({pct_label}% training data)"
    run_dir_name = f"run_{run_idx + 1}_{pct_label}pct"
    run_dir      = Path(ABLATION_DIR) / run_dir_name

    print(f"\n{'='*60}")
    print(f"  {run_label}")
    print(f"{'='*60}")

    # Create run directory structure
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Subsample training data
    n_train = max(1, int(len(shuffled_train) * fraction))
    train_subset = shuffled_train[:n_train]
    print(f"  Training triplets this run: {n_train} / {len(shuffled_train)}")

    # Save the subset used for this run
    with open(run_dir / "train_subset.json", "w") as f:
        json.dump(train_subset, f, indent=2)

    # Train
    t_start = time.time()
    model, history, best_val_loss = train_model(
        train_triplets=train_subset,
        run_dir=run_dir,
        run_label=run_label,
    )
    elapsed = time.time() - t_start

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_run(model, device)
    print(f"  Triplet accuracy : {metrics['triplet_accuracy']:.3f}")
    print(f"  Label accuracy   : {metrics['label_accuracy']:.3f}")
    print(f"  Time elapsed     : {elapsed/60:.1f} min")

    # Save metrics
    run_metrics = {
        "run":              run_idx + 1,
        "pct_data":         pct_label,
        "n_train":          n_train,
        "best_val_loss":    best_val_loss,
        "triplet_accuracy": metrics["triplet_accuracy"],
        "label_accuracy":   metrics["label_accuracy"],
        "elapsed_min":      round(elapsed / 60, 2),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(run_metrics, f, indent=2)
    summary_records.append(run_metrics)

    # Plot loss curves for this run
    plot_run_losses(history, run_dir, run_label)
    print(f"  Saved loss curve to {run_dir}/figures/01_loss_curves.png")

    # Copy settings used for reproducibility
    shutil.copy("settings.yaml", run_dir / "settings.yaml")

# ---------------------------------------------------------------------------
# Save and plot summary
# ---------------------------------------------------------------------------

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(f"{ABLATION_DIR}/ablation_summary.csv", index=False)

print(f"\n{'='*60}")
print("ABLATION STUDY COMPLETE")
print(f"{'='*60}")
print(summary_df[["run", "pct_data", "n_train", "best_val_loss",
                   "triplet_accuracy", "label_accuracy"]].to_string(index=False))

plot_ablation_summary(summary_df, Path(ABLATION_DIR))
print(f"\nAll results saved to {ABLATION_DIR}/")
