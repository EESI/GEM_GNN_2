"""
visualize_individual.py

Generates five separate figures for the individual strain GNN-GLM system:
    1. Training & validation loss curves
    2. Delta vs. embedding cosine similarity (holdout set, OLS regression)
    3. Organism embedding space (UMAP, colored by avg delta)
    4. Delta distribution histogram (holdout set)
    5. Label classification accuracy by category

Requirements:
    pip install matplotlib seaborn umap-learn adjustText scipy torch torch_geometric cobra
"""

import json
import torch
import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress, pearsonr
from torch_geometric.data import Batch

from data import GEMGraph
from models_individual import (
    IndividualGNNConfig,
    IndividualGNNSystem,
    LABEL_TO_INT,
    INT_TO_LABEL,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GEM_DIR          = "/home/eesi/glen/GEM_GNN/SIMBA-GNN/sbml/"
CHECKPOINT       = "checkpoints_individual/best.pt"
PAIRS_CSV        = "/home/eesi/glen/GEM_GNN/SIMBA-GNN/data/pairwise_results.csv"
STRAINS_CSV      = "/home/eesi/glen/GEM_GNN/SIMBA-GNN/data/anaerobic_strains.csv"
MONOCULTURE_JSON = "monoculture_growth.json"
TRAINING_HISTORY = "training_history_individual.json"
VAL_PAIRS_JSON   = "val_pairs_individual.json"
OUTPUT_DIR       = "updated_groupings_label_version"

# Must match your training config exactly
cfg = IndividualGNNConfig(
    node_feature_dim=64,
    gnn_hidden_dim=128,
    gnn_num_layers=6,
    gnn_heads=8,
    gnn_dropout=0.0,
    embedding_dim=256,
    projection_hidden_dim=512,
    use_metabolite_head=False,
    use_gene_expression_head=False,
    use_growth_head=False,
    use_delta_head=True,
    use_label_head=True,
    lambda_delta=1.0,
    lambda_label=1.0,
    triplet_margin=0.5,
    label_mode=False,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

C_TRAIN   = "#4A7C8E"
C_VAL     = "#E07B4F"
C_SCATTER = "#5B8C5A"
C_DIAG    = "#C94040"
C_HIST    = "#7B6FA0"
BG        = "#FAFAF8"
GRID      = "#E8E8E4"
"""
LABEL_COLORS = {
    "very_bad":  "#C94040",
    "bad":       "#E07B4F",
    "neutral":   "#F5C842",
    "good":      "#5B8C5A",
    "very_good": "#2D6A4F",
}
"""
LABEL_COLORS = {
    "worse than -0.5": "#C94040",
    "-0.4:-0.5": "#E07B4F",
    "-0.3:-0.4": "#F5C842",
    "-0.2:-0.3": "#5B8C5A",
    "-0.1:-0.2": "#2D6A4F",
    "0:-0.1": "#F54927",
    "positive": "#39449E",

}

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
# Load model
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = IndividualGNNSystem(cfg)
checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("Model loaded.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

pairs   = pd.read_csv(PAIRS_CSV)
strains = pd.read_csv(STRAINS_CSV)

pairs = pairs[pairs["converged"] == True].copy()
pairs["id1"] = pairs["pairID"].apply(lambda x: int(x.split("__")[0]))
pairs["id2"] = pairs["pairID"].apply(lambda x: int(x.split("__")[1]))
pairs["community_growth"] = pairs["b1 biomass"] + pairs["b2 biomass"]

strain_lookup = dict(zip(strains["ID"], strains["Strain"]))
gem_dir       = Path(GEM_DIR)

with open(MONOCULTURE_JSON) as f:
    monoculture_data = json.load(f)
monoculture_lookup = {int(e["ID"]): e["Optimal Growth Rate"] for e in monoculture_data}

with open(TRAINING_HISTORY) as f:
    history = json.load(f)

with open(VAL_PAIRS_JSON) as f:
    val_pairs_list = json.load(f)

# Build holdout dataframe from val_pairs_individual.json
holdout_records = []
for t in val_pairs_list:
    holdout_records.append({
        "id1":            t["anchor_id"],
        "id2":            t["positive_id"],
        "community_growth": t.get("positive_growth"),
        "delta":          t.get("positive_delta"),
        "label":          t.get("positive_label"),
        "label_int":      t.get("positive_label_int"),
    })
    holdout_records.append({
        "id1":            t["anchor_id"],
        "id2":            t["negative_id"],
        "community_growth": t.get("negative_growth"),
        "delta":          t.get("negative_delta"),
        "label":          t.get("negative_label"),
        "label_int":      t.get("negative_label_int"),
    })

holdout_df = pd.DataFrame(holdout_records).dropna(subset=["delta"])
holdout_df = holdout_df.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)
print(f"Holdout pairs: {len(holdout_df)}")

# ---------------------------------------------------------------------------
# GEM loader + cache
# ---------------------------------------------------------------------------

builder   = GEMGraph(node_feature_dim=cfg.node_feature_dim)
gem_cache = {}

def load_gem(strain_id):
    if strain_id not in gem_cache:
        path = gem_dir / strain_lookup[strain_id]
        m = cobra.io.read_sbml_model(str(path))
        gem_cache[strain_id] = builder.from_cobra(m)
    return gem_cache[strain_id]

# ---------------------------------------------------------------------------
# Plot 1 — Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(ax):
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"],    color=C_TRAIN, linewidth=2,
            label="Train loss")
    ax.plot(epochs, history["val_loss"],      color=C_VAL,   linewidth=2,
            linestyle="--", label="Val loss")

    # Overlay component losses if available
    if any(v > 0 for v in history.get("train_delta_loss", [])):
        ax.plot(epochs, history["train_delta_loss"], color="#7B6FA0",
                linewidth=1.2, linestyle=":", label="Train delta loss")
    if any(v > 0 for v in history.get("train_label_loss", [])):
        ax.plot(epochs, history["train_label_loss"], color="#5B8C5A",
                linewidth=1.2, linestyle=":", label="Train label loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold", pad=12)
    ax.legend(framealpha=0.6, fontsize=9)

# ---------------------------------------------------------------------------
# Plot 2 — Delta vs. cosine similarity (scatter)
# ---------------------------------------------------------------------------

def plot_scatter(ax):
    true_vals, pred_vals = [], []

    for _, row in holdout_df.iterrows():
        try:
            gem_a = Batch.from_data_list([load_gem(row["id1"])]).to(device)
            gem_b = Batch.from_data_list([load_gem(row["id2"])]).to(device)
            with torch.no_grad():
                emb_a = model.encode(gem_a)
                emb_b = model.encode(gem_b)
                sim   = (emb_a * emb_b).sum(dim=-1).item()
            true_vals.append(row["delta"])
            pred_vals.append(sim)
        except Exception as e:
            print(f"  Skipped pair {row['id1']}__{row['id2']}: {e}")
            continue

    if len(true_vals) == 0:
        ax.text(0.5, 0.5, "No holdout pairs could be loaded.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="red")
        ax.set_title("Delta vs. Embedding Similarity", fontweight="bold", pad=12)
        return

    true_arr = np.array(true_vals)
    pred_arr = np.array(pred_vals)

    # Color points by label if available
    if "label" in holdout_df.columns:
        colors = [
            LABEL_COLORS.get(row["label"], "#5B8C5A")
            for _, row in holdout_df.iterrows()
            if not pd.isna(row.get("delta"))
        ]
        ax.scatter(true_arr, pred_arr, c=colors[:len(true_arr)],
                   alpha=0.7, edgecolors="white", linewidths=0.5, s=60, zorder=3)
        # Legend for label colors
        for label, color in LABEL_COLORS.items():
            ax.scatter([], [], color=color, label=label, s=40)
        ax.legend(framealpha=0.6, fontsize=8, title="Label", loc="upper right")
    else:
        ax.scatter(true_arr, pred_arr, color=C_SCATTER, alpha=0.7,
                   edgecolors="white", linewidths=0.5, s=60, zorder=3)

    # OLS regression line
    slope, intercept, r_value, p_value, std_err = linregress(true_arr, pred_arr)
    pearson_r, _ = pearsonr(true_arr, pred_arr)
    x_line = np.linspace(true_arr.min(), true_arr.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color=C_DIAG, linewidth=2.0,
            label=f"OLS fit (slope={slope:.2f}, R\u00b2={r_value**2:.3f})", zorder=4)

    stats_text = (
        f"R\u00b2 = {r_value**2:.3f}\n"
        f"Pearson r = {pearson_r:.3f}\n"
        f"Slope = {slope:.3f}\n"
        f"Intercept = {intercept:.3f}\n"
        f"p = {p_value:.2e}"
    )
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.7, edgecolor=GRID))

    ax.set_xlabel("True delta (co-culture \u2212 monoculture growth)")
    ax.set_ylabel("Embedding cosine similarity")
    ax.set_title("Delta vs. Embedding Similarity\n(Holdout Set, OLS Regression)",
                 fontweight="bold", pad=12)

# ---------------------------------------------------------------------------
# Plot 3 — UMAP of organism embeddings colored by avg delta
# ---------------------------------------------------------------------------

def plot_umap(ax):
    try:
        import umap
    except ImportError:
        ax.text(0.5, 0.5, "Install umap-learn\npip install umap-learn",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Embedding Space (UMAP)", fontweight="bold", pad=12)
        return

    strain_ids = list(
        set(pairs["id1"].tolist()) | set(pairs["id2"].tolist())
    )
    strain_ids = [
        s for s in strain_ids
        if s in strain_lookup and (gem_dir / strain_lookup[s]).exists()
    ]

    embeddings, avg_deltas, names = [], [], []

    for sid in strain_ids:
        try:
            gem = Batch.from_data_list([load_gem(sid)]).to(device)
            with torch.no_grad():
                emb = model.encode(gem).cpu().numpy()[0]
            embeddings.append(emb)

            # Average delta across all pairs where this strain is the anchor
            if sid in monoculture_lookup:
                mono = monoculture_lookup[sid]
                mask1 = pairs["id1"] == sid
                mask2 = pairs["id2"] == sid
                coculture_vals = (
                    list(pairs[mask1]["b1 biomass"]) +
                    list(pairs[mask2]["b2 biomass"])
                )
                avg_delta = np.mean([c - mono for c in coculture_vals]) if coculture_vals else 0.0
            else:
                avg_delta = 0.0

            avg_deltas.append(avg_delta)
            raw_name = strain_lookup[sid]
            names.append(Path(raw_name).stem.replace("_", " "))
        except Exception:
            continue

    if len(embeddings) == 0:
        ax.text(0.5, 0.5, "No embeddings could be generated.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return

    emb_arr   = np.array(embeddings)
    delta_arr = np.array(avg_deltas)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
    coords  = reducer.fit_transform(emb_arr)

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=delta_arr, cmap="RdYlGn",
        s=90, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3
    )
    plt.colorbar(sc, ax=ax,
                 label="Avg delta (co-culture \u2212 monoculture)",
                 shrink=0.85)

    # Annotate organism names
    texts = []
    for (x, y), name in zip(coords, names):
        texts.append(
            ax.annotate(
                name, (x, y),
                fontsize=6.5, color="#333333",
                xytext=(4, 4), textcoords="offset points", zorder=4,
            )
        )
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))
    except ImportError:
        pass

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Organism Embedding Space\n(colored by avg delta, labeled by organism)",
                 fontweight="bold", pad=12)

# ---------------------------------------------------------------------------
# Plot 4 — Delta distribution histogram
# ---------------------------------------------------------------------------

def plot_distribution(ax):
    delta = holdout_df["delta"].dropna()

    if len(delta) == 0:
        ax.text(0.5, 0.5, "No delta values found.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return

    ax.hist(delta, bins=40, color=C_HIST, edgecolor="white",
            linewidth=0.5, alpha=0.9, zorder=3)
    ax.axvline(delta.mean(),   color=C_DIAG,  linewidth=1.8,
               linestyle="--", label=f"Mean ({delta.mean():.3f})")
    ax.axvline(delta.median(), color=C_TRAIN, linewidth=1.8,
               linestyle=":",  label=f"Median ({delta.median():.3f})")
    ax.axvline(0, color="#999999", linewidth=1.2,
               linestyle="-",  label="Monoculture baseline (0)")

    ax.set_xlabel("Delta (co-culture \u2212 monoculture growth rate)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Delta Distribution\n(Holdout Set)",
                 fontweight="bold", pad=12)
    ax.legend(framealpha=0.6)

# ---------------------------------------------------------------------------
# Plot 5 — Label classification accuracy by category
# ---------------------------------------------------------------------------

def plot_label_accuracy(ax):
    if model.label_head is None:
        ax.text(0.5, 0.5, "Label head not enabled in config.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return

    label_correct = {k: 0 for k in LABEL_TO_INT}
    label_total   = {k: 0 for k in LABEL_TO_INT}

    for _, row in holdout_df.iterrows():
        if pd.isna(row.get("label")):
            continue
        try:
            gem_a = Batch.from_data_list([load_gem(row["id1"])]).to(device)
            gem_b = Batch.from_data_list([load_gem(row["id2"])]).to(device)
            with torch.no_grad():
                emb_a      = model.encode(gem_a)
                emb_b      = model.encode(gem_b)
                pred_label = model.label_head.predict_label(emb_a, emb_b)[0]

            true_label = row["label"]
            label_total[true_label]   = label_total.get(true_label, 0) + 1
            if pred_label == true_label:
                label_correct[true_label] = label_correct.get(true_label, 0) + 1
        except Exception:
            continue

    labels   = list(LABEL_TO_INT.keys())
    accuracy = [
        label_correct.get(l, 0) / max(label_total.get(l, 1), 1)
        for l in labels
    ]
    counts = [label_total.get(l, 0) for l in labels]
    colors = [LABEL_COLORS[l] for l in labels]

    bars = ax.bar(labels, accuracy, color=colors,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # Annotate bars with sample counts
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={count}", ha="center", va="bottom", fontsize=8
        )

    overall_acc = (
        sum(label_correct.values()) / max(sum(label_total.values()), 1)
    )

    ax.axhline(1 / len(LABEL_TO_INT), color="#999999", linewidth=1.2,
               linestyle="--", label=f"Random chance ({1/len(LABEL_TO_INT):.2f})")
    ax.axhline(overall_acc, color=C_TRAIN, linewidth=1.5,
               linestyle=":", label=f"Overall accuracy ({overall_acc:.3f})")

    ax.set_xlabel("Label category")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Label Classification Accuracy by Category\n(Holdout Set)",
                 fontweight="bold", pad=12)
    ax.set_ylim(0, 1.12)
    ax.legend(framealpha=0.6, fontsize=9)

    print(f"\nLabel classification summary:")
    for l in labels:
        n   = label_total.get(l, 0)
        acc = label_correct.get(l, 0) / max(n, 1)
        print(f"  {l:<12}: {acc:.3f}  ({label_correct.get(l,0)}/{n})")
    print(f"  Overall    : {overall_acc:.3f}")

# ---------------------------------------------------------------------------
# Generate all five figures separately
# ---------------------------------------------------------------------------

Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("Plotting loss curves...")
fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor=BG)
plot_loss_curves(ax1)
fig1.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/01_loss_curves.png", dpi=200,
             bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print(f"Saved {OUTPUT_DIR}/01_loss_curves.png")

print("Plotting delta vs. cosine similarity scatter...")
fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor=BG)
plot_scatter(ax2)
fig2.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/02_delta_vs_similarity.png", dpi=200,
             bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"Saved {OUTPUT_DIR}/02_delta_vs_similarity.png")

print("Plotting UMAP embeddings (may take a moment)...")
fig3, ax3 = plt.subplots(figsize=(11, 9), facecolor=BG)
plot_umap(ax3)
fig3.tight_layout()
fig3.savefig(f"{OUTPUT_DIR}/03_umap_embeddings.png", dpi=200,
             bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print(f"Saved {OUTPUT_DIR}/03_umap_embeddings.png")

print("Plotting delta distribution...")
fig4, ax4 = plt.subplots(figsize=(8, 6), facecolor=BG)
plot_distribution(ax4)
fig4.tight_layout()
fig4.savefig(f"{OUTPUT_DIR}/04_delta_distribution.png", dpi=200,
             bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print(f"Saved {OUTPUT_DIR}/04_delta_distribution.png")

print("Plotting label classification accuracy...")
fig5, ax5 = plt.subplots(figsize=(9, 6), facecolor=BG)
plot_label_accuracy(ax5)
fig5.tight_layout()
fig5.savefig(f"{OUTPUT_DIR}/05_label_accuracy.png", dpi=200,
             bbox_inches="tight", facecolor=BG)
plt.close(fig5)
print(f"Saved {OUTPUT_DIR}/05_label_accuracy.png")

print(f"\nAll figures saved to {OUTPUT_DIR}/")
