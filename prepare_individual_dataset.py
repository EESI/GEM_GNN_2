"""
prepare_individual_dataset.py

Builds a triplet index JSON for individual strain contrastive learning.

Instead of comparing pairwise community growth rates, this script uses
each strain's monoculture growth rate as the anchor value, then compares
how much better or worse that strain grows in co-culture with a partner.

Triplet logic:
    Anchor   : strain A with its monoculture growth rate as the baseline
    Positive : partner B where co-culture growth of A exceeds monoculture
               growth of A by at least POSITIVE_THRESHOLD
    Negative : partner C where co-culture growth of A is below monoculture
               growth of A by at least NEGATIVE_THRESHOLD

Growth labels (how well anchor grows relative to its monoculture baseline):
    delta > +0.2  : "very_good"
    delta > +0.1  : "good"
    delta > -0.1  : "neutral"
    delta < -0.1  : "bad"
    delta < -0.2  : "very_bad"
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

from config_loader import load_settings, build_individual_config, build_pairwise_config
S = load_settings()

MONOCULTURE_JSON  = S.paths.monoculture_json
STRAINS_CSV       = S.paths.strains_csv
PAIRS_CSV         = S.paths.pairs_csv
GEM_DIR           = S.paths.gem_dir
OUTPUT_JSON       = "triplets_individual.json"

# Positive = anchor co-culture growth is in the top X% across all its partners
TOP_PERCENTILE     = S.prepare_individual.top_percentile   # top third of partners by anchor co-culture growth
# Negative = anchor co-culture growth is in the bottom X% across all its partners  
BOTTOM_PERCENTILE  = S.prepare_individual.bottom_percentile   # bottom third of partners by anchor co-culture growth
# Minimum delta difference between positive and negative partner
MIN_CONTRAST_GAP   = S.prepare_individual.min_contrast_gap   # much smaller since all values are already negative
# Dense mode: sample multiple negatives per positive
DENSE_MODE         = S.prepare_individual.dense_mode
TOP_K              = S.prepare_individual.top_k

RANDOM_SEED        = S.prepare_individual.random_seed

# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def assign_label(delta: float) -> str:
    """
    Assigns a categorical label based on how much the anchor's co-culture
    growth rate differs from its monoculture baseline.

    Args:
        delta: co_culture_growth_of_anchor - monoculture_growth_of_anchor
    Returns:
        label string
    """
    """
    if delta > 0.2: # used to be 0.2
        return "very_good"
    elif delta > 0.1: #used to be 0.1
        return "good"
    elif delta >= -0.1: # used to be -0.1
        return "mediocre"
    elif delta >= -0.2: #used to be -0.2
        return "bad"
    else:
        return "very_bad"
"""
    if delta > 0: # used to be 0.2
        return "positive"
    elif delta > -0.1: #used to be 0.1
        return "0:-0.1"
    elif delta >= -0.2: # used to be -0.1
        return "-0.1:-0.2"
    elif delta >= -0.3: #used to be -0.2
        return "-0.2:-0.3"
    elif delta >= -0.4: #used to be -0.2
        return "-0.3:-0.4"
    elif delta > -0.5: #used to be -0.2
        return "-0.4:-0.5"
    else:
        return "worse than -0.5"
"""
LABEL_TO_INT = {
    "very_bad":  0,
    "bad":       1,
    "mediocre":   2, #could be neutral as well
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
# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading data...")

with open(MONOCULTURE_JSON) as f:
    monoculture_data = json.load(f)

# Build monoculture lookup: strain_id (int) -> optimal growth rate
monoculture_lookup = {
    int(entry["ID"]): entry["Optimal Growth Rate"]
    for entry in monoculture_data
}

strains = pd.read_csv(STRAINS_CSV)
pairs   = pd.read_csv(PAIRS_CSV)

pairs = pairs[pairs["converged"] == True].copy()
pairs["id1"] = pairs["pairID"].apply(lambda x: int(x.split("__")[0]))
pairs["id2"] = pairs["pairID"].apply(lambda x: int(x.split("__")[1]))

strain_lookup = dict(zip(strains["ID"], strains["Strain"]))
gem_dir = Path(GEM_DIR)

print(f"Strains with monoculture data: {len(monoculture_lookup)}")
print(f"Converged pairs: {len(pairs)}")

# ---------------------------------------------------------------------------
# Compute per-anchor deltas
# For each pair (A, B), compute how much A's growth changes in co-culture
# relative to A's monoculture baseline. b1 biomass = A's growth in co-culture.
# ---------------------------------------------------------------------------

def get_anchor_coculture_growth(pairs_df, anchor_id):
    """
    Returns list of (partner_id, anchor_coculture_growth) for all pairs
    where anchor_id appears. Uses b1 biomass when anchor is id1,
    b2 biomass when anchor is id2.
    """
    results = []

    # Anchor is id1
    mask1 = pairs_df["id1"] == anchor_id
    for _, row in pairs_df[mask1].iterrows():
        results.append({
            "partner_id":           row["id2"],
            "anchor_coculture":     row["b1 biomass"],
            "partner_coculture":    row["b2 biomass"],
            "community_growth":     row["b1 biomass"] + row["b2 biomass"],
        })

    # Anchor is id2
    mask2 = pairs_df["id2"] == anchor_id
    for _, row in pairs_df[mask2].iterrows():
        results.append({
            "partner_id":           row["id1"],
            "anchor_coculture":     row["b2 biomass"],
            "partner_coculture":    row["b1 biomass"],
            "community_growth":     row["b1 biomass"] + row["b2 biomass"],
        })

    return results

# ---------------------------------------------------------------------------
# Build triplets
# ---------------------------------------------------------------------------

random.seed(RANDOM_SEED)

def gems_exist(id1, id2):
    return (
        id1 in strain_lookup and (gem_dir / strain_lookup[id1]).exists() and
        id2 in strain_lookup and (gem_dir / strain_lookup[id2]).exists()
    )

all_strain_ids = list(set(pairs["id1"]) | set(pairs["id2"]))
triplets = []
skipped_no_mono    = 0
skipped_no_gem     = 0
skipped_no_pairs   = 0
skipped_no_contrast = 0

for anchor_id in all_strain_ids:

    # Check monoculture data exists
    if anchor_id not in monoculture_lookup:
        skipped_no_mono += 1
        continue

    monoculture_rate = monoculture_lookup[anchor_id]

    # Get all co-culture results for this anchor
    anchor_pairs = get_anchor_coculture_growth(pairs, anchor_id)
    if len(anchor_pairs) < 2:
        skipped_no_pairs += 1
        continue

    # Compute delta for all partners
    for p in anchor_pairs:
        p["delta"]     = p["anchor_coculture"] - monoculture_rate
        p["label"]     = assign_label(p["delta"])
        p["label_int"] = LABEL_TO_INT[p["label"]]

    # Filter to valid GEM partners
    valid_partners = [
        p for p in anchor_pairs
        if gems_exist(anchor_id, p["partner_id"])
    ]
    if len(valid_partners) < 2:
        skipped_no_contrast += 1
        continue

    # Sort by anchor co-culture growth (highest = best partner for anchor)
    valid_partners.sort(key=lambda x: x["anchor_coculture"], reverse=True)

    # Top third = positives, bottom third = negatives
    n = len(valid_partners)
    top_n    = max(1, int(n * TOP_PERCENTILE))
    bottom_n = max(1, int(n * BOTTOM_PERCENTILE))

    positives = valid_partners[:top_n]
    negatives = valid_partners[-bottom_n:]

    # Ensure clear contrast between best positive and worst negative
    if positives[0]["anchor_coculture"] - negatives[-1]["anchor_coculture"] < MIN_CONTRAST_GAP:
        skipped_no_contrast += 1
        continue
    # Sort positives descending, negatives ascending by delta
    positives.sort(key=lambda x: x["delta"], reverse=True)
    negatives.sort(key=lambda x: x["delta"])

    if DENSE_MODE:
        # Multiple triplets per anchor: each positive paired with up to TOP_K negatives
        for pos in positives:
            valid_negs = [
                n for n in negatives
                if pos["delta"] - n["delta"] >= MIN_CONTRAST_GAP
                and n["partner_id"] != pos["partner_id"]
            ]
            chosen_negs = random.sample(valid_negs, min(TOP_K, len(valid_negs)))
            for neg in chosen_negs:
                triplets.append({
                    "anchor":   str(gem_dir / strain_lookup[anchor_id]),
                    "positive": str(gem_dir / strain_lookup[pos["partner_id"]]),
                    "negative": str(gem_dir / strain_lookup[neg["partner_id"]]),
                    "anchor_id":         anchor_id,
                    "positive_id":       pos["partner_id"],
                    "negative_id":       neg["partner_id"],
                    # Monoculture baseline
                    "anchor_monoculture": monoculture_rate,
                    # Co-culture growth of anchor with each partner
                    "positive_anchor_coculture": pos["anchor_coculture"],
                    "negative_anchor_coculture": neg["anchor_coculture"],
                    # Delta from monoculture
                    "positive_delta":    pos["delta"],
                    "negative_delta":    neg["delta"],
                    "contrast_gap":      pos["delta"] - neg["delta"],
                    # Categorical labels
                    "positive_label":    pos["label"],
                    "negative_label":    neg["label"],
                    "positive_label_int": pos["label_int"],
                    "negative_label_int": neg["label_int"],
                    # Community growth (kept for compatibility)
                    "growth_label":      pos["community_growth"],
                    "positive_growth":   pos["community_growth"],
                    "negative_growth":   neg["community_growth"],
                })
    else:
        # One triplet per anchor: best positive vs worst negative
        pos = positives[0]
        neg = negatives[0]

        if pos["delta"] - neg["delta"] < MIN_CONTRAST_GAP:
            skipped_no_contrast += 1
            continue
        if pos["partner_id"] == neg["partner_id"]:
            skipped_no_contrast += 1
            continue

        triplets.append({
            "anchor":            strain_lookup[anchor_id],
            "positive":          strain_lookup[pos["partner_id"]],
            "negative":          strain_lookup[neg["partner_id"]],
            "anchor_id":         anchor_id,
            "positive_id":       pos["partner_id"],
            "negative_id":       neg["partner_id"],
            "anchor_monoculture": monoculture_rate,
            "positive_anchor_coculture": pos["anchor_coculture"],
            "negative_anchor_coculture": neg["anchor_coculture"],
            "positive_delta":    pos["delta"],
            "negative_delta":    neg["delta"],
            "contrast_gap":      pos["delta"] - neg["delta"],
            "positive_label":    pos["label"],
            "negative_label":    neg["label"],
            "positive_label_int": pos["label_int"],
            "negative_label_int": neg["label_int"],
            "growth_label":      pos["community_growth"],
            "positive_growth":   pos["community_growth"],
            "negative_growth":   neg["community_growth"],
        })

random.shuffle(triplets)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nTriplet construction summary:")
print(f"  Built:                   {len(triplets)}")
print(f"  Skipped (no monoculture): {skipped_no_mono}")
print(f"  Skipped (no GEM):         {skipped_no_gem}")
print(f"  Skipped (no pairs):       {skipped_no_pairs}")
print(f"  Skipped (no contrast):    {skipped_no_contrast}")

if triplets:
    label_counts = {}
    for t in triplets:
        label_counts[t["positive_label"]] = label_counts.get(t["positive_label"], 0) + 1
        label_counts[t["negative_label"]] = label_counts.get(t["negative_label"], 0) + 1
    print(f"\n  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:<12}: {count}")

    deltas = [t["positive_delta"] - t["negative_delta"] for t in triplets]
    print(f"\n  Contrast gap stats:")
    print(f"    mean={np.mean(deltas):.4f}  min={np.min(deltas):.4f}  max={np.max(deltas):.4f}")

# ---------------------------------------------------------------------------
# Save stratified train/val split by positive label
# ---------------------------------------------------------------------------

from collections import defaultdict

TRAIN_SPLIT  = S.prepare_individual.train_split
TRAIN_JSON   = S.paths.triplets_individual_train
VAL_JSON     = S.paths.triplets_individual_val

# Group triplets by positive label
label_groups = defaultdict(list)
for t in triplets:
    label_groups[t["positive_label"]].append(t)

print(f"\nLabel distribution before split:")
for label, group in sorted(label_groups.items()):
    print(f"  {label:<12}: {len(group)}")

train_triplets, val_triplets = [], []

for label, group in label_groups.items():
    random.shuffle(group)
    n_val   = max(1, int(len(group) * (1 - TRAIN_SPLIT)))
    n_train = len(group) - n_val
    train_triplets.extend(group[:n_train])
    val_triplets.extend(group[n_train:])

random.shuffle(train_triplets)
random.shuffle(val_triplets)

print(f"\nAfter stratified split:")
print(f"  Train: {len(train_triplets)}")
print(f"  Val:   {len(val_triplets)}")

print(f"\nVal label distribution:")
val_label_counts = defaultdict(int)
for t in val_triplets:
    val_label_counts[t["positive_label"]] += 1
for label in sorted(val_label_counts):
    print(f"  {label:<12}: {val_label_counts[label]}")

# Save
with open(TRAIN_JSON, "w") as f:
    json.dump(train_triplets, f, indent=2)
with open(VAL_JSON, "w") as f:
    json.dump(val_triplets, f, indent=2)

print(f"\nSaved {len(train_triplets)} train triplets to {TRAIN_JSON}")
print(f"Saved {len(val_triplets)} val triplets to {VAL_JSON}")
