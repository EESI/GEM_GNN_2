"""
config_loader.py

Reads settings.yaml and exposes all configuration variables as a single
Settings object. Place this at the top of every script:

    from config_loader import load_settings
    S = load_settings()

Then access any variable with dot notation:
    S.paths.gem_dir
    S.model.gnn_hidden_dim
    S.training.n_epochs
    S.prepare_individual.label_thresholds
    etc.

The settings file is looked for in the following order:
    1. Path passed explicitly to load_settings(path="...")
    2. GEMGNN_SETTINGS environment variable
    3. settings.yaml in the same directory as this script
    4. settings.yaml in the current working directory
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Nested settings dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathSettings:
    gem_dir:                     str = "/path/to/your/gem/xmls"
    strains_csv:                 str = "anaerobic_strains.csv"
    pairs_csv:                   str = "pairwise_results.csv"
    monoculture_json:            str = "monoculture_growth.json"
    triplets_json:               str = "triplets.json"
    val_pairs_json:              str = "val_pairs.json"
    training_history_json:       str = "training_history.json"
    checkpoint_dir:              str = "checkpoints"
    triplets_individual_train:   str = "triplets_individual_train.json"
    triplets_individual_val:     str = "triplets_individual_val.json"
    val_pairs_individual_json:   str = "val_pairs_individual.json"
    training_history_individual: str = "training_history_individual.json"
    checkpoint_dir_individual:   str = "checkpoints_individual"
    figures_dir:                 str = "figures"
    figures_individual_dir:      str = "figures_individual"
    optuna_dir:                  str = "optuna_results"
    embeddings_npy:              str = "embeddings.npy"
    embedding_ids_csv:           str = "embedding_ids.csv"
    outliers_csv:                str = "outliers.csv"


@dataclass
class ModelSettings:
    node_feature_dim:            int   = 64
    gnn_hidden_dim:              int   = 128
    gnn_num_layers:              int   = 6
    gnn_heads:                   int   = 8
    gnn_dropout:                 float = 0.0
    embedding_dim:               int   = 256
    projection_hidden_dim:       int   = 512
    triplet_margin:              float = 0.5
    lambda_aux:                  float = 0.05
    use_metabolite_head:         bool  = False
    use_gene_expression_head:    bool  = False
    use_growth_head:             bool  = False
    metabolite_output_dim:       int   = 32
    gene_expression_output_dim:  int   = 64
    use_delta_head:              bool  = True
    use_label_head:              bool  = True
    lambda_delta:                float = 1.0
    lambda_label:                float = 1.0
    label_mode:                  bool  = False


@dataclass
class TrainingSettings:
    n_epochs:     int   = 100
    batch_size:   int   = 16
    train_split:  float = 0.8
    lr:           float = 0.0005768635139990112
    weight_decay: float = 0.0004922523643135666
    random_seed:  int   = 42
    log_every:    int   = 10


@dataclass
class PreparePairwiseSettings:
    min_growth_gap: float = 0.05
    dense_mode:     bool  = False
    top_k:          int   = 3
    random_seed:    int   = 42


@dataclass
class PrepareIndividualSettings:
    top_percentile:    float = 0.33
    bottom_percentile: float = 0.33
    min_contrast_gap:  float = 0.02
    dense_mode:        bool  = False
    top_k:             int   = 3
    random_seed:       int   = 42
    train_split:       float = 0.8
    label_thresholds:  dict  = field(default_factory=lambda: {
        "very_bad": -0.45,
        "bad":      -0.35,
        "neutral":  -0.25,
        "good":     -0.15,
    })


@dataclass
class EvaluationSettings:
    n_eval_pairs:  int   = 50
    top_k_percent: float = 0.25
    random_seed:   int   = 99
    n_outliers:    int   = 15


@dataclass
class OptunaSearchSpace:
    gnn_hidden_dim:        list  = field(default_factory=lambda: [64, 128, 256, 512])
    gnn_num_layers_min:    int   = 2
    gnn_num_layers_max:    int   = 6
    gnn_heads:             list  = field(default_factory=lambda: [2, 4, 8])
    gnn_dropout_min:       float = 0.0
    gnn_dropout_max:       float = 0.4
    gnn_dropout_step:      float = 0.05
    embedding_dim:         list  = field(default_factory=lambda: [64, 128, 256])
    projection_hidden_dim: list  = field(default_factory=lambda: [128, 256, 512])
    triplet_margin_min:    float = 0.5
    triplet_margin_max:    float = 2.0
    triplet_margin_step:   float = 0.25
    lambda_aux_min:        float = 0.01
    lambda_aux_max:        float = 0.5
    lr_min:                float = 0.0001
    lr_max:                float = 0.01
    weight_decay_min:      float = 0.00001
    weight_decay_max:      float = 0.001


@dataclass
class OptunaSettings:
    n_trials:         int   = 100
    n_epochs:         int   = 30
    n_jobs:           int   = 1
    random_seed:      int   = 42
    direction:        str   = "minimize"
    n_startup_trials: int   = 5
    n_warmup_steps:   int   = 10
    search_space:     OptunaSearchSpace = field(default_factory=OptunaSearchSpace)


@dataclass
class Settings:
    paths:              PathSettings              = field(default_factory=PathSettings)
    model:              ModelSettings             = field(default_factory=ModelSettings)
    training:           TrainingSettings          = field(default_factory=TrainingSettings)
    prepare_pairwise:   PreparePairwiseSettings   = field(default_factory=PreparePairwiseSettings)
    prepare_individual: PrepareIndividualSettings = field(default_factory=PrepareIndividualSettings)
    evaluation:         EvaluationSettings        = field(default_factory=EvaluationSettings)
    optuna:             OptunaSettings            = field(default_factory=OptunaSettings)


# ---------------------------------------------------------------------------
# Helper: populate a dataclass from a dict, ignoring unknown keys
# ---------------------------------------------------------------------------

def _populate(dataclass_instance, data: dict):
    """Recursively populate a dataclass from a dictionary."""
    for key, value in data.items():
        if not hasattr(dataclass_instance, key):
            continue
        current = getattr(dataclass_instance, key)
        if hasattr(current, "__dataclass_fields__"):
            # Nested dataclass — recurse
            _populate(current, value)
        else:
            setattr(dataclass_instance, key, value)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_settings(path: str = None) -> Settings:
    """
    Load settings from a YAML file and return a Settings object.

    Args:
        path: explicit path to settings.yaml. If None, searches in order:
              1. GEMGNN_SETTINGS environment variable
              2. Same directory as config_loader.py
              3. Current working directory

    Returns:
        Settings object with all configuration values.

    Raises:
        FileNotFoundError if no settings file can be located.
    """

    # Locate the settings file
    if path is not None:
        settings_path = Path(path)
    elif "GEMGNN_SETTINGS" in os.environ:
        settings_path = Path(os.environ["GEMGNN_SETTINGS"])
    else:
        candidates = [
            Path(__file__).parent / "settings.yaml",
            Path.cwd() / "settings.yaml",
        ]
        settings_path = next((p for p in candidates if p.exists()), None)
        if settings_path is None:
            raise FileNotFoundError(
                "Could not find settings.yaml. Provide an explicit path with "
                "load_settings(path='...') or set the GEMGNN_SETTINGS "
                "environment variable."
            )

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with open(settings_path) as f:
        raw = yaml.safe_load(f)

    # Build Settings object and populate from YAML
    S = Settings()
    if raw:
        _populate(S, raw)

    print(f"[config_loader] Loaded settings from {settings_path.resolve()}")
    return S


# ---------------------------------------------------------------------------
# Convenience: build GNNGLMConfig / IndividualGNNConfig from settings
# ---------------------------------------------------------------------------

def build_pairwise_config(S: Settings):
    """Build a GNNGLMConfig from settings."""
    from models import GNNGLMConfig
    return GNNGLMConfig(
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
        metabolite_output_dim=S.model.metabolite_output_dim,
        gene_expression_output_dim=S.model.gene_expression_output_dim,
        triplet_margin=S.model.triplet_margin,
        lambda_aux=S.model.lambda_aux,
    )


def build_individual_config(S: Settings):
    """Build an IndividualGNNConfig from settings."""
    from models_individual import IndividualGNNConfig
    return IndividualGNNConfig(
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
        use_delta_head=S.model.use_delta_head,
        use_label_head=S.model.use_label_head,
        lambda_delta=S.model.lambda_delta,
        lambda_label=S.model.lambda_label,
        triplet_margin=S.model.triplet_margin,
        lambda_aux=S.model.lambda_aux,
        label_mode=S.model.label_mode,
    )


def build_label_map(S: Settings) -> tuple[dict, dict, int]:
    """
    Build LABEL_TO_INT and INT_TO_LABEL from settings label thresholds.
    Labels are ordered from worst to best based on threshold order.

    Returns:
        LABEL_TO_INT: dict mapping label string to integer
        INT_TO_LABEL: dict mapping integer to label string
        N_LABELS:     total number of label categories
    """
    thresholds = S.prepare_individual.label_thresholds
    # Labels are ordered worst to best; very_good is always the final label
    ordered_labels = list(thresholds.keys()) + ["very_good"]
    label_to_int = {label: i for i, label in enumerate(ordered_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    return label_to_int, int_to_label, len(ordered_labels)


if __name__ == "__main__":
    # Quick test — prints all loaded settings
    S = load_settings()
    print("\nPaths:")
    for k, v in vars(S.paths).items():
        print(f"  {k}: {v}")
    print("\nModel:")
    for k, v in vars(S.model).items():
        print(f"  {k}: {v}")
    print("\nTraining:")
    for k, v in vars(S.training).items():
        print(f"  {k}: {v}")
    print("\nLabel thresholds:")
    for k, v in S.prepare_individual.label_thresholds.items():
        print(f"  {k}: {v}")
    label_to_int, int_to_label, n = build_label_map(S)
    print(f"\nLabel map ({n} classes):")
    for k, v in label_to_int.items():
        print(f"  {k}: {v}")
