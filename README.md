# GEM-GNN: Graph Neural Network for Microbial Community Growth Prediction

A graph neural network system for predicting microbial community growth using contrastive (triplet) learning on genome-scale metabolic models (GEMs). The system learns biologically meaningful embeddings of microbial organisms from their metabolic network structure and uses those embeddings to predict community growth compatibility between organism pairs.

---

## Overview

Each GEM is represented as a bipartite graph connecting metabolite and reaction nodes, where edges are weighted by stoichiometric coefficients. The GNN encoder processes these graphs through stacked Graph Attention Network (GAT) layers, compressing each organism's metabolic structure into a fixed-size embedding vector. The system is trained using a contrastive triplet learning objective, where groups of three organisms — an anchor, a positive partner with high co-growth compatibility, and a negative partner with low co-growth compatibility — teach the model to place metabolically compatible organisms closer together in embedding space.

Two training systems are implemented:

- **Pairwise system** — learns from community-level co-growth rates between pairs of organisms
- **Individual strain system** — learns from how each organism's growth changes relative to its monoculture baseline when paired with a partner, with categorical interaction labels (very_bad / bad / neutral / good / very_good)

---

## Installation

```bash
# Install PyTorch with the correct CUDA version for your system
# See https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Data

All data used in the example work provided is from the SIMBA-GNN work done by Aminian-Dehkordi et al. (2026) at the Mofrad Lab. The article for SIMBA-GNN can be found here: 
Aminian-Dehkordi, J., Parsa, M., Dickson, A. et al. SIMBA-GNN: mechanistic graph learning for microbiome prediction. npj Syst Biol Appl 12, 8 (2026). https://doi.org/10.1038/s41540-025-00631-w

The data from Aminian-Dehkordi et al. (2026) can be found on their GitHub here: https://github.com/mofradlab/simba


---

## Project Structure

```
GEM_GNN/
│
├── settings.yaml                   # Central configuration file
├── config_loader.py                # Settings loader used by all scripts
├── requirements.txt                # Python dependencies
│
├── models.py                       # Pairwise GNN-GLM architecture
├── models_individual.py            # Individual strain GNN architecture
│
├── data.py                         # Pairwise dataset and dataloader
├── data_individual.py              # Individual strain dataset and dataloader
│
├── prepare_dataset.py              # Builds pairwise triplets.json
├── prepare_individual_dataset.py   # Builds individual strain triplets
│
├── run.py                          # Pairwise system training script
├── run_individual.py               # Individual strain training script
│
├── train.py                        # Trainer class and evaluation utilities
│
├── visualize.py                    # Figures for pairwise system
├── visualize_individual.py         # Figures for individual strain system
│
├── inference.py                    # Single pair growth prediction
├── rank_eval.py                    # Ranking evaluation on holdout set
├── extract_embeddings.py           # Extract and save all organism embeddings
├── collaborator_analysis.py        # Avg growth vs number of collaborators plot
├── outlier_analysis.py             # Identifies worst predictions and their pairings
│
├── optimize_hyperparams.py         # Optuna hyperparameter optimization
├── submit_optimization.sh          # SLURM batch script for optimization
│
├── ablation_study.py               # 5-run ablation with sequential data reduction
│
├── demo.py                         # End-to-end smoke test with synthetic data
│
├── checkpoints/                    # Pairwise system model checkpoints
├── checkpoints_individual/         # Individual strain model checkpoints
├── figures/                        # Pairwise system figures
├── figures_individual/             # Individual strain figures
├── ablation/                       # Ablation study results
└── optuna_results/                 # Hyperparameter optimization results
```

---

## File Descriptions

### Configuration

**`settings.yaml`**
The central configuration file for the entire pipeline. Contains all hardcoded variables including file paths, model architecture parameters, training hyperparameters, dataset preparation thresholds, label definitions, evaluation settings, and Optuna search space bounds. Edit this file to configure any aspect of the pipeline — no other files need to be changed for routine parameter adjustments.

**`config_loader.py`**
Reads `settings.yaml` and exposes all settings as a typed `Settings` object accessible via dot notation (e.g. `S.paths.gem_dir`, `S.model.gnn_hidden_dim`). Also provides convenience functions `build_pairwise_config()`, `build_individual_config()`, and `build_label_map()` that construct the correct model config and label mappings directly from settings. Place at the top of every script with:
```python
from config_loader import load_settings, build_individual_config
S = load_settings()
```

---

### Model Architecture

**`models.py`**
Defines the core GNN-GLM architecture for the pairwise system. Contains:
- `GNNGLMConfig` — dataclass holding all architecture hyperparameters
- `GNNLayer` — single GAT layer with residual connection and LayerNorm
- `GNNGLMEncoder` — stacked GAT encoder with dual mean+sum pooling and MLP projection head
- `AuxiliaryHead` — generic MLP prediction head
- `GrowthHead` — predicts community growth rate from a pair of embeddings using bilinear interaction features
- `GNNGLMSystem` — full triplet training system combining encoder, auxiliary heads, and triplet loss

**`models_individual.py`**
Extends the pairwise architecture for individual strain contrastive learning. Contains:
- `IndividualGNNConfig` — extended config adding delta head, label head, and label mode settings
- `DeltaHead` — predicts continuous delta (co-culture minus monoculture growth rate)
- `LabelHead` — classifies the interaction into categorical labels with ordinal structure
- `LabelTripletLoss` — label-aware triplet loss that scales the margin by the ordinal distance between positive and negative labels
- `IndividualGNNSystem` — full system supporting two training implementations:
  - Implementation 1 (`label_mode=False`): standard triplet loss + delta regression + label classification
  - Implementation 2 (`label_mode=True`): label-aware triplet loss + label classification as primary signal

---

### Data Loading

**`data.py`**
Handles GEM loading and pairwise triplet dataset management. Contains:
- `GEMGraph` — converts a COBRApy SBML model into a PyTorch Geometric bipartite graph with metabolite and reaction node features
- `GEMTripletDataset` — loads triplets from JSON, supports both real GEM files and synthetic data for testing
- `collate_triplets` — custom DataLoader collation function for PyG graph batching
- `make_dataloader` — convenience wrapper for building DataLoader instances

**`data_individual.py`**
Extended dataset for the individual strain system. Handles the additional fields introduced by individual strain triplets including monoculture baseline values, delta values, and categorical label integers.

---

### Dataset Preparation

**`prepare_dataset.py`**
Converts `anaerobic_strains.csv` and `pairwise_results.csv` into `triplets.json` for the pairwise system. For each anchor strain, assigns the highest co-growth partner as positive and lowest co-growth partner as negative. Supports both standard (one triplet per anchor) and dense (multiple negatives per positive) modes. All settings are read from `settings.yaml`.

**`prepare_individual_dataset.py`**
Builds the individual strain triplet dataset from `pairwise_results.csv` and `monoculture_growth.json`. Computes delta = co-culture growth of anchor − monoculture growth of anchor for each partner, assigns categorical labels based on configurable thresholds, and performs a stratified train/val split ensuring all label categories are proportionally represented in both sets. Outputs `triplets_individual_train.json` and `triplets_individual_val.json`.

---

### Training

**`run.py`**
Main training script for the pairwise system. Loads `triplets.json`, performs an 80/20 train/val split, trains the `GNNGLMSystem`, saves `val_pairs.json`, `training_history.json`, and model checkpoints. After training, saves the history with:
```python
history = trainer.train(n_epochs=100)
```

**`run_individual.py`**
Training script for the individual strain system. Loads the pre-split `triplets_individual_train.json` and `triplets_individual_val.json`, trains `IndividualGNNSystem`, and logs triplet loss, delta regression loss, and label classification loss separately. Reports label classification accuracy on the val set after training completes.

**`train.py`**
Contains the `Trainer` class used by the pairwise system, implementing the training loop, AdamW + cosine LR scheduling, gradient clipping, and best-checkpoint saving. Also provides `evaluate_embeddings()` for computing triplet accuracy and cosine similarity metrics on any dataset.

---

### Visualization

**`visualize.py`**
Generates four figures for the pairwise system, each saved as a separate PNG:
1. `01_loss_curves.png` — training and validation loss over epochs
2. `02_true_vs_predicted.png` — true vs. predicted community growth rate with OLS regression line and R² statistics on the holdout set
3. `03_umap_embeddings.png` — UMAP projection of all organism embeddings colored by average community growth rate and annotated with organism names
4. `04_growth_distribution.png` — histogram of community growth rates across all 3,233 converged pairs

**`visualize_individual.py`**
Generates five figures for the individual strain system:
1. `01_loss_curves.png` — total, delta, and label loss components over epochs
2. `02_delta_vs_similarity.png` — true delta vs. embedding cosine similarity with OLS regression, points colored by label category
3. `03_umap_embeddings.png` — UMAP projection colored by average delta from monoculture baseline
4. `04_delta_distribution.png` — histogram of delta values in the holdout set
5. `05_label_accuracy.png` — label classification accuracy broken down by category with sample counts

---

### Evaluation and Analysis

**`inference.py`**
Loads a trained model and predicts the community growth rate (pairwise system) or interaction label and delta (individual system) for a single pair of GEM files. Configure `GEM_DIR`, `GEM_A_FILE`, `GEM_B_FILE`, and `CHECKPOINT` at the top of the file.

**`rank_eval.py`**
Evaluates the model's ability to rank organism pairs by community growth rate on the holdout set. Reports three metrics:
- Spearman rank correlation between predicted and true growth rankings
- Kendall's Tau (fraction of pairwise orderings that are correct)
- Top-K precision (fraction of truly high-growth pairs found in predicted top K%)

**`extract_embeddings.py`**
Encodes all strains in the pairwise dataset using the trained model and saves the results as `embeddings.npy` (shape: [N_strains, embedding_dim]) and `embedding_ids.csv` (strain IDs and names). Used as the database for embedding-based partner retrieval.

**`collaborator_analysis.py`**
Plots average community growth rate versus number of pairwise co-growth partners per strain. Helps identify whether prediction errors correlate with how many training examples a strain appeared in. Points are colored by average growth rate using the same RdYlGn palette as the UMAP figures.

**`outlier_analysis.py`**
Runs inference on all holdout pairs, identifies the top N overestimates and underestimates, and cross-references those pairs with collaborator stats to investigate whether prediction errors are driven by strains with few training examples. Generates three figures: highlighted scatter plot, overestimate bar chart, and underestimate bar chart. Also saves `outliers.csv` with full predictions for all holdout pairs.

---

### Hyperparameter Optimization

**`optimize_hyperparams.py`**
Uses Optuna with TPE sampling and median pruning to find optimal model hyperparameters. Runs `N_TRIALS` trials, each training for `N_EPOCHS` epochs on a fixed train/val split, reporting intermediate validation loss at each epoch for pruning. Uses SQLite persistence so progress is saved if the job is interrupted and can be resumed by resubmitting. Outputs `best_params.json`, `study.pkl`, `trials.csv`, and `optimization.log`.

**`submit_optimization.sh`**
SLURM batch submission script for running hyperparameter optimization on the Purdue Anvil HPC system. Configures GPU resources, loads the conda environment, and runs `optimize_hyperparams.py`. Update `--account`, `--partition`, and `PROJECT_DIR` before submitting.

---

### Ablation Study

**`ablation_study.py`**
Runs five sequential training experiments where 20% of the training data is removed at each step (100% → 80% → 60% → 40% → 20%). The holdout set is fixed across all runs for fair comparison. For each run, saves the trained model checkpoint, loss curves, training history, and evaluation metrics. After all runs complete, generates `ablation_summary.csv` and a three-panel summary plot showing how validation loss, triplet accuracy, and label classification accuracy degrade as training data is reduced.

---

### Other

**`demo.py`**
End-to-end smoke test using fully synthetic GEM data. Requires no real GEM files or COBRApy models. Useful for verifying that the installation is correct and all components connect properly before running on real data.

---

## Workflow

### Pairwise System

```bash
# 1. Configure settings.yaml with your file paths
# 2. Build triplet dataset
python prepare_dataset.py

# 3. Train
python run.py

# 4. Evaluate
python rank_eval.py

# 5. Visualize
python visualize.py

# 6. Analyze outliers
python outlier_analysis.py
python collaborator_analysis.py
```

### Individual Strain System

```bash
# 1. Configure settings.yaml (label_thresholds, paths, model settings)
# 2. Build stratified train/val triplet datasets
python prepare_individual_dataset.py

# 3. Train (set LABEL_MODE in run_individual.py or settings.yaml)
python run_individual.py

# 4. Visualize
python visualize_individual.py
```

### Hyperparameter Optimization

```bash
# Submit to SLURM on Anvil
sbatch submit_optimization.sh

# After completion, update settings.yaml with best_params.json values
# then retrain with run.py or run_individual.py
```

### Ablation Study

```bash
# Requires individual strain train/val splits to already exist
python prepare_individual_dataset.py
python ablation_study.py
```

---

## Key Results

| Metric | Value |
|---|---|
| Triplet accuracy (pairwise) | 0.829 |
| Spearman rank correlation | 0.587 |
| Kendall's Tau | 0.413 |
| Top-25% precision | 0.667 |
| Training strains | 76 |
| Converged pairwise simulations | 3,233 |

---

## Citation

If you use this code in your research, please cite accordingly.

---

## Dependencies

See `requirements.txt` for the full list. Key packages:

| Package | Purpose |
|---|---|
| PyTorch | Deep learning framework |
| PyTorch Geometric | Graph neural network operations |
| COBRApy | Loading SBML genome-scale models |
| Optuna | Hyperparameter optimization |
| UMAP-learn | Embedding space visualization |
| adjustText | Non-overlapping UMAP annotations |
| PyYAML | Settings file parsing |
| SciPy | Ranking metrics and statistics |
