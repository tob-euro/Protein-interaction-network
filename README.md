# Isoform Interaction Network

Link prediction on an isoform-level protein–protein interaction network using a Latent Distance Model (LDM) with random effects.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Train a model

```bash
python scripts/train.py
```

All settings come from `config/config.yaml`. Override anything via CLI:

```bash
python scripts/train.py --latent-dim 64
python scripts/train.py --epochs 50 --lr 0.0001
python scripts/train.py --distance-metric cosine --no-random-effects
```

The trained model and evaluation plots (ROC, PR curve, training curves) are saved to `models/LDM_dim=...`.

### Visualize a trained model

Open `scripts/visualize.py`, set `MODEL_DIR` to your model folder, then run:

```bash
python scripts/visualize.py
```

Saves PCA and hierarchical clustering plots to `<MODEL_DIR>/visualizations/`.

### Data analysis

```bash
python src/data/data_analysis.py
```

Saves degree distribution and adjacency matrix plots to `figures/`.

## Project Structure

```
├── config/
│   └── config.yaml                         # All hyperparameters
├── scripts/
│   ├── train.py                            # Train a model
│   └── visualize.py                        # Visualize a trained model
├── src/
│   ├── data/
│   │   ├── dataset.py                      # Dataset + gene-pair-level splitting
│   │   └── data_analysis.py                # Network statistics and plots
│   ├── models/
│   │   └── latent_distance_model.py        # LDM, BaselineLDM, Trainer
│   ├── training/
│   │   └── evaluate_pretrained_models.py   # load_trained_model, evaluate_model
│   └── visualizations/
│       ├── pca.py                          # PCA of latent space
│       └── hierarchical_clustering.py      # Ward linkage clustering
├── data/                                   # Raw data (not in git)
└── models/                                 # Saved checkpoints (not in git)
```

## Model

**LDM with random effects:** `P(Y_ij = 1) = σ(r_i + r_j − β · ||z_i − z_j||)`

**Baseline (random effects only):** `P(Y_ij = 1) = σ(r_i + r_j)` — useful for measuring how much the latent geometry contributes.

## Data Splitting

Splits at **gene-pair level** to prevent leakage — all isoform pairs for a `(gene_1, gene_2)` block go exclusively into one split, stratified by whether the block contains any positive interaction.