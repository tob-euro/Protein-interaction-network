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

All settings come from `config/config.yaml`.

The trained model and evaluation plots (ROC, PR curve, training curves) are saved to a timestamped directory under `models/`.

### Visualize a trained model

```bash
python scripts/visualize.py models/IND_MM_<timestamp>
```

Saves PCA plots to `<MODEL_DIR>/visualizations/`.

### Data analysis

```bash
python scripts/network_analysis.py
```

Saves degree distribution and adjacency matrix plots to `figures/`.

## Project Structure

```
├── config/
│   └── config.yaml                         # All hyperparameters
├── scripts/
│   ├── train.py                            # Train a model
│   ├── train_repeated.py                   # Train across multiple seeds
│   ├── network_analysis.py                 # Network statistics and plots
│   ├── evaluate_gene_embeddings.py         # Gene-embedding evaluation
│   └── visualize.py                        # Visualize a trained model
├── src/
│   ├── data_scripts/                       # Data loading and split helpers
│   ├── model_classes/                      # LDM and multimodal LDM classes
│   ├── training/
│   │   ├── evaluate.py                     # Model loading and evaluation
│   │   └── runner.py                       # Shared training runner
│   └── visualizations/
│       └── pca.py                          # PCA of latent space
├── data/                                   # Raw data (not in git)
└── models/                                 # Saved checkpoints (not in git)
```

## Model

**LDM with random effects:** `P(Y_ij = 1) = σ(r_i + r_j − β · ||z_i − z_j||)`

**Baseline (random effects only):** `P(Y_ij = 1) = σ(r_i + r_j)` — useful for measuring how much the latent geometry contributes.

## Data Splitting

Splits at **gene-pair level** to prevent leakage — all isoform pairs for a `(gene_1, gene_2)` block go exclusively into one split, stratified by whether the block contains any positive interaction.
