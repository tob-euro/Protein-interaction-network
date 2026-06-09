# Isoform Interaction Network

Link prediction on isoform-level protein-protein interaction networks using
Latent Distance Models (LDMs). The project supports a plain isoform LDM and a
multimodal LDM that combines isoform-isoform interactions, ESM-C isoform
features, gene-isoform membership, and STRING gene-gene interactions.

## Environment

The code in this repository requires a python environment with the listed packages in "requirements.txt" installed.
Alternatively create a new environment, activate it and run:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Configuration

To change configuration for training, use the YAML file `config.yaml`.

Config sections explained:

#### Data
- `iso_path`: path to the CSV file containing the iso-iso network. "data/<file_name>.csv"
- `esmc_path`: path to the CSV file containing the ESM C vectors for each isoform. "data/<file_name>.csv"
- `string_path`: path to the CSV file containing the gene-gene STRING
  interactions.
- `mapping_path`: path to the CSV file containing the ENSP-to-ENSG mapping
  used to aggregate STRING protein pairs to genes.
- `mode`: `transductive` or `inductive`. Transductive splits at gene-pair
  level. Inductive splits genes, so all isoforms of a gene stay in the same
  partition.
- `val_fraction` and `test_fraction`: fraction of data assigned to val/test split.
- `random_state`: The seed controlling the data splitting and model initialization.

#### Model
- `type`: controls the model type: `ldm` or `multimodal`.
- `latent_dim`: controls the latent dimension size of the model. If set to `0`: A random-effects-only model is used. 

#### Training
- `epochs`: number of epochs used in training run.
- `learning_rate`: learning rate for the optimizer.
- `batch_size`: the size of each mini-batch (number of interactions in each optimization step).
- `weight_decay`: weight decay for the optimizer.
- `patience`: controls how many epochs are allowed with no improvement in validation AP before early stopping is applied.

#### Multimodal
- `proj_hidden_dim`: the size of the hidden dimension used in the MLP.
- `lambda_iso_iso`, `lambda_gene_iso`, `lambda_gene_gene`: loss
  weights for the three modalities. A weight of `0` disables that auxiliary
  modality.
- `neg_ratio`: negatives sampled per positive gene-isoform edge.

#### Paths
- `models_dir`: root directory for saved models and plots.
- `figures_dir`: directory for saved figures.


## Project Structure

```text
config.yaml                    Main run configuration
scripts/train.py               Single training run
scripts/train_repeated.py      Repeated training across multiple seeds
scripts/plot_curves.py         Aggregate ROC/PR/box plots across seed groups
scripts/results.py             Thesis result summaries, tests, and figures
scripts/network_analysis.py    Dataset and network statistics
src/data_scripts/              Data loading, splitting, and cache helpers
src/model_classes/             LDM and multimodal LDM implementations
src/training/runner.py         Shared training/evaluation runner
src/training/evaluate.py       Checkpoint loading and evaluation utilities
data/                          Local input data
models/                        Saved model checkpoints and repeated-run groups
figures/                       Generated analysis and comparison figures
```

## Usage

### Single Training Run

```bash
python scripts/train.py --cache
```

`--cache` enables split caching under `.cache/data_splits` by default. This is smart to enable, if you want to run several different models on the same data configuration to not waste unnecessary time loading the data repeatedly. Cache
keys include the split mode, seed, fractions, input file metadata, negative
sampling ratio, and active modalities.

Single runs save to:

```text
models/<TRANS|IND>_<LDM|MM>_<timestamp>/
```

Each run directory contains `config.yaml`, `model.pt`, `training_curves.png`,
ROC/PR plots, prediction histograms, and extra inductive or gene-gene figures
when those evaluations are active.

### Repeated-Seed Runs

```bash
python scripts/train_repeated.py --n_seeds 10 --base_seed 42 --cache
```

This runs seeds `base_seed` through `base_seed + n_seeds - 1`, saves each run in
`seed_<N>/`, prints mean/std/95% CI summaries, and writes
`repeated_results.csv` to a shared output directory. By default that directory
is:

```text
models/repeated_<timestamp>/
```

Use `--output_dir` to choose a specific group name, for example:

```bash
python scripts/train_repeated.py --n_seeds 10 --output_dir models/ind_mm_ldm_seeds=10_latent_dim=2 --cache
```

### Aggregate Curves Across Seed Groups

```bash
python scripts/plot_curves.py "ind_mm_ldm" --models_dir models --output_dir figures/ind_mmldm_seeds=10
```

The plotting script searches model group directory names for the given
substring, loads each `seed_<N>/model.pt` and `seed_<N>/config.yaml`, rebuilds
the matching test splits, and makes aggregated ROC, PR, and boxplot figures.

### Thesis Results

```bash
python scripts/results.py
```

This prints the thesis result tables and extended statistical tests directly in
the terminal, using the repeated-run CSVs under `models/`. It also writes the
inductive ablation figure to `figures/`.

RQ3 STRING reconstruction and checkpoint-based threshold diagnostics are
opt-in because they rescore saved checkpoints:

```bash
python scripts/results.py --rq3
python scripts/results.py --prediction-diagnostics
python scripts/results.py --all
```

All generated result figures are saved directly in `figures/`.

### Network Analysis

```bash
python scripts/network_analysis.py
```

This script analyzes the three different networks and prints out basic stats of each graph together with degree distribution plots. Figures are saved to `figures/`.
