# Isoform Interaction Network

Link prediction on isoform-level protein-protein interaction networks using
Latent Distance Models (LDMs). The project supports a plain isoform LDM and a
multimodal LDM that combines isoform-isoform interactions, ESM-C isoform
features, gene-isoform membership, and optional STRING gene-gene interactions.

## Environment

The project is currently run from the `image_analysis` conda environment.

```bash
conda activate image_analysis
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you are not activating the environment in the shell, use the interpreter
directly:

```bash
/Users/tob/miniforge3/envs/image_analysis/bin/python scripts/train.py --config config/config.yaml
```

In restricted environments where Matplotlib cannot write to the default cache
directory, prefix commands with writable cache paths:

```bash
XDG_CACHE_HOME=/private/tmp/xdg MPLCONFIGDIR=/private/tmp/mpl \
  /Users/tob/miniforge3/envs/image_analysis/bin/python scripts/train.py --config config/config.yaml
```

## Data

Default input paths are configured in `config/config.yaml`:

- `data/results_PHYSICAL_Prob_Model_16_02_26.csv`: isoform-pair data used for
  the main link-prediction task. Training expects `ensp_1`, `ensp_2`, `gene_1`,
  `gene_2`, and binary `interact` columns.
- `data/esmc_globemb_noduplicates_15092025.csv`: ESM-C embeddings for
  multimodal models. The file is indexed by an `ENSP` column; missing isoforms
  are zero-filled.
- `data/STRING_protein_pairs_wscores_physical.csv`: optional STRING
  protein-pair data used when `multimodal.lambda_gene_gene > 0`.
- `data/gene-isoform_mapping_enst_ensp_ensg.csv`: optional ENSP-to-ENSG mapping
  used to aggregate STRING protein pairs to genes.

## Configuration

All training behavior is controlled by YAML config files. The default is
`config/config.yaml`; pass another file with `--config`.

```bash
python scripts/train.py --config config/config.yaml
```

Important config sections:

- `data.mode`: `transductive` or `inductive`. Transductive splits at gene-pair
  level. Inductive splits genes, so all isoforms of a gene stay in the same
  partition.
- `data.val_fraction`, `data.test_fraction`, `data.random_state`: split sizes
  and seed.
- `model.type`: `ldm` for the plain LDM or `multimodal` for the ESM-C-backed
  multimodal model.
- `model.latent_dim`: latent dimension. Set to `0` for a random-effects-only
  baseline.
- `training.*`: epochs, learning rate, batch size, weight decay, and early
  stopping patience.
- `multimodal.proj_hidden_dim`: hidden width for ESM-C projection and random
  effect heads.
- `multimodal.lambda_iso_iso`, `lambda_gene_iso`, `lambda_gene_gene`: loss
  weights for the three modalities. A weight of `0` disables that auxiliary
  modality.
- `multimodal.neg_ratio`: negatives sampled per positive gene-isoform edge.
- `paths.models_dir`: root directory for saved checkpoints.
- `paths.results_dir`: configured figure/results path for downstream outputs.

## Usage

### Single Training Run

```bash
python scripts/train.py --config config/config.yaml --cache
```

`--cache` enables split caching under `.cache/data_splits` by default. Cache
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
python scripts/train_repeated.py --config config/config.yaml --n_seeds 10 --base_seed 42 --cache
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
the matching test splits, and writes aggregate ROC, PR, and boxplot figures.

### Network Analysis

```bash
python scripts/network_analysis.py
```

This analyzes the default isoform-pair CSV and saves network, bipartite, and
STRING gene-gene figures to `figures/`.

## Project Structure

```text
config/config.yaml             Main run configuration
scripts/train.py               Single config-driven training run
scripts/train_repeated.py      Repeated training across multiple seeds
scripts/plot_curves.py         Aggregate ROC/PR/box plots across seed groups
scripts/network_analysis.py    Dataset and network statistics
src/data_scripts/              Data loading, splitting, and cache helpers
src/model_classes/             LDM and multimodal LDM implementations
src/training/runner.py         Shared training/evaluation runner
src/training/evaluate.py       Checkpoint loading and evaluation utilities
data/                          Local input data
models/                        Saved model checkpoints and repeated-run groups
figures/                       Generated analysis and comparison figures
```

## Model Summary

Plain LDM:

```text
P(Y_ij = 1) = sigmoid(alpha + r_i + r_j - ||z_i - z_j||)
```

Random-effects-only baseline (`latent_dim: 0`):

```text
P(Y_ij = 1) = sigmoid(alpha + r_i + r_j)
```

Multimodal LDM uses the same isoform-isoform objective, but obtains isoform
positions and isoform random effects from ESM-C features. It can also train
gene embeddings with gene-isoform membership and STRING gene-gene objectives.

## Splitting

Transductive mode splits at gene-pair level to prevent leakage: every
`(gene_1, gene_2)` block goes entirely to train, validation, or test, stratified
by whether that block contains any positive interaction.

Inductive mode partitions genes first. All isoforms of a gene stay in the same
train/validation/test partition, and interaction pairs are assigned according
to the most held-out endpoint. This creates one-unseen and both-unseen test
subsets, which are evaluated separately after training.
