"""
train_inductive.py — Inductive isoform-level training script.

Mirrors train.py but uses load_and_prepare_data_inductive() to split data at
the isoform level rather than at the gene-pair level.

Each isoform is assigned to exactly one partition (train / val / test).
Val and test interactions each involve at least one isoform never seen during
training.  For MultimodalLDM + ESM-C this is fully inductive: unseen isoform
positions come entirely from the learned ESM-C projection.

Usage:
    python scripts/train_inductive.py --model-type multimodal [options]

All arguments are identical to train.py except:
  --alpha is removed (ClosedWorldTrainDataset is not used in inductive mode).
"""
import argparse
import os
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model_classes.ldm import LatentDistanceModel, LatentDistanceTrainer
from src.model_classes.mm_ldm import MultimodalLDM, MultimodalTrainer
from src.data_scripts.isoform_pairs import (
    ProteinInteractionDataset,
    load_and_prepare_data_inductive,
    diagnose_split_inductive,
)
from src.data_scripts.gene_isoform_pairs import GeneIsoformDataset, prepare_gene_isoform_splits
from src.data_scripts.gene_pairs import GeneGeneDataset, prepare_gene_gene_splits
from src.training.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-type',      type=str,   default='ldm',
                        choices=['ldm', 'multimodal'])
    parser.add_argument('--config',          type=str,   default='config/config.yaml')
    parser.add_argument('--data',            type=str,   default=None)
    parser.add_argument('--latent-dim',      type=int,   default=None)
    parser.add_argument('--distance-metric', type=str,   default=None,
                        choices=['euclidean', 'cosine'])
    parser.add_argument('--epochs',          type=int,   default=None)
    parser.add_argument('--lr',              type=float, default=None)
    parser.add_argument('--batch-size',      type=int,   default=None)
    parser.add_argument('--weight-decay',    type=float, default=None)
    # Multimodal-only
    parser.add_argument('--lambda-iso',         type=float, default=1.0)
    parser.add_argument('--lambda-gene',        type=float, default=0.5)
    parser.add_argument('--neg-ratio',          type=int,   default=5)
    parser.add_argument('--gene-batch-size',    type=int,   default=512)
    parser.add_argument('--lambda-complex',     type=float, default=0.5)
    parser.add_argument('--complex-batch-size', type=int,   default=512)
    parser.add_argument('--esmc',               type=str,
                        default='data/esmc_globemb_noduplicates_15092025.csv',
                        help='Path to ESM-C global embedding CSV. '
                             'Required for truly inductive inference; '
                             'set to empty string to fall back to learned embeddings.')
    return parser.parse_args()


def apply_overrides(cfg, args):
    if args.data:            cfg['data']['path']              = args.data
    if args.latent_dim:      cfg['model']['latent_dim']       = args.latent_dim
    if args.distance_metric: cfg['model']['distance_metric']  = args.distance_metric
    if args.epochs:          cfg['training']['epochs']        = args.epochs
    if args.lr:              cfg['training']['learning_rate'] = args.lr
    if args.batch_size:      cfg['training']['batch_size']    = args.batch_size
    if args.weight_decay:    cfg['training']['weight_decay']  = args.weight_decay
    return cfg


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args)

    d = cfg['data']
    m = cfg['model']
    t = cfg['training']

    device = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"INDUCTIVE TRAINING — {'MULTIMODAL ' if args.model_type == 'multimodal' else ''}LATENT DISTANCE MODEL")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Split: isoform-level (inductive)")
    print(f"{'='*70}\n")

    # =========================================================================
    # 1. Load data — isoform-level inductive split
    # =========================================================================
    print("Step 1: Loading data (isoform-level inductive split)...")
    (train_dataset, train_data, val_data, test_data,
     protein_to_idx, num_proteins, neg_pos_ratio,
     train_proteins, val_proteins, test_proteins) = load_and_prepare_data_inductive(
        d['path'],
        test_size=d['test_size'],
        val_size=d['val_size'],
        random_state=d['random_state'],
    )
    print(f"\n  Isoforms: {num_proteins:,}  |  "
          f"Train pairs: {len(train_dataset):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")
    diagnose_split_inductive(train_data, val_data, test_data,
                              train_proteins, val_proteins, test_proteins)

    # =========================================================================
    # 2. Dataloaders
    # =========================================================================
    print("Step 2: Creating dataloaders...")
    train_loader = DataLoader(train_dataset,
                              batch_size=t['batch_size'], shuffle=True,
                              num_workers=t['num_workers'])
    val_loader   = DataLoader(ProteinInteractionDataset(val_data, protein_to_idx),
                              batch_size=t['batch_size'], shuffle=False,
                              num_workers=t['num_workers'])
    test_loader  = DataLoader(ProteinInteractionDataset(test_data, protein_to_idx),
                              batch_size=t['batch_size'], shuffle=False,
                              num_workers=t['num_workers'])

    gene_train_loader    = None
    complex_train_loader = None
    gene_iso_ratio       = 5.0
    gene_gene_ratio      = 5.0
    num_genes            = 0
    gene_to_idx          = {}

    if args.model_type == 'multimodal':
        print("\nStep 2b: Building gene–isoform bipartite graph...")
        df_full = pd.read_csv(d['path'])
        gene_to_idx, train_g, val_g, _, gene_iso_ratio = prepare_gene_isoform_splits(
            df_full, protein_to_idx,
            train_data, val_data, test_data,
            neg_ratio=args.neg_ratio,
            random_state=d['random_state'],
        )
        num_genes = len(gene_to_idx)
        print(f"  Genes: {num_genes:,}  |  gene-isoform pos_weight (auto): {gene_iso_ratio:.1f}")
        gene_train_loader = DataLoader(GeneIsoformDataset(train_g),
                                       batch_size=args.gene_batch_size, shuffle=True,
                                       num_workers=t['num_workers'])

        if args.lambda_complex > 0:
            print("\nStep 2c: Building gene–gene STRING interaction data...")
            complex_train_t, _, _, gene_to_idx, gene_gene_ratio = prepare_gene_gene_splits(
                gene_to_idx, train_data, val_data, test_data,
            )
            num_genes = len(gene_to_idx)
            print(f"  Total genes (incl. STRING-only): {num_genes:,}  |  "
                  f"gene-gene pos_weight (auto): {gene_gene_ratio:.1f}")
            complex_train_loader = DataLoader(
                GeneGeneDataset(complex_train_t),
                batch_size=args.complex_batch_size, shuffle=True,
                num_workers=t['num_workers'],
            )

    # =========================================================================
    # 3. Model
    # =========================================================================
    print("\nStep 3: Initialising model...")

    esmc_features = None
    if args.model_type == 'multimodal' and args.esmc:
        print(f"  Loading ESM-C embeddings from {args.esmc} ...")
        esmc_df = pd.read_csv(args.esmc).set_index('ENSP')
        ordered_ensp = sorted(protein_to_idx, key=protein_to_idx.get)
        esmc_features = torch.tensor(
            esmc_df.reindex(ordered_ensp).fillna(0.0).values,
            dtype=torch.float32,
        )
        n_missing = (esmc_df.reindex(ordered_ensp).isna().any(axis=1)).sum()
        print(f"  ESM-C: {esmc_features.shape[1]}-dim features for {len(ordered_ensp):,} proteins "
              f"({n_missing:,} missing → zero-filled)")
        n_unseen = len(val_proteins) + len(test_proteins)
        print(f"  Inductive: {n_unseen:,} unseen isoforms will use ESM-C projection only (residual=0)")

    if args.model_type == 'ldm':
        if not args.esmc:
            print("  WARNING: LDM without ESM-C has no inductive capacity — "
                  "unseen isoform embeddings remain at random initialisation.")
        model = LatentDistanceModel(
            num_proteins=num_proteins,
            latent_dim=m['latent_dim'],
            distance_metric=m['distance_metric'],
        )
    else:
        model = MultimodalLDM(
            num_proteins=num_proteins,
            num_genes=num_genes,
            latent_dim=m['latent_dim'],
            distance_metric=m['distance_metric'],
            esmc_features=esmc_features,
        )
        print(f"  Gene embeddings: {num_genes:,} × {m['latent_dim']}")
        if esmc_features is not None:
            print(f"  Isoform positions: ESM-C proj ({esmc_features.shape[1]}→{m['latent_dim']}) + residual")
            # NOTE: projection is NOT frozen in inductive mode.
            # Freezing it (as in the transductive script) causes residuals to overfit
            # training isoforms while unseen isoforms are stuck at the fixed prior.
            # An unfrozen projection learns a generalising encoder that applies
            # equally to seen and unseen isoforms at inference time.

            gene_to_isoforms = {}
            if 'df_full' in dir():
                for g, iso in zip(df_full['gene_1'], df_full['ensp_1']):
                    gene_to_isoforms.setdefault(g, set()).add(iso)
                for g, iso in zip(df_full['gene_2'], df_full['ensp_2']):
                    gene_to_isoforms.setdefault(g, set()).add(iso)
            model.init_gene_centroids(gene_to_idx, gene_to_isoforms, protein_to_idx)
        else:
            print(f"  Isoform positions: learned embeddings ({num_proteins:,} × {m['latent_dim']})")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Latent dim: {m['latent_dim']}  |  Metric: {m['distance_metric']}  |  "
          f"Params: {n_trainable:,} trainable / {n_total:,} total")

    # =========================================================================
    # 4. Train
    # =========================================================================
    print("\nStep 4: Training...")
    if args.model_type == 'ldm':
        trainer = LatentDistanceTrainer(model, device=device)
        best_ap = trainer.train(
            train_loader, val_loader,
            epochs=t['epochs'], lr=t['learning_rate'],
            weight_decay=t['weight_decay'], pos_weight=neg_pos_ratio,
        )
    else:
        trainer = MultimodalTrainer(model, device=device)
        best_ap = trainer.train(
            iso_train_loader     = train_loader,
            gene_train_loader    = gene_train_loader,
            complex_train_loader = complex_train_loader or gene_train_loader,
            val_loader           = val_loader,
            epochs               = t['epochs'],
            lr                   = t['learning_rate'],
            weight_decay         = t['weight_decay'],
            iso_pos_weight       = neg_pos_ratio,
            lambda_iso           = args.lambda_iso,
            gene_pos_weight      = gene_iso_ratio,
            lambda_gene          = args.lambda_gene,
            complex_pos_weight   = gene_gene_ratio,
            lambda_complex       = args.lambda_complex,
        )

    # =========================================================================
    # 5. Save dir
    # =========================================================================
    if args.model_type == 'ldm':
        save_dir = (f"models/INDUCTIVE_LDM_dim={m['latent_dim']}_metric={m['distance_metric']}"
                    f"_epochs={t['epochs']}_lr={t['learning_rate']}_BS={t['batch_size']}")
    else:
        save_dir = (f"models/INDUCTIVE_MM_dim={m['latent_dim']}_metric={m['distance_metric']}"
                    f"_epochs={t['epochs']}_lr={t['learning_rate']}_BS={t['batch_size']}"
                    f"_lIso={args.lambda_iso}_lGene={args.lambda_gene}_negR={args.neg_ratio}")
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================================
    # 6. Training curves
    # =========================================================================
    print("\nStep 5: Saving training curves...")
    fig = trainer.plot_training()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/training_curves.png")

    # =========================================================================
    # 7. Evaluate on test set
    # =========================================================================
    print("\nStep 6: Evaluating on test set...")
    auc, ap, f1, _, _ = evaluate_model(model, test_loader, device=device, save_dir=save_dir)

    # =========================================================================
    # 8. Save checkpoint
    # =========================================================================
    print("\nStep 7: Saving model...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'protein_to_idx':   protein_to_idx,
        'num_proteins':     num_proteins,
        'latent_dim':       m['latent_dim'],
        'distance_metric':  m['distance_metric'],
        'model_type':       args.model_type,
        'split_mode':       'inductive_isoform',
        'train_proteins':   list(train_proteins),
        'val_proteins':     list(val_proteins),
        'test_proteins':    list(test_proteins),
        'test_auc': auc, 'test_ap': ap, 'test_f1': f1,
    }
    if args.model_type == 'multimodal':
        checkpoint.update({
            'gene_to_idx':    gene_to_idx,
            'num_genes':      num_genes,
            'lambda_iso':     args.lambda_iso,
            'lambda_gene':    args.lambda_gene,
            'neg_ratio':      args.neg_ratio,
            'lambda_complex': args.lambda_complex,
            'esmc_path':      args.esmc or None,
        })
        ckpt_name = 'multimodal_ldm_inductive.pt'
    else:
        ckpt_name = 'latent_distance_model_inductive.pt'

    torch.save(checkpoint, f"{save_dir}/{ckpt_name}")
    print(f"  Saved: {save_dir}/{ckpt_name}")

    print(f"\n{'='*70}\nTRAINING COMPLETE (INDUCTIVE)")
    print(f"Best Val AP: {best_ap:.4f}  |  Test AUC: {auc:.4f}  |  AP: {ap:.4f}")
    print(f"Saved to: {save_dir}\n{'='*70}\n")


if __name__ == "__main__":
    main()
