"""
train.py — Unified training script for LDM and MultimodalLDM.

All configuration is loaded from config/config.yaml (or a custom config file).

Split modes (config: split.mode):
  transductive:    gene-pair level split — all isoforms are seen during training.
                   Validation/test pairs involve unseen gene-pair combinations.
  inductive:       isoform-level split — a fraction of isoforms are held out
                   entirely from isoform-pair training. Gene–isoform and gene–gene
                   data always goes entirely to training (prior knowledge).
  inductive_gene:  gene-level split — genes (not individual isoforms) are
                   partitioned so all isoforms of a gene are always in the same
                   split, eliminating within-gene data leakage.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/my_config.yaml
"""
import argparse
import os
import shutil
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader

from src.model_classes.ldm import LatentDistanceModel, LatentDistanceTrainer
from src.model_classes.mm_ldm import MultimodalLDM, MultimodalTrainer
from src.data_scripts.isoform_pairs import (
    ProteinInteractionDataset,
    load_and_prepare_data,
    load_and_prepare_data_inductive,
    diagnose_split,
    diagnose_split_inductive,
)
from src.data_scripts.gene_isoform_pairs import GeneIsoformDataset, prepare_gene_isoform_splits
from src.data_scripts.gene_pairs import GeneGeneDataset, prepare_gene_gene_splits
from src.training.evaluate import evaluate_model, evaluate_inductive_model_separately, evaluate_gene_gene


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d  = cfg['data']
    m  = cfg['model']
    t  = cfg['training']
    sp = cfg['split']
    mm = cfg.get('multimodal', {})

    split_mode    = sp['mode']           # 'transductive' | 'inductive' | 'inductive_gene'
    model_type    = m['type']            # 'ldm' | 'multimodal'
    val_fraction  = sp['val_fraction']
    test_fraction = sp['test_fraction']

    device = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"TRAINING — {'MULTIMODAL ' if model_type == 'multimodal' else ''}LATENT DISTANCE MODEL")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Split: {split_mode}")
    if model_type == 'multimodal':
        print(f"λ_iso_iso={mm['lambda_iso_iso']}  λ_gene_iso={mm['lambda_gene_iso']}  "
              f"λ_gene_gene={mm['lambda_gene_gene']}  neg_ratio={mm['neg_ratio']}")
    print(f"{'='*70}\n")

    # =========================================================================
    # 1. Load isoform-pair data
    # =========================================================================
    print("Step 1: Loading data...")

    if split_mode == 'transductive':
        (train_dataset, train_data, val_data, test_data,
         protein_to_idx, num_proteins, neg_pos_ratio) = load_and_prepare_data(
            d['path'],
            test_size=test_fraction,
            val_size=val_fraction,
            random_state=d['random_state'],
        )
        train_proteins = val_proteins = test_proteins = None
        held_out_isoforms = set()
        print(f"\n  Proteins: {num_proteins:,}  |  "
              f"Train: {len(train_dataset):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")
        diagnose_split(train_dataset, val_data, test_data)

    elif split_mode in ('inductive', 'inductive_gene'):
        (train_dataset, train_data, val_data, test_data,
         protein_to_idx, num_proteins, neg_pos_ratio,
         train_proteins, val_proteins, test_proteins) = load_and_prepare_data_inductive(
            d['path'],
            test_size=test_fraction,
            val_size=val_fraction,
            random_state=d['random_state'],
            gene_level=(split_mode == 'inductive_gene'),
        )
        held_out_isoforms = val_proteins | test_proteins
        print(f"\n  Isoforms: {num_proteins:,}  |  "
              f"Train pairs: {len(train_dataset):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")
        diagnose_split_inductive(train_data, val_data, test_data,
                                  train_proteins, val_proteins, test_proteins)
    else:
        raise ValueError(f"Unknown split mode: {split_mode!r}. "
                         f"Use 'transductive', 'inductive', or 'inductive_gene'.")

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

    gene_iso_loader  = None
    gene_gene_loader = None
    gene_iso_ratio   = 5.0
    gene_gene_ratio  = 5.0
    num_genes            = 0
    gene_to_idx          = {}
    df_full              = None

    if model_type == 'multimodal':
        df_full  = pd.read_csv(d['path'])
        mm_batch = mm.get('batch_size', t['batch_size'])

        # ── Gene–isoform bipartite ──────────────────────────────────────────
        print("\nStep 2b: Building gene–isoform bipartite graph...")
        is_inductive = split_mode in ('inductive', 'inductive_gene')
        gene_to_idx, train_g, _, _, gene_iso_ratio = prepare_gene_isoform_splits(
            df_full, protein_to_idx,
            train_data, val_data, test_data,
            neg_ratio=mm['neg_ratio'],
            random_state=d['random_state'],
            inductive=is_inductive,
            held_out_isoforms=held_out_isoforms if is_inductive else None,
        )
        num_genes = len(gene_to_idx)
        print(f"  Genes: {num_genes:,}  |  gene–iso pos_weight (auto): {gene_iso_ratio:.1f}")
        gene_iso_loader = DataLoader(GeneIsoformDataset(train_g),
                                     batch_size=mm_batch, shuffle=True,
                                     num_workers=t['num_workers'])

        # ── Gene–gene STRING ────────────────────────────────────────────────
        if mm.get('lambda_gene_gene', 0) > 0:
            print("\nStep 2c: Building gene–gene STRING interaction data...")
            gene_gene_train, gene_gene_val, gene_gene_test, gene_to_idx, gene_gene_ratio = prepare_gene_gene_splits(
                gene_to_idx, train_data, val_data, test_data,
                inductive=is_inductive,
            )
            num_genes = len(gene_to_idx)
            print(f"  Total genes (incl. STRING-only): {num_genes:,}  |  "
                  f"gene–gene pos_weight (auto): {gene_gene_ratio:.1f}")
            gene_gene_loader = DataLoader(
                GeneGeneDataset(gene_gene_train),
                batch_size=mm_batch, shuffle=True,
                num_workers=t['num_workers'],
            )
            gene_gene_testloader = DataLoader(
                GeneGeneDataset(gene_gene_test),
                batch_size=mm_batch, shuffle=True,
                num_workers=t['num_workers'],
            )

    # =========================================================================
    # 3. Model
    # =========================================================================
    print("\nStep 3: Initialising model...")

    esmc_features = None
    if model_type == 'multimodal':
        esmc_path = mm.get('esmc_path', '')
        if esmc_path:
            print(f"  Loading ESM-C embeddings from {esmc_path} ...")
            esmc_df      = pd.read_csv(esmc_path).set_index('ENSP')
            ordered_ensp = sorted(protein_to_idx, key=protein_to_idx.get)
            esmc_features = torch.tensor(
                esmc_df.reindex(ordered_ensp).fillna(0.0).values,
                dtype=torch.float32,
            )
            n_missing = (esmc_df.reindex(ordered_ensp).isna().any(axis=1)).sum()
            print(f"  ESM-C: {esmc_features.shape[1]}-dim features for {len(ordered_ensp):,} proteins "
                  f"({n_missing:,} missing → zero-filled)")
            if split_mode in ('inductive', 'inductive_gene'):
                print(f"  {len(held_out_isoforms):,} held-out isoforms: positions from ESM-C projection + zero residual (residual trains to zero for unseen)")

    if model_type == 'ldm':
        model = LatentDistanceModel(
            num_proteins=num_proteins,
            latent_dim=m['latent_dim'],
        )
    else:
        use_residuals   = mm.get('use_residuals', False)
        proj_hidden_dim = mm.get('proj_hidden_dim', 16)
        model = MultimodalLDM(
            num_proteins    = num_proteins,
            num_genes       = num_genes,
            latent_dim      = m['latent_dim'],
            esmc_features   = esmc_features,
            use_residuals   = use_residuals,
            proj_hidden_dim = proj_hidden_dim,
        )
        print(f"  Gene embeddings: {num_genes:,} × {m['latent_dim']}")
        if esmc_features is not None:
            print(f"  Isoform positions: ESM-C proj ({esmc_features.shape[1]}→{proj_hidden_dim}→{m['latent_dim']})"
                  + (" + residual" if use_residuals else ""))
            print(f"  Random effects: re_head({esmc_features.shape[1]}→{proj_hidden_dim}→1)"
                  + (" + re_residual" if use_residuals else ""))
            # Build gene→isoform map restricted to train isoforms in inductive mode
            # to avoid leaking held-out isoform positions into gene embeddings.
            ref_data = train_data if split_mode in ('inductive', 'inductive_gene') else df_full
            gene_to_isoforms = {}
            for g, iso in zip(ref_data['gene_1'], ref_data['ensp_1']):
                gene_to_isoforms.setdefault(g, set()).add(iso)
            for g, iso in zip(ref_data['gene_2'], ref_data['ensp_2']):
                gene_to_isoforms.setdefault(g, set()).add(iso)
            model.init_gene_centroids(gene_to_idx, gene_to_isoforms, protein_to_idx)
        else:
            print(f"  Isoform positions: learned embeddings ({num_proteins:,} × {m['latent_dim']})")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Latent dim: {m['latent_dim']}  |  Params: {n_trainable:,} trainable / {n_total:,} total")

    # =========================================================================
    # 4. Train
    # =========================================================================
    print("\nStep 4: Training...")
    patience = t.get('patience', 10)

    if model_type == 'ldm':
        trainer = LatentDistanceTrainer(model, device=device)
        best_ap = trainer.train(
            train_loader, val_loader,
            epochs=t['epochs'], lr=t['learning_rate'],
            weight_decay=t['weight_decay'], pos_weight=neg_pos_ratio,
            patience=patience,
        )
    else:
        trainer = MultimodalTrainer(model, device=device)
        best_ap = trainer.train(
            iso_iso_loader      = train_loader,
            gene_iso_loader     = gene_iso_loader,
            # When lambda_gene_gene=0, gene_gene loader is unused but trainer always expects one.
            # Pass gene_iso_loader as a harmless dummy in that case.
            gene_gene_loader    = gene_gene_loader or gene_iso_loader,
            val_loader          = val_loader,
            epochs              = t['epochs'],
            lr                  = t['learning_rate'],
            weight_decay        = t['weight_decay'],
            iso_iso_pos_weight  = neg_pos_ratio,
            lambda_iso_iso      = mm['lambda_iso_iso'],
            gene_iso_pos_weight = gene_iso_ratio,
            lambda_gene_iso     = mm['lambda_gene_iso'],
            gene_gene_pos_weight= gene_gene_ratio,
            lambda_gene_gene    = mm.get('lambda_gene_gene', 0.0),
            patience            = patience,
        )

    # =========================================================================
    # 5. Save dir — timestamp + config snapshot so hyperparameters are always recorded
    # =========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    split_tag = ('TRANS' if split_mode == 'transductive' else
                 'IND_G' if split_mode == 'inductive_gene' else 'IND')
    model_tag = 'LDM' if model_type == 'ldm' else 'MM'
    save_dir  = f"models/{split_tag}_{model_tag}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Copy config so the exact hyperparameters used are always available alongside the checkpoint
    shutil.copy(args.config, f"{save_dir}/config.yaml")

    # =========================================================================
    # 6. Training curves
    # =========================================================================
    print("\nStep 5: Saving training curves...")
    trainer.plot_training()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/training_curves.png")

    # =========================================================================
    # 7. Evaluate on test set
    # =========================================================================
    print("\nStep 6: Evaluating on test set...")
    auc, ap, _, _ = evaluate_model(model, test_loader, device=device, save_dir=save_dir)
    if split_mode in ('inductive', 'inductive_gene'):
        auc1, ap1, auc2, ap2 = evaluate_inductive_model_separately(model, test_data, test_proteins, protein_to_idx, batch_size=t['batch_size'], num_workers=t['num_workers'], device=device, save_dir=save_dir)
    if model_type == 'multimodal' and mm.get('lambda_gene_gene', 0) > 0:
        evaluate_gene_gene(model, gene_gene_loader=gene_gene_testloader, device=device, save_dir=save_dir)

    # =========================================================================
    # 8. Save checkpoint
    # =========================================================================
    print("\nStep 7: Saving model...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'protein_to_idx':   protein_to_idx,
        'num_proteins':     num_proteins,
        'latent_dim':       m['latent_dim'],
        'model_type':       model_type,
        'split_mode':       split_mode,
        'test_auc':         auc,
        'test_ap':          ap,
    }
    if split_mode in ('inductive', 'inductive_gene'):
        checkpoint.update({
            'train_proteins': list(train_proteins),
            'val_proteins':   list(val_proteins),
            'test_proteins':  list(test_proteins),
        })
    if model_type == 'multimodal':
        checkpoint.update({
            'gene_to_idx':      gene_to_idx,
            'num_genes':        num_genes,
            'use_residuals':    use_residuals,
            'lambda_iso_iso':   mm['lambda_iso_iso'],
            'lambda_gene_iso':  mm['lambda_gene_iso'],
            'neg_ratio':        mm['neg_ratio'],
            'lambda_gene_gene': mm.get('lambda_gene_gene', 0.0),
        })
        
    ckpt_name = "model.pt"
    torch.save(checkpoint, f"{save_dir}/{ckpt_name}")
    print(f"  Saved: {save_dir}/{ckpt_name}")

    print(f"\n{'='*70}\nTRAINING COMPLETE")
    print(f"Best Val AP: {best_ap:.4f}  |  Test AUC: {auc:.4f}  |  AP: {ap:.4f}")
    print(f"Saved to: {save_dir}\n{'='*70}\n")


if __name__ == "__main__":
    main()
