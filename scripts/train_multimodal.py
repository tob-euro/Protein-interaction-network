import argparse
import os
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model_classes.multimodal_ldm import MultimodalLDM, MultimodalTrainer
from src.data_scripts.dataset import ProteinInteractionDataset, load_and_prepare_data
from src.data_scripts.bipartite_dataset import GeneMembershipDataset, prepare_bipartite_splits
from src.training.evaluate import evaluate_model


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # shared with train.py
    parser.add_argument('--config',          type=str,   default='config/config.yaml')
    parser.add_argument('--data',            type=str,   default=None)
    parser.add_argument('--latent-dim',      type=int,   default=None)
    parser.add_argument('--distance-metric', type=str,   default=None, choices=['euclidean', 'cosine'])
    parser.add_argument('--epochs',          type=int,   default=None)
    parser.add_argument('--lr',              type=float, default=None)
    parser.add_argument('--batch-size',      type=int,   default=None)
    parser.add_argument('--weight-decay',    type=float, default=None)
    # multimodal-specific
    parser.add_argument('--lambda-iso',      type=float, default=1.0,
                        help='Loss weight for isoform–isoform component')
    parser.add_argument('--lambda-gene',     type=float, default=0.5,
                        help='Loss weight for gene–isoform bipartite component')
    parser.add_argument('--gene-pos-weight', type=float, default=5.0,
                        help='BCE pos_weight for gene–isoform loss')
    parser.add_argument('--neg-ratio',       type=int,   default=5,
                        help='Negative samples per positive gene–isoform edge')
    parser.add_argument('--gene-batch-size', type=int,   default=512,
                        help='Batch size for gene–isoform loader')
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


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args)

    d = cfg['data']
    m = cfg['model']
    t = cfg['training']

    # =========================================================================
    # Device
    # =========================================================================
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n{'='*70}\nTRAINING MULTIMODAL LDM\n{'='*70}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"λ_iso: {args.lambda_iso}   λ_gene: {args.lambda_gene}   neg_ratio (gene): {args.neg_ratio}")
    print(f"{'='*70}\n")

    # =========================================================================
    # 1. Load isoform-pair data  (same as train.py)
    # =========================================================================
    print('Step 1: Loading isoform-pair data...')
    train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio = load_and_prepare_data(
        d['path'], test_size=d['test_size'], val_size=d['val_size'], random_state=d['random_state'])
    print(f'\n  Proteins: {num_proteins}  |  '
          f'Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}')

    # =========================================================================
    # 2. Build bipartite gene–isoform splits
    # =========================================================================
    print('\nStep 2: Building gene–isoform bipartite graph...')
    df_full = pd.read_csv(d['path'])

    gene_to_idx, train_gene_triples, val_gene_triples, test_gene_triples = prepare_bipartite_splits(
        df_full, protein_to_idx,
        train_data, val_data, test_data,
        neg_ratio=args.neg_ratio,
        random_state=d['random_state'],
    )
    num_genes = len(gene_to_idx)
    print(f'\n  Genes: {num_genes}')

    # =========================================================================
    # 3. Dataloaders
    # =========================================================================
    print('\nStep 3: Creating dataloaders...')
    iso_train_loader = DataLoader(ProteinInteractionDataset(train_data, protein_to_idx),
                                  batch_size=t['batch_size'], shuffle=True,
                                  num_workers=t['num_workers'], pin_memory=True)
    iso_val_loader   = DataLoader(ProteinInteractionDataset(val_data, protein_to_idx),
                                  batch_size=t['batch_size'], shuffle=False,
                                  num_workers=t['num_workers'], pin_memory=True)
    iso_test_loader  = DataLoader(ProteinInteractionDataset(test_data, protein_to_idx),
                                  batch_size=t['batch_size'], shuffle=False,
                                  num_workers=t['num_workers'], pin_memory=True)

    gene_train_loader = DataLoader(GeneMembershipDataset(train_gene_triples),
                                   batch_size=args.gene_batch_size, shuffle=True,
                                   num_workers=t['num_workers'], pin_memory=True)
    gene_val_loader   = DataLoader(GeneMembershipDataset(val_gene_triples),
                                   batch_size=args.gene_batch_size, shuffle=False,
                                   num_workers=t['num_workers'], pin_memory=True)

    # =========================================================================
    # 4. Model
    # =========================================================================
    print('\nStep 4: Initialising MultimodalLDM...')
    model = MultimodalLDM(
        num_proteins    = num_proteins,
        num_genes       = num_genes,
        latent_dim      = m['latent_dim'],
        distance_metric = m['distance_metric'],
    )
    print(f"  Latent dim: {m['latent_dim']}  |  Metric: {m['distance_metric']}  |  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Isoform embeddings: {num_proteins} × {m['latent_dim']}")
    print(f"  Gene embeddings:    {num_genes}    × {m['latent_dim']}")

    # =========================================================================
    # 5. Train
    # =========================================================================
    print('\nStep 5: Training...')
    trainer = MultimodalTrainer(model, device=device)
    best_f1 = trainer.train(
        iso_train_loader  = iso_train_loader,
        gene_train_loader = gene_train_loader,
        val_loader        = iso_val_loader,
        epochs            = t['epochs'],
        lr                = t['learning_rate'],
        weight_decay      = t['weight_decay'],
        iso_pos_weight    = neg_pos_ratio,
        lambda_iso        = args.lambda_iso,
        gene_pos_weight   = args.gene_pos_weight,
        lambda_gene       = args.lambda_gene,
    )

    save_dir = (f"models/MM_dim={m['latent_dim']}_metric={m['distance_metric']}"
                f"_epochs={t['epochs']}_lr={t['learning_rate']}"
                f"_BS={t['batch_size']}"
                f"_lIso={args.lambda_iso}_lGene={args.lambda_gene}"
                f"_negR={args.neg_ratio}")
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================================
    # 6. Training curves
    # =========================================================================
    print('\nStep 6: Saving training curves...')
    fig = trainer.plot_training()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_dir}/training_curves.png')

    # =========================================================================
    # 7. Evaluate on isoform test set
    # =========================================================================
    print('\nStep 7: Evaluating on isoform test set...')
    auc, ap, f1, _, _ = evaluate_model(model, iso_test_loader, device=device, save_dir=save_dir)

    # =========================================================================
    # 8. Save checkpoint
    # =========================================================================
    print('\nStep 8: Saving model...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'protein_to_idx':   protein_to_idx,
        'gene_to_idx':      gene_to_idx,
        'num_proteins':     num_proteins,
        'num_genes':        num_genes,
        'latent_dim':       m['latent_dim'],
        'distance_metric':  m['distance_metric'],
        'lambda_iso':       args.lambda_iso,
        'lambda_gene':      args.lambda_gene,
        'neg_ratio':        args.neg_ratio,
        'test_auc': auc, 'test_ap': ap, 'test_f1': f1,
    }, f'{save_dir}/multimodal_ldm.pt')
    print(f'  Saved: {save_dir}/multimodal_ldm.pt')

    print(f"\n{'='*70}\nTRAINING COMPLETE")
    print(f'Best Val F1: {best_f1:.4f}  |  Test AUC: {auc:.4f}  |  AP: {ap:.4f}')
    print(f"Saved to: {save_dir}\n{'='*70}\n")


if __name__ == '__main__':
    main()