import argparse
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_scripts.gene_isoform_pairs import (
    GeneIsoformDataset, prepare_gene_isoform_splits,
)
from src.data_scripts.gene_pairs import GeneGeneDataset, prepare_gene_gene_splits
from src.data_scripts.isoform_pairs import (
    ProteinInteractionDataset,
    diagnose_split, diagnose_split_inductive,
    load_and_prepare_data, load_and_prepare_data_inductive,
)
from src.model_classes.ldm import LatentDistanceModel, LatentDistanceTrainer
from src.model_classes.mm_ldm import MultimodalLDM, MultimodalTrainer
from src.training.evaluate import (
    evaluate_gene_gene, evaluate_inductive_model_separately, evaluate_model,
)


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_esmc_features(path, protein_to_idx):
    """Read ESM-C CSV and align rows to protein_to_idx; missing rows zero-filled."""
    print(f"  Loading ESM-C embeddings from {path} ...")
    esmc_df      = pd.read_csv(path).set_index('ENSP')
    ordered_ensp = sorted(protein_to_idx, key=protein_to_idx.get)
    aligned      = esmc_df.reindex(ordered_ensp)
    n_missing    = aligned.isna().any(axis=1).sum()
    print(f"  ESM-C: {aligned.shape[1]}-dim for {len(ordered_ensp):,} proteins "
          f"({n_missing:,} missing → zero-filled)")
    return torch.tensor(aligned.fillna(0.0).values, dtype=torch.float32)


def build_gene_to_isoforms(ref_data):
    mapping = {}
    for g_col, i_col in (('gene_1', 'ensp_1'), ('gene_2', 'ensp_2')):
        for g, iso in zip(ref_data[g_col], ref_data[i_col]):
            mapping.setdefault(g, set()).add(iso)
    return mapping


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='config/config.yaml', help='YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d  = cfg['data']
    m  = cfg['model']
    t  = cfg['training']
    sp = cfg['split']
    mm = cfg.get('multimodal', {})

    split_mode   = sp['mode']        # 'transductive' | 'inductive'
    model_type   = m['type']         # 'ldm' | 'multimodal'
    is_inductive = split_mode == 'inductive'

    device = pick_device()
    bs, nw = t['batch_size'], t['num_workers']

    print(f"\n--- Training {'multimodal ' if model_type == 'multimodal' else ''}LDM ---")
    print(f"Device: {device}", end='')
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    else:
        print()
    print(f"Split: {split_mode}")
    if model_type == 'multimodal':
        print(f"λ_iso_iso={mm['lambda_iso_iso']}  λ_gene_iso={mm['lambda_gene_iso']}  "
              f"λ_gene_gene={mm['lambda_gene_gene']}  neg_ratio={mm['neg_ratio']}")

    # 1. Iso–iso data
    print("\nStep 1: Loading data...")
    if split_mode == 'transductive':
        (train_dataset, train_data, val_data, test_data,
         protein_to_idx, num_proteins, neg_pos_ratio) = load_and_prepare_data(
            d['path'], test_size=sp['test_fraction'],
            val_size=sp['val_fraction'], random_state=d['random_state'])
        train_proteins = val_proteins = test_proteins = None
        held_out_isoforms = set()
        print(f"\n  Proteins: {num_proteins:,}  |  Train: {len(train_dataset):,}  "
              f"Val: {len(val_data):,}  Test: {len(test_data):,}")
        diagnose_split(train_dataset, val_data, test_data)
    elif is_inductive:
        (train_dataset, train_data, val_data, test_data,
         protein_to_idx, num_proteins, neg_pos_ratio,
         train_proteins, val_proteins, test_proteins) = load_and_prepare_data_inductive(
            d['path'], test_size=sp['test_fraction'], val_size=sp['val_fraction'],
            random_state=d['random_state'])
        held_out_isoforms = val_proteins | test_proteins
        print(f"\n  Isoforms: {num_proteins:,}  |  Train pairs: {len(train_dataset):,}  "
              f"Val: {len(val_data):,}  Test: {len(test_data):,}")
        diagnose_split_inductive(train_data, val_data, test_data,
                                 train_proteins, val_proteins, test_proteins)
    else:
        raise ValueError(f"Unknown split mode: {split_mode!r}. "
                         "Use 'transductive' or 'inductive'.")

    # 2. Dataloaders
    print("Step 2: Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader   = DataLoader(ProteinInteractionDataset(val_data, protein_to_idx),
                              batch_size=bs, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(ProteinInteractionDataset(test_data, protein_to_idx),
                              batch_size=bs, shuffle=False, num_workers=nw)

    gene_iso_loader = gene_gene_loader = gene_gene_testloader = None
    gene_iso_ratio  = gene_gene_ratio  = 5.0
    num_genes       = 0
    gene_to_idx     = {}
    df_full         = None

    if model_type == 'multimodal':
        df_full  = pd.read_csv(d['path'])
        mm_batch = mm.get('batch_size', bs)

        print("\nStep 2b: Building gene–isoform bipartite graph...")
        gene_to_idx, train_gi, _, _, gene_iso_ratio = prepare_gene_isoform_splits(
            df_full, protein_to_idx, train_data, val_data, test_data,
            neg_ratio=mm['neg_ratio'], random_state=d['random_state'],
            inductive=is_inductive,
            held_out_isoforms=held_out_isoforms if is_inductive else None)
        num_genes = len(gene_to_idx)
        print(f"  Genes: {num_genes:,}  |  gene–iso pos_weight: {gene_iso_ratio:.1f}")
        gene_iso_loader = DataLoader(GeneIsoformDataset(train_gi),
                                     batch_size=mm_batch, shuffle=True, num_workers=nw)

        if mm.get('lambda_gene_gene', 0) > 0:
            print("\nStep 2c: Building gene–gene STRING data...")
            gg_train, _, gg_test, gene_to_idx, gene_gene_ratio = prepare_gene_gene_splits(
                gene_to_idx, train_data, val_data, test_data, inductive=is_inductive)
            num_genes = len(gene_to_idx)
            print(f"  Total genes: {num_genes:,}  |  gene–gene pos_weight: {gene_gene_ratio:.1f}")
            gene_gene_loader     = DataLoader(GeneGeneDataset(gg_train),
                                              batch_size=mm_batch, shuffle=True, num_workers=nw)
            gene_gene_testloader = DataLoader(GeneGeneDataset(gg_test),
                                              batch_size=mm_batch, shuffle=True, num_workers=nw)

    # 3. Model
    print("\nStep 3: Initialising model...")
    esmc_features = None
    if model_type == 'multimodal' and mm.get('esmc_path'):
        esmc_features = load_esmc_features(mm['esmc_path'], protein_to_idx)
        if is_inductive:
            print(f"  {len(held_out_isoforms):,} held-out isoforms: ESM-C projection only "
                  f"(residual trains to zero for unseen)")

    if model_type == 'ldm':
        model = LatentDistanceModel(num_proteins=num_proteins, latent_dim=m['latent_dim'])
        use_residuals = None
    else:
        use_residuals   = mm.get('use_residuals', False)
        proj_hidden_dim = mm.get('proj_hidden_dim', 16)
        model = MultimodalLDM(
            num_proteins=num_proteins, num_genes=num_genes,
            latent_dim=m['latent_dim'], esmc_features=esmc_features,
            use_residuals=use_residuals, proj_hidden_dim=proj_hidden_dim)
        print(f"  Gene embeddings: {num_genes:,} × {m['latent_dim']}")
        if esmc_features is not None:
            print(f"  Isoform positions: ESM-C proj "
                  f"({esmc_features.shape[1]}→{proj_hidden_dim}→{m['latent_dim']})"
                  + (" + residual" if use_residuals else ""))
            print(f"  Random effects: re_head({esmc_features.shape[1]}→{proj_hidden_dim}→1)"
                  + (" + re_residual" if use_residuals else ""))
            # Restrict gene→isoform map to train isoforms in inductive mode
            # to avoid leaking held-out isoform positions into gene embeddings.
            ref_data = train_data if is_inductive else df_full
            model.init_gene_centroids(gene_to_idx,
                                      build_gene_to_isoforms(ref_data),
                                      protein_to_idx)
        else:
            print(f"  Isoform positions: learned embeddings ({num_proteins:,} × {m['latent_dim']})")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Latent dim: {m['latent_dim']}  |  "
          f"Params: {n_trainable:,} trainable / {n_total:,} total")

    # 4. Train
    print("\nStep 4: Training...")
    patience = t.get('patience', 10)

    if model_type == 'ldm':
        trainer = LatentDistanceTrainer(model, device=device)
        best_ap = trainer.train(
            train_loader, val_loader,
            epochs=t['epochs'], lr=t['learning_rate'],
            weight_decay=t['weight_decay'], pos_weight=neg_pos_ratio,
            patience=patience)
    else:
        trainer = MultimodalTrainer(model, device=device)
        # When λ_gene_gene=0 the gene_gene loader is unused; pass gene_iso_loader
        # as a harmless dummy so the trainer's signature stays simple.
        best_ap = trainer.train(
            iso_iso_loader      = train_loader,
            gene_iso_loader     = gene_iso_loader,
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

    # 5. Save dir + config snapshot (so hyperparameters always travel with the checkpoint)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    split_tag = {'transductive': 'TRANS', 'inductive': 'IND'}[split_mode]
    model_tag = 'LDM' if model_type == 'ldm' else 'MM'
    models_dir = cfg.get('paths', {}).get('models_dir', 'models')
    save_dir = f"{models_dir}/{split_tag}_{model_tag}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.config, f"{save_dir}/config.yaml")

    # 6. Training curves
    print("\nStep 5: Saving training curves...")
    trainer.plot_training()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/training_curves.png")

    # 7. Evaluate
    print("\nStep 6: Evaluating on test set...")
    auc, ap, _, _ = evaluate_model(model, test_loader, device=device, save_dir=save_dir)
    if is_inductive:
        evaluate_inductive_model_separately(
            model, test_data, test_proteins, protein_to_idx,
            batch_size=bs, num_workers=nw, device=device, save_dir=save_dir)
    if model_type == 'multimodal' and mm.get('lambda_gene_gene', 0) > 0:
        evaluate_gene_gene(model, gene_gene_loader=gene_gene_testloader,
                           device=device, save_dir=save_dir)

    # 8. Checkpoint
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
    if is_inductive:
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

    torch.save(checkpoint, f"{save_dir}/model.pt")
    print(f"  Saved: {save_dir}/model.pt")

    print("\n--- Training complete ---")
    print(f"Best val AP: {best_ap:.4f}  |  Test AUC: {auc:.4f}  |  AP: {ap:.4f}")
    print(f"Saved to: {save_dir}\n")


if __name__ == "__main__":
    main()
