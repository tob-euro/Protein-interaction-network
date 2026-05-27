"""Single-run training + evaluation logic shared by train.py and train_repeated.py."""

import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_scripts.gene_isoform_pairs import GeneIsoformDataset, prepare_gene_isoform_splits
from src.data_scripts.gene_pairs import GeneGeneDataset, prepare_gene_gene_splits
from src.data_scripts.isoform_pairs import (
    ProteinInteractionDataset,
    diagnose_split, diagnose_split_inductive,
    load_and_prepare_data, load_and_prepare_data_inductive, load_esmc_features,
)
from src.model_classes.ldm import LatentDistanceModel, LatentDistanceTrainer
from src.model_classes.mm_ldm import MultimodalLDM, MultimodalTrainer
from src.training.evaluate import (
    evaluate_gene_gene, evaluate_inductive_model_separately, evaluate_model,
)


def _build_split_key(d, mm, seed, model_type):
    """Build a dict that uniquely identifies a set of data splits.

    Includes the source CSV's mtime and size so stale cache entries are
    automatically missed when the data file changes.
    """
    iso_stat = os.stat(d['iso_path'])
    key = {
        'mode':          d['mode'],
        'seed':          int(seed),
        'val_fraction':  round(float(d['val_fraction']), 8),
        'test_fraction': round(float(d['test_fraction']), 8),
        'iso_path':      os.path.abspath(d['iso_path']),
        'iso_mtime':     iso_stat.st_mtime,
        'iso_size':      iso_stat.st_size,
    }
    if model_type == 'multimodal':
        key['neg_ratio']     = int(mm['neg_ratio'])
        key['has_gene_iso']  = mm.get('lambda_gene_iso',  0) > 0
        key['has_gene_gene'] = mm.get('lambda_gene_gene', 0) > 0
    return key


def _load_all_splits(d, mm, seed, model_type, is_inductive):
    """Read source CSVs, build all splits, and return a single cacheable dict.

    This is the expensive path: reads the iso-iso CSV, runs train_test_split,
    samples gene–isoform negatives, and (if active) loads STRING gene–gene data.
    The returned dict can be passed directly to SplitCache.put().
    """
    iso_path = d['iso_path']
    val_size = d['val_fraction']
    test_size = d['test_fraction']

    df_full = pd.read_csv(iso_path)

    # ── Iso-iso split ─────────────────────────────────────────────────────────
    if not is_inductive:
        (train_data, val_data, test_data,
         protein_to_idx, num_proteins, neg_pos_ratio) = load_and_prepare_data(
            df=df_full, test_size=test_size, val_size=val_size, random_state=seed)
        train_proteins = val_proteins = test_proteins = None
    else:
        (train_data, val_data, test_data,
         protein_to_idx, num_proteins, neg_pos_ratio,
         train_proteins, val_proteins, test_proteins) = load_and_prepare_data_inductive(
            df=df_full, test_size=test_size, val_size=val_size, random_state=seed)

    data = {
        'train_data':     train_data,
        'val_data':       val_data,
        'test_data':      test_data,
        'protein_to_idx': protein_to_idx,
        'num_proteins':   num_proteins,
        'neg_pos_ratio':  neg_pos_ratio,
        'train_proteins': train_proteins,
        'val_proteins':   val_proteins,
        'test_proteins':  test_proteins,
    }

    if model_type != 'multimodal':
        return data

    # ── Gene–isoform split ────────────────────────────────────────────────────
    neg_ratio = mm['neg_ratio']
    lambda_gi = mm.get('lambda_gene_iso', 0)
    lambda_gg = mm.get('lambda_gene_gene', 0)
    held_out  = (val_proteins | test_proteins) if is_inductive else None
    if lambda_gi > 0:
        gene_to_idx, train_gi, _, _, gene_iso_ratio = prepare_gene_isoform_splits(
            df_full, protein_to_idx, train_data, val_data, test_data,
            neg_ratio=neg_ratio, random_state=seed,
            inductive=is_inductive,
            held_out_isoforms=held_out)
        data.update({
            'gene_to_idx':    gene_to_idx,
            'train_gi':       train_gi,
            'gene_iso_ratio': gene_iso_ratio,
        })

    # ── Gene–gene split (STRING) ──────────────────────────────────────────────
    if lambda_gg > 0:
        gg_train, _, gg_test, gene_to_idx, gene_gene_ratio = prepare_gene_gene_splits(
            gene_to_idx, train_data, val_data, test_data, inductive=is_inductive)
        data.update({
            'gene_to_idx':      gene_to_idx,   # overwrite with STRING-expanded mapping
            'gg_train':         gg_train,
            'gg_test':          gg_test,
            'gene_gene_ratio':  gene_gene_ratio,
        })

    return data


def run_single(cfg, seed, config_path, split_cache=None, parent_dir=None):
    """Run one full training + evaluation with the given random seed.

    Both the data split and model weight initialisation are seeded with `seed`,
    so different seeds produce genuinely different runs.

    Args:
        cfg:          parsed YAML config dict.
        seed:         integer random seed (controls split + weight init).
        config_path:  path to the YAML file (copied into the save dir).
        split_cache:  optional SplitCache instance. When provided, data splits
                      are loaded from disk on a hit and written on a miss,
                      eliminating redundant CSV reads and negative sampling
                      across runs that share the same seed and config.
        parent_dir:   if set, save this run under parent_dir/seed_{seed}/ instead
                      of the default models/<split>_<model>_<timestamp>/. Use this
                      when grouping multiple runs under a shared output directory.

    Returns:
        dict with at minimum:
            seed, test_auc, test_ap, save_dir
        and, when applicable:
            both_unseen_auc, both_unseen_ap, one_unseen_auc, one_unseen_ap
            gene_gene_auc, gene_gene_ap
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    d  = cfg['data']
    m  = cfg['model']
    t  = cfg['training']
    mm = cfg.get('multimodal', {})

    split_mode   = d['mode']
    model_type   = m['type']
    is_inductive = split_mode == 'inductive'

    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
    nw = 4 if device == 'cuda' else 0
    bs = t['batch_size']

    if model_type == 'multimodal':
        lambda_ii = mm['lambda_iso_iso']
        lambda_gi = mm['lambda_gene_iso']
        lambda_gg = mm['lambda_gene_gene']
        neg_ratio = mm['neg_ratio']

    print(f"\n--- Training {'multimodal ' if model_type == 'multimodal' else ''}LDM  "
          f"(seed={seed}) ---")
    print(f"Device: {device}  Split: {split_mode}")
    if model_type == 'multimodal':
        print(f"λ_iso_iso={lambda_ii}  λ_gene_iso={lambda_gi}  "
              f"λ_gene_gene={lambda_gg}  neg_ratio={neg_ratio}")

    # ── 1+2b+2c: Load data (possibly from cache) ─────────────────────────────
    print("\nStep 1: Loading data...")
    split_key  = _build_split_key(d, mm, seed, model_type) if split_cache else None
    split_data = split_cache.get(split_key)          if split_cache else None

    if split_data is not None:
        print(f"  [cache hit — {split_cache.describe(split_key)}]")
    else:
        if split_cache:
            print("  [cache miss — loading from disk and building splits]")
        split_data = _load_all_splits(d, mm, seed, model_type, is_inductive)
        if split_cache:
            cache_path = split_cache.put(split_key, split_data)
            print(f"  [split saved to cache: {cache_path}]")

    # Unpack iso-iso fields
    train_data     = split_data['train_data']
    val_data       = split_data['val_data']
    test_data      = split_data['test_data']
    protein_to_idx = split_data['protein_to_idx']
    num_proteins   = split_data['num_proteins']
    neg_pos_ratio  = split_data['neg_pos_ratio']
    train_proteins = split_data['train_proteins']
    val_proteins   = split_data['val_proteins']
    test_proteins  = split_data['test_proteins']
    held_out_isoforms = (val_proteins | test_proteins) if is_inductive else set()

    print(f"\n  Proteins: {num_proteins:,}  |  Train: {len(train_data):,}  "
          f"Val: {len(val_data):,}  Test: {len(test_data):,}")
    if is_inductive:
        diagnose_split_inductive(train_data, val_data, test_data,
                                 train_proteins, val_proteins, test_proteins)
    else:
        diagnose_split(train_data, val_data, test_data)

    # ── 2. Dataloaders ────────────────────────────────────────────────────────
    print("Step 2: Creating dataloaders...")
    train_loader = DataLoader(ProteinInteractionDataset(train_data, protein_to_idx),
                              batch_size=bs, shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(ProteinInteractionDataset(val_data,   protein_to_idx),
                              batch_size=bs, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(ProteinInteractionDataset(test_data,  protein_to_idx),
                              batch_size=bs, shuffle=False, num_workers=nw)

    # Initialize additional modalities to None and 0 if they are not used
    gene_iso_loader = gene_gene_loader = gene_gene_testloader = gene_to_idx = None
    gene_gene_ratio = gene_iso_ratio = num_genes = 0

    if model_type == 'multimodal':
        if lambda_gi > 0:
            gene_to_idx    = split_data['gene_to_idx']
            train_gi       = split_data['train_gi']
            gene_iso_ratio = split_data['gene_iso_ratio']
            num_genes      = len(gene_to_idx)
            print(f"\nStep 2b: Gene–isoform loader  "
                f"({num_genes:,} genes  |  pos_weight: {gene_iso_ratio:.1f})")
            gene_iso_loader = DataLoader(GeneIsoformDataset(train_gi),
                                        batch_size=bs, shuffle=True, num_workers=nw)

        if lambda_gg > 0:
            gg_train        = split_data['gg_train']
            gg_test         = split_data['gg_test']
            gene_to_idx     = split_data['gene_to_idx']   # STRING-expanded
            gene_gene_ratio = split_data['gene_gene_ratio']
            num_genes       = len(gene_to_idx)
            print(f"\nStep 2c: Gene–gene loader  "
                  f"({num_genes:,} genes  |  pos_weight: {gene_gene_ratio:.1f})")
            gene_gene_loader     = DataLoader(GeneGeneDataset(gg_train),
                                              batch_size=bs, shuffle=True,  num_workers=nw)
            gene_gene_testloader = DataLoader(GeneGeneDataset(gg_test),
                                              batch_size=bs, shuffle=False, num_workers=nw)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("\nStep 3: Initialising model...")
    if model_type == 'multimodal':
        esmc_features = load_esmc_features(d['esmc_path'], protein_to_idx)
        if is_inductive:
            print(f"  {len(held_out_isoforms):,} held-out isoforms: ESM-C projection only")

    if model_type == 'ldm':
        model = LatentDistanceModel(num_proteins=num_proteins, latent_dim=m['latent_dim'])
    else:
        proj_hidden_dim = mm.get('proj_hidden_dim', 42)
        model = MultimodalLDM(
            num_proteins=num_proteins, num_genes=num_genes,
            latent_dim=m['latent_dim'], esmc_features=esmc_features,
            proj_hidden_dim=proj_hidden_dim)
        if m['latent_dim'] > 0:
            print(f"  Gene embeddings: {num_genes:,} × {m['latent_dim']}")
            print(f"  Isoform positions: ESM-C proj "
                  f"({esmc_features.shape[1]}→{proj_hidden_dim}→{m['latent_dim']})")
        else:
            print("  Latent distances disabled: random effects and intercepts only")
        print(f"  Random effects: re_head({esmc_features.shape[1]}→{proj_hidden_dim}→1)")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Latent dim: {m['latent_dim']}  |  "
          f"Params: {n_trainable:,} trainable / {n_total:,} total")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\nStep 4: Training...")
    epochs       = t['epochs']
    lr           = t['learning_rate']
    weight_decay = t['weight_decay']
    patience     = t.get('patience', 10)

    if model_type == 'ldm':
        trainer = LatentDistanceTrainer(model, device=device)
        best_ap = trainer.train(
            train_loader, val_loader,
            epochs=epochs, lr=lr,
            weight_decay=weight_decay, pos_weight=neg_pos_ratio,
            patience=patience)
    else:
        trainer = MultimodalTrainer(model, device=device)
        best_ap = trainer.train(
            iso_iso_loader       = train_loader,
            gene_iso_loader      = gene_iso_loader,
            gene_gene_loader     = gene_gene_loader,
            val_loader           = val_loader,
            epochs               = epochs,
            lr                   = lr,
            weight_decay         = weight_decay,
            iso_iso_pos_weight   = neg_pos_ratio,
            lambda_iso_iso       = lambda_ii,
            gene_iso_pos_weight  = gene_iso_ratio,
            lambda_gene_iso      = lambda_gi,
            gene_gene_pos_weight = gene_gene_ratio,
            lambda_gene_gene     = lambda_gg,
            patience             = patience,
        )

    # ── 5. Save dir + config snapshot ─────────────────────────────────────────
    split_tag  = {'transductive': 'TRANS', 'inductive': 'IND'}[split_mode]
    model_tag  = 'LDM' if model_type == 'ldm' else 'MM'
    if parent_dir is not None:
        save_dir = os.path.join(parent_dir, f"seed_{seed}")
    else:
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        models_dir = cfg.get('paths', {}).get('models_dir', 'models')
        save_dir   = f"{models_dir}/{split_tag}_{model_tag}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_path, f"{save_dir}/config.yaml")

    # ── 6. Training curves ────────────────────────────────────────────────────
    print("\nStep 5: Saving training curves...")
    trainer.plot_training()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/training_curves.png")

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    print("\nStep 6: Evaluating on test set...")
    auc, ap, _, _ = evaluate_model(model, test_loader, device=device, save_dir=save_dir)

    metrics = {'seed': seed, 'test_auc': auc, 'test_ap': ap, 'save_dir': save_dir}

    if is_inductive:
        auc_bu, ap_bu, auc_ou, ap_ou = evaluate_inductive_model_separately(
            model, test_data, test_proteins, protein_to_idx,
            batch_size=bs, num_workers=nw, device=device, save_dir=save_dir)
        metrics.update({
            'both_unseen_auc': auc_bu, 'both_unseen_ap': ap_bu,
            'one_unseen_auc':  auc_ou, 'one_unseen_ap':  ap_ou,
        })

    if model_type == 'multimodal' and lambda_gg > 0:
        auc_gg, ap_gg, _, _ = evaluate_gene_gene(
            model, gene_gene_loader=gene_gene_testloader,
            device=device, save_dir=save_dir)
        metrics.update({'gene_gene_auc': auc_gg, 'gene_gene_ap': ap_gg})

    # ── 8. Checkpoint ─────────────────────────────────────────────────────────
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
        'seed':             seed,
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
            'lambda_iso_iso':   lambda_ii,
            'lambda_gene_iso':  lambda_gi,
            'neg_ratio':        neg_ratio,
            'lambda_gene_gene': lambda_gg,
            'proj_hidden_dim':  proj_hidden_dim,
            'esmc_path':        d['esmc_path'],
        })

    torch.save(checkpoint, f"{save_dir}/model.pt")
    print(f"  Saved: {save_dir}/model.pt")

    print(f"\n--- Run complete (seed={seed}) ---")
    print(f"Best val AP: {best_ap:.4f}  |  Test AUC: {auc:.4f}  |  AP: {ap:.4f}")
    print(f"Saved to: {save_dir}\n")

    return metrics
