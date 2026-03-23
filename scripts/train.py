import argparse
import os
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model_classes.latent_distance_model import LatentDistanceModel, LatentDistanceTrainer, WeightedLatentDistanceTrainer
from src.data_scripts.dataset import ProteinInteractionDataset, WeightedProteinInteractionDataset, load_and_prepare_data
from src.training.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',            type=str,   default='config/config.yaml')
    parser.add_argument('--data',              type=str,   default=None,  help='Path to CSV data file')
    parser.add_argument('--latent-dim',        type=int,   default=None)
    parser.add_argument('--distance-metric',   type=str,   default=None,  choices=['euclidean', 'cosine'])
    parser.add_argument('--epochs',            type=int,   default=None)
    parser.add_argument('--lr',                type=float, default=None)
    parser.add_argument('--batch-size',        type=int,   default=None)
    parser.add_argument('--weight-decay',      type=float, default=None)
    return parser.parse_args()


def apply_overrides(cfg, args):
    if args.data:              cfg['data']['path']              = args.data
    if args.latent_dim:        cfg['model']['latent_dim']       = args.latent_dim
    if args.distance_metric:   cfg['model']['distance_metric']  = args.distance_metric
    if args.epochs:            cfg['training']['epochs']        = args.epochs
    if args.lr:                cfg['training']['learning_rate'] = args.lr
    if args.batch_size:        cfg['training']['batch_size']    = args.batch_size
    if args.weight_decay:      cfg['training']['weight_decay']  = args.weight_decay
    return cfg


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
    print(f"\n{'='*70}\nTRAINING LATENT DISTANCE MODEL\n{'='*70}")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("Step 1: Loading data...")
    train_data, val_data, test_data, protein_to_idx, num_proteins, neg_pos_ratio = load_and_prepare_data(
        d['path'], test_size=d['test_size'], val_size=d['val_size'], random_state=d['random_state'])
    print(f"\n  Proteins: {num_proteins}  |  "
          f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    # =========================================================================
    # 2. Dataloaders
    # =========================================================================
    print("\nStep 2: Creating dataloaders...")
    # train_loader = DataLoader(ProteinInteractionDataset(train_data, protein_to_idx),
    #                           batch_size=t['batch_size'], shuffle=True,
    #                           num_workers=t['num_workers'], pin_memory=True)
    # val_loader   = DataLoader(ProteinInteractionDataset(val_data, protein_to_idx),
    #                           batch_size=t['batch_size'], shuffle=False,
    #                           num_workers=t['num_workers'], pin_memory=True)
    # test_loader  = DataLoader(ProteinInteractionDataset(test_data, protein_to_idx),
    #                           batch_size=t['batch_size'], shuffle=False,
    #                           num_workers=t['num_workers'], pin_memory=True)
    train_loader = DataLoader(WeightedProteinInteractionDataset(train_data, protein_to_idx),
                              batch_size=t['batch_size'], shuffle=True,
                              num_workers=t['num_workers'], pin_memory=True)
    val_loader   = DataLoader(WeightedProteinInteractionDataset(val_data, protein_to_idx),
                              batch_size=t['batch_size'], shuffle=False,
                              num_workers=t['num_workers'], pin_memory=True)
    test_loader  = DataLoader(WeightedProteinInteractionDataset(test_data, protein_to_idx),
                              batch_size=t['batch_size'], shuffle=False,
                              num_workers=t['num_workers'], pin_memory=True)

    # =========================================================================
    # 3. Model
    # =========================================================================
    print("\nStep 3: Initializing model...")
    model = LatentDistanceModel(num_proteins=num_proteins, latent_dim=m['latent_dim'], distance_metric=m['distance_metric'])
    print(f"  Latent dim: {m['latent_dim']}  |  Metric: {m['distance_metric']}  |  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # =========================================================================
    # 4. Train
    # =========================================================================
    print("\nStep 4: Training...")
    trainer = WeightedLatentDistanceTrainer(model, device=device, pos_weight_scale=neg_pos_ratio)
    best_f1 = trainer.train(
        train_loader, val_loader,
        epochs=t['epochs'], lr=t['learning_rate'],
        weight_decay=t['weight_decay'], pos_weight=neg_pos_ratio,
    )

    save_dir = (f"models/WeightedLDM_dim={m['latent_dim']}_metric={m['distance_metric']}"
                f"_epochs={t['epochs']}_lr={t['learning_rate']}"
                f"_BS={t['batch_size']}")
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================================
    # 5. Training curves
    # =========================================================================
    print("\nStep 5: Saving training curves...")
    fig = trainer.plot_training()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/training_curves.png")

    # =========================================================================
    # 6. Evaluate on test set
    # =========================================================================
    print("\nStep 6: Evaluating on test set...")
    auc, ap, f1, _, _ = evaluate_model(model, test_loader, device=device, save_dir=save_dir)

    # =========================================================================
    # 7. Save checkpoint
    # =========================================================================
    print("\nStep 7: Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'protein_to_idx':   protein_to_idx,
        'num_proteins':     num_proteins,
        'latent_dim':       m['latent_dim'],
        'distance_metric':  m['distance_metric'],
        'test_auc': auc, 'test_ap': ap, 'test_f1': f1,
    }, f"{save_dir}/latent_distance_model.pt")
    print(f"  Saved: {save_dir}/latent_distance_model.pt")

    print(f"\n{'='*70}\nTRAINING COMPLETE")
    print(f"Best Val F1: {best_f1:.4f}  |  Test AUC: {auc:.4f}  |  AP: {ap:.4f}")
    print(f"Saved to: {save_dir}\n{'='*70}\n")


if __name__ == "__main__":
    main()