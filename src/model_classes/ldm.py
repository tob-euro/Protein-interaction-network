import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score


class LatentDistanceModel(nn.Module):
    """Latent Distance Model: P(Y_ij=1) = sigmoid(r_i + r_j − β·||z_i − z_j||).

    Args:
        num_proteins: number of distinct isoforms in the embedding table.
        latent_dim: latent space dimensionality.
    """

    def __init__(self, num_proteins, latent_dim=32):
        super().__init__()

        self.latent_dim = latent_dim
        if self.latent_dim > 0:
            self.embeddings     = nn.Embedding(num_proteins, self.latent_dim)
            nn.init.normal_(self.embeddings.weight,     mean=0, std=0.1)
        
        self.random_effects = nn.Embedding(num_proteins, 1)
        self.beta           = nn.Parameter(torch.tensor(1.0))

        nn.init.normal_(self.random_effects.weight, mean=0, std=0.1)

    def compute_distance(self, z1, z2):
        return torch.norm(z1 - z2, p=2, dim=1)

    def forward(self, p1_idx, p2_idx):
        r1 = self.random_effects(p1_idx).squeeze(-1)
        r2 = self.random_effects(p2_idx).squeeze(-1)
        if self.latent_dim == 0:
            return r1 + r2
        else:
            z1 = self.embeddings(p1_idx)
            z2 = self.embeddings(p2_idx)
            return r1 + r2 - self.beta * self.compute_distance(z1, z2)

    def get_embeddings(self):
        assert self.latent_dim > 0, "Latent dimension 0"
        return self.embeddings.weight.detach().cpu().numpy()

    def get_random_effects(self):
        return self.random_effects.weight.detach().cpu().numpy()


class LatentDistanceTrainer:
    """Trainer for LatentDistanceModel.

    Args:
        model: a LatentDistanceModel (or BaselineLDM) instance.
        device: torch device string ('cpu', 'cuda', 'mps').
    """

    def __init__(self, model, device='cpu'):
        self.model  = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses   = []
        self.val_aucs     = []
        self.val_aps      = []

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total = 0.0
        for p1, p2, y in loader:
            p1, p2, y = p1.to(self.device), p2.to(self.device), y.to(self.device)
            loss = criterion(self.model(p1, p2), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def validate(self, loader, criterion):
        self.model.eval()
        total, preds, labels = 0.0, [], []
        with torch.no_grad():
            for p1, p2, y in loader:
                p1, p2, y = p1.to(self.device), p2.to(self.device), y.to(self.device)
                logits = self.model(p1, p2)
                total += criterion(logits, y).item()
                preds.extend(logits.cpu().numpy())
                labels.extend(y.cpu().numpy())
        avg = total / len(loader)
        return (avg,
                roc_auc_score(labels, preds),
                average_precision_score(labels, preds),
                preds, labels)

    def train(self, train_loader, val_loader, epochs=10, lr=1e-3,
              weight_decay=1e-5, pos_weight=222.2, patience=10):
        """Full training loop with early stopping on validation AP.

        Args:
            train_loader, val_loader: iso–iso DataLoaders.
            epochs: maximum number of epochs.
            lr: Adam learning rate.
            weight_decay: Adam L2 weight decay.
            pos_weight: BCE positive-class weight (≈ neg/pos ratio for class imbalance).
            patience: epochs without val-AP improvement before stopping.

        Returns:
            best validation AP achieved during training.
        """
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_ap, best_epoch, best_state = 0.0, 0, None
        n_params = sum(p.numel() for p in self.model.parameters())

        print(f"Training LDM on {self.device}")
        print(f"  pos_weight: {pos_weight:.2f}  steps/epoch: {len(train_loader)}  "
              f"params: {n_params:,}  patience: {patience}")
        print("-" * 70)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_auc, val_ap, _, _ = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)
            scheduler.step(val_ap)

            if val_ap > best_ap:
                best_ap, best_epoch = val_ap, epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1}/{epochs}  loss {train_loss:.4f}/{val_loss:.4f}  "
                  f"AUC {val_auc:.4f}  AP {val_ap:.4f}")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\nBest validation AP: {best_ap:.4f}")
        return best_ap

    def plot_training(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.train_losses, label='Train')
        axes[0].plot(self.val_losses,   label='Val')
        axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training & Validation Loss')
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(self.val_aucs, label='AUC', color='green')
        axes[1].plot(self.val_aps,  label='AP',  color='red')
        axes[1].set(xlabel='Epoch', title='Validation Metrics')
        axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        return fig
