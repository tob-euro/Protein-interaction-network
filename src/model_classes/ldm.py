import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


class LatentDistanceModel(nn.Module):
    """
    Latent Distance Model for Link Prediction.

    Likelihood:    P(Y_ij = 1) = sigmoid(r_i + r_j - beta * ||z_i - z_j||)

    Parameters:
        num_proteins:    number of proteins in the network
        latent_dim:      latent space dimensionality (e.g. 16, 32, 64, 128)
    """

    def __init__(self, num_proteins, latent_dim=32):
        super(LatentDistanceModel, self).__init__()

        self.embeddings     = nn.Embedding(num_proteins, latent_dim)
        self.random_effects = nn.Embedding(num_proteins, 1)
        self.beta           = nn.Parameter(torch.tensor(1.0))

        nn.init.normal_(self.random_effects.weight, mean=0, std=0.1)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.1)

    def compute_distance(self, z1, z2):
        return torch.norm(z1 - z2, p=2, dim=1)

    def forward(self, protein1_idx, protein2_idx):
        z1 = self.embeddings(protein1_idx)
        z2 = self.embeddings(protein2_idx)
        distance = self.compute_distance(z1, z2)

        r1 = self.random_effects(protein1_idx).squeeze(-1)
        r2 = self.random_effects(protein2_idx).squeeze(-1)

        logits = r1 + r2 - self.beta * distance
        return logits

    def get_embeddings(self):
        return self.embeddings.weight.detach().cpu().numpy()

    def get_random_effects(self):
        return self.random_effects.weight.detach().cpu().numpy()


class BaselineLDM(LatentDistanceModel):
    """
    Baseline model using only per-protein random effects — no latent geometry.
    P(Y_ij = 1) = sigmoid(r_i + r_j)

    Useful as a comparison against the full LDM to measure how much the
    latent distance term contributes beyond node-level popularity effects.
    """

    def __init__(self, num_proteins, latent_dim=32):
        super().__init__(num_proteins, latent_dim)

    def forward(self, protein1_idx, protein2_idx):
        r1 = self.random_effects(protein1_idx).squeeze(-1)
        r2 = self.random_effects(protein2_idx).squeeze(-1)
        return r1 + r2


class LatentDistanceTrainer:
    """Trainer for LatentDistanceModel."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses   = []
        self.val_aucs     = []
        self.val_aps      = []

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for protein1_idx, protein2_idx, labels in dataloader:
            protein1_idx = protein1_idx.to(self.device)
            protein2_idx = protein2_idx.to(self.device)
            labels       = labels.to(self.device)
            predictions  = self.model(protein1_idx, protein2_idx)
            loss         = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for protein1_idx, protein2_idx, labels in dataloader:
                protein1_idx = protein1_idx.to(self.device)
                protein2_idx = protein2_idx.to(self.device)
                labels       = labels.to(self.device)
                predictions  = self.model(protein1_idx, protein2_idx)
                loss         = criterion(predictions, labels)
                total_loss  += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        ap  = average_precision_score(all_labels, all_preds)
        return avg_loss, auc, ap, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=10, lr=0.001,
              weight_decay=1e-5, pos_weight=222.2, patience=10):
        """
        Full training loop.

        Args:
            pos_weight: weight applied to positive class in BCEWithLogitsLoss.
                        Set to approx. (num_negatives / num_positives) to handle class imbalance.
            patience:   epochs without val AP improvement before early stopping.
        """
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_ap, best_epoch, best_model_state = 0.0, 0, None

        print(f"Training LDM on {self.device}")
        print(f"pos_weight: {pos_weight:.2f}")
        print(f"Steps/epoch: {len(train_loader)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Early stopping patience: {patience}")
        print("-" * 70)

        for epoch in range(epochs):
            train_loss              = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_auc, val_ap, _, _ = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)

            scheduler.step(val_ap)

            if val_ap > best_ap:
                best_ap          = val_ap
                best_epoch       = epoch
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1}/{epochs}  "
                  f"loss {train_loss:.4f}/{val_loss:.4f}  "
                  f"AUC {val_auc:.4f}  AP {val_ap:.4f}")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break

        self.model.load_state_dict(best_model_state)
        print(f"\nBest validation AP: {best_ap:.4f}")
        return best_ap

    def plot_training(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses,   label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.val_aucs, label='Val AUC', color='green')
        axes[1].plot(self.val_aps,  label='Val AP',  color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        return fig
    