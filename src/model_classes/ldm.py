import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


class LatentDistanceModel(nn.Module):
    """
    Latent Distance Model for Link Prediction.

    Likelihood:    P(Y_ij = 1) = sigmoid(r_i + r_j - beta * ||z_i - z_j||)

    Parameters:
        num_proteins:    number of proteins in the network
        latent_dim:      latent space dimensionality (e.g. 16, 32, 64, 128)
        distance_metric: 'euclidean' or 'cosine'
    """

    def __init__(self, num_proteins, latent_dim=32, distance_metric='euclidean'):
        super(LatentDistanceModel, self).__init__()

        self.embeddings = nn.Embedding(num_proteins, latent_dim)
        self.random_effects = nn.Embedding(num_proteins, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.distance_metric = distance_metric

        nn.init.normal_(self.random_effects.weight, mean=0, std=0.1)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.1)

    def compute_distance(self, z1, z2):
        if self.distance_metric == 'euclidean':
            return torch.norm(z1 - z2, p=2, dim=1)
        elif self.distance_metric == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(z1, z2, dim=1)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

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

    def __init__(self, num_proteins, latent_dim=32, distance_metric='euclidean'):
        super().__init__(num_proteins, latent_dim, distance_metric)

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
        self.val_losses = []
        self.val_aucs = []
        self.val_aps = []
        self.val_f1s = []

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for protein1_idx, protein2_idx, labels in dataloader:
            protein1_idx = protein1_idx.to(self.device)
            protein2_idx = protein2_idx.to(self.device)
            labels = labels.to(self.device)
            predictions = self.model(protein1_idx, protein2_idx)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for protein1_idx, protein2_idx, labels in dataloader:
                protein1_idx = protein1_idx.to(self.device)
                protein2_idx = protein2_idx.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(protein1_idx, protein2_idx)
                loss = criterion(predictions, labels)
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, np.array(all_preds) > 0.5)
        return avg_loss, auc, ap, f1, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=1e-5, pos_weight=222.2):
        """
        Full training loop.

        Args:
            pos_weight: weight applied to positive class in BCEWithLogitsLoss.
                        Set to approx. (num_negatives / num_positives) to handle class imbalance.
        """
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_ap = 0
        best_model_state = None

        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(epochs):
            if hasattr(train_loader.dataset, '_resample'):
                train_loader.dataset._resample()

            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_auc, val_ap, val_f1, _, _ = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)
            self.val_f1s.append(val_f1)

            scheduler.step(val_ap)

            if val_ap > best_ap:
                best_ap = val_ap
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1}/{epochs}  "
                  f"loss {train_loss:.4f}/{val_loss:.4f}  "
                  f"AUC {val_auc:.4f}  AP {val_ap:.4f}  F1 {val_f1:.4f}")

        self.model.load_state_dict(best_model_state)
        print(f"\nBest validation AP: {best_ap:.4f}")
        return best_ap

    def plot_training(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(self.val_aucs, label='Val AUC', color='green')
        axes[1].plot(self.val_aps, label='Val AP', color='red')
        axes[1].plot(self.val_f1s, label='Val F1', color='blue')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        return fig