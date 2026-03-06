import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# =============================================================================
# Model
# =============================================================================

class MultimodalLDM(nn.Module):
    """
    Multimodal Latent Distance Model.

    Isoform–isoform:  P(Y_ij = 1) = sigmoid(r_i + r_j − β_iso  · d(z_i, z_j))
    Gene–isoform:     P(E_gi = 1) = sigmoid(γ_g        − β_gene · d(u_g, z_i))

    Parameters:
        num_proteins:    number of unique isoforms/proteins in the network
        num_genes:       number of unique canonical genes
        latent_dim:      latent space dimensionality (e.g. 16, 32, 64, 128)
        distance_metric: 'euclidean' or 'cosine'
    """

    def __init__(self, num_proteins, num_genes, latent_dim=32, distance_metric='euclidean'):
        super().__init__()

        self.distance_metric = distance_metric

        # ── Shared isoform latent space ───────────────────────────────────────
        self.isoform_embeddings = nn.Embedding(num_proteins, latent_dim)
        nn.init.normal_(self.isoform_embeddings.weight, mean=0, std=0.1)

        # ── Gene latent space (bipartite component) ───────────────────────────
        self.gene_embeddings = nn.Embedding(num_genes, latent_dim)
        nn.init.normal_(self.gene_embeddings.weight, mean=0, std=0.1)

        # ── Isoform–isoform parameters ────────────────────────────────────────
        self.beta_iso     = nn.Parameter(torch.tensor(1.0))   # distance weight
        self.random_effects = nn.Embedding(num_proteins, 1)   # per-isoform random effects r_i
        nn.init.normal_(self.random_effects.weight, mean=0, std=0.1)

        # ── Gene–isoform (bipartite) parameters ──────────────────────────────
        self.beta_gene      = nn.Parameter(torch.tensor(1.0))  # distance weight
        self.gene_intercept = nn.Embedding(num_genes, 1)       # per-gene popularity γ_g
        nn.init.normal_(self.gene_intercept.weight, mean=0, std=0.1)

    def compute_distance(self, z1, z2):
        if self.distance_metric == 'euclidean':
            return torch.norm(z1 - z2, p=2, dim=1)
        elif self.distance_metric == 'cosine':
            return 1.0 - F.cosine_similarity(z1, z2, dim=1)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def forward_isoform(self, protein1_idx, protein2_idx):
        """
        Isoform–isoform interaction logits.
        Identical in form to the base LDM forward pass.

        Returns: logits of shape (batch,)
        """
        z1 = self.isoform_embeddings(protein1_idx)
        z2 = self.isoform_embeddings(protein2_idx)
        dist = self.compute_distance(z1, z2)

        r1 = self.random_effects(protein1_idx).squeeze(-1)
        r2 = self.random_effects(protein2_idx).squeeze(-1)
        return r1 + r2 - self.beta_iso * dist

    def forward_bipartite(self, gene_idx, protein_idx):
        """
        Gene–isoform membership logits.
            logit = γ_g  −  β_gene · d(u_g, z_i)

        z_i is the *shared* isoform embedding — gradients here feed
        directly back into the isoform–isoform likelihood during training.

        Returns: logits of shape (batch,)
        """
        u_g = self.gene_embeddings(gene_idx)
        z_i = self.isoform_embeddings(protein_idx)          # shared!
        dist = self.compute_distance(u_g, z_i)
        gamma_g = self.gene_intercept(gene_idx).squeeze(-1)
        return gamma_g - self.beta_gene * dist

    # kept for API compatibility with existing visualize.py / evaluate.py
    def forward(self, protein1_idx, protein2_idx):
        """Alias for forward_isoform so existing evaluate / visualize code works."""
        return self.forward_isoform(protein1_idx, protein2_idx)

    def get_embeddings(self):
        """Isoform embeddings — alias used by pca.py / hierarchical_clustering.py."""
        return self.isoform_embeddings.weight.detach().cpu().numpy()

    def get_isoform_embeddings(self):
        return self.isoform_embeddings.weight.detach().cpu().numpy()

    def get_gene_embeddings(self):
        return self.gene_embeddings.weight.detach().cpu().numpy()

    def get_random_effects(self):
        return self.random_effects.weight.detach().cpu().numpy()


# =============================================================================
# Trainer
# =============================================================================

class MultimodalTrainer:
    """Trainer for MultimodalLDM."""

    def __init__(self, model, device='cpu'):
        self.model  = model.to(device)
        self.device = device

        # Logged per epoch
        self.train_iso_losses  = []
        self.train_gene_losses = []
        self.val_losses        = []
        self.val_aucs          = []
        self.val_aps           = []
        self.val_f1s           = []

    def train_epoch(self, iso_loader, gene_loader, optimizer, crit_iso, crit_gene, lambda_iso, lambda_gene):
        """
        One epoch of joint training.

        Both loaders are iterated in lock-step; the shorter one is cycled so
        every batch of the longer loader is used each epoch. A single backward
        pass per step sums the two weighted losses, giving the shared isoform
        embeddings a coherent gradient from both tasks.
        """
        self.model.train()

        # Cycle the shorter loader so both streams stay in sync for the full epoch
        n_steps   = max(len(iso_loader), len(gene_loader))
        iso_iter  = itertools.cycle(iso_loader)  if len(iso_loader)  < n_steps else iter(iso_loader)
        gene_iter = itertools.cycle(gene_loader) if len(gene_loader) < n_steps else iter(gene_loader)

        total_iso, total_gene = 0.0, 0.0

        for _ in range(n_steps):
            p1, p2, iso_labels           = next(iso_iter)
            gene_idx, prot_idx, g_labels = next(gene_iter)

            p1         = p1.to(self.device)
            p2         = p2.to(self.device)
            iso_labels = iso_labels.to(self.device)
            gene_idx   = gene_idx.to(self.device)
            prot_idx   = prot_idx.to(self.device)
            g_labels   = g_labels.to(self.device)

            iso_logits  = self.model.forward_isoform(p1, p2)
            gene_logits = self.model.forward_bipartite(gene_idx, prot_idx)

            loss_iso  = crit_iso(iso_logits, iso_labels)
            loss_gene = crit_gene(gene_logits, g_labels)

            # Combined loss — single backward pass over both tasks
            loss = lambda_iso * loss_iso + lambda_gene * loss_gene

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iso  += loss_iso.item()
            total_gene += loss_gene.item()

        return total_iso / n_steps, total_gene / n_steps

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for p1, p2, labels in val_loader:
                p1, p2, labels = p1.to(self.device), p2.to(self.device), labels.to(self.device)
                logits = self.model.forward_isoform(p1, p2)
                total_loss += criterion(logits, labels).item()
                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss  = total_loss / max(len(val_loader), 1)
        preds_arr = np.array(all_preds)
        auc = roc_auc_score(all_labels, preds_arr)
        ap  = average_precision_score(all_labels, preds_arr)
        f1  = f1_score(all_labels, preds_arr > 0.5)
        return avg_loss, auc, ap, f1, all_preds, all_labels

    def train(
        self,
        iso_train_loader,
        gene_train_loader,
        val_loader,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-5,
        # Isoform–isoform loss
        iso_pos_weight=222.2,
        lambda_iso=1.0,
        # Gene–isoform loss
        # gene_pos_weight should equal neg_ratio (negatives per positive in the
        # gene–isoform dataset).  With the default neg_ratio=5 this is 5.0.
        gene_pos_weight=5.0,
        lambda_gene=0.5,
    ):
        """
        Full multimodal training loop.

        Args:
            iso_pos_weight:  BCE pos_weight for the isoform–isoform loss.
                             Set to approx. (num_neg / num_pos) in the isoform dataset (~222.2).
            lambda_iso:      Scaling factor for L_isoform in the combined loss.
            gene_pos_weight: BCE pos_weight for the gene–isoform loss.
                             Should equal neg_ratio (num_neg_gene / num_pos_gene).
            lambda_gene:     Scaling factor for L_bipartite in the combined loss.
        """
        crit_iso  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(iso_pos_weight).to(self.device))
        crit_gene = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gene_pos_weight).to(self.device))

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_f1, best_state = 0.0, None

        print(f"Training MultimodalLDM on {self.device}")
        print(f"  λ_iso={lambda_iso}  λ_gene={lambda_gene}  "
              f"pos_weight_iso={iso_pos_weight}  pos_weight_gene={gene_pos_weight}")
        print(f"  Steps/epoch: {max(len(iso_train_loader), len(gene_train_loader))} "
              f"(iso={len(iso_train_loader)}  gene={len(gene_train_loader)})")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 70)

        for epoch in range(epochs):
            iso_loss, gene_loss = self.train_epoch(
                iso_train_loader, gene_train_loader,
                optimizer, crit_iso, crit_gene,
                lambda_iso, lambda_gene,
            )
            val_loss, val_auc, val_ap, val_f1, _, _ = self.validate(val_loader, crit_iso)

            self.train_iso_losses.append(iso_loss)
            self.train_gene_losses.append(gene_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)
            self.val_f1s.append(val_f1)

            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1:3d}/{epochs}  "
                  f"L_iso={iso_loss:.4f}  L_gene={gene_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"AUC={val_auc:.4f}  AP={val_ap:.4f}  F1={val_f1:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\nBest validation F1: {best_f1:.4f}")
        return best_f1

    def plot_training(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(self.train_iso_losses,  label='Train iso loss')
        axes[0].plot(self.train_gene_losses, label='Train gene loss')
        axes[0].plot(self.val_losses,        label='Val iso loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.val_aucs, label='Val AUC', color='green')
        axes[1].plot(self.val_aps,  label='Val AP',  color='red')
        axes[1].plot(self.val_f1s,  label='Val F1',  color='blue')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics (Isoform–Isoform)')
        axes[1].legend()
        axes[1].grid(True)

        ratio = [g / (i + 1e-9) for i, g in zip(self.train_iso_losses, self.train_gene_losses)]
        axes[2].plot(ratio, color='purple', label='gene loss / iso loss')
        axes[2].axhline(1.0, color='grey', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Epoch')
        axes[2].set_title('Loss Ratio (gene / isoform)')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        return fig