import itertools

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score


class ProjectionMLP(nn.Module):
    """Two-layer MLP: in_dim → hidden_dim (Dropout, ReLU) → out_dim."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class MultimodalLDM(nn.Module):
    """Three-modality Latent Distance Model: iso–iso, gene–iso, gene–gene.

    Isoform positions come from an ESM-C projection (with optional per-isoform
    residual); falls back to a learned embedding table when no ESM-C is provided.

    Args:
        num_proteins: number of isoforms (size of the isoform vocabulary).
        num_genes: number of genes (incl. STRING-only entries).
        latent_dim: latent space dimensionality.
        esmc_features: float tensor (num_proteins, esmc_dim). If None, falls back to a fully-learned embedding table — no inductive capacity.
        use_residuals: if True, add per-isoform corrections on top of the ESM-C projection. Set False in inductive mode to prevent memorization of seen training isoforms.
        proj_hidden_dim: hidden width of the ESM-C projection and re_head MLPs.
    """

    def __init__(self, num_proteins, num_genes, latent_dim=32, esmc_features=None, proj_hidden_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Isoform branch ────────────────────────────────────────────────────
        self.register_buffer('esmc_features', esmc_features.float(), persistent=False)
        esmc_dim       = esmc_features.shape[1]
        if self.latent_dim > 0:
            self.esmc_proj = ProjectionMLP(esmc_dim, proj_hidden_dim, latent_dim)
        self.re_head   = ProjectionMLP(esmc_dim, proj_hidden_dim, 1)

        # ── Gene branch (shared across gene–iso and gene–gene tasks) ──────────
        if self.latent_dim > 0:
            self.gene_embeddings     = nn.Embedding(num_genes, latent_dim)
            for emb in (self.gene_embeddings, self.gene_iso_intercept, self.gene_gene_intercept):
                nn.init.normal_(emb.weight, mean=0, std=0.1)
        self.gene_iso_intercept  = nn.Embedding(num_genes, 1)
        self.gene_gene_intercept = nn.Embedding(num_genes, 1)
        for emb in (self.gene_embeddings, self.gene_iso_intercept, self.gene_gene_intercept):
            nn.init.normal_(emb.weight, mean=0, std=0.1)

        # ── Per-modality global intercepts ────────────────────────────────────
        self.alpha_iso_iso   = nn.Parameter(torch.tensor(0.0))
        self.alpha_gene_iso  = nn.Parameter(torch.tensor(0.0))
        self.alpha_gene_gene = nn.Parameter(torch.tensor(0.0))

        # ── Per-modality distance scales ──────────────────────────────────────
        if self.latent_dim > 0:
            self.beta_iso_iso   = nn.Parameter(torch.tensor(1.0))
            self.beta_gene_iso  = nn.Parameter(torch.tensor(1.0))
            self.beta_gene_gene = nn.Parameter(torch.tensor(1.0))

    def _isoform_latent(self, p_idx):
        return self.esmc_proj(self.esmc_features[p_idx])

    def _random_effect(self, p_idx):
        return self.re_head(self.esmc_features[p_idx]).squeeze(-1)

    def compute_distance(self, z1, z2):
        return torch.norm(z1 - z2, p=2, dim=1)

    def forward_iso_iso(self, p1, p2):
        r1, r2 = self._random_effect(p1),  self._random_effect(p2)
        if self.latent_dim == 0:
            return r1 + r2
        z1, z2 = self._isoform_latent(p1), self._isoform_latent(p2)
        r1, r2 = self._random_effect(p1),  self._random_effect(p2)
        return self.alpha_iso_iso + r1 + r2 - F.softplus(self.beta_iso_iso) * self.compute_distance(z1, z2)

    def forward_gene_iso(self, g_idx, p_idx):
        gamma = self.gene_iso_intercept(g_idx).squeeze(-1)
        if self.latent_dim == 0:
            return gamma
        u_g   = self.gene_embeddings(g_idx)
        z_i   = self._isoform_latent(p_idx)
        gamma = self.gene_iso_intercept(g_idx).squeeze(-1)
        return self.alpha_gene_iso + gamma - F.softplus(self.beta_gene_iso) * self.compute_distance(u_g, z_i)

    def forward_gene_gene(self, g_a, g_b):
        d_a = self.gene_gene_intercept(g_a).squeeze(-1)
        d_b = self.gene_gene_intercept(g_b).squeeze(-1)
        return self.alpha_gene_gene + d_a + d_b - F.softplus(self.beta_gene_gene) * self.compute_distance(u_a, u_b)

    def forward(self, p1, p2):
        return self.forward_iso_iso(p1, p2)

    def get_embeddings(self):
        return self.get_isoform_embeddings()

    def get_isoform_embeddings(self):
        with torch.no_grad():
            idx = torch.arange(self.esmc_features.shape[0], device=self.esmc_features.device)
            return self._isoform_latent(idx).cpu().numpy()

    def get_gene_embeddings(self):
        return self.gene_embeddings.weight.detach().cpu().numpy()

    def get_random_effects(self):
        with torch.no_grad():
            idx = torch.arange(self.esmc_features.shape[0], device=self.esmc_features.device)
            return self._random_effect(idx).unsqueeze(-1).cpu().numpy()

    @torch.no_grad()
    def init_gene_centroids(self, gene_to_idx, gene_to_isoforms, protein_to_idx):
        """Initialize each gene at its isoforms' mean latent position;
        STRING-only genes are placed near the global mean.

        Args:
            gene_to_idx: gene → index mapping (full gene vocabulary).
            gene_to_isoforms: gene → set of member isoforms. In inductive mode,
                pass only training-set isoforms to avoid leaking held-out positions.
            protein_to_idx: isoform → index mapping.
        """
        device = self.gene_embeddings.weight.device
        n_iso  = self.esmc_features.shape[0]
        iso_positions = self._isoform_latent(torch.arange(n_iso, device=device))

        n_init = 0
        for gene_id, isoforms in gene_to_isoforms.items():
            if gene_id not in gene_to_idx:
                continue
            iso_indices = [protein_to_idx[i] for i in isoforms if i in protein_to_idx]
            if not iso_indices:
                continue
            idx_t = torch.tensor(iso_indices, device=device)
            self.gene_embeddings.weight.data[gene_to_idx[gene_id]] = iso_positions[idx_t].mean(dim=0)
            n_init += 1

        if n_init > 0:
            global_mean = self.gene_embeddings.weight.data[:max(gene_to_idx.values()) + 1].mean(dim=0)
            for gene_id, gene_idx in gene_to_idx.items():
                if gene_id not in gene_to_isoforms:
                    self.gene_embeddings.weight.data[gene_idx] = (
                        global_mean + torch.randn_like(global_mean) * 0.1)

        print(f"  Gene centroids initialized: {n_init:,} / {len(gene_to_idx):,} genes")


class MultimodalTrainer:
    """Trainer for MultimodalLDM.

    Iso–iso is the validation task (held-out interactions); gene–iso and
    gene–gene are auxiliary signals always trained on full data.

    Args:
        model: a MultimodalLDM instance.
        device: torch device string ('cpu', 'cuda', 'mps').
    """

    def __init__(self, model, device='cpu'):
        self.model  = model.to(device)
        self.device = device
        self.train_iso_iso_losses   = []
        self.train_gene_iso_losses  = []
        self.train_gene_gene_losses = []
        self.val_losses = []
        self.val_aucs   = []
        self.val_aps    = []

    @staticmethod
    def _make_iter(loader, n_steps, active):
        if not active:
            return None
        return itertools.cycle(loader) if len(loader) < n_steps else iter(loader)

    def train_epoch(self, iso_iso_loader, gene_iso_loader, gene_gene_loader,
                    optimizer, crit_iso_iso, crit_gene_iso, crit_gene_gene,
                    lambda_iso_iso, lambda_gene_iso, lambda_gene_gene):
        self.model.train()
        n_steps = max(len(iso_iso_loader), len(gene_iso_loader), len(gene_gene_loader))

        # Cycle shorter loaders to match the longest, then scale each modality's λ
        # by its cycling factor so unique samples contribute equal weight per epoch.
        eff_iso_iso   = lambda_iso_iso   / (n_steps / len(iso_iso_loader))
        eff_gene_iso  = lambda_gene_iso  / (n_steps / len(gene_iso_loader))
        eff_gene_gene = lambda_gene_gene / (n_steps / len(gene_gene_loader))

        iso_iso_iter   = self._make_iter(iso_iso_loader,   n_steps, True)
        gene_iso_iter  = self._make_iter(gene_iso_loader,  n_steps, lambda_gene_iso  > 0)
        gene_gene_iter = self._make_iter(gene_gene_loader, n_steps, lambda_gene_gene > 0)

        total_ii = total_gi = total_gg = 0.0

        for _ in range(n_steps):
            p1, p2, y = (t.to(self.device) for t in next(iso_iso_iter))
            loss_ii = crit_iso_iso(self.model.forward_iso_iso(p1, p2), y)
            total_ii += loss_ii.item()
            loss = eff_iso_iso * loss_ii

            if gene_iso_iter is not None:
                g, p, y = (t.to(self.device) for t in next(gene_iso_iter))
                loss_gi = crit_gene_iso(self.model.forward_gene_iso(g, p), y)
                total_gi += loss_gi.item()
                loss = loss + eff_gene_iso * loss_gi

            if gene_gene_iter is not None:
                ga, gb, y = (t.to(self.device) for t in next(gene_gene_iter))
                loss_gg = crit_gene_gene(self.model.forward_gene_gene(ga, gb), y)
                total_gg += loss_gg.item()
                loss = loss + eff_gene_gene * loss_gg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_ii / n_steps, total_gi / n_steps, total_gg / n_steps

    def validate(self, val_loader, criterion):
        self.model.eval()
        total, preds, labels = 0.0, [], []
        with torch.no_grad():
            for p1, p2, y in val_loader:
                p1, p2, y = p1.to(self.device), p2.to(self.device), y.to(self.device)
                logits = self.model.forward_iso_iso(p1, p2)
                total += criterion(logits, y).item()
                preds.extend(logits.cpu().numpy())
                labels.extend(y.cpu().numpy())
        avg = total / max(len(val_loader), 1)
        return avg, roc_auc_score(labels, preds), average_precision_score(labels, preds)

    def train(self, iso_iso_loader, gene_iso_loader, gene_gene_loader, val_loader,
              epochs=30, lr=1e-3, weight_decay=1e-5,
              iso_iso_pos_weight=222.2,  lambda_iso_iso=1.0,
              gene_iso_pos_weight=5.0,   lambda_gene_iso=0.5,
              gene_gene_pos_weight=5.0,  lambda_gene_gene=0.3,
              patience=10):
        """Three-modality training loop with early stopping on iso–iso val AP.

        Args:
            iso_iso_loader, gene_iso_loader, gene_gene_loader: training loaders
                for each modality. Shorter loaders are cycled to match the longest.
            val_loader: iso–iso validation loader (the actual prediction task).
            epochs: maximum number of epochs.
            lr: Adam learning rate.
            weight_decay: Adam L2 weight decay.
            *_pos_weight: BCE positive-class weights per modality.
            lambda_*: loss weights per modality (0 disables that modality).
            patience: epochs without iso–iso val-AP improvement before stopping.

        Returns:
            best iso–iso validation AP achieved during training.
        """
        device = self.device
        crit_ii = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(iso_iso_pos_weight,   dtype=torch.float32, device=device))
        crit_gi = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gene_iso_pos_weight,  dtype=torch.float32, device=device))
        crit_gg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gene_gene_pos_weight, dtype=torch.float32, device=device))

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_ap, best_epoch, best_state = 0.0, 0, None
        n_ii, n_gi, n_gg = len(iso_iso_loader), len(gene_iso_loader), len(gene_gene_loader)
        n_steps = max(n_ii, n_gi, n_gg)

        print(f"Training MultimodalLDM (3 modalities) on {device}")
        print(f"  λ — iso_iso={lambda_iso_iso}  gene_iso={lambda_gene_iso}  "
              f"gene_gene={lambda_gene_gene}")
        print(f"  pos_weights — iso_iso: {iso_iso_pos_weight:.2f}  "
              f"gene_iso: {gene_iso_pos_weight:.2f}  gene_gene: {gene_gene_pos_weight:.2f}")
        print(f"  Steps/epoch: {n_steps}  (ii={n_ii}  gi={n_gi}  gg={n_gg}; shorter loaders cycle)")
        print(f"  Effective λ — iso_iso: {lambda_iso_iso:.3f}  "
              f"gene_iso: {lambda_gene_iso / (n_steps / n_gi):.3f}  "
              f"gene_gene: {lambda_gene_gene / (n_steps / n_gg):.3f}")
        print(f"  Patience: {patience}")
        print("-" * 70)

        for epoch in range(epochs):
            ii_loss, gi_loss, gg_loss = self.train_epoch(
                iso_iso_loader, gene_iso_loader, gene_gene_loader,
                optimizer, crit_ii, crit_gi, crit_gg,
                lambda_iso_iso, lambda_gene_iso, lambda_gene_gene)

            val_loss, val_auc, val_ap = self.validate(val_loader, crit_ii)

            self.train_iso_iso_losses.append(ii_loss)
            self.train_gene_iso_losses.append(gi_loss)
            self.train_gene_gene_losses.append(gg_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)
            scheduler.step(val_ap)

            if val_ap > best_ap:
                best_ap, best_epoch = val_ap, epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1}/{epochs}  "
                  f"L_ii={ii_loss:.4f}  L_gi={gi_loss:.4f}  L_gg={gg_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  AUC={val_auc:.4f}  AP={val_ap:.4f}")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\nBest validation AP: {best_ap:.4f}")
        return best_ap

    def plot_training(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.train_iso_iso_losses,   label='Train iso–iso')
        axes[0].plot(self.train_gene_iso_losses,  label='Train gene–iso')
        axes[0].plot(self.train_gene_gene_losses, label='Train gene–gene')
        axes[0].plot(self.val_losses,             label='Val iso–iso', linestyle='--')
        axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training Losses')
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(self.val_aucs, label='AUC', color='green')
        axes[1].plot(self.val_aps,  label='AP',  color='red')
        axes[1].set(xlabel='Epoch', title='Validation Metrics (Iso–Iso)')
        axes[1].legend(); axes[1].grid(True)

        plt.tight_layout()
        return fig
