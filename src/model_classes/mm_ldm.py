import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


# =============================================================================
# Shared MLP building block
# =============================================================================

class ProjectionMLP(nn.Module):
    """Two-layer MLP: in_dim → hidden_dim (ReLU) → out_dim."""

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


# =============================================================================
# Model
# =============================================================================

class MultimodalLDM(nn.Module):
    """
    Multimodal Latent Distance Model.

    Iso–iso:      P(Y_ij = 1) = sigmoid(r_i + r_j       − β_iso_iso  · d(z_i, z_j))
    Gene–iso:     P(E_gi = 1) = sigmoid(γ_g              − β_gene_iso · d(u_g, z_i))
    Gene–gene:    P(C_gh = 1) = sigmoid(δ_g + δ_h        − β_gene_gene · d(u_g, u_h))

    Isoform latent positions (use_residuals=False, inductive):
        z_i = esmc_proj(esmc_i)                          purely ESM-C driven
    Isoform latent positions (use_residuals=True, transductive):
        z_i = esmc_proj(esmc_i) + isoform_residual_i    ESM-C + per-isoform correction
    Fallback (no ESM-C):
        z_i = isoform_embeddings_i

    Isoform random effects (interaction propensity):
        r_i = re_head(esmc_i) [+ re_residual_i]   (ESM-C mode, residual optional)
        r_i = random_effects_i                      (fallback — transductive only)

    Parameters:
        num_proteins:   number of unique isoforms/proteins in the network
        num_genes:      number of unique canonical genes
        latent_dim:     latent space dimensionality (e.g. 16, 32, 64, 128)
        esmc_features:  float32 tensor of shape (num_proteins, esmc_dim). If None,
                          falls back to learned embedding tables (no inductive capacity).
        use_residuals:  if True, add per-isoform correction embeddings on top of
                          the ESM-C projection. Set False in inductive mode to prevent
                          per-isoform memorization of seen training pairs.
    """

    def __init__(self, num_proteins, num_genes, latent_dim=32,
                 esmc_features=None, use_residuals=True, proj_hidden_dim=16):
        super().__init__()

        self.use_esmc      = esmc_features is not None
        self.use_residuals = use_residuals

        if self.use_esmc:
            self.register_buffer('esmc_features', esmc_features.float())
            esmc_dim = esmc_features.shape[1]

            # ESM-C projection: maps sequence embeddings to latent positions
            self.esmc_proj = ProjectionMLP(esmc_dim, proj_hidden_dim, latent_dim)

            # Random effect head: predicts propensity from ESM-C features
            self.re_head = ProjectionMLP(esmc_dim, proj_hidden_dim, 1)

            if self.use_residuals:
                # Per-isoform corrections on top of the ESM-C projection.
                # Only used in transductive mode — in inductive mode these
                # would memorize seen isoforms without benefiting unseen ones.
                self.isoform_residual = nn.Embedding(num_proteins, latent_dim)
                nn.init.zeros_(self.isoform_residual.weight)
                self.re_residual = nn.Embedding(num_proteins, 1)
                nn.init.zeros_(self.re_residual.weight)
        else:
            # Fallback: fully learned embedding table (transductive only)
            self.isoform_embeddings = nn.Embedding(num_proteins, latent_dim)
            nn.init.normal_(self.isoform_embeddings.weight, mean=0, std=0.1)

            self.random_effects = nn.Embedding(num_proteins, 1)
            nn.init.normal_(self.random_effects.weight, mean=0, std=0.1)

        # ── Gene latent space ─────────────────────────────────────────────────
        self.gene_embeddings = nn.Embedding(num_genes, latent_dim)
        nn.init.normal_(self.gene_embeddings.weight, mean=0, std=0.1)

        # ── Iso–iso parameters ────────────────────────────────────────────────
        self.beta_iso_iso = nn.Parameter(torch.tensor(1.0))

        # ── Gene–iso (bipartite) parameters ──────────────────────────────────
        self.beta_gene_iso      = nn.Parameter(torch.tensor(1.0))
        self.gene_iso_intercept = nn.Embedding(num_genes, 1)
        nn.init.normal_(self.gene_iso_intercept.weight, mean=0, std=0.1)

        # ── Gene–gene (STRING) parameters ─────────────────────────────────────
        self.beta_gene_gene      = nn.Parameter(torch.tensor(1.0))
        self.gene_gene_intercept = nn.Embedding(num_genes, 1)
        nn.init.normal_(self.gene_gene_intercept.weight, mean=0, std=0.1)

    def _isoform_latent(self, protein_idx):
        """Isoform latent position: ESM-C projection (+ optional residual), or learned embedding."""
        if self.use_esmc:
            z = self.esmc_proj(self.esmc_features[protein_idx])
            if self.use_residuals:
                z = z + self.isoform_residual(protein_idx)
            return z
        return self.isoform_embeddings(protein_idx)

    def _random_effect(self, protein_idx):
        """Per-isoform interaction propensity r_i."""
        if self.use_esmc:
            r = self.re_head(self.esmc_features[protein_idx]).squeeze(-1)
            if self.use_residuals:
                r = r + self.re_residual(protein_idx).squeeze(-1)
            return r
        return self.random_effects(protein_idx).squeeze(-1)

    def compute_distance(self, z1, z2):
        return torch.norm(z1 - z2, p=2, dim=1)

    def forward_iso_iso(self, protein1_idx, protein2_idx):
        z1   = self._isoform_latent(protein1_idx)
        z2   = self._isoform_latent(protein2_idx)
        dist = self.compute_distance(z1, z2)
        r1   = self._random_effect(protein1_idx)
        r2   = self._random_effect(protein2_idx)
        return r1 + r2 - nn.functional.softplus(self.beta_iso_iso) * dist

    def forward_gene_iso(self, gene_idx, protein_idx):
        u_g     = self.gene_embeddings(gene_idx)
        z_i     = self._isoform_latent(protein_idx)
        dist    = self.compute_distance(u_g, z_i)
        gamma_g = self.gene_iso_intercept(gene_idx).squeeze(-1)
        return gamma_g - nn.functional.softplus(self.beta_gene_iso) * dist

    def forward_gene_gene(self, gene_idx_a, gene_idx_b):
        u_g  = self.gene_embeddings(gene_idx_a)
        u_h  = self.gene_embeddings(gene_idx_b)
        dist = self.compute_distance(u_g, u_h)
        d_g  = self.gene_gene_intercept(gene_idx_a).squeeze(-1)
        d_h  = self.gene_gene_intercept(gene_idx_b).squeeze(-1)
        return d_g + d_h - nn.functional.softplus(self.beta_gene_gene) * dist

    def forward(self, protein1_idx, protein2_idx):
        return self.forward_iso_iso(protein1_idx, protein2_idx)

    def get_embeddings(self):
        return self.get_isoform_embeddings()

    def get_isoform_embeddings(self):
        if self.use_esmc:
            with torch.no_grad():
                all_idx = torch.arange(self.esmc_features.shape[0], device=self.esmc_features.device)
                return self._isoform_latent(all_idx).cpu().numpy()
        return self.isoform_embeddings.weight.detach().cpu().numpy()

    def get_gene_embeddings(self):
        return self.gene_embeddings.weight.detach().cpu().numpy()

    def get_random_effects(self):
        if self.use_esmc:
            with torch.no_grad():
                all_idx = torch.arange(self.esmc_features.shape[0], device=self.esmc_features.device)
                return self._random_effect(all_idx).unsqueeze(-1).cpu().numpy()
        return self.random_effects.weight.detach().cpu().numpy()

    @torch.no_grad()
    def init_gene_centroids(self, gene_to_idx, gene_to_isoforms, protein_to_idx):
        """
        Initialize each gene embedding at the mean projected position of its
        isoforms so the gene–isoform distance signal is immediately discriminative.

        Only pass isoforms from the training split to avoid leaking position
        information for held-out isoforms in inductive mode.
        """
        device = self.gene_embeddings.weight.device
        n_iso  = (self.esmc_features.shape[0] if self.use_esmc
                  else self.isoform_embeddings.weight.shape[0])

        all_idx       = torch.arange(n_iso, device=device)
        iso_positions = self._isoform_latent(all_idx)

        n_init = 0
        for gene_id, isoforms in gene_to_isoforms.items():
            if gene_id not in gene_to_idx:
                continue
            gene_idx    = gene_to_idx[gene_id]
            iso_indices = [protein_to_idx[iso] for iso in isoforms if iso in protein_to_idx]
            if not iso_indices:
                continue
            idx_tensor = torch.tensor(iso_indices, device=device)
            self.gene_embeddings.weight.data[gene_idx] = iso_positions[idx_tensor].mean(dim=0)
            n_init += 1

        # STRING-only genes (no isoforms) → place near global mean
        if n_init > 0:
            global_mean = self.gene_embeddings.weight.data[:max(gene_to_idx.values()) + 1].mean(dim=0)
            for gene_id, gene_idx in gene_to_idx.items():
                if gene_id not in gene_to_isoforms:
                    self.gene_embeddings.weight.data[gene_idx] = (
                        global_mean + torch.randn_like(global_mean) * 0.1)

        print(f"  Gene centroids initialized: {n_init:,} / {len(gene_to_idx):,} genes")



# =============================================================================
# Trainer
# =============================================================================

class MultimodalTrainer:
    """Trainer for MultimodalLDM."""

    def __init__(self, model, device='cpu'):
        self.model  = model.to(device)
        self.device = device

        self.train_iso_iso_losses   = []
        self.train_gene_iso_losses  = []
        self.train_gene_gene_losses = []
        self.val_losses             = []
        self.val_aucs               = []
        self.val_aps                = []

    def train_epoch(self, iso_iso_loader, gene_iso_loader, gene_gene_loader,
                    optimizer, crit_iso_iso, crit_gene_iso, crit_gene_gene,
                    lambda_iso_iso, lambda_gene_iso, lambda_gene_gene):
        self.model.train()

        n_steps = max(len(iso_iso_loader), len(gene_iso_loader), len(gene_gene_loader))

        # Cycle shorter loaders to match the longest so no signal is discarded.
        # Scale each modality's lambda down by its cycling factor so each unique
        # sample contributes equal total gradient weight per epoch regardless of
        # dataset size.
        eff_lambda_iso_iso   = lambda_iso_iso   / (n_steps / len(iso_iso_loader))
        eff_lambda_gene_iso  = lambda_gene_iso  / (n_steps / len(gene_iso_loader))
        eff_lambda_gene_gene = lambda_gene_gene / (n_steps / len(gene_gene_loader))

        iso_iso_iter   = (itertools.cycle(iso_iso_loader)
                          if len(iso_iso_loader) < n_steps else iter(iso_iso_loader))
        gene_iso_iter  = ((itertools.cycle(gene_iso_loader)
                           if len(gene_iso_loader) < n_steps else iter(gene_iso_loader))
                          if lambda_gene_iso > 0 else None)
        gene_gene_iter = ((itertools.cycle(gene_gene_loader)
                           if len(gene_gene_loader) < n_steps else iter(gene_gene_loader))
                          if lambda_gene_gene > 0 else None)

        total_iso_iso, total_gene_iso, total_gene_gene = 0.0, 0.0, 0.0

        for _ in range(n_steps):
            p1, p2, iso_iso_labels = next(iso_iso_iter)
            p1, p2, iso_iso_labels = (p1.to(self.device), p2.to(self.device),
                                      iso_iso_labels.to(self.device))

            iso_iso_logits  = self.model.forward_iso_iso(p1, p2)
            loss_iso_iso    = crit_iso_iso(iso_iso_logits, iso_iso_labels)
            # loss            = eff_lambda_iso_iso * loss_iso_iso
            total_iso_iso  += loss_iso_iso.item()

            if gene_iso_iter is not None:
                gene_idx, prot_idx, gene_iso_labels = next(gene_iso_iter)
                gene_idx, prot_idx, gene_iso_labels = (gene_idx.to(self.device),
                                                       prot_idx.to(self.device),
                                                       gene_iso_labels.to(self.device))
                gene_iso_logits  = self.model.forward_gene_iso(gene_idx, prot_idx)
                loss_gene_iso    = crit_gene_iso(gene_iso_logits, gene_iso_labels)
                total_gene_iso  += loss_gene_iso.item()

            if gene_gene_iter is not None:
                gene_idx_a, gene_idx_b, gene_gene_labels = next(gene_gene_iter)
                gene_idx_a, gene_idx_b, gene_gene_labels = (gene_idx_a.to(self.device),
                                                             gene_idx_b.to(self.device),
                                                             gene_gene_labels.to(self.device))
                gene_gene_logits  = self.model.forward_gene_gene(gene_idx_a, gene_idx_b)
                loss_gene_gene    = crit_gene_gene(gene_gene_logits, gene_gene_labels)
                total_gene_gene  += loss_gene_gene.item()
            
            # Combine losses from the three modalities
            loss = eff_lambda_iso_iso * loss_iso_iso + eff_lambda_gene_iso * loss_gene_iso + eff_lambda_gene_gene * loss_gene_gene

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_iso_iso / n_steps, total_gene_iso / n_steps, total_gene_gene / n_steps

    def validate(self, val_loader, criterion):
        """Validate on iso–iso pairs. Returns loss, AUC, AP."""
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for p1, p2, labels in val_loader:
                p1, p2, labels = p1.to(self.device), p2.to(self.device), labels.to(self.device)
                logits = self.model.forward_iso_iso(p1, p2)
                total_loss += criterion(logits, labels).item()
                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss  = total_loss / max(len(val_loader), 1)
        preds_arr = np.array(all_preds)
        auc = roc_auc_score(all_labels, preds_arr)
        ap  = average_precision_score(all_labels, preds_arr)
        return avg_loss, auc, ap

    def train(
        self,
        iso_iso_loader,
        gene_iso_loader,
        gene_gene_loader,
        val_loader,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-5,
        iso_iso_pos_weight=222.2,
        lambda_iso_iso=1.0,
        gene_iso_pos_weight=5.0,
        lambda_gene_iso=0.5,
        gene_gene_pos_weight=5.0,
        lambda_gene_gene=0.3,
        patience=10,
    ):
        """
        Full three-modality training loop.

        Gene–iso and gene–gene data are always fully used for training (no
        val/test split) — they are annotation/prior data, not experimental targets.
        Validation is iso–iso only, which is the actual prediction task.

        Args:
            patience: epochs without iso–iso val AP improvement before early stopping.
        """
        crit_iso_iso   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(iso_iso_pos_weight,   dtype=torch.float32).to(self.device))
        crit_gene_iso  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gene_iso_pos_weight,  dtype=torch.float32).to(self.device))
        crit_gene_gene = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gene_gene_pos_weight, dtype=torch.float32).to(self.device))

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_ap, best_epoch, best_state = 0.0, 0, None

        n_iso_iso   = len(iso_iso_loader)
        n_gene_iso  = len(gene_iso_loader)
        n_gene_gene = len(gene_gene_loader)
        n_steps     = max(n_iso_iso, n_gene_iso, n_gene_gene)

        print(f"Training MultimodalLDM (3 modalities) on {self.device}")
        print(f"  λ_iso_iso={lambda_iso_iso}  λ_gene_iso={lambda_gene_iso}  λ_gene_gene={lambda_gene_gene}")
        print(f"  pos_weights — iso_iso: {iso_iso_pos_weight:.2f}  gene_iso: {gene_iso_pos_weight:.2f}  gene_gene: {gene_gene_pos_weight:.2f}")
        print(f"  Steps/epoch: {n_steps}  (iso_iso={n_iso_iso}  gene_iso={n_gene_iso}  gene_gene={n_gene_gene}  → max, shorter loaders cycle)")
        print(f"  Effective λ — iso_iso: {lambda_iso_iso:.3f}  "
              f"gene_iso: {lambda_gene_iso / (n_steps/n_gene_iso):.3f}  "
              f"gene_gene: {lambda_gene_gene / (n_steps/n_gene_gene):.3f}  (scaled by cycling factor)")
        print(f"  Early stopping patience: {patience}")
        print("-" * 70)

        for epoch in range(epochs):
            iso_iso_loss, gene_iso_loss, gene_gene_loss = self.train_epoch(
                iso_iso_loader, gene_iso_loader, gene_gene_loader,
                optimizer, crit_iso_iso, crit_gene_iso, crit_gene_gene,
                lambda_iso_iso, lambda_gene_iso, lambda_gene_gene,
            )

            val_loss, val_auc, val_ap = self.validate(val_loader, crit_iso_iso)

            self.train_iso_iso_losses.append(iso_iso_loss)
            self.train_gene_iso_losses.append(gene_iso_loss)
            self.train_gene_gene_losses.append(gene_gene_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)

            scheduler.step(val_ap)

            if val_ap > best_ap:
                best_ap    = val_ap
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1}/{epochs}  "
                  f"L_ii={iso_iso_loss:.4f}  L_gi={gene_iso_loss:.4f}  L_gg={gene_gene_loss:.4f}  "
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
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.val_aucs, label='Val AUC', color='green')
        axes[1].plot(self.val_aps,  label='Val AP',  color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics (Iso–Iso)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        return fig
