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
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Model
# =============================================================================

class MultimodalLDM(nn.Module):
    """
    Multimodal Latent Distance Model.

    Isoform–isoform:  P(Y_ij = 1) = sigmoid(r_i + r_j − β_iso  · d(z_i, z_j))
    Gene–isoform:     P(E_gi = 1) = sigmoid(γ_g        − β_gene · d(u_g, z_i))
    Gene–gene:        P(C_gh = 1) = sigmoid(δ_g + δ_h  − β_gg   · d(u_g, u_h))

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
                 esmc_features=None, use_residuals=True):
        super().__init__()

        self.use_esmc      = esmc_features is not None
        self.use_residuals = use_residuals

        if self.use_esmc:
            self.register_buffer('esmc_features', esmc_features.float())
            esmc_dim = esmc_features.shape[1]

            # ESM-C projection: maps sequence embeddings to latent positions
            self.esmc_proj = ProjectionMLP(esmc_dim, 256, latent_dim)

            # Random effect head: predicts propensity from ESM-C features
            self.re_head = ProjectionMLP(esmc_dim, 128, 1)

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

        # ── Isoform–isoform parameters ────────────────────────────────────────
        self.beta_iso = nn.Parameter(torch.tensor(1.0))

        # ── Gene–isoform (bipartite) parameters ──────────────────────────────
        self.beta_gene      = nn.Parameter(torch.tensor(1.0))
        self.gene_intercept = nn.Embedding(num_genes, 1)
        nn.init.normal_(self.gene_intercept.weight, mean=0, std=0.1)

        # ── Gene–gene (complex co-membership) parameters ─────────────────────
        self.beta_complex = nn.Parameter(torch.tensor(1.0))
        self.gene_re      = nn.Embedding(num_genes, 1)
        nn.init.normal_(self.gene_re.weight, mean=0, std=0.1)

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

    def forward_isoform(self, protein1_idx, protein2_idx):
        z1   = self._isoform_latent(protein1_idx)
        z2   = self._isoform_latent(protein2_idx)
        dist = self.compute_distance(z1, z2)
        r1   = self._random_effect(protein1_idx)
        r2   = self._random_effect(protein2_idx)
        return r1 + r2 - nn.functional.softplus(self.beta_iso) * dist

    def forward_bipartite(self, gene_idx, protein_idx):
        u_g     = self.gene_embeddings(gene_idx)
        z_i     = self._isoform_latent(protein_idx)
        dist    = self.compute_distance(u_g, z_i)
        gamma_g = self.gene_intercept(gene_idx).squeeze(-1)
        return gamma_g - nn.functional.softplus(self.beta_gene) * dist

    def forward_complex(self, gene_idx_a, gene_idx_b):
        u_g  = self.gene_embeddings(gene_idx_a)
        u_h  = self.gene_embeddings(gene_idx_b)
        dist = self.compute_distance(u_g, u_h)
        d_g  = self.gene_re(gene_idx_a).squeeze(-1)
        d_h  = self.gene_re(gene_idx_b).squeeze(-1)
        return d_g + d_h - nn.functional.softplus(self.beta_complex) * dist

    # API compatibility with evaluate.py / visualize.py
    def forward(self, protein1_idx, protein2_idx):
        return self.forward_isoform(protein1_idx, protein2_idx)

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

        self.train_iso_losses     = []
        self.train_gene_losses    = []
        self.train_complex_losses = []
        self.val_losses           = []
        self.val_aucs             = []
        self.val_aps              = []

    def train_epoch(self, iso_loader, gene_loader, complex_loader,
                    optimizer, crit_iso, crit_gene, crit_complex,
                    lambda_iso, lambda_gene, lambda_complex):
        self.model.train()

        n_steps = max(len(iso_loader), len(gene_loader), len(complex_loader))

        # Cycle shorter loaders to match the longest so no signal is discarded.
        # Scale each modality's lambda down by its cycling factor so each unique
        # sample contributes equal total gradient weight per epoch regardless of
        # dataset size.
        eff_lambda_iso     = lambda_iso     / (n_steps / len(iso_loader))
        eff_lambda_gene    = lambda_gene    / (n_steps / len(gene_loader))
        eff_lambda_complex = lambda_complex / (n_steps / len(complex_loader))

        iso_iter     = itertools.cycle(iso_loader)     if len(iso_loader)     < n_steps else iter(iso_loader)
        gene_iter    = itertools.cycle(gene_loader)    if len(gene_loader)    < n_steps else iter(gene_loader)
        complex_iter = itertools.cycle(complex_loader) if len(complex_loader) < n_steps else iter(complex_loader)

        total_iso, total_gene, total_complex = 0.0, 0.0, 0.0

        for _ in range(n_steps):
            p1, p2, iso_labels           = next(iso_iter)
            gene_idx, prot_idx, g_labels = next(gene_iter)
            ga, gb, c_labels             = next(complex_iter)

            p1, p2, iso_labels = p1.to(self.device), p2.to(self.device), iso_labels.to(self.device)
            gene_idx, prot_idx, g_labels = (gene_idx.to(self.device),
                                            prot_idx.to(self.device),
                                            g_labels.to(self.device))
            ga, gb, c_labels = ga.to(self.device), gb.to(self.device), c_labels.to(self.device)

            iso_logits     = self.model.forward_isoform(p1, p2)
            gene_logits    = self.model.forward_bipartite(gene_idx, prot_idx)
            complex_logits = self.model.forward_complex(ga, gb)

            loss_iso     = crit_iso(iso_logits, iso_labels)
            loss_gene    = crit_gene(gene_logits, g_labels)
            loss_complex = crit_complex(complex_logits, c_labels)

            loss = eff_lambda_iso * loss_iso + eff_lambda_gene * loss_gene + eff_lambda_complex * loss_complex

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iso     += loss_iso.item()
            total_gene    += loss_gene.item()
            total_complex += loss_complex.item()

        return total_iso / n_steps, total_gene / n_steps, total_complex / n_steps

    def validate(self, val_loader, criterion):
        """Validate on isoform–isoform pairs. Returns loss, AUC, AP."""
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
        return avg_loss, auc, ap

    def train(
        self,
        iso_train_loader,
        gene_train_loader,
        complex_train_loader,
        val_loader,
        epochs=30,
        lr=1e-3,
        weight_decay=1e-5,
        iso_pos_weight=222.2,
        lambda_iso=1.0,
        gene_pos_weight=5.0,
        lambda_gene=0.5,
        complex_pos_weight=5.0,
        lambda_complex=0.3,
        patience=10,
    ):
        """
        Full three-modality training loop.

        Gene-isoform and gene-gene data are always fully used for training (no
        val/test split) — they are annotation/prior data, not experimental targets.
        Validation is isoform-isoform only, which is the actual prediction task.

        Args:
            patience: epochs without iso val AP improvement before early stopping.
        """
        crit_iso     = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(iso_pos_weight,     dtype=torch.float32).to(self.device))
        crit_gene    = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(gene_pos_weight,    dtype=torch.float32).to(self.device))
        crit_complex = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(complex_pos_weight, dtype=torch.float32).to(self.device))

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_ap, best_epoch, best_state = 0.0, 0, None

        n_iso     = len(iso_train_loader)
        n_gene    = len(gene_train_loader)
        n_complex = len(complex_train_loader)
        n_steps   = max(n_iso, n_gene, n_complex)

        print(f"Training MultimodalLDM (3 modalities) on {self.device}")
        print(f"  λ_iso={lambda_iso}  λ_gene={lambda_gene}  λ_complex={lambda_complex}")
        print(f"  pos_weights — iso: {iso_pos_weight:.2f}  gene: {gene_pos_weight:.2f}  complex: {complex_pos_weight:.2f}")
        print(f"  Steps/epoch: {n_steps}  (iso={n_iso}  gene={n_gene}  complex={n_complex}  → max, shorter loaders cycle)")
        print(f"  Effective λ — iso: {lambda_iso:.3f}  "
              f"gene: {lambda_gene / (n_steps/n_gene):.3f}  "
              f"complex: {lambda_complex / (n_steps/n_complex):.3f}  (scaled by cycling factor)")
        print(f"  Early stopping patience: {patience}")
        print("-" * 70)

        for epoch in range(epochs):
            iso_loss, gene_loss, complex_loss = self.train_epoch(
                iso_train_loader, gene_train_loader, complex_train_loader,
                optimizer, crit_iso, crit_gene, crit_complex,
                lambda_iso, lambda_gene, lambda_complex,
            )

            val_loss, val_auc, val_ap = self.validate(val_loader, crit_iso)

            self.train_iso_losses.append(iso_loss)
            self.train_gene_losses.append(gene_loss)
            self.train_complex_losses.append(complex_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_aps.append(val_ap)

            scheduler.step(val_ap)

            if val_ap > best_ap:
                best_ap    = val_ap
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            print(f"Epoch {epoch+1:3d}/{epochs}  "
                  f"L_iso={iso_loss:.4f}  L_gene={gene_loss:.4f}  L_cplx={complex_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  AUC={val_auc:.4f}  AP={val_ap:.4f}")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\nBest validation AP: {best_ap:.4f}")
        return best_ap

    def plot_training(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(self.train_iso_losses,     label='Train iso')
        axes[0].plot(self.train_gene_losses,    label='Train gene')
        axes[0].plot(self.train_complex_losses, label='Train complex')
        axes[0].plot(self.val_losses,           label='Val iso', linestyle='--')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.val_aucs, label='Val AUC', color='green')
        axes[1].plot(self.val_aps,  label='Val AP',  color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('Validation Metrics (Isoform–Isoform)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot([g / (i + 1e-9) for i, g in zip(self.train_iso_losses, self.train_gene_losses)],
                     color='steelblue', label='gene / iso')
        axes[2].plot([c / (i + 1e-9) for i, c in zip(self.train_iso_losses, self.train_complex_losses)],
                     color='tomato', label='complex / iso')
        axes[2].axhline(1.0, color='grey', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Epoch')
        axes[2].set_title('Loss Ratios (relative to iso loss)')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        return fig
