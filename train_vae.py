"""
BioLACE — VAE Training Script

Inputs (CLI)
--base            Directory containing data (e.g Slideseq).
--counts_csv      Raw counts (genes × barcodes). Default: MappedDGEForR.csv
--coords_csv      Bead coordinates. Default: BeadLocationsForR.csv
--sim_dir         Precomputed similarity matrix (.npy), aligned to barcodes.
--outdir          Output directory.

Data Processing
1. Load counts + spatial coordinates, construct AnnData.
2. Normalize + log-transform expression.
3. Load similarity matrix, reorder to match AnnData cells, build pos/neg masks.
4. Construct kNN Laplacian from spatial coordinates (optional if coords missing).

Training Procedure
• Per epoch:
    - Compute full-batch mean embeddings Z (used for Laplacian + contrastive).
    - After warmup, apply:
         * Laplacian loss: trace(Zᵀ L Z)/(2N)
         * Contrastive loss (InfoNCE over mined pos/neg pairs)
    - Iterate over batches to compute:
         * Reconstruction loss (MSE)
         * KL divergence
    - Optimize total loss:
         L_total = L_rec + β * L_KL + λ_lap * L_lap + L_contrastive

Outputs
    latent_embeddings.npy   (N × d latent matrix)
    latent_similarity.npy   (cosine similarity in latent space)
    latent_pred.npy         (Leiden clusters from latent space)
    barcodes.txt            (barcode ordering)
    *_spatial_clusters.png  (spatial plot colored by latent clusters)
    *_umap_clusters.png     (UMAP plot colored by latent clusters)
"""

import os
import gc
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import pairwise_distances

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inputs and hyperparameters for BioLACE")
    # input/output paths
    parser.add_argument('--base', type=str, required=True, help='Base directory for Slide-seq data')
    parser.add_argument('--counts_csv', type=str, default='MappedDGEForR.csv', help='Counts CSV filename')
    parser.add_argument('--coords_csv', type=str, default='BeadLocationsForR.csv', help='Coords CSV filename')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    # similarity
    parser.add_argument('--sim_dir', type=str, required=True, help='Path to similarity .npy file')
    parser.add_argument('--high_thresh', type=float, required=True, help='High threshold for similarity')
    parser.add_argument('--low_thresh', type=float, required=True, help='Low threshold for similarity')
    # hyperparameters
    parser.add_argument('--latent_dim', type=int, default=9, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=1000, help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='Epochs of pure VAE loss before regularization')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--beta_kl', type=float, default=1e-3, help='Beta for KL loss')
    parser.add_argument('--lambda_lap', type=float, required=True, help='Lambda for Laplacian loss')
    parser.add_argument('--lambda_cont', type=float, required=True, help='Lambda for Contrastive loss')
    parser.add_argument('--knn_k_graph', type=int, default=6, help='KNN k for Laplacian graph')
    parser.add_argument('--n_neighbors_graph', type=int, default=6, help='Neighbors for Scanpy preprocessing')
    parser.add_argument('--leiden_resolution', type=float, default=1.0, help='Leiden clustering resolution')
    # reproducibility
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser.parse_args()

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def read_slideseq(counts_csv, coords_csv, min_counts=40):
    """
    Read Slide-seq counts and coordinates, align barcodes, and return an AnnData.
    """
    # coordinates
    df_loc = pd.read_csv(coords_csv)
    bar_col = [c for c in df_loc.columns if 'bar' in c.lower()][0]
    x_col   = [c for c in df_loc.columns if c.lower().startswith('x')][0]
    y_col   = [c for c in df_loc.columns if c.lower().startswith('y')][0]
    loc = df_loc[[bar_col, x_col, y_col]].copy()
    loc.columns = ['barcode', 'xcoord', 'ycoord']
    loc['barcode'] = loc['barcode'].astype(str)

    # counts (chunked to sparse)
    reader = pd.read_csv(counts_csv, chunksize=20000)
    rows, cols, data = [], [], []
    genes, barcodes, offset = None, None, 0
    for chunk in reader:
        if genes is None:
            genes = chunk.iloc[:, 0].astype(str).values
        if barcodes is None:
            barcodes = chunk.columns[1:].astype(str).values
        M = chunk.iloc[:, 1:].to_numpy(dtype=np.float32)
        rr, cc = np.nonzero(M)
        rows.append(rr + offset)
        cols.append(cc)
        data.append(M[rr, cc])
        offset += M.shape[0]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    X = sp.csr_matrix((data, (rows, cols)), shape=(offset, len(barcodes)), dtype=np.float32).T

    # keep only barcodes with coordinates
    keep_mask = np.isin(barcodes, loc['barcode'].values)
    X = X[keep_mask]
    barcodes = barcodes[keep_mask]
    totals = np.asarray(X.sum(1)).ravel()
    good = totals >= min_counts
    X = X[good]
    barcodes = barcodes[good]
    totals = totals[good]

    obs = pd.DataFrame(index=barcodes)
    var = pd.DataFrame(index=genes)
    loc = loc.set_index('barcode').loc[obs.index]
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs['xcoord'] = loc['xcoord'].values.astype(np.float32)
    adata.obs['ycoord'] = loc['ycoord'].values.astype(np.float32)
    adata.obsm['spatial'] = np.c_[adata.obs['xcoord'].values,
                                  adata.obs['ycoord'].values].astype(np.float32)
    adata.obs['total_counts'] = totals.astype(np.float32)
    return adata

# ----------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.enc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.enc_fc(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def encode_mu(self, x):
        mu, _ = self.encode(x)
        return mu

# ----------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------
def recon_loss_mse(x, x_hat):
    return F.mse_loss(x_hat, x, reduction='sum')

def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def laplacian_from_coords(coords, k=6, device="cpu"):
    """
    Build an unweighted kNN graph from coordinates and return its Laplacian.
    """
    from sklearn.neighbors import NearestNeighbors
    N = coords.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    A = torch.zeros((N, N), device=device)
    for i, neigh in enumerate(idx):
        A[i, neigh] = 1
        A[neigh, i] = 1
    D = torch.diag(A.sum(1))
    L = (D - A)
    return L

def laplacian_loss(Z, L):
    if L is None:
        return torch.tensor(0., device=Z.device)
    # trace(Z^T L Z) / (2N)
    return torch.trace(Z.t() @ L @ Z) / (2 * Z.size(0))

def info_nce_loss(Z, pos_mask, neg_mask, tau=0.5, alpha=1.0):
    """
    Marker-guided InfoNCE:
      - Z: [N, d] latent embeddings
      - pos_mask, neg_mask: [N, N] boolean masks for positives/negatives
    Uses a per-anchor temperature based on the number of positives.
    """
    N = Z.size(0)
    Z = F.normalize(Z, p=2, dim=1)
    raw_sim = Z @ Z.T

    # per-anchor temperatures
    pos_counts = pos_mask.sum(dim=1).clamp(min=1).float()
    max_cnt    = pos_counts.max()
    tau_i      = (tau * (pos_counts / max_cnt).pow(alpha)).to(Z.device)

    # scaled similarities
    sim = raw_sim / tau_i[:, None]

    # denominator mask (pos ∪ neg)
    bg  = pos_mask | neg_mask
    den = sim.masked_fill(~bg, float('-inf'))
    num = sim.masked_fill(~pos_mask, float('-inf'))

    log_den = torch.logsumexp(den, dim=1)
    log_num = torch.logsumexp(num, dim=1)
    losses  = -(log_num - log_den)

    valid = pos_mask.sum(dim=1) > 0
    if not valid.any():
        return torch.tensor(0., device=Z.device)

    mean_cnt = pos_counts.mean()
    weights  = (mean_cnt / pos_counts).to(Z.device)

    l_valid = losses[valid]
    w_valid = weights[valid]
    return (w_valid * l_valid).sum() / w_valid.sum()

# ----------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------ Paths and hyperparameters ------------------
    BASE = Path(args.base)
    COUNTS_CSV = Path(args.counts_csv) if args.counts_csv else BASE / "MappedDGEForR.csv"
    COORDS_CSV = Path(args.coords_csv) if args.coords_csv else BASE / "BeadLocationsForR.csv"

    SIM_DIR = args.sim_dir
    high_thresh = args.high_thresh
    low_thresh = args.low_thresh

    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim
    epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    batch_size = args.batch_size
    beta_kl = args.beta_kl
    lambda_lap = args.lambda_lap
    lambda_cont = args.lambda_cont
    knn_k_graph = args.knn_k_graph
    n_neighbors_graph = args.n_neighbors_graph
    leiden_resolution = args.leiden_resolution

    # ------------------ Reproducibility and device ------------------
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------ Scanpy setup and preprocessing --------------
    sc.settings.verbosity = 2
    sc.logging.print_header()
    sc.set_figure_params(facecolor="white", figsize=(6, 6))

    adata = read_slideseq(COUNTS_CSV, COORDS_CSV, min_counts=40)
    print(f"[INFO] AnnData: {adata.n_obs} beads × {adata.n_vars} genes")

    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ------------------ Tensor construction ------------------------
    X_mat = adata.X.toarray() if sp.isspmatrix(adata.X) else np.asarray(adata.X)
    expr_cpu = torch.tensor(X_mat, dtype=torch.float32)
    expr_dev_full = expr_cpu.to(device, non_blocking=True)

    N, G = expr_cpu.shape
    print(f"[INFO] Tensor shape: {N}×{G}")

    # ------------------ Load and align similarity -------------------
    coord_all = pd.read_csv(COORDS_CSV, index_col=0)
    full_order = coord_all.index.astype(str).to_numpy()

    full_idx = pd.Index(full_order)
    keep_idx = full_idx.get_indexer(adata.obs_names.astype(str))

    sim_np = np.load(SIM_DIR).astype(np.float32)
    if sim_np.shape[0] != sim_np.shape[1]:
        raise ValueError(f"sim must be square; got {sim_np.shape}")
    sim_np = sim_np[np.ix_(keep_idx, keep_idx)]
    sim = torch.from_numpy(sim_np).to(device)

    N = adata.n_obs
    eye      = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = (sim >= high_thresh) & ~eye
    neg_mask = (sim <= low_thresh) & ~eye

    print(f"[OK] Aligned sim to adata: sim={tuple(sim.shape)}  adata={adata.n_obs}×{adata.n_vars}")

    # ------------------ Spatial Laplacian (optional) ---------------
    L = None
    if "spatial" in adata.obsm and adata.obsm["spatial"] is not None:
        try:
            L = laplacian_from_coords(adata.obsm["spatial"], k=knn_k_graph, device=device)
        except Exception as e:
            print(f"[WARN] Could not build Laplacian (continuing without): {e}")
            L = None

    # ------------------ Dataloader ---------------------------------
    dataset = TensorDataset(expr_cpu)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------ Model and optimizer -------------------------
    model = VAE(G, hidden_dim, latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------ Training loop ------------------------------
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # full-batch mean embeddings for Laplacian and contrastive terms
        with torch.no_grad():
            Z_mu_full = model.encode_mu(expr_dev_full)
        Z_full = F.normalize(Z_mu_full, p=2, dim=1)

        # regularizers (Laplacian + contrastive) after warmup
        if ep <= warmup_epochs:
            loss_lap = torch.tensor(0., device=device)
            cont_loss = torch.tensor(0., device=device)
        else:
            loss_lap = laplacian_loss(Z_full, L)
            cont_loss = info_nce_loss(Z_full, pos_mask, neg_mask)

        # minibatch reconstruction + KL
        loss_rec = torch.tensor(0., device=device)
        loss_kl  = torch.tensor(0., device=device)
        for (xb_cpu,) in loader:
            xb = xb_cpu.to(device, non_blocking=True)
            x_hat, mu_b, logvar_b, _ = model(xb)
            loss_rec += recon_loss_mse(xb, x_hat)
            loss_kl  += kld_loss(mu_b, logvar_b)
        if len(loader) > 0:
            loss_rec /= len(loader)
            loss_kl  /= len(loader)

        total = loss_rec + beta_kl * loss_kl + lambda_lap * loss_lap + lambda_cont * cont_loss
        total.backward()
        opt.step()

        if ep == 1 or ep % 10 == 0:
            print(
                f"[{ep:04d}] REC={loss_rec.item():.2f}  "
                f"KL={loss_kl.item():.2f}  LAP={loss_lap.item():.4f}  "
                f"Cont={cont_loss.item():.4f}"
            )

    # ------------------ Embedding and outputs ----------------------
    model.eval()
    with torch.no_grad():
        Z_eval = model.encode_mu(expr_dev_full)
    Z_np = F.normalize(Z_eval, p=2, dim=1).detach().cpu().numpy()  # [N, d]

    # latent cosine similarity
    Dcos = pairwise_distances(Z_np, Z_np, metric='cosine', n_jobs=1)
    S_lat = (1.0 - Dcos).astype(np.float32)
    np.fill_diagonal(S_lat, 0.0)

    np.save(OUTDIR / "latent_embeddings.npy", Z_np)
    np.save(OUTDIR / "latent_similarity.npy", S_lat)
    np.savetxt(OUTDIR / "barcodes.txt", adata.obs_names.astype(str).to_numpy(), fmt="%s")
    print(f"[SAVE] latent_embeddings.npy, latent_similarity.npy, barcodes.txt → {OUTDIR}")

    # Scanpy neighbors + Leiden on VAE latent space
    adata.obsm["X_vae"] = Z_np
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_graph, use_rep="X_vae", metric="cosine")
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added="leiden_vae")

    # Save cluster assignments as numeric .npy (original behavior)
    clusters = adata.obs["leiden_vae"].astype(int, errors="ignore").to_numpy()
    np.save(OUTDIR / "latent_pred.npy", clusters)
    print(f"[SAVE] latent_pred.npy → {OUTDIR}")

    # Also save human-readable barcode → cluster mapping
    cluster_csv = pd.DataFrame({
        "barcode": adata.obs_names.astype(str),
        "cluster": adata.obs["leiden_vae"].astype(str)
    })
    cluster_csv.to_csv(OUTDIR / "cluster_assignments.csv", index=False)

    print(f"[SAVE] cluster_assignments.csv → {OUTDIR}")

    # spatial view of latent clusters
    sc.pl.scatter(
        adata,
        x="xcoord",
        y="ycoord",
        color="leiden_vae",
        title="VAE latent clusters (spatial)",
        size=5,
        save="_spatial_clusters.png",
        show=False,
    )
    sc.pl.umap(
        adata,
        color="leiden_vae",
        title="VAE latent clusters (UMAP)",
        size=10,
        save="_umap_clusters.png",
        show=False,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
