# **BioLACE**
Unifying spatial geometry and marker-gene priors for cohesive cell-type clustering in spatial transcriptomics.

BioLACE is a biologically guided representation learning framework for spatial transcriptomics. It integrates:
- **VAE-based transcriptomic embeddings**
- **Spatial graph Laplacian regularization**
- **Marker-informed contrastive learning**

The result is a latent space that preserves tissue geometry while separating biologically distinct cell types—resolving the classic *oversmoothing vs. over-segmentation* trade-off.

---

## **Project Structure**

| File | Description |
|------|------------|
| `train_vae.py` | Main training loop for BioLACE (VAE + Laplacian + contrastive objectives). |
| `similarity_construction.ipynb` | Notebook demonstrating similarity matrices (marker-only, spatial-only, fused). |
| `post_vae_clustering.ipynb` | Notebook for clustering analysis, plotting, and downstream visualization. |
| `README.md` | Project documentation (this file). |

---

## **Key Ideas for Marker Similarity**

### **1. Marker-Guided Similarity**
We extract cluster-enriched marker genes using Leiden + Wilcoxon testing to form a biological similarity matrix:

`S_marker = cosine( X_markers )`

### **2. Spatial-Neighbor Aggregated Similarity**
We smooth features based on graph neighbors with geometric decay:

`Aggregated = X_self + λ ⋅ mean(X_neighbors)`

Then compute cosine similarity:

`S_spatial = cosine( Aggregated )`

### **3. Fused Similarity for Contrastive Learning**
We combine spatial and biological similarity using a Hadamard product:

`S_fused = S_marker ⊙ S_spatial`

Thresholds are identified via **two-centroid K-means**, defining positive and negative contrastive pairs.

---

## **Similarity Demonstration (MERFISH Example)**

We demonstrate similarity construction on **MERFISH mouse spinal cord** where anatomical ground-truth is reliable.

Three matrices are computed:

1. **Leiden-Marker Similarity**  
   - Uses markers derived from pseudo-Leiden clusters

2. **Spatial-Neighbor Aggregated Similarity**  
   - Captures denoised spatial structure

3. **Final Fused Similarity**
   - High-sim pairs pulled together
   - Low-sim pairs pushed apart via InfoNCE

Plots are generated in `similarity_construction.ipynb`.

---

## **Dependencies**

We recommend a clean conda environment:

```bash
conda create -n biolace python=3.10
conda activate biolace
pip install scanpy squidpy torch scikit-learn seaborn matplotlib numpy
```

---

## **Running Training**
An example command to run training:
```bash
python train_vae.py --input path/to/data.h5ad --sim_dir path/to/similarity.npy --high_thresh 0.65 --low_thresh 0.35 --lambda_lap 1e3 --lambda_cont 1e4
```

---

## **Data Format**

Input must be an `.h5ad` file containing:

| Field | Required? | Description |
|-------|----------|-------------|
| `adata.X` | ✓ | gene expression (normalized) |
| `adata.obsm["spatial"]` | ✓ | (x,y) coordinates |
| `adata.obs[celltype]` | optional | used for evaluation only |
| `adata.var` | optional | stores marker metadata |

## **Maintainers**

Developed by **Haoran (Hunter) Qin** and the Zhou Lab @ Vanderbilt University.  
Contact: `maizie.zhou@vanderbilt.edu`

---

