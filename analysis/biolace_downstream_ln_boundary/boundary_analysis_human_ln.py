#!/usr/bin/env python3
"""
Boundary program analysis around user-selected cancer cluster(s).

This script is designed for public display and intentionally uses the clusters
produced by a method such as BioLACE. It does not infer which cluster is cancer
and it does not use ground-truth annotation to define the boundary.

Workflow
--------
[1] Data loading
[2] Cluster loading
[3] Cluster selection: use only the cluster IDs passed by --tumor-clusters
[4] Boundary calculation: signed distance to selected cancer-cluster boundary
[5] Marker selection: four Open-ST boundary programs
[6] Module score calculation: log1p, gene-wise z-score, program average
[7] Boundary curves and spatial program maps

Interpretation
--------------
Negative signed distance = inside the selected method-predicted cancer cluster(s).
Positive signed distance = outside the selected method-predicted cancer cluster(s).
Distance near zero = boundary of the selected cancer cluster(s).

Example
-------
python boundary_program_analysis_from_predicted_clusters_v2.py \
  --h5ad /maiziezhou_lab/Datasets/ST_datasets/humanMetastaticLymphNode/GSE251926_metastatic_lymph_node_3d.h5ad \
  --backed \
  --section-col n_section \
  --section-id 19 \
  --cluster-path ./vae_human_ln_slice19_gt_similarity \
  --tumor-clusters 3,7 \
  --outdir ./human_ln_boundary_from_biolace_selected_cancer_clusters
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Utilities
# =============================================================================


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")
    return text[:max_len] if text else "NA"


def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    if hasattr(x, "A"):
        return x.A
    return np.asarray(x)


def robust_zscore(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score with safe handling of zero-variance genes."""
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def read_table_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def parse_cluster_ids(text: str) -> List[str]:
    ids = [x.strip() for x in re.split(r"[,;\s]+", str(text)) if x.strip()]
    if not ids:
        raise ValueError("--tumor-clusters must contain at least one cluster ID")
    return ids


# =============================================================================
# [1] Data loading
# =============================================================================


def load_h5ad_section(args):
    """Load one tissue section, cell IDs, coordinates, and optional annotation."""
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover
        raise ImportError("scanpy is required to read h5ad files") from exc

    log("[1/7] Data loading: reading h5ad and selecting section")
    adata = sc.read_h5ad(args.h5ad, backed="r" if args.backed else None)

    obs_all = adata.obs
    if args.section_col and args.section_id is not None:
        if args.section_col not in obs_all.columns:
            raise KeyError(f"Section column {args.section_col!r} not found in adata.obs")
        section_values = obs_all[args.section_col].astype(str).values
        section_index = np.where(section_values == str(args.section_id))[0]
    else:
        section_index = np.arange(adata.n_obs)

    if len(section_index) == 0:
        raise ValueError(f"No cells matched {args.section_col} == {args.section_id}")

    obs = adata.obs.iloc[section_index].copy()
    obs_names = np.asarray(obs.index.astype(str))

    if args.spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[args.spatial_key][section_index, :2], dtype=np.float32)
    elif {"x", "y"}.issubset(obs.columns):
        coords = obs[["x", "y"]].values.astype(np.float32)
    elif {"center_x", "center_y"}.issubset(obs.columns):
        coords = obs[["center_x", "center_y"]].values.astype(np.float32)
    else:
        raise KeyError(
            f"Could not find spatial coordinates in obsm[{args.spatial_key!r}], "
            "obs['x','y'], or obs['center_x','center_y']"
        )

    annotation = None
    if args.annotation_col:
        if args.annotation_col in obs.columns:
            annotation = obs[args.annotation_col].astype(str).values
        else:
            warnings.warn(
                f"Annotation column {args.annotation_col!r} not found; "
                "optional validation crosstab will be skipped"
            )

    log(f"      selected {len(section_index):,} cells")
    return adata, section_index, obs, obs_names, coords, annotation


# =============================================================================
# [2] Cluster loading
# =============================================================================


def align_table_to_obs(df: pd.DataFrame, labels: np.ndarray, obs_names: np.ndarray) -> np.ndarray:
    """Align labels to h5ad obs_names using an ID column when available."""
    labels = np.asarray(labels).astype(str)
    key = first_existing_column(df, ["obs_name", "cell_id", "cell_ids", "barcode"])

    if key is None:
        if len(labels) != len(obs_names):
            raise ValueError(
                "Label table has no obs_name/cell_id/barcode column and row count does not "
                "match the selected h5ad section. Provide a label table with cell IDs or a "
                "label vector already aligned to this section."
            )
        return labels

    mapping = pd.Series(labels, index=df[key].astype(str).values)
    aligned = mapping.reindex(pd.Series(obs_names).astype(str).values)
    n_missing = int(aligned.isna().sum())
    if n_missing:
        warnings.warn(f"{n_missing} cells missing when aligning labels by {key}; filling as Missing")
    return aligned.fillna("Missing").astype(str).values


def choose_label_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    """Choose the method-generated cluster column from an assignment table."""
    if requested:
        if requested not in df.columns:
            raise KeyError(f"Requested cluster column {requested!r} not found. Available columns: {list(df.columns)}")
        return requested

    pred_cols = [c for c in df.columns if str(c).startswith("pred_")]
    if pred_cols:
        return pred_cols[0]

    col = first_existing_column(df, ["pred", "cluster", "label", "leiden", "kmeans", "gmm", "prediction"])
    if col is None:
        raise KeyError(
            "Could not infer a method-generated cluster column. "
            "Pass --cluster-column COLUMN. Available columns: " + ", ".join(map(str, df.columns))
        )
    return col


def load_labels_from_directory(path: Path, obs_names: np.ndarray, cluster_column: Optional[str]) -> np.ndarray:
    """Load labels from VAE output, reclustering output, or generic assignment directory."""
    # Recluster output from recluster_vae_result.py.
    rec_dir = path / "recluster" if (path / "recluster").exists() else path
    assignments = rec_dir / "recluster_assignments.csv"
    summary = rec_dir / "recluster_summary.csv"
    if assignments.exists():
        df = pd.read_csv(assignments)
        col = cluster_column
        if col is None and summary.exists():
            s = pd.read_csv(summary)
            if not s.empty and "method" in s.columns:
                best_method = str(s.iloc[0]["method"])
                candidate = f"pred_{best_method}"
                if candidate in df.columns:
                    col = candidate
        col = choose_label_column(df, col)
        return align_table_to_obs(df, df[col].values, obs_names)

    # VAE output directory with cells.csv + pred.npy.
    cells = path / "cells.csv"
    for pred_name in ["pred.npy", "best_pred.npy", "cluster_labels.npy", "labels.npy"]:
        pred_path = path / pred_name
        if pred_path.exists():
            labels = np.load(pred_path, allow_pickle=True).astype(str)
            if cells.exists():
                df = pd.read_csv(cells)
                return align_table_to_obs(df, labels, obs_names)
            if len(labels) != len(obs_names):
                raise ValueError(f"{pred_path} has {len(labels)} labels but selected h5ad section has {len(obs_names)} cells")
            return labels

    # Generic CSVs.
    for csv_name in ["cells.csv", "assignments.csv", "clusters.csv", "labels.csv", "metadata.csv"]:
        csv_path = path / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = choose_label_column(df, cluster_column)
            return align_table_to_obs(df, df[col].values, obs_names)

    raise FileNotFoundError(
        f"No recognized cluster assignment found in {path}. Expected recluster_assignments.csv, "
        "cells.csv + pred.npy, or a CSV with cluster labels."
    )


def load_method_clusters(args, obs_names: np.ndarray) -> np.ndarray:
    """Load cluster labels generated by BioLACE or another method."""
    log("[2/7] Cluster loading: loading method-generated cluster labels")
    path = Path(args.cluster_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        labels = load_labels_from_directory(path, obs_names, args.cluster_column)
    elif path.suffix.lower() == ".npy":
        labels = np.load(path, allow_pickle=True).astype(str)
        if len(labels) != len(obs_names):
            raise ValueError(f"{path} has {len(labels)} labels but selected h5ad section has {len(obs_names)} cells")
    else:
        df = read_table_auto(path)
        col = choose_label_column(df, args.cluster_column)
        labels = align_table_to_obs(df, df[col].values, obs_names)

    labels = pd.Series(labels).astype(str).values
    log(f"      loaded {len(np.unique(labels)):,} predicted clusters from {path}")
    return labels


# =============================================================================
# [3] Cluster selection: explicit cancer cluster IDs only
# =============================================================================


def select_cancer_clusters(
    labels: np.ndarray,
    tumor_clusters: str,
    outdir: Path,
    annotation: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Create the cancer mask from user-specified cluster IDs."""
    log("[3/7] Cluster selection: using explicitly supplied cancer cluster ID(s)")
    labels = pd.Series(labels).astype(str).values
    selected = parse_cluster_ids(tumor_clusters)

    available = set(pd.unique(labels))
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(
            f"The following --tumor-clusters were not found in method labels: {missing}. "
            f"Available cluster IDs include: {sorted(list(available))[:30]}"
        )

    cancer_mask = pd.Series(labels).isin(selected).values
    if cancer_mask.sum() == 0 or (~cancer_mask).sum() == 0:
        raise ValueError(
            f"Selected mask has {cancer_mask.sum()} cancer cells and {(~cancer_mask).sum()} non-cancer cells; "
            "cannot compute a boundary."
        )

    rows = []
    for cluster_id, sub_idx in pd.Series(np.arange(len(labels))).groupby(labels):
        ids = sub_idx.values.astype(int)
        row = {
            "cluster": str(cluster_id),
            "n_cells": int(len(ids)),
            "selected_as_cancer_cluster": str(cluster_id) in selected,
        }
        if annotation is not None:
            vc = pd.Series(annotation[ids]).value_counts()
            row["dominant_annotation_optional_validation_only"] = str(vc.index[0]) if len(vc) else "NA"
            row["dominant_annotation_fraction_optional_validation_only"] = float(vc.iloc[0] / len(ids)) if len(vc) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["selected_as_cancer_cluster", "n_cells"], ascending=[False, False])
    summary.to_csv(outdir / "cluster_selection_summary.csv", index=False)

    if annotation is not None:
        pd.crosstab(pd.Series(labels, name="method_cluster"), pd.Series(annotation, name="annotation")).to_csv(
            outdir / "cluster_annotation_crosstab_optional_validation_only.csv"
        )

    with open(outdir / "selected_cancer_clusters.json", "w") as f:
        json.dump(
            {
                "selected_clusters": selected,
                "n_selected_cancer_cells": int(cancer_mask.sum()),
                "n_other_cells": int((~cancer_mask).sum()),
                "boundary_definition": "negative = inside selected method-predicted cluster(s); positive = outside selected cluster(s)",
                "annotation_used_for_boundary": False,
            },
            f,
            indent=2,
        )

    log(f"      selected cluster(s): {', '.join(selected)}")
    log(f"      selected cancer cells: {cancer_mask.sum():,}; other cells: {(~cancer_mask).sum():,}")
    return cancer_mask, selected, summary


# =============================================================================
# [4] Boundary calculation
# =============================================================================


def signed_boundary_distance(coords: np.ndarray, cancer_mask: np.ndarray) -> np.ndarray:
    """Compute signed distance to the selected cancer-cluster boundary."""
    log("[4/7] Boundary calculation: nearest distance to selected cancer-cluster interface")
    if cancer_mask.sum() == 0 or (~cancer_mask).sum() == 0:
        raise ValueError("Need at least one selected and one non-selected cell to compute signed boundary distance")

    nn_cancer = NearestNeighbors(n_neighbors=1).fit(coords[cancer_mask])
    nn_other = NearestNeighbors(n_neighbors=1).fit(coords[~cancer_mask])

    distance_to_cancer = nn_cancer.kneighbors(coords, return_distance=True)[0].ravel().astype(np.float32)
    distance_to_other = nn_other.kneighbors(coords, return_distance=True)[0].ravel().astype(np.float32)

    signed = distance_to_cancer.copy()
    signed[cancer_mask] = -distance_to_other[cancer_mask]

    finite = np.isfinite(signed)
    log(f"      signed distance range: {np.nanmin(signed[finite]):.3f} to {np.nanmax(signed[finite]):.3f}")
    return signed


# =============================================================================
# [5] Marker selection and [6] module score calculation
# =============================================================================


def boundary_program_gene_sets() -> Dict[str, List[str]]:
    """Four compact programs for tumor-boundary biology in the Open-ST lymph node."""
    return {
        "cholesterol_biosynthesis": ["HMGCR", "HMGCS1", "DHCR7", "DHCR24", "SQLE", "FDFT1", "MSMO1", "FDPS", "MVD", "IDI1"],
        "immune_activation": ["GZMA", "GZMB", "LYZ", "CXCL9", "CXCL10", "IFNG", "PRF1", "NKG7"],
        "proliferation": ["MKI67", "TOP2A", "CENPF", "UBE2C", "CDK1", "RRM2", "HMGB2", "CDCA5"],
        "hypoxia_stress": ["HIF1A", "VEGFA", "CA9", "SLC2A1", "LDHA", "BNIP3", "NDRG1", "MMP9", "MMP12"],
    }


def load_programs_json(path: Optional[str]) -> Dict[str, List[str]]:
    programs = boundary_program_gene_sets()
    if not path:
        return programs
    with open(path) as f:
        user = json.load(f)
    for key, value in user.items():
        programs[key] = list(value)
    return programs


def compute_module_scores(
    adata,
    section_index: np.ndarray,
    programs: Dict[str, List[str]],
    layer: Optional[str],
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-cell module scores by averaging z-scored log-expression."""
    log("[5/7] Marker selection: matching program genes to h5ad var_names")
    var_names = np.asarray(adata.var_names.astype(str))
    upper_to_gene = {gene.upper(): gene for gene in var_names}
    gene_to_index = {gene: i for i, gene in enumerate(var_names)}

    presence_rows = []
    score_df = pd.DataFrame(index=np.arange(len(section_index)))

    log("[6/7] Module score calculation: log1p, gene-wise z-score, program average")
    for program, requested_genes in programs.items():
        present = []
        for gene in requested_genes:
            hit = upper_to_gene.get(gene.upper())
            if hit is not None:
                present.append(hit)
        present = list(dict.fromkeys(present))
        missing = [gene for gene in requested_genes if gene.upper() not in upper_to_gene]

        presence_rows.append(
            {
                "program": program,
                "n_requested": len(requested_genes),
                "n_present": len(present),
                "present_genes": ";".join(present),
                "missing_genes": ";".join(missing),
            }
        )

        if not present:
            warnings.warn(f"No genes found for program {program!r}; score will be NaN")
            score_df[program] = np.nan
            continue

        gene_indices = [gene_to_index[g] for g in present]
        sub = adata[section_index, gene_indices]
        if layer is not None and layer in adata.layers:
            X = sub.layers[layer]
        else:
            X = sub.X
        X = to_dense(X).astype(np.float32)
        X = np.log1p(np.maximum(X, 0))
        score_df[program] = np.nanmean(robust_zscore(X), axis=1)

    presence = pd.DataFrame(presence_rows)
    presence.to_csv(outdir / "module_gene_presence.csv", index=False)
    return score_df, presence


# =============================================================================
# [7] Boundary curves and spatial maps
# =============================================================================


def select_distance_window(
    signed_distance: np.ndarray,
    distance_window: Optional[float],
    clip_quantile: float,
) -> Tuple[float, float, str]:
    """
    Match the original downstream script exactly.

    The original boundary analysis does not use a fixed symmetric distance window.
    It clips the signed-distance distribution to the central `clip_quantile`
    fraction using asymmetric quantiles of the signed distances.
    """
    finite = signed_distance[np.isfinite(signed_distance)]
    if finite.size < 20:
        raise ValueError("Too few finite signed distances")

    # Optional manual override, only if explicitly requested.
    if distance_window is not None and distance_window > 0:
        w = float(distance_window)
        return -w, w, f"manual fixed symmetric window +/- {w:g}"

    lo, hi = np.nanquantile(
        finite,
        [(1.0 - clip_quantile) / 2.0, 1.0 - (1.0 - clip_quantile) / 2.0],
    )
    return float(lo), float(hi), f"central signed-distance quantile {clip_quantile:.3f} [{lo:g}, {hi:g}]"


def boundary_curves(
    scores: pd.DataFrame,
    signed_distance: np.ndarray,
    programs_to_plot: Sequence[str],
    outdir: Path,
    n_bins: int,
    distance_window: Optional[float],
    clip_quantile: float,
) -> pd.DataFrame:
    """Bin cells by signed boundary distance and average module scores."""
    log("[7/7] Boundary curve calculation: binning cells near selected cancer-cluster boundary")
    lo, hi, window_note = select_distance_window(signed_distance, distance_window, clip_quantile)
    valid = np.isfinite(signed_distance) & (signed_distance >= lo) & (signed_distance <= hi)
    if valid.sum() < 20:
        raise ValueError(f"Only {valid.sum()} cells fall inside the boundary window [{lo}, {hi}]")

    bins = np.linspace(lo, hi, n_bins + 1)
    bin_id = np.digitize(signed_distance[valid], bins) - 1
    x_mid = (bins[:-1] + bins[1:]) / 2

    rows = []
    for b in range(n_bins):
        in_bin = bin_id == b
        if not in_bin.any():
            continue
        row = {"bin": b, "distance_mid": float(x_mid[b]), "n_cells": int(in_bin.sum())}
        for program in programs_to_plot:
            vals = scores.loc[valid, program].values[in_bin]
            finite = np.isfinite(vals)
            row[f"{program}_mean"] = float(np.nanmean(vals)) if finite.any() else np.nan
            row[f"{program}_sem"] = float(np.nanstd(vals) / max(1, math.sqrt(int(finite.sum())))) if finite.any() else np.nan
        rows.append(row)

    curves = pd.DataFrame(rows)
    curves.to_csv(outdir / "boundary_module_curves.csv", index=False)
    with open(outdir / "boundary_window_used.json", "w") as f:
        json.dump({"distance_min": lo, "distance_max": hi, "note": window_note}, f, indent=2)

    # Individual boundary curves.
    for program in programs_to_plot:
        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        x = curves["distance_mid"].to_numpy()
        y = curves[f"{program}_mean"].to_numpy()
        sem = curves[f"{program}_sem"].to_numpy()
        ax.plot(x, y, linewidth=2.2)
        ax.fill_between(x, y - sem, y + sem, alpha=0.20)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Signed distance to selected cancer-cluster boundary")
        ax.set_ylabel("Mean module score")
        ax.set_title(program.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(outdir / f"boundary_curve_{sanitize(program)}.png", dpi=300)
        plt.close(fig)

    # Public-display 2x2 overview.
    ncols = 2
    nrows = int(math.ceil(len(programs_to_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 4.6 * nrows), squeeze=False)
    for ax, program in zip(axes.ravel(), programs_to_plot):
        x = curves["distance_mid"].to_numpy()
        y = curves[f"{program}_mean"].to_numpy()
        sem = curves[f"{program}_sem"].to_numpy()
        ax.plot(x, y, linewidth=2.2)
        ax.fill_between(x, y - sem, y + sem, alpha=0.20)
        ax.axvline(0, linestyle="--", linewidth=1.2)
        ax.axhline(0, linestyle=":", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_title(program.replace("_", " ").title())
        ax.set_xlabel("Signed distance to boundary")
        ax.set_ylabel("Mean module score")
    for ax in axes.ravel()[len(programs_to_plot):]:
        ax.axis("off")
    fig.suptitle("Boundary-aligned programs around selected cancer cluster(s)", y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(outdir / "boundary_curves_all_programs.png", dpi=300)
    plt.close(fig)

    return curves


def make_plot_subset(n: int, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    return idx


def add_cancer_outline(
    ax,
    coords: np.ndarray,
    cancer_mask: np.ndarray,
    grid_size: int = 350,
    smooth_sigma: float = 1.2,
    color: str = "red",
    linewidth: float = 2.0,
) -> None:
    """Overlay a dotted contour around the selected cancer cluster(s)."""
    if cancer_mask.sum() < 5 or (~cancer_mask).sum() < 5:
        return

    x = coords[:, 0]
    y = coords[:, 1]
    pad_x = 0.01 * max(1e-6, float(np.nanmax(x) - np.nanmin(x)))
    pad_y = 0.01 * max(1e-6, float(np.nanmax(y) - np.nanmin(y)))
    x_edges = np.linspace(np.nanmin(x) - pad_x, np.nanmax(x) + pad_x, grid_size + 1)
    y_edges = np.linspace(np.nanmin(y) - pad_y, np.nanmax(y) + pad_y, grid_size + 1)

    all_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    cancer_counts, _, _ = np.histogram2d(x[cancer_mask], y[cancer_mask], bins=[x_edges, y_edges])

    all_smooth = gaussian_filter(all_counts.astype(np.float32), sigma=smooth_sigma)
    cancer_smooth = gaussian_filter(cancer_counts.astype(np.float32), sigma=smooth_sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        cancer_fraction = cancer_smooth / np.maximum(all_smooth, 1e-6)

    # Mask bins with almost no nearby cells to avoid contours over empty whitespace.
    cancer_fraction[all_smooth < 0.05] = np.nan
    if not np.isfinite(cancer_fraction).any():
        return

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    try:
        ax.contour(
            x_centers,
            y_centers,
            cancer_fraction.T,
            levels=[0.5],
            colors=color,
            linestyles=":",
            linewidths=linewidth,
            zorder=10,
        )
    except Exception:
        # Fallback: plot boundary cells as red dots if contouring fails.
        nn = NearestNeighbors(n_neighbors=min(9, len(coords))).fit(coords)
        neigh = nn.kneighbors(coords, return_distance=False)
        boundary = cancer_mask & np.any(~cancer_mask[neigh], axis=1)
        ax.scatter(coords[boundary, 0], coords[boundary, 1], s=5, c=color, linewidths=0, zorder=10, rasterized=True)


def plot_cancer_mask(
    coords: np.ndarray,
    cancer_mask: np.ndarray,
    out_path: Path,
    selected_clusters: Sequence[str],
    max_points: int,
    seed: int,
    point_size: float,
) -> None:
    idx = make_plot_subset(len(cancer_mask), max_points, seed)
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.scatter(coords[idx, 0], coords[idx, 1], c="lightgray", s=point_size, linewidths=0, rasterized=True)
    in_sel = idx[cancer_mask[idx]]
    ax.scatter(coords[in_sel, 0], coords[in_sel, 1], c="red", s=point_size + 1.0, linewidths=0, rasterized=True)
    add_cancer_outline(ax, coords, cancer_mask, color="red", linewidth=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Selected cancer cluster(s): {', '.join(map(str, selected_clusters))}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spatial_module_map(
    coords: np.ndarray,
    score: np.ndarray,
    cancer_mask: np.ndarray,
    out_path: Path,
    title: str,
    max_points: int,
    seed: int,
    point_size: float,
    vmin_q: float,
    vmax_q: float,
) -> None:
    """Plot a module score with light low values, dark high values, and cancer boundary outline."""
    idx = make_plot_subset(len(score), max_points, seed)
    values = np.asarray(score, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        warnings.warn(f"Skipping spatial map for {title}: no finite values")
        return

    vmin = float(np.nanquantile(values[finite], vmin_q))
    vmax = float(np.nanquantile(values[finite], vmax_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values[finite])), float(np.nanmax(values[finite]))
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Plot low-score cells first and high-score cells last, so activation remains visible.
    idx = idx[np.argsort(np.nan_to_num(values[idx], nan=vmin))]

    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    sc = ax.scatter(
        coords[idx, 0],
        coords[idx, 1],
        c=np.clip(values[idx], vmin, vmax),
        cmap="Blues",
        norm=Normalize(vmin=vmin, vmax=vmax),
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    add_cancer_outline(ax, coords, cancer_mask, color="red", linewidth=2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("module score", fontsize=11)
    cb.ax.tick_params(labelsize=9)
    # Make the outline meaning explicit without cluttering the plot.
    ax.text(
        0.02,
        0.02,
        "red dotted outline: selected cancer cluster boundary",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_signed_distance_map(
    coords: np.ndarray,
    signed_distance: np.ndarray,
    cancer_mask: np.ndarray,
    out_path: Path,
    max_points: int,
    seed: int,
    point_size: float,
    distance_window: Optional[float],
    clip_quantile: float,
) -> None:
    idx = make_plot_subset(len(signed_distance), max_points, seed)
    lo, hi, _ = select_distance_window(signed_distance, distance_window, clip_quantile)
    w = max(abs(lo), abs(hi))
    idx = idx[np.argsort(np.abs(signed_distance[idx]))[::-1]]  # boundary cells plotted later after reverse below
    idx = idx[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    sc = ax.scatter(
        coords[idx, 0],
        coords[idx, 1],
        c=np.clip(signed_distance[idx], -w, w),
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-w, vcenter=0.0, vmax=w),
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    add_cancer_outline(ax, coords, cancer_mask, color="red", linewidth=2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Signed distance to selected cancer-cluster boundary", fontsize=14)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("signed distance", fontsize=11)
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Boundary analysis for cholesterol, immune, proliferation, and hypoxia programs around explicitly selected method-predicted cancer clusters."
    )

    # h5ad inputs.
    parser.add_argument("--h5ad", required=True, help="Input h5ad file")
    parser.add_argument("--backed", action="store_true", help="Read h5ad in backed mode")
    parser.add_argument("--section-col", default="n_section", help="obs column specifying tissue section")
    parser.add_argument("--section-id", default="19", help="section value to select")
    parser.add_argument("--spatial-key", default="spatial", help="obsm key containing spatial coordinates")
    parser.add_argument("--layer", default="raw", help="Expression layer to use if present; otherwise uses X")
    parser.add_argument(
        "--annotation-col",
        default="annotation",
        help="Optional annotation column used only for validation outputs, never for boundary definition",
    )

    # Method-generated clusters.
    parser.add_argument("--cluster-path", required=True, help="Path to method cluster output: VAE dir, recluster dir, CSV, or .npy labels")
    parser.add_argument("--cluster-column", default=None, help="Column containing method-generated cluster labels, if cluster-path is a table")
    parser.add_argument(
        "--tumor-clusters",
        "--cancer-clusters",
        dest="tumor_clusters",
        required=True,
        help="Comma/space-separated method-generated cluster ID(s) to use as cancer. This is required; no automatic cluster inference is performed.",
    )

    # Programs and outputs.
    parser.add_argument("--programs-json", default=None, help="Optional JSON dict overriding/extending the four program gene sets")
    parser.add_argument("--outdir", default="./boundary_program_analysis_from_selected_clusters")

    # Boundary curve window. Default is intentionally local around boundary for presentation.
    parser.add_argument("--boundary-bins", type=int, default=40, help="Number of signed-distance bins")
    parser.add_argument(
        "--distance-window",
        "--max-abs-distance",
        dest="distance_window",
        type=float,
        default=None,
        help="Optional manual symmetric distance window. By default, the script matches the original analysis and uses central signed-distance quantile clipping.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.99,
        help="Central signed-distance quantile used for boundary curves; matches the original downstream analysis default.",
    )

    # Spatial plotting.
    parser.add_argument("--max-points-plot", type=int, default=100000, help="Maximum points shown in spatial maps")
    parser.add_argument("--point-size", type=float, default=3.0, help="Point size for spatial maps")
    parser.add_argument("--score-vmin-quantile", type=float, default=0.02, help="Lower quantile for module-map color clipping")
    parser.add_argument("--score-vmax-quantile", type=float, default=0.995, help="Upper quantile for module-map color clipping")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for plotting subsampling")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    outdir = mkdir(Path(args.outdir))
    spatial_dir = mkdir(outdir / "spatial_module_maps")

    distance_window = None if args.distance_window is None or args.distance_window <= 0 else float(args.distance_window)

    adata, section_index, obs, obs_names, coords, annotation = load_h5ad_section(args)
    labels = load_method_clusters(args, obs_names)
    cancer_mask, selected_clusters, selection_summary = select_cancer_clusters(
        labels=labels,
        tumor_clusters=args.tumor_clusters,
        outdir=outdir,
        annotation=annotation,
    )
    signed_distance = signed_boundary_distance(coords, cancer_mask)

    programs = load_programs_json(args.programs_json)
    boundary_programs = ["cholesterol_biosynthesis", "immune_activation", "proliferation", "hypoxia_stress"]
    missing = [p for p in boundary_programs if p not in programs]
    if missing:
        raise KeyError(f"Missing required boundary programs: {missing}")

    scores, presence = compute_module_scores(adata, section_index, programs, args.layer, outdir)

    curves = boundary_curves(
        scores=scores,
        signed_distance=signed_distance,
        programs_to_plot=boundary_programs,
        outdir=outdir,
        n_bins=args.boundary_bins,
        distance_window=distance_window,
        clip_quantile=args.clip_quantile,
    )

    # Export cell-level table.
    cell_table = pd.DataFrame(
        {
            "obs_name": obs_names,
            "method_cluster": labels,
            "selected_as_cancer_cluster": cancer_mask,
            "signed_distance_to_selected_cancer_cluster_boundary": signed_distance,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    if annotation is not None:
        cell_table["annotation_optional_validation_only"] = annotation
    for program in boundary_programs:
        cell_table[f"score_{program}"] = scores[program].values
    cell_table.to_csv(outdir / "cell_boundary_module_scores.csv", index=False)

    # Spatial plots.
    plot_cancer_mask(
        coords=coords,
        cancer_mask=cancer_mask,
        out_path=outdir / "selected_cancer_cluster_mask.png",
        selected_clusters=selected_clusters,
        max_points=args.max_points_plot,
        seed=args.seed,
        point_size=args.point_size,
    )
    plot_signed_distance_map(
        coords=coords,
        signed_distance=signed_distance,
        cancer_mask=cancer_mask,
        out_path=outdir / "signed_boundary_distance_map.png",
        max_points=args.max_points_plot,
        seed=args.seed,
        point_size=args.point_size,
        distance_window=distance_window,
        clip_quantile=args.clip_quantile,
    )
    for program in boundary_programs:
        plot_spatial_module_map(
            coords=coords,
            score=scores[program].values,
            cancer_mask=cancer_mask,
            out_path=spatial_dir / f"spatial_score_{sanitize(program)}.png",
            title=program.replace("_", " ").title(),
            max_points=args.max_points_plot,
            seed=args.seed,
            point_size=args.point_size,
            vmin_q=args.score_vmin_quantile,
            vmax_q=args.score_vmax_quantile,
        )

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    finite = np.isfinite(signed_distance)
    ax.hist(signed_distance[finite], bins=80)
    ax.axvline(0, linestyle="--", linewidth=1.2)
    if distance_window is not None:
        ax.set_xlim(-distance_window, distance_window)
    ax.set_xlabel("Signed distance to selected cancer-cluster boundary")
    ax.set_ylabel("Cell count")
    ax.set_title("Signed boundary-distance distribution")
    fig.tight_layout()
    fig.savefig(outdir / "signed_boundary_distance_histogram.png", dpi=300)
    plt.close(fig)

    with open(outdir / "analysis_metadata.json", "w") as f:
        json.dump(
            {
                "h5ad": args.h5ad,
                "section_col": args.section_col,
                "section_id": args.section_id,
                "cluster_path": args.cluster_path,
                "cluster_column": args.cluster_column,
                "selected_cancer_clusters": selected_clusters,
                "n_cells": int(len(obs_names)),
                "n_clusters": int(len(np.unique(labels))),
                "n_selected_cancer_cells": int(cancer_mask.sum()),
                "distance_window": distance_window,
                "clip_quantile_if_no_fixed_window": float(args.clip_quantile),
                "annotation_used_for_boundary": False,
                "programs": boundary_programs,
            },
            f,
            indent=2,
        )

    log(f"Done. Outputs written to: {outdir}")
    print("Key files:")
    for path in [
        outdir / "selected_cancer_clusters.json",
        outdir / "cluster_selection_summary.csv",
        outdir / "cell_boundary_module_scores.csv",
        outdir / "boundary_module_curves.csv",
        outdir / "boundary_curves_all_programs.png",
        outdir / "selected_cancer_cluster_mask.png",
        outdir / "signed_boundary_distance_map.png",
        spatial_dir / "spatial_score_cholesterol_biosynthesis.png",
        spatial_dir / "spatial_score_immune_activation.png",
        spatial_dir / "spatial_score_proliferation.png",
        spatial_dir / "spatial_score_hypoxia_stress.png",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
