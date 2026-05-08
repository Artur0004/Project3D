"""
=============================================================================
cordatum_connected_pores.py
=============================================================================
Connected Pore Network Analysis of Cordatum Shell (micro-CT 3D dataset)

WHAT THIS SCRIPT DOES
---------------------
Loads a 16-bit TIFF micro-CT volume, segments the shell object and the pore
space inside it, performs 3D connected-component analysis on the pore network,
classifies pores as open (boundary-connected) or closed, and saves all figures
and tables needed for a scientific poster.

REQUIRED PACKAGES
-----------------
  numpy, scipy, scikit-image, matplotlib, pandas, tifffile

Optional (used if present):
  pyvista  — high-quality 3D surface rendering
  dask     — out-of-core loading (not used by default)

Install via:
  pip install numpy scipy scikit-image matplotlib pandas tifffile

HOW TO RUN
----------
1. Open this file in PyCharm.
2. Adjust the PARAMETERS block below (especially USE_DOWNSAMPLED and the
   threshold modes) to match your data.
3. Run the script (Shift+F10).  All output goes to OUTPUT_DIR.

PARAMETERS TO ADJUST FIRST IF SEGMENTATION LOOKS WRONG
-------------------------------------------------------
- USE_DOWNSAMPLED / DOWNSAMPLE_FACTOR  — start with downsampled=True, factor=2
- SHELL_THRESHOLD_MODE = "otsu" works well; override with "manual" +
  SHELL_MANUAL_THRESHOLD if the shell boundary is misclassified.
- PORE_THRESHOLD_MODE = "otsu_inside_shell" is usually best; use "manual" +
  PORE_MANUAL_THRESHOLD to fix over/under-segmentation.
- MIN_PORE_SIZE — increase to remove noise, decrease to keep small pores.
- GAUSSIAN_SIGMA — increase slightly (e.g. 1.5) if the raw data is very noisy.
=============================================================================
"""

# ============================================================
#  PARAMETERS  — adjust these before running
# ============================================================
from pathlib import Path

DATA_PATH   = Path(r"C:\Users\artxm\PycharmProjects\3d2\Cordatum_Shell.tif")
OUTPUT_DIR  = Path(r"C:\Users\artxm\PycharmProjects\3d2\cordatum_results")

# --- downsampling -------------------------------------------------------
USE_DOWNSAMPLED  = True   # set False for full-resolution run
DOWNSAMPLE_FACTOR = 2     # only used when USE_DOWNSAMPLED is True

# --- optional cropping --------------------------------------------------
USE_CROP = False
CROP_Z   = (None, None)
CROP_Y   = (None, None)
CROP_X   = (None, None)

# --- preprocessing ------------------------------------------------------
GAUSSIAN_SIGMA = 1.0      # 0 = skip smoothing

# --- shell segmentation -------------------------------------------------
SHELL_THRESHOLD_MODE     = "otsu"   # "otsu" | "manual"
SHELL_MANUAL_THRESHOLD   = 0.10     # used only when mode="manual"
SHELL_MIN_SIZE           = 5000     # voxels; remove small shell fragments

# --- pore segmentation --------------------------------------------------
PORE_THRESHOLD_MODE    = "otsu_inside_shell"  # "otsu_inside_shell" | "manual"
PORE_MANUAL_THRESHOLD  = 0.35                 # used only when mode="manual"
MIN_PORE_SIZE          = 50                   # voxels; remove tiny pore noise

# --- optional analyses --------------------------------------------------
RUN_THRESHOLD_SENSITIVITY = True   # takes extra time; set False to skip
RUN_3D_RENDERING          = True   # marching-cubes 3D figure; set False if slow
RUN_DISTANCE_TRANSFORM    = True   # local pore thickness; set False to skip

# ============================================================
#  IMPORTS
# ============================================================
import sys
import os
import time
import logging
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import tifffile
import scipy.ndimage as ndi
from skimage import filters, morphology, measure, segmentation

# ============================================================
#  SETUP  — directories, logging
# ============================================================
DIRS = {
    "figures": OUTPUT_DIR / "figures",
    "tables":  OUTPUT_DIR / "tables",
    "masks":   OUTPUT_DIR / "masks",
    "logs":    OUTPUT_DIR / "logs",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = DIRS["logs"] / "analysis.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SUMMARY_LINES = []   # collects lines for analysis_summary.txt

def _note(msg: str):
    """Print, log, and store a line for the summary file."""
    log.info(msg)
    SUMMARY_LINES.append(msg)


# ============================================================
#  HELPER UTILITIES
# ============================================================

def _savefig(name: str, fig=None, dpi: int = 150):
    path = DIRS["figures"] / name
    (fig or plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig or plt.gcf())
    log.info(f"  saved figure: {path.name}")


def _pct_clip(img: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    """Return (vmin, vmax) for display clipping."""
    vmin, vmax = np.percentile(img, [lo, hi])
    return float(vmin), float(vmax)


def _show_slice(ax, img2d, title="", cmap="gray", vmin=None, vmax=None):
    if vmin is None:
        vmin, vmax = _pct_clip(img2d)
    ax.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _random_label_cmap(n: int = 256):
    """Random colormap for labelled images (label 0 = black)."""
    rng = np.random.default_rng(42)
    colors = rng.random((n, 4))
    colors[:, 3] = 1.0
    colors[0] = [0, 0, 0, 1]   # background = black
    return mcolors.ListedColormap(colors)


# ============================================================
#  1.  LOAD VOLUME
# ============================================================

def load_volume() -> np.ndarray:
    """Load the TIFF stack from DATA_PATH, optionally crop and downsample."""
    log.info("=" * 60)
    log.info("LOADING VOLUME")
    log.info("=" * 60)
    log.info(f"  File : {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"TIFF not found: {DATA_PATH}")

    t0 = time.time()
    # memmap avoids loading into RAM during metadata inspection
    try:
        vol = tifffile.memmap(DATA_PATH)
    except Exception:
        log.warning("  memmap failed — falling back to imread (may use more RAM)")
        vol = tifffile.imread(str(DATA_PATH))

    log.info(f"  Raw shape : {vol.shape}  dtype={vol.dtype}  "
             f"({time.time()-t0:.1f} s to open)")

    # --- optional crop --------------------------------------------------
    if USE_CROP:
        slices = (
            slice(*CROP_Z),
            slice(*CROP_Y),
            slice(*CROP_X),
        )
        log.info(f"  Cropping to {slices}")
        vol = np.array(vol[slices])

    # --- downsample ------------------------------------------------------
    if USE_DOWNSAMPLED:
        f = DOWNSAMPLE_FACTOR
        log.info(f"  Downsampling by factor {f} (slice indexing)")
        vol = np.array(vol[::f, ::f, ::f])
        log.info(f"  Downsampled shape : {vol.shape}")
    else:
        vol = np.array(vol)   # materialise memmap into RAM

    # --- stats -----------------------------------------------------------
    vmin, vmax = int(vol.min()), int(vol.max())
    vmean = float(vol.mean())
    mem_gb = vol.nbytes / 1e9
    log.info(f"  Shape : {vol.shape}")
    log.info(f"  dtype : {vol.dtype}")
    log.info(f"  Intensity range : [{vmin}, {vmax}]  mean={vmean:.1f}")
    log.info(f"  RAM   : {mem_gb:.2f} GB")

    _note(f"Dataset path        : {DATA_PATH}")
    _note(f"Shape used          : {vol.shape}  (downsampled={USE_DOWNSAMPLED}, "
          f"factor={DOWNSAMPLE_FACTOR if USE_DOWNSAMPLED else 1})")
    _note(f"dtype               : {vol.dtype}")
    _note(f"Intensity range     : [{vmin}, {vmax}]  mean={vmean:.1f}")
    _note(f"RAM (array)         : {mem_gb:.2f} GB")

    return vol


# ============================================================
#  2.  BASIC VISUALISATION
# ============================================================

def make_slice_figures(vol: np.ndarray):
    """Save raw_slice_examples.png and intensity_histogram.png."""
    log.info("-" * 40)
    log.info("RAW SLICE FIGURES")

    Z, Y, X = vol.shape
    sz, sy, sx = Z // 2, Y // 2, X // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Raw micro-CT slices (16-bit, percentile-clipped)", fontsize=11)
    _show_slice(axes[0], vol[sz, :, :], title=f"Axial  z={sz}")
    _show_slice(axes[1], vol[:, sy, :], title=f"Coronal  y={sy}")
    _show_slice(axes[2], vol[:, :, sx], title=f"Sagittal  x={sx}")
    plt.tight_layout()
    _savefig("raw_slice_examples.png", fig)

    # intensity histogram — random sample to keep it fast
    log.info("  Computing intensity histogram")
    rng = np.random.default_rng(0)
    flat = vol.ravel()
    n_sample = min(len(flat), 2_000_000)
    sample = rng.choice(flat, size=n_sample, replace=False).astype(np.float32)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sample, bins=512, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Intensity (raw)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Intensity histogram (random sample)")
    plt.tight_layout()
    _savefig("intensity_histogram.png", fig)


# ============================================================
#  3.  PREPROCESSING — normalise + smooth
# ============================================================

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """
    Percentile-normalise to float32 [0, 1] then apply Gaussian smoothing.
    Returns smoothed float32 volume.
    """
    log.info("-" * 40)
    log.info("PREPROCESSING")

    p_low  = float(np.percentile(vol, 1.0))
    p_high = float(np.percentile(vol, 99.8))
    log.info(f"  Normalisation percentiles: p1={p_low:.1f}  p99.8={p_high:.1f}")

    norm = (vol.astype(np.float32) - p_low) / (p_high - p_low)
    norm = np.clip(norm, 0.0, 1.0)

    if GAUSSIAN_SIGMA > 0:
        log.info(f"  Gaussian smoothing sigma={GAUSSIAN_SIGMA}")
        smooth = ndi.gaussian_filter(norm, sigma=GAUSSIAN_SIGMA).astype(np.float32)
    else:
        smooth = norm

    # --- preprocessing comparison figure --------------------------------
    Z = vol.shape[0]
    sz = Z // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Preprocessing comparison (central axial slice)")
    _show_slice(axes[0], vol[sz],    title="Raw")
    _show_slice(axes[1], norm[sz],   title="Normalised [0–1]", vmin=0, vmax=1)
    _show_slice(axes[2], smooth[sz], title=f"Smoothed σ={GAUSSIAN_SIGMA}", vmin=0, vmax=1)
    plt.tight_layout()
    _savefig("preprocessing_comparison.png", fig)

    _note(f"Normalisation p1    : {p_low:.2f}")
    _note(f"Normalisation p99.8 : {p_high:.2f}")
    _note(f"Gaussian sigma      : {GAUSSIAN_SIGMA}")

    return smooth, p_low, p_high


# ============================================================
#  4.  SHELL SEGMENTATION
# ============================================================

def segment_shell(smooth: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Returns (shell_mask [bool], threshold_used).
    shell_mask defines the complete shell region (solid + enclosed pores).
    """
    log.info("-" * 40)
    log.info("SHELL SEGMENTATION")

    # --- threshold -------------------------------------------------------
    if SHELL_THRESHOLD_MODE == "otsu":
        rng = np.random.default_rng(1)
        flat = smooth.ravel()
        n_sample = min(len(flat), 5_000_000)
        sample = rng.choice(flat, size=n_sample, replace=False)
        threshold = float(filters.threshold_otsu(sample))
        log.info(f"  Otsu threshold (shell) = {threshold:.4f}")
    else:
        threshold = float(SHELL_MANUAL_THRESHOLD)
        log.info(f"  Manual threshold (shell) = {threshold:.4f}")

    shell_initial = smooth > threshold

    # --- morphological clean-up -----------------------------------------
    log.info("  Removing small objects …")
    shell_clean = morphology.remove_small_objects(shell_initial, min_size=SHELL_MIN_SIZE)

    log.info("  Binary closing (3D) …")
    struct = ndi.generate_binary_structure(3, 1)   # 6-connectivity kernel
    shell_closed = ndi.binary_closing(shell_clean, structure=ndi.iterate_structure(struct, 5))

    # fill holes slice-by-slice (cheaper than 3D fill and preserves large cavities)
    log.info("  Filling holes slice-by-slice …")
    shell_filled = np.zeros_like(shell_closed)
    for i in range(shell_closed.shape[0]):
        shell_filled[i] = ndi.binary_fill_holes(shell_closed[i])

    # keep the single largest connected component as the main shell object
    log.info("  Keeping largest connected component …")
    labeled, _ = ndi.label(shell_filled)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0   # ignore background
    main_label = counts.argmax()
    shell_mask = labeled == main_label

    shell_vox = int(shell_mask.sum())
    log.info(f"  Shell mask voxels : {shell_vox:,}")
    _note(f"Shell threshold     : {threshold:.4f}  (mode={SHELL_THRESHOLD_MODE})")
    _note(f"Shell volume        : {shell_vox:,} voxels")

    # --- figure ----------------------------------------------------------
    sz = smooth.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Shell segmentation (central axial slice)")
    _show_slice(axes[0], smooth[sz],      title="Smoothed")
    axes[1].imshow(shell_mask[sz], cmap="gray", interpolation="nearest")
    axes[1].set_title("Shell mask"); axes[1].axis("off")
    _show_slice(axes[2], smooth[sz],      title="Overlay")
    axes[2].imshow(shell_mask[sz], alpha=0.35, cmap="Reds", interpolation="nearest")
    plt.tight_layout()
    _savefig("shell_segmentation.png", fig)

    return shell_mask.astype(bool), threshold


# ============================================================
#  5.  PORE SEGMENTATION
# ============================================================

def segment_pores(smooth: np.ndarray, shell_mask: np.ndarray,
                  threshold_override: float = None) -> tuple[np.ndarray, float]:
    """
    Segment pore (low-intensity) voxels inside shell_mask.
    Returns (pore_mask [bool], threshold_used).
    """
    # --- threshold -------------------------------------------------------
    if threshold_override is not None:
        threshold = float(threshold_override)
    elif PORE_THRESHOLD_MODE == "otsu_inside_shell":
        inside_vals = smooth[shell_mask]
        rng = np.random.default_rng(2)
        n_sample = min(len(inside_vals), 5_000_000)
        sample = rng.choice(inside_vals, size=n_sample, replace=False)
        threshold = float(filters.threshold_otsu(sample))
        log.info(f"  Otsu threshold (pore, inside shell) = {threshold:.4f}")
    else:
        threshold = float(PORE_MANUAL_THRESHOLD)
        log.info(f"  Manual threshold (pore) = {threshold:.4f}")

    pore_raw = (smooth < threshold) & shell_mask

    # remove small noise fragments
    pore_clean = morphology.remove_small_objects(pore_raw, min_size=MIN_PORE_SIZE)

    return pore_clean.astype(bool), threshold


def make_pore_figures(smooth: np.ndarray, pore_mask: np.ndarray,
                      pore_threshold: float):
    """Save pore segmentation overlay figures."""
    log.info("-" * 40)
    log.info("PORE SEGMENTATION FIGURES")
    _note(f"Pore threshold      : {pore_threshold:.4f}  (mode={PORE_THRESHOLD_MODE})")
    _note(f"Min pore size       : {MIN_PORE_SIZE} voxels")

    sz = smooth.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Pore segmentation (central axial slice)")
    _show_slice(axes[0], smooth[sz],     title="Smoothed")
    axes[1].imshow(pore_mask[sz], cmap="Blues", interpolation="nearest")
    axes[1].set_title("Pore mask"); axes[1].axis("off")
    _show_slice(axes[2], smooth[sz],     title="Overlay (pores in blue)")
    axes[2].imshow(pore_mask[sz], alpha=0.5, cmap="Blues", interpolation="nearest")
    plt.tight_layout()
    _savefig("pore_segmentation_overlay.png", fig)

    # zoomed-in view — central quarter
    Z, Y, X = smooth.shape
    zy0, zy1 = Y // 4, 3 * Y // 4
    zx0, zx1 = X // 4, 3 * X // 4
    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
    fig2.suptitle("Pore segmentation — zoomed (central axial slice)")
    _show_slice(axes2[0], smooth[sz, zy0:zy1, zx0:zx1], title="Smoothed (zoom)")
    _show_slice(axes2[1], smooth[sz, zy0:zy1, zx0:zx1], title="Pores overlay (zoom)")
    axes2[1].imshow(pore_mask[sz, zy0:zy1, zx0:zx1],
                    alpha=0.55, cmap="Blues", interpolation="nearest")
    plt.tight_layout()
    _savefig("pore_segmentation_zoom.png", fig2)


# ============================================================
#  6.  3D CONNECTED COMPONENT ANALYSIS
# ============================================================

def connected_component_analysis(pore_mask: np.ndarray,
                                  shell_mask: np.ndarray) -> tuple:
    """
    26-connectivity labelling.  Returns (labeled, comp_stats_df, global_metrics).
    """
    log.info("-" * 40)
    log.info("3D CONNECTED COMPONENT ANALYSIS")

    struct26 = np.ones((3, 3, 3), dtype=np.int8)
    log.info("  Labelling (26-connectivity) …")
    labeled, n_comp = ndi.label(pore_mask, structure=struct26)
    log.info(f"  Found {n_comp:,} components")

    # voxel counts per label  (label 0 = background)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    total_pore_vox  = int(pore_mask.sum())
    total_shell_vox = int(shell_mask.sum())
    total_vol_vox   = int(pore_mask.size)
    porosity_pct    = 100.0 * total_pore_vox / total_shell_vox if total_shell_vox else 0.0

    # sort labels by size
    label_ids   = np.arange(1, n_comp + 1)
    comp_counts = counts[label_ids]
    sort_idx    = np.argsort(comp_counts)[::-1]
    sorted_labels  = label_ids[sort_idx]
    sorted_counts  = comp_counts[sort_idx]

    largest_label  = int(sorted_labels[0])
    largest_vol    = int(sorted_counts[0])
    largest_frac   = 100.0 * largest_vol / total_pore_vox if total_pore_vox else 0.0

    log.info(f"  Total pore voxels : {total_pore_vox:,}")
    log.info(f"  Shell voxels      : {total_shell_vox:,}")
    log.info(f"  Porosity          : {porosity_pct:.2f} %")
    log.info(f"  Largest component : {largest_vol:,} voxels  "
             f"({largest_frac:.1f} % of pore volume)")

    _note(f"Shell volume        : {total_shell_vox:,} voxels")
    _note(f"Pore volume         : {total_pore_vox:,} voxels")
    _note(f"Solid volume        : {total_shell_vox - total_pore_vox:,} voxels")
    _note(f"Porosity            : {porosity_pct:.2f} %")
    _note(f"Number of components: {n_comp:,}")
    _note(f"Largest component   : {largest_vol:,} voxels  ({largest_frac:.1f} %)")

    # ---------- per-component stats table --------------------------------
    # boundary touch: does any voxel of this component sit on the array edge?
    log.info("  Computing boundary-contact flags …")
    Z, Y, X = labeled.shape
    boundary_mask = np.zeros(labeled.shape, dtype=bool)
    boundary_mask[0, :, :]  = True;  boundary_mask[-1, :, :]  = True
    boundary_mask[:, 0, :]  = True;  boundary_mask[:, -1, :]  = True
    boundary_mask[:, :, 0]  = True;  boundary_mask[:, :, -1]  = True

    boundary_labels = set(np.unique(labeled[boundary_mask])) - {0}

    rows = []
    for rank, (lbl, vol_v) in enumerate(zip(sorted_labels, sorted_counts), start=1):
        rows.append({
            "label":                  int(lbl),
            "rank_by_size":           rank,
            "volume_voxels":          int(vol_v),
            "fraction_of_pore_volume": float(vol_v) / total_pore_vox if total_pore_vox else 0.0,
            "touches_volume_boundary": int(lbl) in boundary_labels,
        })
    comp_df = pd.DataFrame(rows)

    # ---------- global metrics dict --------------------------------------
    n_boundary = int(comp_df["touches_volume_boundary"].sum())
    vol_boundary = int(comp_df.loc[comp_df["touches_volume_boundary"], "volume_voxels"].sum())
    vol_closed   = int(total_pore_vox - vol_boundary)

    global_metrics = {
        "volume_shape":                        str(pore_mask.shape),
        "downsample_factor":                   DOWNSAMPLE_FACTOR if USE_DOWNSAMPLED else 1,
        "shell_volume_voxels":                 total_shell_vox,
        "pore_volume_voxels":                  total_pore_vox,
        "solid_volume_voxels":                 total_shell_vox - total_pore_vox,
        "porosity_percent":                    round(porosity_pct, 4),
        "number_of_components":                n_comp,
        "largest_component_label":             largest_label,
        "largest_component_volume_voxels":     largest_vol,
        "largest_component_fraction_percent":  round(largest_frac, 4),
        "mean_component_volume_voxels":        round(float(sorted_counts.mean()), 2),
        "median_component_volume_voxels":      round(float(np.median(sorted_counts)), 2),
        "min_component_volume_voxels":         int(sorted_counts.min()),
        "max_component_volume_voxels":         int(sorted_counts.max()),
        "components_above_100_vox":            int((sorted_counts > 100).sum()),
        "components_above_1000_vox":           int((sorted_counts > 1000).sum()),
        "components_above_10000_vox":          int((sorted_counts > 10000).sum()),
        "boundary_connected_component_count":  n_boundary,
        "boundary_connected_pore_fraction_percent": round(100.0 * vol_boundary / total_pore_vox, 4)
                                                     if total_pore_vox else 0.0,
        "closed_pore_volume_voxels":           vol_closed,
        "closed_pore_fraction_percent":        round(100.0 * vol_closed / total_pore_vox, 4)
                                               if total_pore_vox else 0.0,
        "shell_threshold_mode":                SHELL_THRESHOLD_MODE,
        "pore_threshold_mode":                 PORE_THRESHOLD_MODE,
        "min_pore_size_voxels":                MIN_PORE_SIZE,
        "gaussian_sigma":                      GAUSSIAN_SIGMA,
    }

    _note(f"Boundary-connected  : {n_boundary} components  "
          f"({global_metrics['boundary_connected_pore_fraction_percent']:.1f} % of pore vol)")
    _note(f"Closed pore frac.   : {global_metrics['closed_pore_fraction_percent']:.1f} %")

    return labeled, comp_df, global_metrics, sorted_labels, sorted_counts, boundary_labels


# ============================================================
#  7.  OPEN vs CLOSED POROSITY FIGURES
# ============================================================

def compute_boundary_connected_components(comp_df: pd.DataFrame,
                                          global_metrics: dict):
    """Save open_vs_closed_porosity.png bar chart."""
    log.info("-" * 40)
    log.info("OPEN VS CLOSED POROSITY FIGURE")

    vol_open   = float(comp_df.loc[comp_df["touches_volume_boundary"], "volume_voxels"].sum())
    vol_closed = float(global_metrics["closed_pore_volume_voxels"])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Boundary-connected\n(open)", "Internal\n(closed)"],
                  [vol_open, vol_closed],
                  color=["#2196F3", "#FF9800"], edgecolor="black", width=0.5)
    for bar, val in zip(bars, [vol_open, vol_closed]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:,.0f}\n({100*val/(vol_open+vol_closed):.1f} %)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Pore volume [voxels]")
    ax.set_title("Open vs Closed Porosity")
    ax.set_ylim(0, max(vol_open, vol_closed) * 1.18)
    plt.tight_layout()
    _savefig("open_vs_closed_porosity.png", fig)


# ============================================================
#  8.  CONNECTED COMPONENT FIGURES
# ============================================================

def plot_connected_components_slice(smooth: np.ndarray, pore_mask: np.ndarray,
                                    labeled: np.ndarray, sorted_labels: np.ndarray):
    """Save connected_components_slice.png."""
    log.info("  Plotting connected_components_slice …")
    sz = smooth.shape[0] // 2

    slice_label = labeled[sz].copy()
    # re-map to top-N for colourmap clarity (show top 50 by size)
    top_n = 50
    top_labels = set(sorted_labels[:top_n].tolist())
    display_label = np.where(np.isin(slice_label, list(top_labels)), slice_label, 0)

    cmap = _random_label_cmap(256)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Connected components — central axial slice")
    _show_slice(axes[0], smooth[sz], title="Smoothed")
    axes[1].imshow(pore_mask[sz], cmap="gray", interpolation="nearest")
    axes[1].set_title("Pore mask"); axes[1].axis("off")
    axes[2].imshow(display_label % 255, cmap=cmap, interpolation="nearest")
    axes[2].set_title(f"Top {top_n} components (random colours)"); axes[2].axis("off")
    plt.tight_layout()
    _savefig("connected_components_slice.png", fig)


def plot_largest_component_3d(labeled: np.ndarray, sorted_labels: np.ndarray,
                               sorted_counts: np.ndarray):
    """Save largest_component_3d.png using marching cubes."""
    if not RUN_3D_RENDERING:
        log.info("  3D rendering skipped (RUN_3D_RENDERING=False)")
        return

    log.info("  3D marching-cubes rendering of largest component …")
    largest_lbl = int(sorted_labels[0])
    mask_large  = (labeled == largest_lbl)

    # downsample for speed if the volume is big
    ds = 1
    if mask_large.sum() > 5_000_000:
        ds = 2
        mask_large = mask_large[::ds, ::ds, ::ds]
        log.info(f"    Extra downsample ×{ds} for 3D rendering")

    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(mask_large.astype(np.float32),
                                             level=0.5, step_size=1)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(8, 7))
        ax  = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces] * ds,
                                alpha=0.60, linewidths=0,
                                facecolor="#2196F3", edgecolor="none")
        ax.add_collection3d(mesh)
        ax.set_xlim(0, labeled.shape[2])
        ax.set_ylim(0, labeled.shape[1])
        ax.set_zlim(0, labeled.shape[0])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        vol_v = int(sorted_counts[0])
        ax.set_title(f"Largest connected pore component\n"
                     f"({vol_v:,} voxels)", fontsize=10)
        plt.tight_layout()
        _savefig("largest_component_3d.png", fig)
    except Exception as e:
        log.warning(f"  3D rendering failed: {e}")


def plot_component_histogram(sorted_counts: np.ndarray):
    """Save component_size_distribution.png."""
    log.info("  Plotting component_size_distribution …")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sorted_counts, bins=min(100, len(sorted_counts)),
            color="steelblue", edgecolor="none", log=True)
    ax.set_xlabel("Component volume [voxels]")
    ax.set_ylabel("Number of components (log scale)")
    ax.set_title("Pore component size distribution")
    plt.tight_layout()
    _savefig("component_size_distribution.png", fig)


def plot_rank_size(sorted_counts: np.ndarray):
    """Save component_rank_size_plot.png."""
    log.info("  Plotting component_rank_size_plot …")
    ranks = np.arange(1, len(sorted_counts) + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, sorted_counts, "o-", ms=3, color="darkorange", lw=1.2)
    ax.set_xlabel("Rank (1 = largest)")
    ax.set_ylabel("Component volume [voxels] (log scale)")
    ax.set_title("Rank–size plot of pore components")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    _savefig("component_rank_size_plot.png", fig)


def plot_largest_component_fraction(global_metrics: dict):
    """Save largest_component_fraction.png."""
    log.info("  Plotting largest_component_fraction …")
    lc_frac   = global_metrics["largest_component_fraction_percent"]
    rest_frac = 100.0 - lc_frac

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Largest\ncomponent", "All other\ncomponents"],
                  [lc_frac, rest_frac],
                  color=["#E53935", "#78909C"], edgecolor="black", width=0.5)
    for bar, val in zip(bars, [lc_frac, rest_frac]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f} %", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Fraction of total pore volume [%]")
    ax.set_ylim(0, 115)
    ax.set_title("Largest connected component fraction")
    plt.tight_layout()
    _savefig("largest_component_fraction.png", fig)

    # scientific interpretation annotation
    if lc_frac > 70:
        interp = "DOMINANT NETWORK (>70 %): pore space is one connected network."
    elif lc_frac > 30:
        interp = "MIXED (~30–70 %): pore space is partly connected, partly isolated."
    else:
        interp = "FRAGMENTED (<30 %): pore space consists mostly of isolated pores."
    log.info(f"  Interpretation: {interp}")
    _note(f"Connectivity interp : {interp}")


# ============================================================
#  9.  THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

def run_threshold_sensitivity(smooth: np.ndarray, shell_mask: np.ndarray,
                               base_threshold: float):
    """
    Re-run pore segmentation + CCA at 5 threshold multiples and save results.
    """
    log.info("-" * 40)
    log.info("THRESHOLD SENSITIVITY ANALYSIS")

    multipliers = [0.90, 0.95, 1.00, 1.05, 1.10]
    thresholds  = [base_threshold * m for m in multipliers]
    struct26    = np.ones((3, 3, 3), dtype=np.int8)

    records = []
    for mult, thr in zip(multipliers, thresholds):
        log.info(f"  T × {mult:.2f} = {thr:.4f}")
        pm = (smooth < thr) & shell_mask
        pm = morphology.remove_small_objects(pm, min_size=MIN_PORE_SIZE)

        labeled_s, n_s = ndi.label(pm, structure=struct26)
        if n_s == 0:
            records.append({"threshold": thr, "multiplier": mult,
                             "porosity_percent": 0, "n_components": 0,
                             "largest_frac_percent": 0,
                             "boundary_frac_percent": 0})
            continue

        counts_s = np.bincount(labeled_s.ravel()); counts_s[0] = 0
        total_pore_s = int(pm.sum())
        total_shell_s = int(shell_mask.sum())
        por_s = 100.0 * total_pore_s / total_shell_s if total_shell_s else 0.0
        largest_s = int(counts_s.max())
        lf_s = 100.0 * largest_s / total_pore_s if total_pore_s else 0.0

        # boundary fraction
        Z, Y, X = labeled_s.shape
        bm = np.zeros(labeled_s.shape, bool)
        bm[0,:,:]=True; bm[-1,:,:]=True
        bm[:,0,:]=True; bm[:,-1,:]=True
        bm[:,:,0]=True; bm[:,:,-1]=True
        b_lbls = set(np.unique(labeled_s[bm])) - {0}
        vol_b  = int(sum(counts_s[l] for l in b_lbls))
        bf_s   = 100.0 * vol_b / total_pore_s if total_pore_s else 0.0

        records.append({"threshold": round(thr, 5), "multiplier": mult,
                        "porosity_percent": round(por_s, 3),
                        "n_components": n_s,
                        "largest_frac_percent": round(lf_s, 3),
                        "boundary_frac_percent": round(bf_s, 3)})

    df = pd.DataFrame(records)
    df.to_csv(DIRS["tables"] / "threshold_sensitivity_metrics.csv", index=False)
    log.info("  Saved threshold_sensitivity_metrics.csv")

    # figure
    thrs = [r["threshold"] for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    fig.suptitle("Threshold sensitivity analysis")

    axes[0].plot(thrs, [r["porosity_percent"]     for r in records], "o-", color="green")
    axes[0].set_ylabel("Porosity [%]"); axes[0].set_title("Porosity")
    axes[0].set_xlabel("Threshold")

    axes[1].plot(thrs, [r["n_components"]          for r in records], "o-", color="steelblue")
    axes[1].set_ylabel("N components"); axes[1].set_title("Number of components")
    axes[1].set_xlabel("Threshold")
    if max(r["n_components"] for r in records) > 0:
        axes[1].set_yscale("log")

    axes[2].plot(thrs, [r["largest_frac_percent"]  for r in records], "o-", color="red")
    axes[2].set_ylabel("Largest component [%]"); axes[2].set_title("Largest component fraction")
    axes[2].set_xlabel("Threshold")

    for ax in axes:
        ax.axvline(base_threshold, color="gray", ls="--", lw=1, label="base T")
        ax.grid(True, ls="--", alpha=0.4)
    axes[0].legend()
    plt.tight_layout()
    _savefig("threshold_sensitivity.png", fig)

    # sensitivity note
    range_lf = max(r["largest_frac_percent"] for r in records) - \
               min(r["largest_frac_percent"] for r in records)
    if range_lf > 20:
        sens_note = (f"HIGH sensitivity: largest component fraction varies "
                     f"{range_lf:.1f} pp across ±10 % threshold range. "
                     f"Connectivity result is threshold-sensitive.")
    else:
        sens_note = (f"LOW sensitivity: largest component fraction varies "
                     f"only {range_lf:.1f} pp across ±10 % threshold range. "
                     f"Result is robust.")
    log.info(f"  {sens_note}")
    _note(f"Threshold sensitivity: {sens_note}")

    return df


# ============================================================
# 10.  DISTANCE TRANSFORM / LOCAL THICKNESS
# ============================================================

def run_distance_transform(pore_mask: np.ndarray):
    """Compute EDT inside pore mask and save figures + metrics."""
    log.info("-" * 40)
    log.info("DISTANCE TRANSFORM (local pore radius)")

    dist = ndi.distance_transform_edt(pore_mask).astype(np.float32)
    dist_inside = dist[pore_mask]

    mean_r   = float(dist_inside.mean())
    median_r = float(np.median(dist_inside))
    max_r    = float(dist_inside.max())

    log.info(f"  Mean pore radius   : {mean_r:.2f} vox")
    log.info(f"  Median pore radius : {median_r:.2f} vox")
    log.info(f"  Max pore radius    : {max_r:.2f} vox")
    _note(f"Mean pore radius    : {mean_r:.2f} voxels")
    _note(f"Median pore radius  : {median_r:.2f} voxels")
    _note(f"Max pore radius     : {max_r:.2f} voxels")

    # slice figure
    sz  = dist.shape[0] // 2
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist[sz], cmap="hot", interpolation="nearest")
    ax.set_title("Distance transform — central axial slice\n"
                 "(value = distance to nearest solid [vox])")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Distance [voxels]")
    plt.tight_layout()
    _savefig("pore_distance_transform_slice.png", fig)

    # radius histogram
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(dist_inside, bins=200, color="tomato", edgecolor="none", log=True)
    ax2.axvline(mean_r,   color="navy",  ls="--", lw=1.5, label=f"mean={mean_r:.1f}")
    ax2.axvline(median_r, color="black", ls=":",  lw=1.5, label=f"median={median_r:.1f}")
    ax2.set_xlabel("Local pore radius [voxels]")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Distribution of local pore radii")
    ax2.legend()
    plt.tight_layout()
    _savefig("pore_radius_histogram.png", fig2)


# ============================================================
# 11.  SAVE METRICS / TABLES
# ============================================================

def save_metrics(comp_df: pd.DataFrame, global_metrics: dict):
    """Write component_statistics.csv and pore_connectivity_metrics.csv."""
    comp_path = DIRS["tables"] / "component_statistics.csv"
    comp_df.to_csv(comp_path, index=False)
    log.info(f"  Saved {comp_path.name}")

    gm_df = pd.DataFrame([global_metrics])
    gm_path = DIRS["tables"] / "pore_connectivity_metrics.csv"
    gm_df.to_csv(gm_path, index=False)
    log.info(f"  Saved {gm_path.name}")


# ============================================================
# 12.  ANALYSIS SUMMARY
# ============================================================

def write_summary():
    """Write analysis_summary.txt."""
    path = OUTPUT_DIR / "analysis_summary.txt"
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CORDATUM SHELL — CONNECTED PORE NETWORK ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        for line in SUMMARY_LINES:
            f.write(line + "\n")
        f.write("\n" + "=" * 60 + "\n")
    log.info(f"Summary written to: {path}")

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    for line in SUMMARY_LINES:
        print(line)
    print("=" * 60 + "\n")


# ============================================================
# 13.  MAIN
# ============================================================

def main():
    _note("=" * 58)
    _note("CORDATUM SHELL — CONNECTED PORE NETWORK ANALYSIS")
    _note("=" * 58)

    # 1. Load
    vol = load_volume()

    # 2. Raw visualisation
    make_slice_figures(vol)

    # 3. Normalise + smooth
    smooth, p_low, p_high = normalize_volume(vol)

    # free raw volume memory
    del vol

    # 4. Shell segmentation
    shell_mask, shell_threshold = segment_shell(smooth)

    # 5. Pore segmentation
    log.info("-" * 40)
    log.info("PORE SEGMENTATION")
    pore_mask, pore_threshold = segment_pores(smooth, shell_mask)
    make_pore_figures(smooth, pore_mask, pore_threshold)

    # Store pore threshold for sensitivity analysis
    _note(f"Pore threshold used : {pore_threshold:.4f}")

    # 6. Connected component analysis
    (labeled, comp_df, global_metrics,
     sorted_labels, sorted_counts, boundary_labels) = connected_component_analysis(
         pore_mask, shell_mask)

    # 7. Open vs closed
    compute_boundary_connected_components(comp_df, global_metrics)

    # 8. All component figures
    plot_connected_components_slice(smooth, pore_mask, labeled, sorted_labels)
    plot_largest_component_3d(labeled, sorted_labels, sorted_counts)
    plot_component_histogram(sorted_counts)
    plot_rank_size(sorted_counts)
    plot_largest_component_fraction(global_metrics)

    # 9. Save tables
    save_metrics(comp_df, global_metrics)

    # 10. Threshold sensitivity
    if RUN_THRESHOLD_SENSITIVITY:
        run_threshold_sensitivity(smooth, shell_mask, pore_threshold)

    # 11. Distance transform
    if RUN_DISTANCE_TRANSFORM:
        try:
            run_distance_transform(pore_mask)
        except Exception as e:
            log.warning(f"  Distance transform failed: {e}")

    # 12. Summary
    write_summary()

    log.info("=" * 60)
    log.info(f"All outputs saved to: {OUTPUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("FATAL ERROR:\n" + traceback.format_exc())
        sys.exit(1)
