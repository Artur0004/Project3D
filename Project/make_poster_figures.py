"""
make_poster_figures.py
======================
Generates poster-ready figures for Connected Pore Network Analysis
of the Cordatum Shell micro-CT dataset.

Outputs saved to OUTPUT_DIR:
  poster_fig1_data_overview.png / .pdf
  poster_fig2_pore_segmentation.png / .pdf
  poster_fig3_connectivity_result.png / .pdf
  poster_fig4_quantitative_summary.png / .pdf
  poster_main_metrics.csv
  poster_summary_text.txt
  poster_references.txt
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS  — adjust before running
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

DATA_PATH          = Path(r"C:\Users\artxm\PycharmProjects\3d2\Cordatum_Shell.tif")
RESULTS_DIR        = Path(r"C:\Users\artxm\PycharmProjects\3DImageAnalysis\cordatum_results")
OUTPUT_DIR         = RESULTS_DIR / "poster_final"

# Existing pre-computed mask files (binary uint8, 0/255)
USE_EXISTING_MASKS = True
EXISTING_MASKS_DIR = Path(r"C:\Users\artxm\PycharmProjects\3DImageAnalysis")

# Downsampling applied to both CT volume and masks (1 = full resolution)
DOWNSAMPLE_FACTOR  = 2

# Gaussian smoothing sigma for CT before thresholding (0 = skip)
GAUSSIAN_SIGMA     = 1.0

# Segmentation thresholds; None = auto Otsu
SHELL_THRESHOLD    = None    # float in [0, 1] or None
PORE_THRESHOLD     = None    # float in [0, 1] or None

# Minimum pore component size in voxels (after downsampling)
MIN_PORE_SIZE      = 50

# Output quality
DPI                = 300
SAVE_PDF           = True

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
import time
import warnings
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import scipy.ndimage as ndi
from skimage import filters, morphology, measure
import tifffile

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output -> {OUTPUT_DIR}\n")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    15,
    "axes.labelsize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.grid":         False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pct_clip(img, lo=1.0, hi=99.5):
    """Return (vmin, vmax) percentile clip values for display."""
    v1, v2 = np.percentile(img.ravel(), [lo, hi])
    return float(v1), float(v2)


def savefig(fig, stem):
    """Save figure as PNG (and optionally PDF)."""
    png = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    print(f"  Saved {png.name}")
    if SAVE_PDF:
        pdf = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print(f"  Saved {pdf.name}")
    plt.close(fig)


def panel_label(ax, letter, x=-0.04, y=1.03):
    """Add bold panel label (A, B, …) to an axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="right",
            color="black")


def show_raw(ax, img2d, title="", lo=1.0, hi=99.5):
    """Display a 2-D grayscale slice with percentile clipping."""
    vmin, vmax = pct_clip(img2d, lo=lo, hi=hi)
    ax.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title, fontsize=15, pad=5)
    ax.axis("off")


def show_binary(ax, mask2d, title="", color="white", bg="black"):
    """Display a binary 2-D mask."""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([bg, color])
    ax.imshow(mask2d.astype(np.uint8), cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title, fontsize=15, pad=5)
    ax.axis("off")


def dilate_display(mask2d, radius=3):
    """
    Dilate a binary 2-D mask by `radius` pixels for visual clarity only.
    Single-pixel pores become visible dots at poster scale.
    Does NOT modify the analysis mask.
    """
    struct = morphology.disk(radius)
    return morphology.dilation(mask2d.astype(bool), struct).astype(np.uint8)


def dense_pore_box(pore_slice, shell_slice=None, patch=120, pad=20):
    """
    Find the patch×patch region with the highest pore-per-shell-area density.
    Returns (r0, r1, c0, c1).
    """
    H, W = pore_slice.shape
    if not pore_slice.any():
        cy, cx = H // 2, W // 2
        r0 = max(0, cy - patch // 2); r1 = min(H, r0 + patch)
        c0 = max(0, cx - patch // 2); c1 = min(W, c0 + patch)
        return r0, r1, c0, c1
    # raw pore density via sliding window
    density = ndi.uniform_filter(pore_slice.astype(np.float32), size=patch)
    # exclude image borders so the box center never snaps to the edge
    border = patch // 2 + pad + 5
    density[:border, :]  = 0; density[H - border:, :]  = 0
    density[:, :border]  = 0; density[:, W - border:]  = 0
    # also exclude deep-background (outside shell) if shell available
    if shell_slice is not None and shell_slice.any():
        # erode shell mask so we prefer boxes inside the shell body
        eroded = ndi.binary_erosion(shell_slice > 0,
                                    iterations=max(1, patch // 8))
        density[~eroded] *= 0.01
    if density.max() == 0:                  # no valid candidate found
        cy, cx = H // 2, W // 2
        r0 = max(0, cy - patch // 2 - pad)
        r1 = min(H, r0 + patch + 2 * pad)
        c0 = max(0, cx - patch // 2 - pad)
        c1 = min(W, c0 + patch + 2 * pad)
        return int(r0), int(r1), int(c0), int(c1)
    r, c = np.unravel_index(density.argmax(), density.shape)
    r0 = max(0, int(r) - patch // 2 - pad)
    r1 = min(H, r0 + patch + 2 * pad)
    c0 = max(0, int(c) - patch // 2 - pad)
    c1 = min(W, c0 + patch + 2 * pad)
    return int(r0), int(r1), int(c0), int(c1)


def make_overlay_rgba(raw2d, pore2d, alpha_pore=0.55):
    """
    Overlay pore mask on grayscale CT slice.
    Pores rendered in cyan (#00BFFF) at alpha_pore opacity.
    """
    vmin, vmax = pct_clip(raw2d)
    gray = np.clip((raw2d.astype(np.float32) - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1).astype(np.float32)
    pore = pore2d > 0
    rgba[pore, 0] = 0.0 * alpha_pore + rgba[pore, 0] * (1 - alpha_pore)
    rgba[pore, 1] = 0.75 * alpha_pore + rgba[pore, 1] * (1 - alpha_pore)
    rgba[pore, 2] = 1.0 * alpha_pore + rgba[pore, 2] * (1 - alpha_pore)
    rgba = np.clip(rgba, 0, 1)
    return rgba


def crop_pad(arr2d, r0, r1, c0, c1):
    """Safely crop a 2-D slice to [r0:r1, c0:c1]."""
    H, W = arr2d.shape[:2]
    r0 = max(0, r0); r1 = min(H, r1)
    c0 = max(0, c0); c1 = min(W, c1)
    return arr2d[r0:r1, c0:c1]


def add_scale_note(fig, text, y=0.01, fontsize=9):
    """Add a small caption line at the bottom of the figure."""
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=fontsize, color="#555555",
             style="italic", wrap=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load CT volume + shell and pore masks.
    Returns (vol_raw, vol_norm, shell_mask, pore_mask) all at the same
    downsampled resolution.
    """
    f = DOWNSAMPLE_FACTOR

    # ── CT volume ────────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CT volume not found: {DATA_PATH}\n"
            f"Update DATA_PATH at the top of this script."
        )
    print("Loading CT volume ...")
    t0 = time.time()
    try:
        vol_mm = tifffile.memmap(DATA_PATH)
    except Exception:
        vol_mm = tifffile.imread(str(DATA_PATH))
    print(f"  Raw shape : {vol_mm.shape}  dtype={vol_mm.dtype}  "
          f"({time.time()-t0:.1f} s)")

    vol_raw = np.array(vol_mm[::f, ::f, ::f])
    print(f"  Downsampled shape : {vol_raw.shape}")
    del vol_mm

    # normalise to float32 [0, 1]
    p1, p999 = np.percentile(vol_raw, [1.0, 99.8])
    vol_norm = np.clip(
        (vol_raw.astype(np.float32) - p1) / max(p999 - p1, 1e-9),
        0.0, 1.0
    ).astype(np.float32)

    # ── masks ─────────────────────────────────────────────────────────────────
    shell_mask, pore_mask = _load_or_generate_masks(vol_raw, vol_norm, f)

    # sanity: shapes must agree
    if shell_mask.shape != vol_raw.shape:
        raise RuntimeError(
            f"Shape mismatch: vol={vol_raw.shape}, shell_mask={shell_mask.shape}. "
            f"Check DOWNSAMPLE_FACTOR or mask files."
        )

    print(f"  Shell voxels : {shell_mask.sum():,}")
    print(f"  Pore voxels  : {pore_mask.sum():,}")
    return vol_raw, vol_norm, shell_mask, pore_mask


def _load_or_generate_masks(vol_raw, vol_norm, f):
    """Load existing binary masks or regenerate from CT."""
    shell_path = EXISTING_MASKS_DIR / "cordatum_matter_mask.tif"
    pore_path  = EXISTING_MASKS_DIR / "cordatum_air_masked.tif"

    if USE_EXISTING_MASKS and shell_path.exists() and pore_path.exists():
        print("Loading existing masks ...")
        sm_full = tifffile.imread(str(shell_path))
        pm_full = tifffile.imread(str(pore_path))
        shell_mask = (sm_full[::f, ::f, ::f] > 0).astype(np.uint8)
        pore_mask  = (pm_full[::f, ::f, ::f] > 0).astype(np.uint8)
        del sm_full, pm_full
        return shell_mask, pore_mask

    print("Regenerating masks from CT ...")
    return _generate_masks(vol_norm)


def _generate_masks(vol_norm):
    """Otsu-based shell + pore segmentation from normalised CT."""
    smooth = (ndi.gaussian_filter(vol_norm, sigma=GAUSSIAN_SIGMA)
              if GAUSSIAN_SIGMA > 0 else vol_norm)

    # shell
    t_sh = (SHELL_THRESHOLD if SHELL_THRESHOLD is not None
            else filters.threshold_otsu(smooth))
    raw_shell = smooth >= t_sh
    lbl_sh, _ = ndi.label(raw_shell)
    props_sh = measure.regionprops(lbl_sh)
    largest_sh = max(props_sh, key=lambda p: p.area)
    shell_mask = ndi.binary_fill_holes(lbl_sh == largest_sh.label).astype(np.uint8)

    # pore inside shell
    inside_vals = smooth[shell_mask > 0]
    t_po = (PORE_THRESHOLD if PORE_THRESHOLD is not None
            else filters.threshold_otsu(inside_vals))
    raw_pore = (smooth <= t_po) & (shell_mask > 0)
    pore_mask = morphology.remove_small_objects(
        raw_pore, min_size=MIN_PORE_SIZE
    ).astype(np.uint8)

    return shell_mask, pore_mask


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTED-COMPONENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_cc_analysis(pore_mask, shell_mask):
    """
    3-D connected-component analysis (26-connectivity).
    Returns (labeled, metrics_dict).
    Label 1 = largest component, 2 = second-largest, …, 0 = background.
    """
    print("\nRunning CC analysis ...")
    struct26 = ndi.generate_binary_structure(3, 3)
    labeled_raw, n_raw = ndi.label(pore_mask > 0, structure=struct26)
    print(f"  Raw components: {n_raw}")

    # remove components smaller than MIN_PORE_SIZE
    props = measure.regionprops(labeled_raw)
    small_lbls = [p.label for p in props if p.area < MIN_PORE_SIZE]
    if small_lbls:
        small_mask = np.isin(labeled_raw, small_lbls)
        labeled_raw[small_mask] = 0
        labeled_raw, n_comp = ndi.label(labeled_raw > 0, structure=struct26)
        props = measure.regionprops(labeled_raw)
    else:
        n_comp = n_raw
    print(f"  After min-size filter: {n_comp} components")

    total_pore  = int((pore_mask > 0).sum())
    total_shell = int((shell_mask > 0).sum())
    porosity    = total_pore / max(total_shell, 1) * 100.0

    if n_comp == 0:
        metrics = dict(
            shell_volume=total_shell, pore_volume=total_pore,
            porosity_percent=round(porosity, 4), n_components=0,
            largest_vol=0, largest_frac=0.0, other_frac=0.0,
            boundary_frac=0.0, sizes=[],
        )
        return labeled_raw, metrics

    # sort by size (descending); re-label so 1 = largest
    comp_sizes = {p.label: p.area for p in props}
    sorted_labels = sorted(comp_sizes, key=comp_sizes.__getitem__, reverse=True)

    labeled = np.zeros_like(labeled_raw)
    for new_lbl, old_lbl in enumerate(sorted_labels, start=1):
        labeled[labeled_raw == old_lbl] = new_lbl

    sizes_sorted = [comp_sizes[l] for l in sorted_labels]
    largest_vol  = sizes_sorted[0]
    largest_frac = largest_vol / max(total_pore, 1) * 100.0
    other_frac   = 100.0 - largest_frac

    # boundary-connected: any voxel touching volume border
    Z, Y, X = labeled.shape
    border = np.zeros((Z, Y, X), dtype=bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    b_labels = np.unique(labeled[border & (labeled > 0)])
    boundary_vol  = sum(sizes_sorted[l - 1] for l in b_labels if 1 <= l <= len(sizes_sorted))
    boundary_frac = boundary_vol / max(total_pore, 1) * 100.0

    print(f"  Porosity              : {porosity:.3f}%")
    print(f"  Largest component     : {largest_frac:.1f}% of pore volume")
    print(f"  Boundary-connected    : {boundary_frac:.1f}% of pore volume")

    metrics = dict(
        shell_volume=total_shell,
        pore_volume=total_pore,
        porosity_percent=round(porosity, 4),
        n_components=n_comp,
        largest_vol=largest_vol,
        largest_frac=round(largest_frac, 2),
        other_frac=round(other_frac, 2),
        boundary_frac=round(boundary_frac, 2),
        sizes=sizes_sorted,
    )
    return labeled, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  REPRESENTATIVE SLICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_slices(pore_mask, shell_mask, labeled):
    """
    Choose representative z-slices and a zoom bounding box.
    Returns (z_center, z_high_pore, z_largest_cc, zoom_box)
    where zoom_box = (z, r0, r1, c0, c1).
    """
    Z, Y, X = pore_mask.shape
    margin  = max(Z // 10, 5)
    z_lo, z_hi = margin, Z - margin

    # central slice
    z_center = Z // 2

    # highest pore density inside the shell — use pore-to-shell ratio per slice
    shell_per_z = shell_mask.sum(axis=(1, 2)).astype(float)
    pore_per_z  = pore_mask.sum(axis=(1, 2)).astype(float)
    ratio = np.where(shell_per_z > 0, pore_per_z / shell_per_z, 0.0)
    ratio[:margin] = 0; ratio[Z - margin:] = 0
    z_high = int(np.argmax(ratio))
    if z_high == 0:                     # fallback: highest raw pore count
        pore_per_z[:margin] = 0; pore_per_z[Z - margin:] = 0
        z_high = int(np.argmax(pore_per_z))
    z_high = max(z_lo, min(z_hi, z_high))

    # slice where largest CC has the most 2-D area
    z_largest = z_center
    if labeled.max() >= 1:
        lc_per_z = (labeled == 1).sum(axis=(1, 2))
        lc_per_z[:margin] = 0; lc_per_z[Z - margin:] = 0
        z_largest = int(np.argmax(lc_per_z))
        z_largest = max(z_lo, min(z_hi, z_largest))

    # zoom box: search z_center first (shows main body), then z_high, then z_largest
    patch_size = max(80, min(Y, X) // 5)
    best_zoom  = None
    best_score = -1
    for z_cand in [z_center, z_high, z_largest]:
        z_cand = max(z_lo, min(z_hi, z_cand))
        r0c, r1c, c0c, c1c = dense_pore_box(
            pore_mask[z_cand],
            shell_slice=shell_mask[z_cand],
            patch=patch_size, pad=15,
        )
        score = int(pore_mask[z_cand, r0c:r1c, c0c:c1c].sum())
        if score > best_score:
            best_score = score
            best_zoom  = (z_cand, r0c, r1c, c0c, c1c)
    zoom_box = best_zoom

    print(f"\nSlices selected:")
    print(f"  z_center    = {z_center}")
    print(f"  z_high_pore = {z_high}")
    print(f"  z_largest   = {z_largest}")
    print(f"  zoom_box    = {zoom_box}")
    return z_center, z_high, z_largest, zoom_box


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1: DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def fig1_data_overview(vol_raw, shell_mask, pore_mask, slices):
    """
    2 × 3 composite: three orthogonal CT views + three zoomed pore regions.
    """
    print("\n-- Figure 1: Data Overview --")
    z_c, z_h, z_l, zoom_box = slices
    Z, Y, X = vol_raw.shape
    z_bx, r0, r1, c0, c1 = zoom_box

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.10, wspace=0.06,
                           left=0.04, right=0.98, top=0.90, bottom=0.06)

    axs = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    # ── top row: three orthogonal full slices ──
    show_raw(axs[0][0], vol_raw[z_c, :, :],
             title=f"Axial  (z = {z_c})")
    show_raw(axs[0][1], vol_raw[:, Y // 2, :],
             title=f"Coronal  (y = {Y // 2})")
    show_raw(axs[0][2], vol_raw[:, :, X // 2],
             title=f"Sagittal  (x = {X // 2})")

    # ── bottom row: zoomed pore regions ──
    # dilate pore mask for display (pores are ~1 px at full-slice scale)
    pore_disp_full = dilate_display(pore_mask[z_h, :, :], radius=2)

    # panel D: high-pore slice, full view with dilated pores highlighted
    overlay_full = make_overlay_rgba(vol_raw[z_h, :, :], pore_disp_full, alpha_pore=0.72)
    axs[1][0].imshow(overlay_full, interpolation="nearest", aspect="equal")
    axs[1][0].set_title(f"High-pore slice  (z = {z_h})  — pores in cyan",
                         fontsize=15, pad=5)
    axs[1][0].axis("off")

    # panel E: zoomed raw CT on the zoom-box slice
    raw_crop = crop_pad(vol_raw[z_bx, :, :], r0, r1, c0, c1)
    show_raw(axs[1][1], raw_crop,
             title=f"Zoomed raw CT  (z = {z_bx})")

    # panel F: zoomed pore overlay (mild dilation for clarity)
    pore_crop      = crop_pad(pore_mask[z_bx, :, :], r0, r1, c0, c1)
    pore_crop_disp = dilate_display(pore_crop, radius=1)
    overlay_crop   = make_overlay_rgba(raw_crop, pore_crop_disp, alpha_pore=0.72)
    axs[1][2].imshow(overlay_crop, interpolation="nearest", aspect="equal")
    axs[1][2].set_title("Zoomed — pores highlighted", fontsize=15, pad=5)
    axs[1][2].axis("off")

    # panel labels
    letters = [["A", "B", "C"], ["D", "E", "F"]]
    for r in range(2):
        for c in range(3):
            panel_label(axs[r][c], letters[r][c])

    fig.suptitle("Cordatum Shell — Micro-CT Data Overview",
                 fontsize=22, fontweight="bold", y=0.97)
    add_scale_note(
        fig,
        f"16-bit micro-CT volume, {Z * DOWNSAMPLE_FACTOR} × "
        f"{Y * DOWNSAMPLE_FACTOR} × {X * DOWNSAMPLE_FACTOR} voxels "
        f"(downsampled ×{DOWNSAMPLE_FACTOR} for display). Voxel size unknown.",
    )
    savefig(fig, "poster_fig1_data_overview")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2: PORE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def fig2_pore_segmentation(vol_raw, shell_mask, pore_mask, slices):
    """
    2 × 3 composite: full-slice row + zoomed-region row.
    Columns: raw CT / shell mask / pore overlay.
    """
    print("\n-- Figure 2: Pore Segmentation --")
    z_c, z_h, z_l, zoom_box = slices
    z_bx, r0, r1, c0, c1 = zoom_box

    # use central slice for the main segmentation illustration:
    # it shows the full shell cross-section, which is more informative than the
    # aperture slice z_h which is thin and near the edge of the volume.
    z_seg = z_c

    raw_full  = vol_raw[z_seg, :, :]
    po_full   = pore_mask[z_seg, :, :]
    # dilated pore for full-slice display (radius 2 avoids ring artefacts)
    po_full_d = dilate_display(po_full, radius=2)
    ov_full   = make_overlay_rgba(raw_full, po_full_d, alpha_pore=0.78)

    # for the zoomed row, pull from the zoom_box (which may be a different z)
    z_bx2, r0, r1, c0, c1 = zoom_box
    raw_full_bx = vol_raw[z_bx2, :, :]
    po_full_bx  = pore_mask[z_bx2, :, :]
    raw_crop  = crop_pad(raw_full_bx, r0, r1, c0, c1)
    po_crop   = crop_pad(po_full_bx,  r0, r1, c0, c1)
    po_crop_d = dilate_display(po_crop, radius=1)
    ov_crop   = make_overlay_rgba(raw_crop, po_crop_d, alpha_pore=0.75)

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.06,
                           left=0.04, right=0.98, top=0.90, bottom=0.06)
    axs = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    # ── top row: full slice ──
    # Panel A: raw CT
    show_raw(axs[0][0], raw_full, title="Raw CT slice")
    # Panel B: dilated pore mask (white-on-black); more informative than shell mask
    show_binary(axs[0][1], po_full_d, title="Segmented pore mask",
                color="white", bg="black")
    # Panel C: overlay
    axs[0][2].imshow(ov_full, interpolation="nearest", aspect="equal")
    axs[0][2].set_title("Pore overlay on CT", fontsize=15, pad=5)
    axs[0][2].axis("off")

    # ── bottom row: zoomed region ──
    show_raw(axs[1][0], raw_crop, title=f"Zoomed raw CT  (z = {z_bx2})")
    show_binary(axs[1][1], po_crop_d, title="Zoomed — Pore mask",
                color="white", bg="black")
    axs[1][2].imshow(ov_crop, interpolation="nearest", aspect="equal")
    axs[1][2].set_title("Zoomed — Pore overlay", fontsize=15, pad=5)
    axs[1][2].axis("off")

    # row annotations
    for ax, lbl in zip(axs[0], ["A", "B", "C"]):
        panel_label(ax, lbl)
    for ax, lbl in zip(axs[1], ["D", "E", "F"]):
        panel_label(ax, lbl)

    # row separating line
    fig.add_artist(plt.Line2D([0.03, 0.98], [0.50, 0.50],
                              transform=fig.transFigure,
                              color="#CCCCCC", linewidth=0.8))

    fig.suptitle("Pore Segmentation — Cordatum Shell",
                 fontsize=22, fontweight="bold", y=0.97)
    add_scale_note(
        fig,
        f"Top row: axial slice z = {z_seg} | Bottom row: zoomed region z = {z_bx2} "
        f"(both downsampled x{DOWNSAMPLE_FACTOR}). "
        "Pore mask: Otsu threshold inside shell. Pores shown in cyan (dilated 1-2 px for visibility).",
    )
    savefig(fig, "poster_fig2_pore_segmentation")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3: CONNECTIVITY RESULT
# ─────────────────────────────────────────────────────────────────────────────

def _build_label_cmap(n_show=20):
    """
    Colormap for labeled slices.
      0       → white (background)
      1       → strong orange (largest component)
      2..n    → muted tab10 colours
      n+1..   → light gray (small components)
    """
    tab10 = plt.cm.tab10.colors
    n_total = max(n_show + 2, 22)
    cols = np.ones((n_total, 4), dtype=float)          # RGBA white default
    cols[0]  = [1.0, 1.0, 1.0, 1.0]                   # background
    cols[1]  = [0.98, 0.42, 0.02, 1.0]                # largest → strong orange
    for i in range(2, n_show + 1):
        c = tab10[(i - 2) % 10]
        cols[i] = [c[0], c[1], c[2], 0.85]
    # everything above n_show → light gray
    if n_total > n_show + 1:
        cols[n_show + 1:] = [0.80, 0.80, 0.80, 0.6]
    return mcolors.ListedColormap(cols)


def _mip_projection(lc_mask, other_mask, shell_mask, axis):
    """
    Create a 2-D projection image (MIP-style).
      largest component → tab blue (#1f77b4)
      other pores       → light gray
      shell outline     → very light gray background
    Returns RGBA float32 array.
    """
    lc_proj    = lc_mask.any(axis=axis)
    other_proj = other_mask.any(axis=axis)
    sh_proj    = shell_mask.any(axis=axis)

    H, W = lc_proj.shape
    rgba = np.ones((H, W, 4), dtype=np.float32)    # white background

    rgba[sh_proj, :3]    = 0.93                     # very light gray shell silhouette
    rgba[other_proj, :3] = [0.70, 0.70, 0.70]      # other pores: gray
    rgba[lc_proj,   :3]  = [0.12, 0.47, 0.71]      # largest CC: tab blue

    return rgba


def fig3_connectivity_result(vol_raw, pore_mask, shell_mask, labeled,
                              metrics, slices):
    """
    1 × 3 composite:
      A. Binary pore mask on selected slice
      B. Top-N labelled components on same slice
      C. Three-view (XY / XZ / YZ) MIP of largest connected component
    """
    print("\n-- Figure 3: Connectivity Result --")
    z_c, z_h, z_l, zoom_box = slices
    n_comp        = metrics["n_components"]
    largest_frac  = metrics["largest_frac"]

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("white")

    # outer 1×3 grid; panel C gets its own inner 1×3 sub-grid
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08,
                              left=0.04, right=0.98, top=0.88, bottom=0.06)
    ax_A = fig.add_subplot(outer[0])
    ax_B = fig.add_subplot(outer[1])
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.04)
    ax_C0 = fig.add_subplot(inner[0])
    ax_C1 = fig.add_subplot(inner[1])
    ax_C2 = fig.add_subplot(inner[2])

    # masks for projection panels
    if n_comp > 0:
        lc_mask    = (labeled == 1).astype(bool)
        other_mask = (labeled > 1).astype(bool)
    else:
        lc_mask    = np.zeros_like(pore_mask, dtype=bool)
        other_mask = (pore_mask > 0)
    all_pore_mask = (pore_mask > 0)

    # ── Panel A: XY maximum-intensity projection of all pores (binary) ──
    # Project along Z (axis 0) to show plan view of all pore voxels
    pore_proj_xy = all_pore_mask.any(axis=0)           # shape (Y, X)
    shell_proj_xy = (shell_mask > 0).any(axis=0)
    # build RGBA: white bg, shell=light gray, pores=white dots on dark
    H_xy, W_xy = pore_proj_xy.shape
    rgba_A = np.ones((H_xy, W_xy, 4), dtype=np.float32)
    rgba_A[shell_proj_xy, :3] = 0.88                    # light gray shell
    rgba_A[pore_proj_xy,  :3] = [0.08, 0.08, 0.08]     # near-black pore dots
    ax_A.imshow(rgba_A, interpolation="nearest", aspect="equal")
    ax_A.set_title("All pore voxels\n(top-view projection, Z-axis)", fontsize=14, pad=5)
    ax_A.axis("off")
    panel_label(ax_A, "A")

    # ── Panel B: XY projection coloured by component (largest=orange, others=gray) ──
    lc_proj   = lc_mask.any(axis=0)
    oth_proj  = other_mask.any(axis=0)
    rgba_B = np.ones((H_xy, W_xy, 4), dtype=np.float32)
    rgba_B[shell_proj_xy, :3] = 0.92
    rgba_B[oth_proj,  :3]     = [0.65, 0.65, 0.65]     # gray = other pores
    rgba_B[lc_proj,   :3]     = [0.98, 0.42, 0.02]     # orange = largest CC
    ax_B.imshow(rgba_B, interpolation="nearest", aspect="equal")
    ax_B.set_title(f"Connected components\n(top-view projection, Z-axis)\n"
                   f"{n_comp} components  |  largest = {largest_frac:.1f}% of pore volume",
                   fontsize=13, pad=5)
    ax_B.axis("off")
    if n_comp > 0:
        patches = [
            mpatches.Patch(color="#FA6B05", label="Largest component"),
            mpatches.Patch(color="#AAAAAA", label="Other components"),
        ]
        ax_B.legend(handles=patches, loc="lower right", fontsize=10,
                    framealpha=0.9, edgecolor="#CCCCCC")
    panel_label(ax_B, "B")

    # ── Panel C: three-view MIP of largest CC ──
    if n_comp > 0:
        lc_mask    = (labeled == 1).astype(bool)
        other_mask = ((labeled > 1)).astype(bool)
    else:
        lc_mask    = np.zeros_like(pore_mask, dtype=bool)
        other_mask = (pore_mask > 0)

    axis_labels = [
        ("XY projection\n(top view)",   0, ax_C0),
        ("XZ projection\n(front view)", 1, ax_C1),
        ("YZ projection\n(side view)",  2, ax_C2),
    ]
    for title, axis, ax in axis_labels:
        rgba = _mip_projection(lc_mask, other_mask, shell_mask, axis)
        ax.imshow(rgba, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=12, pad=4)
        ax.axis("off")

    # shared label for panel C
    ax_C0.text(-0.06, 1.03, "C", transform=ax_C0.transAxes,
               fontsize=20, fontweight="bold", va="bottom", ha="right")

    # shared annotation below C
    fig.text(
        (outer[2].get_position(fig).x0 + outer[2].get_position(fig).x1) / 2,
        0.03,
        f"Largest connected component = {largest_frac:.1f}% of total pore volume\n"
        "(blue = largest CC,  gray = other pore components,  "
        "light background = shell silhouette)",
        ha="center", va="bottom", fontsize=10, color="#333333",
    )

    fig.suptitle("3-D Connected Pore Network — Cordatum Shell",
                 fontsize=22, fontweight="bold", y=0.97)
    savefig(fig, "poster_fig3_connectivity_result")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4: QUANTITATIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def fig4_quantitative_summary(metrics):
    """
    Two-panel summary figure.
      A. Bar chart: largest CC vs. other pores (% of pore volume)
      B. Clean metrics table
    Optionally adds a small component-size log-distribution inset.
    """
    print("\n-- Figure 4: Quantitative Summary --")

    lf  = metrics["largest_frac"]
    of  = metrics["other_frac"]
    bf  = metrics["boundary_frac"]
    por = metrics["porosity_percent"]
    n   = metrics["n_components"]
    sizes = metrics["sizes"]

    # connectivity interpretation
    if lf >= 70:
        interpretation = "connected"
        interp_color   = "#2196F3"
    elif lf <= 30:
        interpretation = "fragmented"
        interp_color   = "#E53935"
    else:
        interpretation = "mixed"
        interp_color   = "#FF9800"

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                           left=0.06, right=0.97, top=0.88, bottom=0.12)
    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])

    # ── Panel A: bar chart ──
    bars       = ["Largest\ncomponent", "All other\ncomponents", "Boundary-\nconnected"]
    values     = [lf, of, bf]
    bar_colors = ["#1f77b4", "#AAAAAA", "#FF9800"]
    bar_hatch  = ["", "", "//"]

    b = ax_A.bar(bars, values, color=bar_colors, hatch=bar_hatch,
                 edgecolor="white", linewidth=1.2, width=0.55, zorder=2)
    ax_A.set_ylim(0, max(110, max(values) * 1.18))
    ax_A.set_ylabel("Fraction of total pore volume [%]", fontsize=13)
    ax_A.set_title("Pore Volume Partitioning", fontsize=16, pad=8)
    ax_A.tick_params(axis="x", labelsize=12)
    ax_A.tick_params(axis="y", labelsize=11)
    ax_A.spines["top"].set_visible(False)
    ax_A.spines["right"].set_visible(False)
    ax_A.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax_A.axhline(y=0, color="black", linewidth=0.8)
    ax_A.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#DDDDDD", zorder=0)

    # value labels on bars
    for bar, val in zip(b, values):
        ax_A.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                  f"{val:.1f}%", ha="center", va="bottom",
                  fontsize=12, fontweight="bold", color="#222222")

    panel_label(ax_A, "A")

    # inset: component size distribution (log scale) if > 1 component
    if len(sizes) > 1:
        ax_ins = ax_A.inset_axes([0.62, 0.45, 0.36, 0.48])
        ranks  = np.arange(1, len(sizes) + 1)
        ax_ins.loglog(ranks, sizes, "o-", color="#555555",
                      markersize=3, linewidth=1.0)
        ax_ins.set_xlabel("Rank", fontsize=8)
        ax_ins.set_ylabel("Size [vox]", fontsize=8)
        ax_ins.set_title("Size distribution", fontsize=8, pad=2)
        ax_ins.tick_params(labelsize=7)
        ax_ins.spines["top"].set_visible(False)
        ax_ins.spines["right"].set_visible(False)

    # ── Panel B: metrics table ──
    ax_B.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["Porosity  φ",            f"{por:.3f} %"],
        ["Shell volume",           f"{metrics['shell_volume']:,} vox"],
        ["Pore volume",            f"{metrics['pore_volume']:,} vox"],
        ["No. pore components",    f"{n:,}"],
        ["Largest component",      f"{metrics['largest_vol']:,} vox"],
        ["Largest fraction  f₁",   f"{lf:.1f} %"],
        ["Other components",       f"{of:.1f} %"],
        ["Boundary-connected",     f"{bf:.1f} %"],
        ["Downsample factor",      f"×{DOWNSAMPLE_FACTOR}"],
        ["Min. component size",    f"{MIN_PORE_SIZE} vox"],
    ]

    n_rows = len(table_data)
    col_w  = [0.58, 0.42]
    y_start = 0.98

    for i, row in enumerate(table_data):
        y = y_start - i * (0.88 / (n_rows - 1))
        is_header = (i == 0)
        bg_color  = "#E8EFF8" if is_header else ("#F5F5F5" if i % 2 == 0 else "white")
        # row background
        rect = mpatches.FancyBboxPatch(
            (0.01, y - 0.045), 0.98, 0.08,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, edgecolor="none",
            transform=ax_B.transAxes, clip_on=False,
        )
        ax_B.add_patch(rect)
        ax_B.text(0.05, y, row[0], transform=ax_B.transAxes,
                  fontsize=12, va="center",
                  fontweight="bold" if is_header else "normal",
                  color="#111111")
        ax_B.text(0.60, y, row[1], transform=ax_B.transAxes,
                  fontsize=12, va="center", ha="left",
                  fontweight="bold" if is_header else "normal",
                  color="#111111")

    ax_B.set_title("Analysis Metrics", fontsize=16, pad=10)
    panel_label(ax_B, "B")

    # interpretation label
    fig.text(0.5, 0.02,
             f"Pore network interpretation: "
             f"{'CONNECTED' if interpretation == 'connected' else interpretation.upper()}  "
             f"(largest component fraction = {lf:.1f}%)",
             ha="center", va="bottom", fontsize=13,
             color=interp_color, fontweight="bold")

    fig.suptitle("Cordatum Shell — Quantitative Pore Analysis Summary",
                 fontsize=22, fontweight="bold", y=0.97)
    savefig(fig, "poster_fig4_quantitative_summary")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV METRICS
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(metrics):
    print("\nSaving metrics CSV ...")
    rows = [
        ("metric", "value", "unit"),
        ("shell_volume",                    metrics["shell_volume"],         "voxels"),
        ("pore_volume",                     metrics["pore_volume"],          "voxels"),
        ("porosity_percent",                metrics["porosity_percent"],     "%"),
        ("number_of_pore_components",       metrics["n_components"],         ""),
        ("largest_component_volume",        metrics["largest_vol"],          "voxels"),
        ("largest_component_fraction_percent", metrics["largest_frac"],      "%"),
        ("other_components_fraction_percent",  metrics["other_frac"],        "%"),
        ("boundary_connected_pore_fraction_percent", metrics["boundary_frac"], "%"),
        ("downsample_factor",               DOWNSAMPLE_FACTOR,               ""),
        ("pore_threshold",
         PORE_THRESHOLD if PORE_THRESHOLD is not None else "otsu_inside_shell", ""),
        ("shell_threshold",
         SHELL_THRESHOLD if SHELL_THRESHOLD is not None else "otsu",        ""),
        ("minimum_pore_size",               MIN_PORE_SIZE,                  "voxels"),
    ]
    path = OUTPUT_DIR / "poster_main_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(metrics):
    lf  = metrics["largest_frac"]
    bf  = metrics["boundary_frac"]
    por = metrics["porosity_percent"]
    n   = metrics["n_components"]

    if lf >= 70:
        interpretation = "connected"
    elif lf <= 30:
        interpretation = "fragmented"
    else:
        interpretation = "mixed (partially connected)"

    text = f"""\
Connected Pore Network Analysis — Cordatum Shell
=================================================

The Cordatum shell micro-CT volume was loaded from:
  {DATA_PATH}

A shell mask was generated using Otsu thresholding on the percentile-
normalised, Gaussian-smoothed (σ = {GAUSSIAN_SIGMA}) CT volume, with the
largest connected component retained and holes filled.

Pore space inside the shell was segmented using a second Otsu threshold
applied only to voxels within the shell mask. Connected-component analysis
was performed in 3-D with 26-connectivity (full neighbourhood).

Downsampling factor : ×{DOWNSAMPLE_FACTOR}
Minimum pore size   : {MIN_PORE_SIZE} voxels (after downsampling)

Key Results
-----------
Total shell volume            : {metrics['shell_volume']:,} voxels
Total pore volume             : {metrics['pore_volume']:,} voxels
Porosity  φ                   : {por:.4f} %
Number of pore components     : {n}
Largest component volume      : {metrics['largest_vol']:,} voxels
Largest component fraction f₁ : {lf:.2f} %
Other components fraction     : {metrics['other_frac']:.2f} %
Boundary-connected fraction   : {bf:.2f} %

Interpretation
--------------
The pore structure of the Cordatum shell is characterised as {interpretation}.
The largest connected pore component contains {lf:.1f}% of the total pore
volume. {"This suggests that pores form a dominant interconnected network."
if lf >= 70 else
"This suggests that pore space is distributed across many isolated cavities."
if lf <= 30 else
"Pore space consists of a mix of connected and isolated components."}
{"A significant fraction ({:.1f}%) of pore volume is boundary-connected (open pores).".format(bf) if bf > 10 else ""}

Formulas
--------
  Porosity:                φ = V_pore / V_shell × 100%
  Largest CC fraction:  f₁ = V_largest / V_pore × 100%
"""
    path = OUTPUT_DIR / "poster_summary_text.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  REFERENCES
# ─────────────────────────────────────────────────────────────────────────────

def save_references():
    text = """\
References
==========

1. Reedy, C. L. and Reedy, C. L. (2022). High-resolution micro-CT with 3D
   image analysis for porosity characterization of historic bricks.
   Heritage Science, 10, 83. DOI: 10.1186/s40494-022-00723-4

2. van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F.,
   Warner, J. D., Yager, N., Gouillart, E., Yu, T., and the scikit-image
   contributors. (2014). scikit-image: Image processing in Python.
   PeerJ, 2, e453. DOI: 10.7717/peerj.453

3. Otsu, N. (1979). A threshold selection method from gray-level histograms.
   IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62–66.

4. University of Helsinki X-Ray Micro-CT Laboratory. Porosity Analysis
   service description. Used as inspiration for pore size, pore distribution,
   and connected-component analysis outputs.
"""
    path = OUTPUT_DIR / "poster_references.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("  Cordatum Shell - Poster Figure Generator")
    print("=" * 60)

    # 1. load data
    vol_raw, vol_norm, shell_mask, pore_mask = load_data()

    # 2. connected-component analysis
    labeled, metrics = run_cc_analysis(pore_mask, shell_mask)

    # 3. choose representative slices
    slices = select_slices(pore_mask, shell_mask, labeled)

    # 4. generate figures
    fig1_data_overview(vol_raw, shell_mask, pore_mask, slices)
    fig2_pore_segmentation(vol_raw, shell_mask, pore_mask, slices)
    fig3_connectivity_result(vol_raw, pore_mask, shell_mask, labeled,
                             metrics, slices)
    fig4_quantitative_summary(metrics)

    # 5. save text outputs
    save_metrics(metrics)
    save_summary(metrics)
    save_references()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed:.1f} s")
    print(f"  All outputs -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
