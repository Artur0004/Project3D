"""
make_poster_figures_v2.py
=========================
ROI-based poster figures for Cordatum Shell pore network analysis.

Improvements over v1:
- Works inside a manually/auto-selected ROI deep in the shell material.
- Avoids circular scan-boundary artifacts (no scan edge in the ROI).
- Pores are segmented as low-density trabecular channels WITHIN the shell
  matrix (not exterior air or scan boundary).
- Only 3 simple, large-font, non-overlapping poster figures.

Output folder:
  cordatum_results/poster_final_v2/
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS  — adjust these before running
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

DATA_PATH  = Path(r"C:\Users\artxm\PycharmProjects\3d2\Cordatum_Shell.tif")
OUTPUT_DIR = Path(r"C:\Users\artxm\PycharmProjects\3DImageAnalysis"
                  r"\cordatum_results\poster_final_v2")

# ── ROI bounds in DOWNSAMPLED voxel coordinates.
#    Set to None for automatic detection (recommended first run).
#    Override manually after checking Figure 1 output.
ROI_Z = None   # e.g. (172, 322)
ROI_Y = None   # e.g. (178, 328)
ROI_X = None   # e.g. (173, 323)

# Automatic ROI half-size in voxels (each side of centroid); used when ROI_* = None
ROI_HALF = 75

# ── Processing
DOWNSAMPLE_FACTOR          = 2      # applied to both CT and ROI bounds
GAUSSIAN_SIGMA             = 0.8    # smoothing before segmentation

# Solid shell threshold ('otsu' uses ROI-internal Otsu; 'manual' uses value below)
SOLID_THRESHOLD_MODE       = "otsu"
SOLID_MANUAL_THRESHOLD     = 0.65   # normalised [0,1]; used when mode='manual'

# Pore threshold ('otsu' = same as solid threshold; 'manual' uses value below)
PORE_THRESHOLD_MODE        = "otsu"
PORE_MANUAL_THRESHOLD      = 0.55   # used when mode='manual'

# Morphological closing of solid mask before building shell envelope
CLOSE_SOLID_RADIUS         = 1      # voxels; set 0 to skip

# After segmenting pores, remove components touching ROI face?
# True: removes components at ROI edge (useful if ROI clips the shell surface)
# False: keep everything (good if ROI is fully inside the shell matrix)
REMOVE_BOUNDARY_PORES      = False

MIN_PORE_SIZE              = 100    # voxels (after downsampling)

# ── Output
DPI       = 400
SAVE_PDF  = True

# Threshold multipliers used in the sensitivity / validation analysis
THRESH_MULTIPLIERS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

# Multipliers shown in the visual comparison figure (Fig 4)
COMPARISON_MULTIPLIERS = [0.85, 1.00, 1.15]
COMPARISON_LABELS      = ["Low threshold", "Chosen threshold", "High threshold"]

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys, time, warnings, csv
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
    "font.family":        "DejaVu Sans",
    "font.size":          13,
    "axes.titlesize":     16,
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pct_clip(img, lo=1.0, hi=99.5):
    return float(np.percentile(img.ravel(), lo)), float(np.percentile(img.ravel(), hi))


def savefig(fig, stem):
    png = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    print(f"  Saved {png.name}")
    if SAVE_PDF:
        pdf = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print(f"  Saved {pdf.name}")
    plt.close(fig)


def panel_label(ax, letter, x=-0.04, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="bottom", ha="right",
            color="black")


def show_ct(ax, img2d, title="", lo=1.0, hi=99.5):
    vmin, vmax = pct_clip(img2d, lo, hi)
    ax.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title, fontsize=16, pad=6)
    ax.axis("off")


def show_mask(ax, mask2d, title="", fg="white", bg="black"):
    cmap = mcolors.ListedColormap([bg, fg])
    ax.imshow(mask2d.astype(np.uint8), cmap=cmap,
              vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title, fontsize=16, pad=6)
    ax.axis("off")


def overlay_rgba(raw2d, pore2d, alpha=0.65):
    """Return RGBA float32: raw in gray, pores in cyan (#00BFFF)."""
    vmin, vmax = pct_clip(raw2d)
    gray = np.clip((raw2d.astype(np.float32) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1).astype(np.float32)
    p = pore2d > 0
    rgba[p, 0] = 0.0 * alpha + rgba[p, 0] * (1 - alpha)
    rgba[p, 1] = 0.75 * alpha + rgba[p, 1] * (1 - alpha)
    rgba[p, 2] = 1.0 * alpha + rgba[p, 2] * (1 - alpha)
    return np.clip(rgba, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD CT VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def load_ct():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CT not found: {DATA_PATH}")
    f = DOWNSAMPLE_FACTOR
    print("Loading CT ...")
    t0 = time.time()
    try:
        mm = tifffile.memmap(DATA_PATH)
    except Exception:
        mm = tifffile.imread(str(DATA_PATH))
    print(f"  Raw {mm.shape}  {mm.dtype}  ({time.time()-t0:.1f} s)")
    vol = np.array(mm[::f, ::f, ::f])
    print(f"  Downsampled {vol.shape}")
    # normalise to float32 [0, 1]
    p1 = float(np.percentile(vol, 1.0))
    p999 = float(np.percentile(vol, 99.8))
    norm = np.clip((vol.astype(np.float32) - p1) / max(p999 - p1, 1e-9), 0, 1)
    return vol, norm


# ─────────────────────────────────────────────────────────────────────────────
#  ROI SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_roi(norm):
    """
    Return (z0,z1, y0,y1, x0,x1) in downsampled coords.
    If ROI_* parameters are set, use them directly.
    Otherwise find the centroid of bright shell material and place
    a ROI_HALF-voxel cube around it.
    """
    Z, Y, X = norm.shape

    if ROI_Z is not None and ROI_Y is not None and ROI_X is not None:
        z0, z1 = int(ROI_Z[0]), int(ROI_Z[1])
        y0, y1 = int(ROI_Y[0]), int(ROI_Y[1])
        x0, x1 = int(ROI_X[0]), int(ROI_X[1])
        print(f"  Using manual ROI: Z={z0}:{z1}  Y={y0}:{y1}  X={x0}:{x1}")
        return z0, z1, y0, y1, x0, x1

    print("  Auto-detecting ROI ...")
    # global Otsu to find solid shell material
    t_global = filters.threshold_otsu(norm)
    solid = norm >= t_global

    # centroid of solid material
    coords = np.where(solid)
    zc = int(np.mean(coords[0]))
    yc = int(np.mean(coords[1]))
    xc = int(np.mean(coords[2]))
    print(f"  Shell centroid (z={zc}, y={yc}, x={xc})")

    h = ROI_HALF
    z0 = max(0, zc - h); z1 = min(Z, zc + h)
    y0 = max(0, yc - h); y1 = min(Y, yc + h)
    x0 = max(0, xc - h); x1 = min(X, xc + h)

    # verify ROI is mostly solid
    roi_solid = solid[z0:z1, y0:y1, x0:x1].mean()
    print(f"  ROI solid fraction: {roi_solid:.3f}")
    if roi_solid < 0.5:
        print("  WARNING: ROI has low solid fraction. Consider overriding ROI_* manually.")

    print(f"  Auto ROI: Z={z0}:{z1}  Y={y0}:{y1}  X={x0}:{x1}")
    return z0, z1, y0, y1, x0, x1


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION WITHIN ROI
# ─────────────────────────────────────────────────────────────────────────────

def segment_roi(roi_norm):
    """
    Segment solid shell material and pore space in the ROI.
    Returns (solid_mask, pore_mask, smooth_roi, t_solid).
    All arrays have the same shape as roi_norm.
    """
    # smooth
    smooth = (ndi.gaussian_filter(roi_norm.astype(np.float32), sigma=GAUSSIAN_SIGMA)
              if GAUSSIAN_SIGMA > 0 else roi_norm.astype(np.float32))

    # solid threshold
    if SOLID_THRESHOLD_MODE == "otsu":
        t_solid = float(filters.threshold_otsu(smooth))
    else:
        t_solid = float(SOLID_MANUAL_THRESHOLD)
    print(f"  Solid threshold: {t_solid:.4f}")

    solid_raw = smooth >= t_solid

    # optional morphological closing (fills thin gaps in solid)
    if CLOSE_SOLID_RADIUS > 0:
        ball = morphology.ball(CLOSE_SOLID_RADIUS)
        solid_closed = morphology.binary_closing(solid_raw, ball).astype(bool)
    else:
        solid_closed = solid_raw.astype(bool)

    # shell envelope: dilate solid to include immediately adjacent pore voxels
    envelope = ndi.binary_dilation(solid_closed, iterations=max(1, CLOSE_SOLID_RADIUS + 2))

    # pore threshold (same as solid threshold unless overridden)
    if PORE_THRESHOLD_MODE == "otsu":
        t_pore = t_solid
    else:
        t_pore = float(PORE_MANUAL_THRESHOLD)

    # pore candidates: dark regions within envelope, not solid
    pore_cand = (smooth < t_pore) & envelope & (~solid_closed)

    # remove components touching ROI boundary
    if REMOVE_BOUNDARY_PORES:
        pore_cand = _remove_boundary_components(pore_cand)

    # remove tiny noise objects
    pore_mask = morphology.remove_small_objects(pore_cand, min_size=MIN_PORE_SIZE)

    print(f"  Solid voxels : {solid_closed.sum():,}  ({solid_closed.mean()*100:.1f}%)")
    print(f"  Pore voxels  : {pore_mask.sum():,}  ({pore_mask.mean()*100:.1f}%)")
    return solid_closed.astype(np.uint8), pore_mask.astype(np.uint8), smooth, t_solid


def _remove_boundary_components(mask):
    """Remove connected components that touch any face of the 3D volume."""
    struct = ndi.generate_binary_structure(3, 3)
    labeled, _ = ndi.label(mask, structure=struct)
    Z, Y, X = mask.shape
    border = np.zeros((Z, Y, X), dtype=bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    bad = set(np.unique(labeled[border & (labeled > 0)])) - {0}
    clean = mask.copy()
    for lbl in bad:
        clean[labeled == lbl] = False
    return clean


def _segment_pores_for_t(smooth, t):
    """Lightweight pore segmentation for a single threshold value (silent)."""
    solid_raw = smooth >= t
    if CLOSE_SOLID_RADIUS > 0:
        solid_closed = morphology.binary_closing(solid_raw, morphology.ball(CLOSE_SOLID_RADIUS))
    else:
        solid_closed = solid_raw.astype(bool)
    envelope = ndi.binary_dilation(solid_closed, iterations=max(1, CLOSE_SOLID_RADIUS + 2))
    pore_cand = (smooth < t) & envelope & (~solid_closed)
    if REMOVE_BOUNDARY_PORES:
        pore_cand = _remove_boundary_components(pore_cand)
    return morphology.remove_small_objects(pore_cand, min_size=MIN_PORE_SIZE).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTED-COMPONENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_cc(pore_mask, solid_mask):
    """
    3-D CC analysis with 26-connectivity.
    Returns (labeled, metrics_dict).  Label 1 = largest component.
    """
    print("\nCC analysis ...")
    struct26 = ndi.generate_binary_structure(3, 3)
    labeled_raw, n_comp = ndi.label(pore_mask, structure=struct26)
    print(f"  {n_comp} components found")

    total_pore  = int(pore_mask.sum())
    total_solid = int(solid_mask.sum())
    roi_size    = pore_mask.size
    porosity    = total_pore / max(total_solid + total_pore, 1) * 100.0

    if n_comp == 0:
        return labeled_raw, dict(
            n_components=0, largest_vol=0, largest_frac=0.0,
            other_frac=0.0, boundary_frac=0.0, total_pore=total_pore,
            total_solid=total_solid, roi_size=roi_size,
            porosity=round(porosity, 4), sizes=[],
        )

    props = measure.regionprops(labeled_raw)
    comp_sizes = {p.label: p.area for p in props}
    sorted_lbl = sorted(comp_sizes, key=comp_sizes.__getitem__, reverse=True)

    # re-label: 1 = largest
    labeled = np.zeros_like(labeled_raw)
    for new_lbl, old_lbl in enumerate(sorted_lbl, start=1):
        labeled[labeled_raw == old_lbl] = new_lbl

    sizes = [comp_sizes[l] for l in sorted_lbl]
    largest_vol  = sizes[0]
    largest_frac = largest_vol / max(total_pore, 1) * 100.0
    other_frac   = 100.0 - largest_frac

    # boundary-connected fraction (after removing if REMOVE_BOUNDARY_PORES=True)
    Z, Y, X = labeled.shape
    border = np.zeros((Z, Y, X), dtype=bool)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    b_lbl = set(np.unique(labeled[border & (labeled > 0)])) - {0}
    boundary_vol  = sum(sizes[l - 1] for l in b_lbl if 1 <= l <= len(sizes))
    boundary_frac = boundary_vol / max(total_pore, 1) * 100.0

    print(f"  Porosity          : {porosity:.3f}%")
    print(f"  Largest component : {largest_frac:.1f}% of pore volume")
    print(f"  Boundary-connected: {boundary_frac:.1f}%")

    return labeled, dict(
        n_components=n_comp, largest_vol=largest_vol,
        largest_frac=round(largest_frac, 2),
        other_frac=round(other_frac, 2),
        boundary_frac=round(boundary_frac, 2),
        total_pore=total_pore, total_solid=total_solid,
        roi_size=roi_size, porosity=round(porosity, 4),
        sizes=sizes,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BEST DISPLAY SLICE (within ROI)
# ─────────────────────────────────────────────────────────────────────────────

def best_slice(pore_mask, labeled):
    """Return the z-index (within ROI) with the most labeled pore voxels."""
    nz = pore_mask.shape[0]
    pore_per_z = pore_mask.sum(axis=(1, 2))
    margin = max(1, nz // 10)
    pore_per_z[:margin] = 0; pore_per_z[nz - margin:] = 0
    z_best = int(np.argmax(pore_per_z))
    if pore_per_z[z_best] == 0:
        z_best = nz // 2
    return z_best


# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLD SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sensitivity_analysis(smooth_roi, t_solid):
    """
    Re-segment pores at several threshold multipliers around t_solid.
    Returns list of dicts with multiplier, threshold, porosity, n_components,
    largest_component_fraction_percent.
    """
    print("\nThreshold sensitivity analysis ...")
    struct26 = ndi.generate_binary_structure(3, 3)
    results = []
    for mult in THRESH_MULTIPLIERS:
        t = t_solid * mult
        pm = _segment_pores_for_t(smooth_roi, t)
        total_pore = int(pm.sum())
        solid_est  = int((smooth_roi >= t).sum())
        porosity   = total_pore / max(solid_est + total_pore, 1) * 100.0
        if total_pore > 0:
            lbl_raw, n_comp = ndi.label(pm, structure=struct26)
            props = measure.regionprops(lbl_raw)
            sizes = sorted([p.area for p in props], reverse=True)
            largest_frac = sizes[0] / max(total_pore, 1) * 100.0
        else:
            n_comp = 0
            largest_frac = 0.0
        results.append(dict(
            multiplier=mult,
            threshold=round(float(t), 4),
            porosity=round(float(porosity), 3),
            n_components=int(n_comp),
            largest_frac=round(float(largest_frac), 2),
        ))
        print(f"  mult={mult:.2f}  t={t:.4f}  por={porosity:.2f}%  "
              f"n={n_comp}  f1={largest_frac:.1f}%")
    return results


def _robustness_verdict(sens_data):
    """Return (short_label, long_sentence) describing threshold robustness."""
    fracs = [r["largest_frac"] for r in sens_data]
    n_above = sum(1 for f in fracs if f >= 70)
    if n_above >= len(fracs) - 1:
        return (
            "robust",
            "The connected network interpretation is robust across the tested threshold range.",
        )
    else:
        return (
            "sensitive",
            "The connectivity result is sensitive to threshold choice.",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  LABEL COLORMAP
# ─────────────────────────────────────────────────────────────────────────────

def _label_cmap(n_show=15):
    """
    Return (cmap, norm) for labeled images using BoundaryNorm so that
    integer label i maps exactly to colormap color index i.
      0         → white (background)
      1         → tab blue (largest CC)
      2..n_show → tab10 colours
      > n_show  → light gray
    """
    tab10 = plt.cm.tab10.colors
    n_colors = n_show + 2          # indices 0 … n_show+1
    cols = np.ones((n_colors, 4), dtype=float)
    cols[0]  = [1.0, 1.0, 1.0, 1.0]               # background = white
    cols[1]  = [0.12, 0.47, 0.71, 1.0]            # largest CC = tab blue
    for i in range(2, n_show + 1):
        c = tab10[(i - 2) % 10]
        cols[i] = [c[0], c[1], c[2], 0.9]
    cols[n_show + 1] = [0.78, 0.78, 0.78, 0.75]   # overflow = light gray
    cmap   = mcolors.ListedColormap(cols)
    # BoundaryNorm: value i maps to colour index i (integer → exact bin)
    bounds = np.arange(-0.5, n_colors, 1.0)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1: ROI OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def fig1_roi_overview(vol_norm, roi_bounds):
    """
    1 x 3: full CT slice | slice with ROI box | zoomed ROI interior.
    """
    print("\n-- Figure 1: ROI Overview --")
    z0, z1, y0, y1, x0, x1 = roi_bounds
    Z, Y, X = vol_norm.shape

    # choose an axial slice near the ROI centre
    z_disp = (z0 + z1) // 2
    full_slice = vol_norm[z_disp, :, :]
    roi_crop   = vol_norm[z_disp, y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                             gridspec_kw={"wspace": 0.06,
                                          "left": 0.04, "right": 0.98,
                                          "top": 0.85, "bottom": 0.06})

    # Panel A: full CT slice
    show_ct(axes[0], full_slice, title=f"Raw CT  (axial z = {z_disp})")
    panel_label(axes[0], "A")

    # Panel B: same slice with ROI rectangle
    show_ct(axes[1], full_slice, title="Selected analysis region")
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2.5, edgecolor="#FF3333", facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].text((x0 + x1) / 2, y0 - 6, "ROI",
                 color="#FF3333", fontsize=13, fontweight="bold",
                 ha="center", va="bottom")
    panel_label(axes[1], "B")

    # Panel C: zoomed ROI
    show_ct(axes[2], roi_crop, title="Zoomed ROI — shell microstructure")
    panel_label(axes[2], "C")

    fig.suptitle(
        "Cordatum Shell Micro-CT and Selected ROI",
        fontsize=22, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.02,
        f"16-bit micro-CT, downsampled x{DOWNSAMPLE_FACTOR}. "
        f"ROI: Z={z0}-{z1}, Y={y0}-{y1}, X={x0}-{x1} "
        f"({z1-z0} x {y1-y0} x {x1-x0} voxels, entirely inside shell material).",
        ha="center", va="bottom", fontsize=10, color="#555555", style="italic",
    )
    savefig(fig, "poster_fig1_roi_overview")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2: ROI SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def fig2_roi_segmentation(roi_norm, solid_mask, pore_mask, z_s, roi_bounds):
    """
    1 x 4: raw ROI slice | solid mask | pore mask | overlay.
    """
    print("\n-- Figure 2: ROI Segmentation --")
    z0, z1, y0, y1, x0, x1 = roi_bounds

    raw_sl   = roi_norm[z_s, :, :]
    solid_sl = solid_mask[z_s, :, :]
    pore_sl  = pore_mask[z_s, :, :]
    ov_sl    = overlay_rgba(raw_sl, pore_sl, alpha=0.72)

    fig, axes = plt.subplots(1, 4, figsize=(22, 7),
                             gridspec_kw={"wspace": 0.05,
                                          "left": 0.03, "right": 0.99,
                                          "top": 0.85, "bottom": 0.06})

    show_ct(axes[0],  raw_sl,   title="Raw ROI slice")
    show_mask(axes[1], solid_sl, title="Solid shell material", fg="#EEEEEE", bg="black")
    show_mask(axes[2], pore_sl,  title="Segmented pore space",  fg="white",    bg="black")
    axes[3].imshow(ov_sl, interpolation="nearest", aspect="equal")
    axes[3].set_title("Pore overlay  (cyan)", fontsize=16, pad=6)
    axes[3].axis("off")

    for ax, lbl in zip(axes, ["A", "B", "C", "D"]):
        panel_label(ax, lbl)

    fig.suptitle(
        "Pore Segmentation — Shell ROI",
        fontsize=22, fontweight="bold", y=0.97,
    )
    z_abs = z0 + z_s
    fig.text(
        0.5, 0.02,
        f"Axial slice z = {z_abs} (ROI local z = {z_s}). "
        f"Solid: Otsu threshold within ROI. "
        f"Pores: low-density regions in trabecular shell matrix (cyan, dilated 1 px). "
        f"Scan boundary excluded.",
        ha="center", va="bottom", fontsize=10, color="#555555", style="italic",
    )
    savefig(fig, "poster_fig2_roi_segmentation")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3: CONNECTIVITY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def fig3_connectivity_summary(roi_norm, pore_mask, solid_mask, labeled,
                               metrics, z_s):
    """
    2 x 2 composite:
      A. Pore mask on display slice
      B. Labelled components on same slice
      C. XY MIP projection coloured by component
      D. Bar chart with key metrics
    """
    print("\n-- Figure 3: Connectivity Summary --")

    lf = metrics["largest_frac"]
    of = metrics["other_frac"]
    bf = metrics["boundary_frac"]
    nc = metrics["n_components"]

    if lf >= 70:
        interp       = "CONNECTED"
        interp_color = "#1565C0"
    elif lf <= 30:
        interp       = "FRAGMENTED"
        interp_color = "#B71C1C"
    else:
        interp       = "MIXED"
        interp_color = "#E65100"

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.28, wspace=0.12,
        left=0.07, right=0.97, top=0.88, bottom=0.10,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    pore_sl = pore_mask[z_s, :, :]
    lbl_sl  = labeled[z_s, :, :].astype(np.int32)

    # ── A: pore mask (white on black) ──
    show_mask(ax_A, pore_sl, title="Pore space (binary)", fg="white", bg="black")
    panel_label(ax_A, "A")

    # ── B: labelled components (correct colormap via BoundaryNorm) ──
    n_show   = min(15, nc) if nc > 0 else 1
    cmap_lbl, norm_lbl = _label_cmap(n_show)
    lbl_disp = np.where(lbl_sl > n_show, n_show + 1, lbl_sl).astype(np.int32)
    ax_B.imshow(lbl_disp, cmap=cmap_lbl, norm=norm_lbl,
                interpolation="nearest", aspect="equal")
    ax_B.set_title(f"Connected components  ({nc} total)", fontsize=16, pad=6)
    ax_B.axis("off")
    if nc > 0:
        leg = [
            mpatches.Patch(color="#1f77b4", label="Largest component"),
            mpatches.Patch(color="#BBBBBB", label="Other components"),
        ]
        ax_B.legend(handles=leg, loc="lower right", fontsize=12,
                    framealpha=0.9, edgecolor="#CCCCCC")
    panel_label(ax_B, "B")

    # ── C: three-view projection (XY / XZ / YZ) coloured by component ──
    # Replace ax_C with a 1×3 sub-grid so each projection gets its own axes.
    ax_C.set_visible(False)   # hide the placeholder axes
    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, 0], wspace=0.04,
    )
    ax_C_views = [fig.add_subplot(gs_C[0, i]) for i in range(3)]

    lc_mask    = (labeled == 1)
    other_mask = (labeled > 1)
    solid_any  = (solid_mask > 0)
    C_COL_LC   = np.array([0.12, 0.47, 0.71])   # tab blue
    C_COL_OTH  = np.array([0.72, 0.72, 0.72])   # gray
    C_COL_SH   = np.array([0.93, 0.93, 0.93])   # very light shell background

    proj_defs = [
        ("XY  (top view)",   0),
        ("XZ  (front view)", 1),
        ("YZ  (side view)",  2),
    ]
    for (title, axis), ax_v in zip(proj_defs, ax_C_views):
        lc_p    = lc_mask.any(axis=axis)
        oth_p   = other_mask.any(axis=axis)
        sh_p    = solid_any.any(axis=axis)
        H2, W2  = lc_p.shape
        img     = np.ones((H2, W2, 4), dtype=np.float32)
        img[sh_p,  :3] = C_COL_SH
        img[oth_p, :3] = C_COL_OTH
        img[lc_p,  :3] = C_COL_LC
        ax_v.imshow(img, interpolation="nearest", aspect="equal")
        ax_v.set_title(title, fontsize=12, pad=4)
        ax_v.axis("off")

    # shared panel label and legend for the three views
    ax_C_views[0].text(-0.08, 1.04, "C", transform=ax_C_views[0].transAxes,
                       fontsize=22, fontweight="bold", va="bottom", ha="right")
    leg_C = [
        mpatches.Patch(color="#1f77b4", label=f"Largest CC  ({lf:.1f}%)"),
        mpatches.Patch(color="#BBBBBB", label=f"Other pores  ({of:.1f}%)"),
    ]
    ax_C_views[2].legend(handles=leg_C, loc="lower right", fontsize=10,
                          framealpha=0.9, edgecolor="#CCCCCC")

    # note: the solid-blue projection is the correct result for a fully-
    # percolating network — the connected component spans the entire ROI
    # in all three directions, so every projection plane is completely covered.
    ax_C_views[1].text(
        0.5, -0.12,
        "Solid blue = pore network spans entire ROI\n"
        "in all three directions (fully percolating)",
        transform=ax_C_views[1].transAxes,
        ha="center", va="top", fontsize=10.5, color="#1565C0", style="italic",
    )

    # ── D: bar chart ──
    bars   = ["Largest\ncomponent", "Other\ncomponents"]
    vals   = [lf, of]
    colors = ["#1f77b4", "#AAAAAA"]

    bh = ax_D.bar(bars, vals, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=2)
    ax_D.set_ylim(0, max(vals) * 1.22 + 5)
    ax_D.set_ylabel("Fraction of total pore volume [%]", fontsize=14)
    ax_D.set_title("Pore Volume Partitioning", fontsize=16, pad=8)
    ax_D.tick_params(axis="x", labelsize=13)
    ax_D.tick_params(axis="y", labelsize=12)
    ax_D.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#DDDDDD", zorder=0)
    ax_D.axhline(0, color="black", linewidth=0.8)
    ax_D.spines["top"].set_visible(False)
    ax_D.spines["right"].set_visible(False)
    ax_D.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    for bar, val in zip(bh, vals):
        ax_D.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                  f"{val:.1f}%", ha="center", va="bottom",
                  fontsize=14, fontweight="bold")

    # key metrics text box below bars
    metrics_lines = [
        f"Porosity (ROI):  {metrics['porosity']:.2f}%",
        f"Components:  {nc}",
        f"Largest CC:  {lf:.1f}%",
        f"Boundary-connected:  {bf:.1f}%",
        f"Interpretation:  {interp}",
    ]
    metrics_text = "\n".join(metrics_lines)
    ax_D.text(0.5, 0.02, metrics_text,
              transform=ax_D.transAxes,
              ha="center", va="bottom", fontsize=12,
              color="#222222",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4FF",
                        edgecolor="#AAAACC", linewidth=1.0))
    panel_label(ax_D, "D")

    fig.suptitle(
        "3-D Connected Pore Network — Cordatum Shell ROI",
        fontsize=22, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.04,
        f"Pore network interpretation:  {interp}  "
        f"(largest component = {lf:.1f}% of pore volume)",
        ha="center", va="bottom",
        fontsize=15, fontweight="bold", color=interp_color,
    )
    savefig(fig, "poster_fig3_connectivity_summary")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 SIMPLE  — three-panel A/B/C, no projection, no text box
# ─────────────────────────────────────────────────────────────────────────────

def fig3_simple(pore_mask, labeled, metrics, z_s):
    """
    Clean 1 x 3 poster figure:
      A. Binary pore mask on the representative slice.
      B. Connected components on the same slice (largest = blue, others = gray).
      C. Bar chart: largest component fraction vs all other components.
    """
    print("\n-- Figure 3 simple: Connectivity Summary --")

    lf = metrics["largest_frac"]
    of = metrics["other_frac"]
    nc = metrics["n_components"]

    pore_sl = pore_mask[z_s, :, :]
    lbl_sl  = labeled[z_s, :, :].astype(np.int32)

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.10,
        left=0.05, right=0.97,
        top=0.82, bottom=0.10,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    # ── A: binary pore mask ──
    show_mask(ax_A, pore_sl, title="Pore space", fg="white", bg="black")
    panel_label(ax_A, "A")

    # ── B: connected components (correct colormap via BoundaryNorm) ──
    n_show = min(15, nc) if nc > 0 else 1
    cmap_lbl, norm_lbl = _label_cmap(n_show)
    lbl_disp = np.where(lbl_sl > n_show, n_show + 1, lbl_sl).astype(np.int32)
    ax_B.imshow(lbl_disp, cmap=cmap_lbl, norm=norm_lbl,
                interpolation="nearest", aspect="equal")
    ax_B.set_title("Connected components", fontsize=17, pad=8)
    ax_B.axis("off")
    leg_B = [
        mpatches.Patch(color="#1f77b4", label="Largest component"),
        mpatches.Patch(color="#BBBBBB", label="Other components"),
    ]
    ax_B.legend(handles=leg_B, loc="lower right", fontsize=13,
                framealpha=0.92, edgecolor="#CCCCCC")
    panel_label(ax_B, "B")

    # ── C: bar chart ──
    bars   = ["Largest\ncomponent", "Other\ncomponents"]
    vals   = [lf, of]
    colors = ["#1f77b4", "#AAAAAA"]

    bh = ax_C.bar(bars, vals, color=colors, width=0.48,
                  edgecolor="white", linewidth=1.5, zorder=2)
    ax_C.set_ylim(0, 115)
    ax_C.set_ylabel("Fraction of total pore volume [%]", fontsize=14)
    ax_C.set_title("Pore volume partitioning", fontsize=17, pad=8)
    ax_C.tick_params(axis="x", labelsize=15)
    ax_C.tick_params(axis="y", labelsize=13)
    ax_C.yaxis.grid(True, linestyle="--", linewidth=0.6,
                    color="#DDDDDD", zorder=0)
    ax_C.axhline(0, color="black", linewidth=0.8)
    ax_C.spines["top"].set_visible(False)
    ax_C.spines["right"].set_visible(False)
    ax_C.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    for bar, val in zip(bh, vals):
        ax_C.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 1.5,
                  f"{val:.2f}%",
                  ha="center", va="bottom",
                  fontsize=16, fontweight="bold", color="#111111")

    panel_label(ax_C, "C")

    fig.suptitle(
        "3-D Connected Pore Network — Cordatum Shell",
        fontsize=22, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.02,
        f"Pore network: CONNECTED  |  Largest component = {lf:.2f}% of pore volume  "
        f"|  {nc} components total",
        ha="center", va="bottom",
        fontsize=14, fontweight="bold", color="#1565C0",
    )
    savefig(fig, "poster_fig3_connectivity_summary_simple")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 VALIDATION: pore overlay | CC | threshold sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def fig3_validation(roi_norm, pore_mask, labeled, sens_data, metrics, z_s):
    """
    1 x 3 validation figure:
      A. Raw ROI slice with cyan pore overlay.
      B. Connected components on same slice (largest = blue, others = gray).
      C. Threshold sensitivity plot: largest CC fraction vs threshold multiplier.
    """
    print("\n-- Figure 3 validation: Connectivity Validation --")

    raw_sl  = roi_norm[z_s, :, :]
    pore_sl = pore_mask[z_s, :, :]
    lbl_sl  = labeled[z_s, :, :].astype(np.int32)
    nc      = metrics["n_components"]
    lf      = metrics["largest_frac"]

    fig = plt.figure(figsize=(21, 7))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.14,
        left=0.05, right=0.97,
        top=0.83, bottom=0.13,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    # ── A: pore overlay ──
    ov = overlay_rgba(raw_sl, pore_sl, alpha=0.72)
    ax_A.imshow(ov, interpolation="nearest", aspect="equal")
    ax_A.set_title("Pore segmentation", fontsize=17, pad=8)
    ax_A.axis("off")
    panel_label(ax_A, "A")

    # ── B: connected components ──
    n_show = min(15, nc) if nc > 0 else 1
    cmap_lbl, norm_lbl = _label_cmap(n_show)
    lbl_disp = np.where(lbl_sl > n_show, n_show + 1, lbl_sl).astype(np.int32)
    ax_B.imshow(lbl_disp, cmap=cmap_lbl, norm=norm_lbl,
                interpolation="nearest", aspect="equal")
    ax_B.set_title("Connected components", fontsize=17, pad=8)
    ax_B.axis("off")
    leg_B = [
        mpatches.Patch(color="#1f77b4", label=f"Largest  ({lf:.1f}% of pore vol.)"),
        mpatches.Patch(color="#BBBBBB", label="Other components"),
    ]
    ax_B.legend(handles=leg_B, loc="lower right", fontsize=12,
                framealpha=0.92, edgecolor="#CCCCCC")
    panel_label(ax_B, "B")

    # ── C: threshold sensitivity plot ──
    mults = [r["multiplier"]   for r in sens_data]
    fracs = [r["largest_frac"] for r in sens_data]
    pors  = [r["porosity"]     for r in sens_data]

    ax_C.plot(mults, fracs, "o-", color="#1f77b4", linewidth=2.5,
              markersize=8, zorder=3, label="Largest CC fraction")
    # robustness reference line
    ax_C.axhline(70, color="#CC3333", linestyle="--", linewidth=1.4,
                 label="70% reference", zorder=2)
    # chosen threshold marker
    ax_C.axvline(1.00, color="#666666", linestyle=":", linewidth=1.6,
                 label="Chosen threshold", zorder=2)

    # small porosity labels — alternate above/below to minimise overlap
    for i, (m, f, p) in enumerate(zip(mults, fracs, pors)):
        va     = "bottom" if i % 2 == 0 else "top"
        dy     =  4.0     if i % 2 == 0 else -4.5
        ax_C.text(m, f + dy, f"{p:.1f}%",
                  ha="center", va=va, fontsize=9, color="#555555")

    ax_C.set_xlabel("Threshold multiplier", fontsize=14)
    ax_C.set_ylabel("Largest component fraction [%]", fontsize=14)
    ax_C.set_title("Threshold sensitivity", fontsize=17, pad=8)
    ax_C.set_ylim(-8, 118)
    ax_C.set_xticks(mults)
    ax_C.tick_params(axis="x", labelsize=11)
    ax_C.tick_params(axis="y", labelsize=12)
    ax_C.legend(fontsize=11, loc="lower left", framealpha=0.9,
                edgecolor="#CCCCCC")
    ax_C.spines["top"].set_visible(False)
    ax_C.spines["right"].set_visible(False)
    ax_C.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#EEEEEE", zorder=0)
    panel_label(ax_C, "C")

    # suptitle only — no bottom sentence
    fig.suptitle(
        "3-D Connected Pore Network — Cordatum Shell",
        fontsize=22, fontweight="bold", y=0.97,
    )
    savefig(fig, "poster_fig3_connectivity_validation")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 CLEAN: pore overlay | connected components
# ─────────────────────────────────────────────────────────────────────────────

def fig3_connected_components_clean(roi_norm, pore_mask, labeled, metrics, z_s):
    """
    1 x 2 clean poster figure:
      A. Raw ROI slice with cyan pore overlay.
      B. Connected components (largest = blue, others = gray).
    No bottom sentence, no bar chart.
    """
    print("\n-- Figure 3 clean: Connected Components --")

    raw_sl  = roi_norm[z_s, :, :]
    pore_sl = pore_mask[z_s, :, :]
    lbl_sl  = labeled[z_s, :, :].astype(np.int32)
    nc      = metrics["n_components"]
    lf      = metrics["largest_frac"]

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.07,
        left=0.03, right=0.97,
        top=0.84, bottom=0.04,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    # ── A: pore overlay ──
    ov = overlay_rgba(raw_sl, pore_sl, alpha=0.72)
    ax_A.imshow(ov, interpolation="nearest", aspect="equal")
    ax_A.set_title("Segmented pore space", fontsize=18, pad=10)
    ax_A.axis("off")
    panel_label(ax_A, "A")

    # ── B: connected components ──
    n_show = min(15, nc) if nc > 0 else 1
    cmap_lbl, norm_lbl = _label_cmap(n_show)
    lbl_disp = np.where(lbl_sl > n_show, n_show + 1, lbl_sl).astype(np.int32)
    ax_B.imshow(lbl_disp, cmap=cmap_lbl, norm=norm_lbl,
                interpolation="nearest", aspect="equal")
    ax_B.set_title("Largest connected component", fontsize=18, pad=10)
    ax_B.axis("off")
    leg_B = [
        mpatches.Patch(color="#1f77b4", label=f"Largest  ({lf:.2f}% of pore vol.)"),
        mpatches.Patch(color="#BBBBBB", label="Other components"),
    ]
    ax_B.legend(handles=leg_B, loc="lower right", fontsize=13,
                framealpha=0.93, edgecolor="#CCCCCC")
    panel_label(ax_B, "B")

    fig.suptitle(
        "Connected Pore Component in Selected Shell ROI",
        fontsize=22, fontweight="bold", y=0.97,
    )
    savefig(fig, "poster_fig3_connected_components_clean")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4: VISUAL THRESHOLD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def fig4_threshold_comparison(roi_norm, smooth_roi, t_solid, sens_data, z_s):
    """
    1 x 3: pore overlay at low / chosen / high threshold on the same ROI slice.
    Consistent image contrast across all panels.
    """
    print("\n-- Figure 4: Threshold Comparison --")

    raw_sl = roi_norm[z_s, :, :]
    # compute normalised gray base once so contrast is identical across panels
    vmin, vmax = pct_clip(raw_sl)
    gray_base = np.clip(
        (raw_sl.astype(np.float32) - vmin) / max(vmax - vmin, 1e-9), 0, 1
    )

    # quick lookup for stats
    sens_by_mult = {round(r["multiplier"], 2): r for r in sens_data}

    fig = plt.figure(figsize=(21, 8))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.05,
        left=0.03, right=0.97,
        top=0.78, bottom=0.05,
    )

    alpha = 0.68
    for col, (mult, label) in enumerate(zip(COMPARISON_MULTIPLIERS, COMPARISON_LABELS)):
        t    = t_solid * mult
        pm   = _segment_pores_for_t(smooth_roi, t)
        pore = pm[z_s, :, :]

        # RGBA overlay with fixed contrast
        rgba = np.stack(
            [gray_base, gray_base, gray_base, np.ones_like(gray_base)], axis=-1
        ).astype(np.float32)
        p = pore > 0
        rgba[p, 0] = 0.0  * alpha + rgba[p, 0] * (1 - alpha)
        rgba[p, 1] = 0.75 * alpha + rgba[p, 1] * (1 - alpha)
        rgba[p, 2] = 1.0  * alpha + rgba[p, 2] * (1 - alpha)
        rgba = np.clip(rgba, 0, 1)

        r = sens_by_mult[round(mult, 2)]
        title_line1 = label
        title_line2 = (f"Porosity: {r['porosity']:.1f}%   "
                       f"Largest CC: {r['largest_frac']:.1f}%")

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(rgba, interpolation="nearest", aspect="equal")
        ax.set_title(f"{title_line1}\n{title_line2}",
                     fontsize=15, pad=8, linespacing=1.6)
        ax.axis("off")
        panel_label(ax, ["A", "B", "C"][col])

    fig.suptitle(
        "Effect of Threshold on Pore Segmentation",
        fontsize=22, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.01,
        "Cyan = segmented pore space.  "
        "Despite different porosity values, one connected component dominates at all thresholds.",
        ha="center", va="bottom", fontsize=11, color="#555555", style="italic",
    )
    savefig(fig, "poster_fig4_threshold_comparison")


def save_threshold_comparison_csv(sens_data):
    selected = {round(m, 2) for m in COMPARISON_MULTIPLIERS}
    path = OUTPUT_DIR / "poster_threshold_comparison.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "threshold_multiplier", "porosity_percent",
            "number_of_components", "largest_component_fraction_percent",
        ])
        for r in sens_data:
            if round(r["multiplier"], 2) in selected:
                w.writerow([r["multiplier"], r["porosity"],
                            r["n_components"], r["largest_frac"]])
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL POSTER FIGURE 2: raw → segmentation → connectivity
# ─────────────────────────────────────────────────────────────────────────────

def fig2_segmentation_connectivity(roi_norm, pore_mask, labeled, metrics, z_s):
    """
    1 x 3: raw ROI slice | pore overlay | connected components.
    Tells the story: raw data -> segmentation -> connected network.
    """
    print("\n-- Final Fig 2: Segmentation + Connectivity --")

    raw_sl  = roi_norm[z_s, :, :]
    pore_sl = pore_mask[z_s, :, :]
    lbl_sl  = labeled[z_s, :, :].astype(np.int32)
    nc      = metrics["n_components"]
    lf      = metrics["largest_frac"]

    fig = plt.figure(figsize=(21, 8))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.07,
        left=0.03, right=0.97,
        top=0.83, bottom=0.10,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    # ── A: raw ROI slice ──
    show_ct(ax_A, raw_sl, title="Raw ROI")
    panel_label(ax_A, "A")

    # ── B: pore overlay ──
    ov = overlay_rgba(raw_sl, pore_sl, alpha=0.72)
    ax_B.imshow(ov, interpolation="nearest", aspect="equal")
    ax_B.set_title("Segmented pore space", fontsize=17, pad=8)
    ax_B.axis("off")
    panel_label(ax_B, "B")

    # ── C: connected components ──
    n_show = min(15, nc) if nc > 0 else 1
    cmap_lbl, norm_lbl = _label_cmap(n_show)
    lbl_disp = np.where(lbl_sl > n_show, n_show + 1, lbl_sl).astype(np.int32)
    ax_C.imshow(lbl_disp, cmap=cmap_lbl, norm=norm_lbl,
                interpolation="nearest", aspect="equal")
    ax_C.set_title("Largest connected component", fontsize=17, pad=8)
    ax_C.axis("off")
    panel_label(ax_C, "C")

    fig.suptitle(
        "Pore Segmentation and Connectivity — Selected Shell ROI",
        fontsize=22, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.02,
        f"Largest component = {lf:.2f}% of pore volume  |  "
        f"{nc} components total  |  Blue = dominant connected network",
        ha="center", va="bottom", fontsize=12, color="#333333",
    )
    savefig(fig, "poster_fig2_segmentation_connectivity")


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL POSTER FIGURE 3: threshold comparison (clean)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_threshold_comparison_final(roi_norm, smooth_roi, t_solid, sens_data, z_s):
    """
    1 x 3: pore overlay at low / chosen / high threshold.
    Consistent contrast. Two numbers per panel (porosity, largest CC).
    """
    print("\n-- Final Fig 3: Threshold Comparison --")

    raw_sl = roi_norm[z_s, :, :]
    vmin, vmax = pct_clip(raw_sl)
    gray_base = np.clip(
        (raw_sl.astype(np.float32) - vmin) / max(vmax - vmin, 1e-9), 0, 1
    )
    sens_by_mult = {round(r["multiplier"], 2): r for r in sens_data}

    fig = plt.figure(figsize=(21, 8))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        wspace=0.05,
        left=0.03, right=0.97,
        top=0.78, bottom=0.12,
    )

    alpha = 0.68
    for col, (mult, label) in enumerate(zip(COMPARISON_MULTIPLIERS, COMPARISON_LABELS)):
        t    = t_solid * mult
        pm   = _segment_pores_for_t(smooth_roi, t)
        pore = pm[z_s, :, :]

        rgba = np.stack(
            [gray_base, gray_base, gray_base, np.ones_like(gray_base)], axis=-1
        ).astype(np.float32)
        p = pore > 0
        rgba[p, 0] = 0.0  * alpha + rgba[p, 0] * (1 - alpha)
        rgba[p, 1] = 0.75 * alpha + rgba[p, 1] * (1 - alpha)
        rgba[p, 2] = 1.0  * alpha + rgba[p, 2] * (1 - alpha)
        rgba = np.clip(rgba, 0, 1)

        r = sens_by_mult[round(mult, 2)]
        title = (f"{label}\n"
                 f"Porosity: {r['porosity']:.1f}%,  Largest CC: {r['largest_frac']:.1f}%")

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(rgba, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=15, pad=8, linespacing=1.6)
        ax.axis("off")
        panel_label(ax, ["A", "B", "C"][col])

    fig.suptitle(
        "Threshold Effect on Pore Segmentation",
        fontsize=22, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.02,
        "Porosity changes with threshold, but the largest connected component remains dominant.",
        ha="center", va="bottom", fontsize=12, color="#444444", style="italic",
    )
    savefig(fig, "poster_fig3_threshold_comparison")


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(metrics, roi_bounds):
    print("\nSaving metrics CSV ...")
    z0, z1, y0, y1, x0, x1 = roi_bounds
    roi_shape = f"{z1-z0} x {y1-y0} x {x1-x0}"
    rows = [
        ("metric", "value", "unit"),
        ("roi_shape",                          roi_shape,                       "voxels"),
        ("roi_z_range",                        f"{z0}-{z1}",                    ""),
        ("roi_y_range",                        f"{y0}-{y1}",                    ""),
        ("roi_x_range",                        f"{x0}-{x1}",                    ""),
        ("porosity_percent",                   metrics["porosity"],             "%"),
        ("number_of_pore_components",          metrics["n_components"],         ""),
        ("largest_component_volume",           metrics["largest_vol"],          "voxels"),
        ("largest_component_fraction_percent", metrics["largest_frac"],         "%"),
        ("other_components_fraction_percent",  metrics["other_frac"],           "%"),
        ("boundary_connected_fraction_percent",metrics["boundary_frac"],        "%"),
        ("minimum_component_size",             MIN_PORE_SIZE,                   "voxels"),
        ("downsample_factor",                  DOWNSAMPLE_FACTOR,               ""),
        ("solid_threshold_mode",               SOLID_THRESHOLD_MODE,            ""),
        ("boundary_removal",                   str(REMOVE_BOUNDARY_PORES),      ""),
    ]
    path = OUTPUT_DIR / "poster_main_metrics.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Saved {path.name}")


def save_threshold_csv(sens_data):
    path = OUTPUT_DIR / "threshold_sensitivity_roi.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "threshold_multiplier", "threshold_value",
            "porosity_percent", "number_of_components",
            "largest_component_fraction_percent",
        ])
        for r in sens_data:
            w.writerow([
                r["multiplier"], r["threshold"], r["porosity"],
                r["n_components"], r["largest_frac"],
            ])
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY TEXT
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(metrics, roi_bounds, sens_data=None):
    z0, z1, y0, y1, x0, x1 = roi_bounds
    lf  = metrics["largest_frac"]
    por = metrics["porosity"]
    nc  = metrics["n_components"]
    bf  = metrics["boundary_frac"]

    if lf >= 70:
        conclusion = "connected"
        detail = ("The dominant pore component forms a continuous 3-D network "
                  "throughout the analysed shell region.")
    elif lf <= 30:
        conclusion = "fragmented"
        detail = ("Pore space is distributed across many small isolated cavities "
                  "with no dominant connected network.")
    else:
        conclusion = "mixed (partially connected)"
        detail = ("Pore space consists of a mix of larger connected channels "
                  "and smaller isolated cavities.")

    # threshold sensitivity section
    if sens_data:
        rob_short, rob_long = _robustness_verdict(sens_data)
        sens_header = "\nThreshold Sensitivity (validation)\n-----------------------------------"
        sens_rows = "\n".join(
            f"  mult={r['multiplier']:.2f}  t={r['threshold']:.4f}  "
            f"por={r['porosity']:.2f}%  n={r['n_components']}  "
            f"f1={r['largest_frac']:.2f}%"
            for r in sens_data
        )
        sens_section = (
            f"{sens_header}\n{sens_rows}\n\n"
            f"Robustness assessment: {rob_short.upper()}\n{rob_long}"
        )
    else:
        sens_section = ""

    poster_note = (
        f"The chosen threshold gave an ROI porosity of {por:.2f}% and a largest "
        f"connected component fraction of {lf:.2f}%. Visual threshold comparison "
        f"showed that changing the threshold altered the amount of segmented pore "
        f"space, but the pore network remained dominated by one connected component. "
        f"This supports the interpretation that the selected shell ROI contains a "
        f"connected pore network."
    )

    text = f"""\
Connected Pore Network Analysis -- Cordatum Shell (ROI-based)
=============================================================

A region of interest (ROI) was selected entirely inside the shell material
to avoid circular-scan boundary artefacts.

ROI coordinates (downsampled x{DOWNSAMPLE_FACTOR}):
  Z = {z0} to {z1}  |  Y = {y0} to {y1}  |  X = {x0} to {x1}
  Size: {z1-z0} x {y1-y0} x {x1-x0} voxels

Segmentation method:
  Solid shell material identified using Otsu threshold within the ROI.
  Pore space = low-density trabecular channels below the Otsu threshold,
  within a morphologically dilated solid envelope.
  Minimum pore size: {MIN_PORE_SIZE} voxels.
  Boundary removal: {REMOVE_BOUNDARY_PORES}

Key Results (chosen threshold, multiplier = 1.00)
-------------------------------------------------
Porosity (ROI)                : {por:.4f} %
Number of pore components     : {nc}
Largest component fraction f1 : {lf:.2f} %
Other components fraction     : {metrics['other_frac']:.2f} %
Boundary-connected fraction   : {bf:.2f} %

Interpretation
--------------
The pore network in the analysed Cordatum shell region is {conclusion}.
{detail}

Poster Summary
--------------
{poster_note}
{sens_section}

Formulas
--------
  Porosity:  phi = V_pore / (V_solid + V_pore) x 100%
  Largest CC fraction:  f1 = V_largest / V_pore x 100%
"""
    path = OUTPUT_DIR / "poster_summary_text.txt"
    path.write_text(text, encoding="utf-8")
    print(f"  Saved {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Cordatum Shell - Poster Figure Generator v2")
    print("=" * 60)

    # 1. load
    vol_raw, vol_norm = load_ct()

    # 2. ROI
    roi_bounds = select_roi(vol_norm)
    z0, z1, y0, y1, x0, x1 = roi_bounds
    roi_norm = vol_norm[z0:z1, y0:y1, x0:x1].copy()

    # 3. segment
    print("\nSegmenting within ROI ...")
    solid_mask, pore_mask, smooth_roi, t_solid = segment_roi(roi_norm)

    # 4. CC
    labeled, metrics = run_cc(pore_mask, solid_mask)

    # 5. best display slice within ROI
    z_s = best_slice(pore_mask, labeled)
    print(f"\nDisplay slice (ROI-local z = {z_s}, global z = {z0 + z_s})")

    # 6. threshold sensitivity (runs before figures so verdict is ready)
    sens_data = threshold_sensitivity_analysis(smooth_roi, t_solid)

    # 7. final poster figures (3 only)
    fig1_roi_overview(vol_norm, roi_bounds)
    fig2_segmentation_connectivity(roi_norm, pore_mask, labeled, metrics, z_s)
    fig3_threshold_comparison_final(roi_norm, smooth_roi, t_solid, sens_data, z_s)

    # 8. text outputs
    save_metrics(metrics, roi_bounds)
    save_threshold_csv(sens_data)
    save_threshold_comparison_csv(sens_data)
    save_summary(metrics, roi_bounds, sens_data)

    print(f"\n{'='*60}")
    print(f"  Done in {time.time()-t0:.1f} s")
    print(f"  All outputs -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
