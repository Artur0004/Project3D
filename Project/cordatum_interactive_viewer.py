"""
cordatum_interactive_viewer.py
==============================
Interactive napari viewer for Cordatum Shell micro-CT pore network inspection.

Run:
    python cordatum_interactive_viewer.py

Requirements:
    pip install "napari[all]" magicgui scikit-image tifffile imageio

Napari navigation tips:
  - Scroll mouse wheel or drag the bottom dimension sliders to move through slices.
  - Click the 2D/3D toggle button (top-left of canvas) to switch views.
  - Toggle layer visibility with the eye icon in the layer list.
  - Press Ctrl+Shift+T to reset camera.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

DATA_PATH = Path(r"C:\Users\artxm\PycharmProjects\3DImageAnalysis\Cordatum_Shell.tif")
if not DATA_PATH.exists():
    DATA_PATH = Path(r"C:\Users\artxm\PycharmProjects\3d2\Cordatum_Shell.tif")

OUTPUT_DIR = Path(
    r"C:\Users\artxm\PycharmProjects\3DImageAnalysis\cordatum_results\interactive_viewer"
)

USE_DOWNSAMPLED   = True
DOWNSAMPLE_FACTOR = 2

USE_ROI = True
ROI_Z   = (172, 322)
ROI_Y   = (178, 328)
ROI_X   = (173, 323)

# Optional: path to a manually labelled binary TIFF for Dice comparison
MANUAL_LABEL_PATH = None    # e.g. Path(r"C:\...\manual_labels.tif")

# Animation rendering (used by Export buttons)
ANIM_FPS    = 12
ANIM_DPI    = 100
ANIM_FIGSIZE = (8, 8)

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys, time, csv, datetime, warnings, traceback
import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology, measure
import tifffile

warnings.filterwarnings("ignore")

# napari / magicgui — deferred until main() so import errors are clear
napari   = None
widgets  = None
Colormap = None

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT FOLDERS
# ─────────────────────────────────────────────────────────────────────────────
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
ANIMATIONS_DIR  = OUTPUT_DIR / "animations"

def _ensure_dirs():
    for d in (OUTPUT_DIR, SCREENSHOTS_DIR, ANIMATIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP GUIDE
# ─────────────────────────────────────────────────────────────────────────────

def print_startup_guide(otsu_t):
    print()
    print("=" * 64)
    print("  Interactive Cordatum Shell Viewer")
    print("=" * 64)
    print()
    print("  Dataset :", DATA_PATH.name)
    print("  ROI     : Z=%d-%d  Y=%d-%d  X=%d-%d" % (*ROI_Z, *ROI_Y, *ROI_X))
    print(f"  Otsu threshold (ROI-internal) : {otsu_t:.4f}")
    print(f"  Default threshold (mult=1.00) : {otsu_t:.4f}")
    print()
    print("  Controls:")
    print("  1. Adjust 'Threshold multiplier' in the right panel.")
    print("  2. Press 'Recompute segmentation' to update all layers.")
    print("  3. Use the napari dimension sliders (bottom) for z/y/x navigation.")
    print("  4. Click the 2D/3D toggle (top-left of canvas) for 3D view.")
    print("  5. Toggle layer visibility with the eye icon in the layer list.")
    print("  6. Export screenshots, metrics CSV, or flythrough animations.")
    print()
    print("  For 3D inspection: hide 'Raw CT ROI', show 'Largest component'.")
    print("=" * 64)
    print()

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load CT, downsample, crop to ROI, normalise.  Returns (roi_norm, otsu_t)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CT file not found: {DATA_PATH}")

    print("Loading CT volume ...")
    t0 = time.time()
    try:
        mm = tifffile.memmap(DATA_PATH)
    except Exception:
        mm = tifffile.imread(str(DATA_PATH))

    f = DOWNSAMPLE_FACTOR if USE_DOWNSAMPLED else 1
    vol = np.array(mm[::f, ::f, ::f], dtype=np.float32)
    print(f"  Downsampled shape: {vol.shape}  ({time.time()-t0:.1f}s)")

    # global percentile normalise
    p1, p998 = float(np.percentile(vol, 1.0)), float(np.percentile(vol, 99.8))
    vol = np.clip((vol - p1) / max(p998 - p1, 1e-9), 0.0, 1.0)

    # ROI crop
    if USE_ROI:
        z0, z1 = ROI_Z;  y0, y1 = ROI_Y;  x0, x1 = ROI_X
        roi = vol[z0:z1, y0:y1, x0:x1].copy()
    else:
        roi = vol

    print(f"  ROI shape        : {roi.shape}")

    # ROI-internal percentile normalise (improves contrast)
    r1, r99 = float(np.percentile(roi, 1.0)), float(np.percentile(roi, 99.5))
    roi_norm = np.clip((roi - r1) / max(r99 - r1, 1e-9), 0.0, 1.0).astype(np.float32)

    # Otsu threshold inside the ROI (on Gaussian-smoothed copy)
    smooth_otsu = ndi.gaussian_filter(roi_norm, sigma=0.8)
    otsu_t = float(filters.threshold_otsu(smooth_otsu))
    print(f"  Otsu threshold   : {otsu_t:.4f}")

    # Optional manual label
    manual_label = None
    if MANUAL_LABEL_PATH and Path(MANUAL_LABEL_PATH).exists():
        ml = tifffile.imread(str(MANUAL_LABEL_PATH)).astype(bool)
        if ml.shape == roi_norm.shape:
            manual_label = ml
            print(f"  Manual label loaded: {ml.sum():,} positive voxels")
        else:
            print(f"  WARNING: manual label shape {ml.shape} != ROI {roi_norm.shape}, ignoring.")

    return roi_norm, otsu_t, manual_label

# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _remove_boundary_comps(mask):
    """Remove connected components that touch any face of the 3-D volume."""
    struct = ndi.generate_binary_structure(3, 3)
    lbl, _ = ndi.label(mask, structure=struct)
    Z, Y, X = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    border[:, :, 0] = border[:, :, -1] = True
    bad = set(np.unique(lbl[border & (lbl > 0)])) - {0}
    clean = mask.copy()
    for b in bad:
        clean[lbl == b] = False
    return clean


def compute_segmentation(roi_norm, threshold, gaussian_sigma,
                         min_size, remove_boundary):
    """
    Segment pore space within the ROI and run CC analysis.
    Returns (pore_mask uint8, labeled int32, metrics dict).
    """
    smooth = (ndi.gaussian_filter(roi_norm.astype(np.float32), sigma=gaussian_sigma)
              if gaussian_sigma > 0 else roi_norm.astype(np.float32))

    solid_raw    = smooth >= threshold
    solid_closed = morphology.binary_closing(solid_raw, morphology.ball(1))
    envelope     = ndi.binary_dilation(solid_closed, iterations=3)
    pore_cand    = (smooth < threshold) & envelope & (~solid_closed)

    if remove_boundary:
        pore_cand = _remove_boundary_comps(pore_cand)

    pore_mask = morphology.remove_small_objects(pore_cand, min_size=max(1, min_size))

    labeled, metrics = _run_cc(pore_mask, solid_closed)
    return pore_mask.astype(np.uint8), labeled.astype(np.int32), metrics


def _run_cc(pore_mask, solid_mask):
    """26-connectivity CC analysis.  Label 1 = largest component."""
    struct26 = ndi.generate_binary_structure(3, 3)
    lbl_raw, n_comp = ndi.label(pore_mask, structure=struct26)

    total_pore  = int(pore_mask.sum())
    total_solid = int(solid_mask.sum())
    porosity    = total_pore / max(total_solid + total_pore, 1) * 100.0

    if n_comp == 0:
        return np.zeros_like(pore_mask, dtype=np.int32), dict(
            n_components=0, porosity=round(porosity, 3),
            largest_frac=0.0, other_frac=0.0,
            boundary_frac=0.0, total_pore=total_pore,
            interpretation="FRAGMENTED", sizes=[],
        )

    props    = measure.regionprops(lbl_raw)
    size_map = {p.label: p.area for p in props}
    ordered  = sorted(size_map, key=size_map.__getitem__, reverse=True)

    # relabel: 1 = largest
    labeled = np.zeros_like(lbl_raw)
    for new, old in enumerate(ordered, start=1):
        labeled[lbl_raw == old] = new

    sizes        = [size_map[l] for l in ordered]
    largest_frac = sizes[0] / max(total_pore, 1) * 100.0
    other_frac   = 100.0 - largest_frac

    # boundary-connected fraction
    Z, Y, X = labeled.shape
    border = np.zeros_like(labeled, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    border[:, :, 0] = border[:, :, -1] = True
    b_set        = set(np.unique(labeled[border & (labeled > 0)])) - {0}
    boundary_vol = sum(sizes[l - 1] for l in b_set if 1 <= l <= len(sizes))
    boundary_frac = boundary_vol / max(total_pore, 1) * 100.0

    interp = ("CONNECTED"  if largest_frac >= 70 else
              "FRAGMENTED" if largest_frac <= 30 else "MIXED")

    return labeled, dict(
        n_components=n_comp,
        porosity=round(porosity, 3),
        largest_frac=round(largest_frac, 2),
        other_frac=round(other_frac, 2),
        boundary_frac=round(boundary_frac, 2),
        total_pore=total_pore,
        interpretation=interp,
        sizes=sizes,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  METRICS FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics(metrics, threshold, thresh_mult, dice=None):
    m = metrics
    mult_str = f"{thresh_mult:.3f}" if isinstance(thresh_mult, float) else thresh_mult
    lines = [
        f"Threshold        : {threshold:.4f}  (x{mult_str})",
        f"Porosity         : {m['porosity']:.3f} %",
        f"Components       : {m['n_components']}",
        f"Largest CC       : {m['largest_frac']:.2f} % of pore vol.",
        f"Other CCs        : {m['other_frac']:.2f} %",
        f"Boundary-connect.: {m['boundary_frac']:.2f} %",
        f"Interpretation   : {m['interpretation']}",
    ]
    if dice is not None:
        lines.append(f"Dice vs manual   : {dice:.3f}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER SETUP AND UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def _make_colormap(color_rgba, name):
    """Return a napari Colormap: transparent at 0, solid color at 1."""
    return Colormap(colors=[[0, 0, 0, 0], color_rgba], name=name)

LAYER_NAMES = {
    "raw":     "Raw CT ROI",
    "pore":    "Pore mask",
    "largest": "Largest component",
    "others":  "Other components",
    "labels":  "CC labels",
}

# colours for the three overlay layers [R, G, B, A]
_CYAN   = [0.00, 1.00, 1.00, 1.0]
_BLUE   = [0.12, 0.47, 0.71, 1.0]
_LGRAY  = [0.70, 0.70, 0.70, 1.0]


def add_initial_layers(viewer, roi_norm, pore_mask, labeled):
    """Called once at startup to create all napari layers."""
    nz, ny, nx = roi_norm.shape
    empty_img = np.zeros((nz, ny, nx), dtype=np.float32)
    empty_lbl = np.zeros((nz, ny, nx), dtype=np.int32)

    # 1. Raw CT
    viewer.add_image(
        roi_norm,
        name=LAYER_NAMES["raw"],
        colormap="gray",
        contrast_limits=[0, 1],
        blending="translucent_no_depth",
    )

    # 2. Pore mask (cyan)
    viewer.add_image(
        empty_img,
        name=LAYER_NAMES["pore"],
        colormap=_make_colormap(_CYAN,  "cyan_mask"),
        blending="additive",
        opacity=0.55,
    )

    # 3. Largest CC (blue)
    viewer.add_image(
        empty_img,
        name=LAYER_NAMES["largest"],
        colormap=_make_colormap(_BLUE,  "blue_mask"),
        blending="additive",
        opacity=0.55,
    )

    # 4. Other CCs (light gray)
    viewer.add_image(
        empty_img,
        name=LAYER_NAMES["others"],
        colormap=_make_colormap(_LGRAY, "gray_mask"),
        blending="additive",
        opacity=0.35,
    )

    # 5. CC labels (auto-coloured per label)
    lbl_layer = viewer.add_labels(
        empty_lbl,
        name=LAYER_NAMES["labels"],
    )
    lbl_layer.visible = False   # hidden by default; user can toggle


def update_layers(viewer, pore_mask, labeled, opacity,
                  show_largest, show_others):
    """Update overlay layer data and visibility. Never creates new layers."""
    pore_f    = pore_mask.astype(np.float32)
    largest_f = (labeled == 1).astype(np.float32)
    others_f  = (labeled > 1).astype(np.float32)

    ll = viewer.layers  # shorthand

    ll[LAYER_NAMES["pore"]].data    = pore_f
    ll[LAYER_NAMES["pore"]].opacity = opacity

    ll[LAYER_NAMES["largest"]].data    = largest_f
    ll[LAYER_NAMES["largest"]].opacity = opacity
    ll[LAYER_NAMES["largest"]].visible = show_largest

    ll[LAYER_NAMES["others"]].data    = others_f
    ll[LAYER_NAMES["others"]].opacity = max(0.1, opacity - 0.15)
    ll[LAYER_NAMES["others"]].visible = show_others

    ll[LAYER_NAMES["labels"]].data = labeled.astype(np.int32)

# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT: SCREENSHOT
# ─────────────────────────────────────────────────────────────────────────────

def save_screenshot(viewer, threshold):
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"screenshot_t{threshold:.4f}_{ts}.png"
    path = SCREENSHOTS_DIR / name
    try:
        img = viewer.screenshot(canvas_only=True, flash=False)
        import imageio
        imageio.imwrite(str(path), img)
        print(f"  Screenshot saved: {path.name}")
    except Exception as e:
        print(f"  Screenshot failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT: METRICS CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(metrics, threshold, thresh_mult, gaussian_sigma,
                     min_size, dice=None):
    path = OUTPUT_DIR / "interactive_metrics_log.csv"
    header = [
        "timestamp", "threshold", "threshold_multiplier",
        "gaussian_sigma", "min_component_size",
        "porosity_percent", "number_of_components",
        "largest_component_fraction_percent",
        "boundary_connected_fraction_percent",
        "dice_vs_manual", "interpretation",
    ]
    row = [
        datetime.datetime.now().isoformat(timespec="seconds"),
        round(threshold, 4),
        round(thresh_mult, 3) if isinstance(thresh_mult, float) else thresh_mult,
        gaussian_sigma, min_size,
        metrics["porosity"], metrics["n_components"],
        metrics["largest_frac"], metrics["boundary_frac"],
        round(dice, 4) if dice is not None else "",
        metrics["interpretation"],
    ]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"  Metrics appended: {path.name}")

# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT: FLYTHROUGH ANIMATION
# ─────────────────────────────────────────────────────────────────────────────

def _anim_make_overlay(gray2d, pore2d, alpha=0.55):
    rgba = np.stack([gray2d, gray2d, gray2d, np.ones_like(gray2d)], axis=-1).astype(np.float32)
    p = pore2d > 0
    rgba[p, 0] = 0.0  * alpha + rgba[p, 0] * (1 - alpha)
    rgba[p, 1] = 0.75 * alpha + rgba[p, 1] * (1 - alpha)
    rgba[p, 2] = 1.0  * alpha + rgba[p, 2] * (1 - alpha)
    return np.clip(rgba, 0, 1)


def _anim_make_cc_overlay(gray2d, cc2d, alpha=0.65):
    """Largest CC in blue on grayscale."""
    rgba = np.stack([gray2d, gray2d, gray2d, np.ones_like(gray2d)], axis=-1).astype(np.float32)
    p = cc2d > 0
    rgba[p, 0] = 0.12 * alpha + rgba[p, 0] * (1 - alpha)
    rgba[p, 1] = 0.47 * alpha + rgba[p, 1] * (1 - alpha)
    rgba[p, 2] = 0.71 * alpha + rgba[p, 2] * (1 - alpha)
    return np.clip(rgba, 0, 1)


def _canvas_to_rgb(fig):
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return buf.reshape(h, w, 4)[:, :, :3].copy()
    except AttributeError:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3).copy()


def export_animation(roi_norm, pore_mask, labeled, mode="overlay",
                     axis=0, threshold=0.664, fps=ANIM_FPS):
    """
    Render a flythrough animation as MP4 (+ optional GIF).
    mode: "raw" | "overlay" | "largest_cc"
    axis: 0=z, 1=y, 2=x
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pore_mask is None:
        print("  Run Recompute first before exporting animation.")
        return

    ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
    axis_name  = ["z", "y", "x"][axis]
    mode_label = {"raw": "raw", "overlay": "pore_overlay",
                  "largest_cc": "largest_component"}[mode]
    stem = f"{mode_label}_flythrough_axis_{axis_name}_threshold_{threshold:.4f}"

    n_slices = roi_norm.shape[axis]

    # fixed contrast for the raw CT
    p1, p99 = float(np.percentile(roi_norm, 1.0)), float(np.percentile(roi_norm, 99.5))
    roi_disp = np.clip((roi_norm - p1) / max(p99 - p1, 1e-9), 0, 1).astype(np.float32)

    largest_mask = (labeled == 1).astype(np.uint8) if labeled is not None else None

    def _sl(vol, i):
        if axis == 0: return vol[i]
        if axis == 1: return vol[:, i, :]
        return vol[:, :, i]

    title_map = {
        "raw":        "Raw CT ROI",
        "overlay":    "Segmented pore space",
        "largest_cc": "Largest connected component",
    }

    print(f"\n  Rendering {n_slices} frames ({mode}, axis={axis_name}) ...")
    fig, ax = plt.subplots(1, 1, figsize=ANIM_FIGSIZE, dpi=ANIM_DPI)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.02, right=0.98)
    ax.axis("off")

    gray0 = _sl(roi_disp, 0)
    if mode == "raw":
        im = ax.imshow(gray0, cmap="gray", vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal")
    else:
        im = ax.imshow(np.zeros((*gray0.shape, 4), dtype=np.float32),
                       interpolation="nearest", aspect="equal")

    ttl = fig.text(0.5, 0.96, title_map[mode], ha="center", va="top",
                   color="white", fontsize=18, fontweight="bold")
    cnt = fig.text(0.5, 0.03, "", ha="center", va="bottom",
                   color="#BBBBBB", fontsize=13)
    fig.canvas.draw()

    frames = []
    t_start = time.time()
    for i in range(n_slices):
        if (i + 1) % 30 == 0 or i == n_slices - 1:
            print(f"    Frame {i+1:3d}/{n_slices}  ({time.time()-t_start:.1f}s)")
        cnt.set_text(f"Slice  {i+1:3d} / {n_slices}")

        gray = _sl(roi_disp, i)
        if mode == "raw":
            im.set_data(gray)
        elif mode == "overlay":
            pore = _sl(pore_mask, i)
            im.set_data(_anim_make_overlay(gray, pore))
        else:
            cc = _sl(largest_mask, i)
            im.set_data(_anim_make_cc_overlay(gray, cc))

        frames.append(_canvas_to_rgb(fig))

    plt.close(fig)
    print(f"    Frame size: {frames[0].shape[1]}x{frames[0].shape[0]} px")

    # Ensure even dimensions for libx264
    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        h2, w2 = h - h % 2, w - w % 2
        frames = [f[:h2, :w2] for f in frames]

    # Write MP4
    try:
        import imageio
        mp4_path = ANIMATIONS_DIR / f"{stem}.mp4"
        imageio.mimwrite(str(mp4_path), frames, fps=fps, quality=8, macro_block_size=None)
        print(f"  Saved {mp4_path.name}  ({len(frames)} frames, {fps} fps)")
    except Exception as e:
        print(f"  MP4 write failed: {e}")

    # Write GIF (every 3rd frame, 35% size)
    try:
        from PIL import Image as PILImage
        subset  = frames[::3]
        gif_fps = max(1, fps // 3)
        pil_frames = []
        for f in subset:
            img = PILImage.fromarray(f)
            nw  = max(4, int(f.shape[1] * 0.35))
            nh  = max(4, int(f.shape[0] * 0.35))
            img = img.resize((nw, nh), PILImage.LANCZOS)
            pil_frames.append(img.convert("P", palette=PILImage.ADAPTIVE, colors=128))
        gif_path = ANIMATIONS_DIR / f"{stem}.gif"
        pil_frames[0].save(
            str(gif_path),
            save_all=True, append_images=pil_frames[1:],
            duration=int(1000 / gif_fps), loop=0, optimize=True,
        )
        print(f"  Saved {gif_path.name}")
    except Exception as e:
        print(f"  GIF write failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_MULTIPLIERS = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]


def run_threshold_sweep(roi_norm, otsu_t, gaussian_sigma, min_size,
                        metrics_text_widget):
    print("\nRunning threshold sweep ...")
    path = OUTPUT_DIR / "threshold_sweep.csv"
    struct26 = ndi.generate_binary_structure(3, 3)
    results  = []

    smooth = (ndi.gaussian_filter(roi_norm.astype(np.float32), sigma=gaussian_sigma)
              if gaussian_sigma > 0 else roi_norm.astype(np.float32))

    for mult in SWEEP_MULTIPLIERS:
        t = otsu_t * mult
        solid_raw    = smooth >= t
        solid_closed = morphology.binary_closing(solid_raw, morphology.ball(1))
        envelope     = ndi.binary_dilation(solid_closed, iterations=3)
        pore_cand    = (smooth < t) & envelope & (~solid_closed)
        pore_mask    = morphology.remove_small_objects(pore_cand, min_size=max(1, min_size))

        total_pore  = int(pore_mask.sum())
        total_solid = int(solid_closed.sum())
        porosity    = total_pore / max(total_solid + total_pore, 1) * 100.0

        if total_pore > 0:
            lbl_raw, nc = ndi.label(pore_mask, structure=struct26)
            props  = measure.regionprops(lbl_raw)
            sizes  = sorted([p.area for p in props], reverse=True)
            lf     = sizes[0] / max(total_pore, 1) * 100.0
        else:
            nc, lf = 0, 0.0

        results.append(dict(multiplier=mult, threshold=round(float(t), 4),
                            porosity=round(float(porosity), 3),
                            n_components=int(nc),
                            largest_frac=round(float(lf), 2)))
        print(f"  mult={mult:.2f}  t={t:.4f}  por={porosity:.1f}%  "
              f"n={nc}  f1={lf:.1f}%")

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold_multiplier", "threshold_value", "porosity_percent",
                    "number_of_components", "largest_component_fraction_percent"])
        for r in results:
            w.writerow([r["multiplier"], r["threshold"], r["porosity"],
                        r["n_components"], r["largest_frac"]])
    print(f"  Saved {path.name}")

    fracs   = [r["largest_frac"] for r in results]
    robust  = all(f >= 70 for f in fracs)
    min_f   = min(fracs)
    max_f   = max(fracs)
    verdict = "ROBUST — connected interpretation holds at all tested thresholds." \
              if robust else \
              "SENSITIVE — connectivity changes with threshold."

    summary_lines = [
        "── Threshold Sweep Results ──",
        f"Multipliers tested : {', '.join(str(m) for m in SWEEP_MULTIPLIERS)}",
        f"Largest CC range   : {min_f:.1f}% – {max_f:.1f}%",
        f"Verdict            : {verdict}",
        "",
        "mult   t       por%   n    f1%",
    ] + [f"{r['multiplier']:.2f}   {r['threshold']:.4f}  "
         f"{r['porosity']:6.1f}  {r['n_components']:2d}  {r['largest_frac']:6.2f}"
         for r in results]

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    try:
        metrics_text_widget.value = summary_text
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
#  DICE (if manual label provided)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dice(auto_mask, manual_mask):
    a = auto_mask.astype(bool)
    m = manual_mask.astype(bool)
    intersection = (a & m).sum()
    return 2 * intersection / max(a.sum() + m.sum(), 1)

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MAGICGUI WIDGET
# ─────────────────────────────────────────────────────────────────────────────

def build_widget(viewer, roi_norm, otsu_t, manual_label):
    """
    Construct the control panel as a magicgui Container and wire all callbacks.
    Returns the container (already added to viewer as a dock widget by main()).
    """
    # ── slider / checkbox widgets ──────────────────────────────────────────
    w_thresh_mult = widgets.FloatSlider(
        value=1.00, min=0.70, max=1.30, step=0.01,
        label="Threshold multiplier",
    )
    w_manual_thresh = widgets.FloatSlider(
        value=round(otsu_t, 4), min=0.0, max=1.0, step=0.005,
        label="Manual threshold",
    )
    w_manual_thresh.enabled = False

    w_use_manual = widgets.CheckBox(value=False, label="Use manual threshold")
    w_sigma = widgets.FloatSlider(
        value=0.8, min=0.0, max=2.0, step=0.1,
        label="Gaussian sigma",
    )
    w_min_size = widgets.SpinBox(
        value=100, min=0, max=5000, step=50,
        label="Min component size (vx)",
    )
    w_opacity = widgets.FloatSlider(
        value=0.55, min=0.0, max=1.0, step=0.05,
        label="Overlay opacity",
    )
    w_remove_boundary = widgets.CheckBox(
        value=False, label="Remove boundary components",
    )
    w_show_largest = widgets.CheckBox(value=True,  label="Show largest component")
    w_show_others  = widgets.CheckBox(value=True,  label="Show other components")

    # ── axis / animation controls ──────────────────────────────────────────
    w_anim_axis = widgets.ComboBox(
        choices=["z", "y", "x"], value="z", label="Animation axis",
    )
    w_scroll_axis = widgets.ComboBox(
        choices=["z (dim 0)", "y (dim 1)", "x (dim 2)"],
        value="z (dim 0)", label="Scroll axis",
    )

    # ── buttons ───────────────────────────────────────────────────────────
    btn_recompute   = widgets.PushButton(text="  Recompute segmentation  ")
    btn_screenshot  = widgets.PushButton(text="  Save screenshot  ")
    btn_metrics_csv = widgets.PushButton(text="  Save metrics CSV  ")
    btn_anim_raw    = widgets.PushButton(text="  Export raw flythrough  ")
    btn_anim_ov     = widgets.PushButton(text="  Export pore overlay flythrough  ")
    btn_anim_cc     = widgets.PushButton(text="  Export largest CC flythrough  ")
    btn_sweep       = widgets.PushButton(text="  Run threshold sweep  ")

    # ── metrics text area ─────────────────────────────────────────────────
    w_metrics = widgets.TextEdit(
        value="Press  'Recompute segmentation'  to see metrics.",
        label="",
    )
    try:
        w_metrics.native.setReadOnly(True)
    except Exception:
        pass

    # ── shared mutable state ──────────────────────────────────────────────
    state = {
        "pore_mask": None,
        "labeled":   None,
        "metrics":   {},
        "threshold": otsu_t,
        "mult":      1.00,
        "dice":      None,
    }

    # ── helper: resolve current threshold ─────────────────────────────────
    def _get_threshold():
        if w_use_manual.value:
            return float(w_manual_thresh.value), "manual"
        mult = float(w_thresh_mult.value)
        return float(otsu_t * mult), mult

    # ── recompute callback ────────────────────────────────────────────────
    def _recompute():
        t, mult = _get_threshold()
        state["threshold"] = t
        state["mult"]       = mult

        print(f"\nRecomputing  t={t:.4f}  sigma={w_sigma.value:.1f}  "
              f"min_size={w_min_size.value}  remove_boundary={w_remove_boundary.value}")
        try:
            pm, lbl, m = compute_segmentation(
                roi_norm, t,
                float(w_sigma.value),
                int(w_min_size.value),
                bool(w_remove_boundary.value),
            )
        except Exception as exc:
            print(f"  ERROR during segmentation: {exc}")
            traceback.print_exc()
            return

        state["pore_mask"] = pm
        state["labeled"]   = lbl
        state["metrics"]   = m

        dice = None
        if manual_label is not None:
            dice = compute_dice(pm, manual_label)
            state["dice"] = dice

        update_layers(
            viewer, pm, lbl,
            float(w_opacity.value),
            bool(w_show_largest.value),
            bool(w_show_others.value),
        )

        txt = format_metrics(m, t, mult, dice)
        w_metrics.value = txt
        print(txt)

    # ── use_manual toggle ─────────────────────────────────────────────────
    def _toggle_manual(val):
        w_thresh_mult.enabled   = not val
        w_manual_thresh.enabled = val
        if val:
            # sync manual slider to the current effective threshold
            w_manual_thresh.value = round(float(otsu_t * float(w_thresh_mult.value)), 4)
        _recompute()

    w_use_manual.changed.connect(_toggle_manual)

    # live recompute when either threshold control changes
    w_manual_thresh.changed.connect(
        lambda _: _recompute() if w_use_manual.value else None
    )
    w_thresh_mult.changed.connect(
        lambda _: _recompute() if not w_use_manual.value else None
    )

    # ── opacity / visibility sliders update layers live ───────────────────
    def _update_visibility():
        if state["pore_mask"] is None:
            return
        update_layers(
            viewer, state["pore_mask"], state["labeled"],
            float(w_opacity.value),
            bool(w_show_largest.value),
            bool(w_show_others.value),
        )

    w_opacity.changed.connect(lambda _: _update_visibility())
    w_show_largest.changed.connect(lambda _: _update_visibility())
    w_show_others.changed.connect(lambda _: _update_visibility())

    # ── scroll axis control ───────────────────────────────────────────────
    _axis_order_map = {
        "z (dim 0)": (0, 1, 2),
        "y (dim 1)": (1, 0, 2),
        "x (dim 2)": (2, 0, 1),
    }

    def _on_scroll_axis(val):
        try:
            viewer.dims.order = _axis_order_map[val]
        except Exception as e:
            print(f"  Could not change dims order: {e}")

    w_scroll_axis.changed.connect(_on_scroll_axis)

    # ── button callbacks ──────────────────────────────────────────────────
    btn_recompute.clicked.connect(_recompute)

    def _screenshot():
        save_screenshot(viewer, state["threshold"])

    btn_screenshot.clicked.connect(_screenshot)

    def _save_csv():
        if not state["metrics"]:
            print("  Run Recompute first.")
            return
        save_metrics_csv(
            state["metrics"], state["threshold"], state["mult"],
            float(w_sigma.value), int(w_min_size.value), state["dice"],
        )

    btn_metrics_csv.clicked.connect(_save_csv)

    def _anim_raw():
        axis = ["z", "y", "x"].index(w_anim_axis.value)
        export_animation(roi_norm, state["pore_mask"], state["labeled"],
                         mode="raw", axis=axis, threshold=state["threshold"])

    def _anim_overlay():
        axis = ["z", "y", "x"].index(w_anim_axis.value)
        export_animation(roi_norm, state["pore_mask"], state["labeled"],
                         mode="overlay", axis=axis, threshold=state["threshold"])

    def _anim_cc():
        axis = ["z", "y", "x"].index(w_anim_axis.value)
        export_animation(roi_norm, state["pore_mask"], state["labeled"],
                         mode="largest_cc", axis=axis, threshold=state["threshold"])

    btn_anim_raw.clicked.connect(_anim_raw)
    btn_anim_ov.clicked.connect(_anim_overlay)
    btn_anim_cc.clicked.connect(_anim_cc)

    def _sweep():
        run_threshold_sweep(
            roi_norm, otsu_t,
            float(w_sigma.value), int(w_min_size.value),
            w_metrics,
        )

    btn_sweep.clicked.connect(_sweep)

    # ── assemble container ────────────────────────────────────────────────
    container = widgets.Container(
        widgets=[
            widgets.Label(value="── Threshold ──────────────────"),
            w_thresh_mult,
            w_manual_thresh,
            w_use_manual,
            widgets.Label(value="── Segmentation ────────────────"),
            w_sigma,
            w_min_size,
            w_remove_boundary,
            widgets.Label(value="── Display ─────────────────────"),
            w_opacity,
            w_show_largest,
            w_show_others,
            widgets.Label(value="── Navigation ──────────────────"),
            w_scroll_axis,
            widgets.Label(value="── Actions ─────────────────────"),
            btn_recompute,
            btn_screenshot,
            btn_metrics_csv,
            widgets.Label(value="── Animation ───────────────────"),
            w_anim_axis,
            btn_anim_raw,
            btn_anim_ov,
            btn_anim_cc,
            widgets.Label(value="── Validation ──────────────────"),
            btn_sweep,
            widgets.Label(value="── Metrics ─────────────────────"),
            w_metrics,
        ],
        scrollable=True,
    )

    # Trigger initial recompute to populate all layers
    _recompute()

    return container

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global napari, widgets, Colormap

    # Deferred imports with clear error messages
    try:
        import napari as _napari
        napari = _napari
    except ImportError:
        print("ERROR: napari not installed.")
        print("  Run (in this project's venv):")
        print("    pip install napari magicgui PySide6")
        print("  Do NOT use napari[all] — it requires llvmlite which fails on ARM64.")
        sys.exit(1)

    try:
        from magicgui import widgets as _widgets
        widgets = _widgets
    except ImportError:
        print("ERROR: magicgui not installed.  Run:  pip install magicgui")
        sys.exit(1)

    try:
        from napari.utils.colormaps import Colormap as _Colormap
        Colormap = _Colormap
    except ImportError:
        try:
            from napari.utils import Colormap as _Colormap
            Colormap = _Colormap
        except ImportError:
            print("ERROR: Could not import napari Colormap.")
            sys.exit(1)

    _ensure_dirs()

    # Load data
    roi_norm, otsu_t, manual_label = load_data()
    print_startup_guide(otsu_t)

    # Create viewer
    viewer = napari.Viewer(title="Cordatum Shell — Pore Network Inspector")

    # Create all placeholder layers (will be populated by widget's initial recompute)
    nz, ny, nx = roi_norm.shape
    empty = np.zeros((nz, ny, nx), dtype=np.float32)
    add_initial_layers(viewer, roi_norm, empty.astype(np.uint8), empty.astype(np.int32))

    # Build control widget and dock it
    widget = build_widget(viewer, roi_norm, otsu_t, manual_label)
    viewer.window.add_dock_widget(widget, area="right", name="Segmentation Controls",
                                  allowed_areas=["right", "left"])

    print("\nViewer ready.")
    print("  Use the right panel to adjust threshold and recompute.")
    print("  Use napari's dimension sliders at the bottom to navigate slices.")
    print("  Toggle 2D / 3D with the button at the top-left of the canvas.")
    print()

    napari.run()


if __name__ == "__main__":
    main()
