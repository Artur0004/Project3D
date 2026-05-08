"""
make_animations.py
==================
Flythrough animations of the Cordatum Shell micro-CT ROI.

Uses the same ROI and threshold as the poster analysis.

Outputs (in cordatum_results/animations/):
  cordatum_roi_raw_flythrough.mp4
  cordatum_roi_pore_overlay_flythrough.mp4
  cordatum_roi_raw_flythrough.gif          (optional)
  cordatum_roi_pore_overlay_flythrough.gif (optional)
  raw_frame_start/middle/end.png
  overlay_frame_start/middle/end.png
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

DATA_PATH  = Path(r"C:\Users\artxm\PycharmProjects\3d2\Cordatum_Shell.tif")
OUTPUT_DIR = Path(r"C:\Users\artxm\PycharmProjects\3DImageAnalysis"
                  r"\cordatum_results\animations")

# ROI — downsampled x2 coordinates (identical to poster analysis)
DOWNSAMPLE_FACTOR = 2
ROI_Z = (172, 322)
ROI_Y = (178, 328)
ROI_X = (173, 323)

# Segmentation — same as poster analysis
THRESHOLD_VALUE    = 0.6640
GAUSSIAN_SIGMA     = 0.8
CLOSE_SOLID_RADIUS = 1
MIN_PORE_SIZE      = 100

# Animation
FPS        = 12          # frames per second for both videos
PORE_ALPHA = 0.55        # cyan overlay opacity  [0.45 – 0.65]

# Frame rendering resolution
DPI      = 130           # matplotlib render DPI  →  ~1300 x 1300 px per frame
FIGSIZE  = (10.0, 10.0)  # figure size in inches (square = square ROI slices)

# GIF options
SAVE_GIF     = True
GIF_EVERY_N  = 3         # sample every Nth frame to keep GIF size manageable
GIF_SCALE    = 0.35      # downscale factor for GIF frames

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import morphology
import tifffile

warnings.filterwarnings("ignore")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD CT AND SEGMENT PORES
# ─────────────────────────────────────────────────────────────────────────────

def load_and_segment():
    print("Loading CT ...")
    f = DOWNSAMPLE_FACTOR
    try:
        mm = tifffile.memmap(DATA_PATH)
    except Exception:
        mm = tifffile.imread(str(DATA_PATH))
    vol = np.array(mm[::f, ::f, ::f])
    p1   = float(np.percentile(vol, 1.0))
    p999 = float(np.percentile(vol, 99.8))
    norm = np.clip((vol.astype(np.float32) - p1) / max(p999 - p1, 1e-9), 0.0, 1.0)
    print(f"  Downsampled volume: {vol.shape}")

    z0, z1 = ROI_Z;  y0, y1 = ROI_Y;  x0, x1 = ROI_X
    roi = norm[z0:z1, y0:y1, x0:x1].copy()
    print(f"  ROI: Z={z0}-{z1}, Y={y0}-{y1}, X={x0}-{x1}  →  {roi.shape}")

    print("Segmenting pores ...")
    smooth = (ndi.gaussian_filter(roi, sigma=GAUSSIAN_SIGMA)
              if GAUSSIAN_SIGMA > 0 else roi)
    solid_raw = smooth >= THRESHOLD_VALUE
    if CLOSE_SOLID_RADIUS > 0:
        solid = morphology.binary_closing(solid_raw, morphology.ball(CLOSE_SOLID_RADIUS))
    else:
        solid = solid_raw.astype(bool)
    envelope  = ndi.binary_dilation(solid, iterations=max(1, CLOSE_SOLID_RADIUS + 2))
    pore_cand = (smooth < THRESHOLD_VALUE) & envelope & (~solid)
    pore_mask = morphology.remove_small_objects(pore_cand, min_size=MIN_PORE_SIZE)
    print(f"  Pore voxels: {pore_mask.sum():,}  ({pore_mask.mean() * 100:.1f}%)")

    # normalise ROI for display (fixed contrast across all frames)
    p1r  = float(np.percentile(roi, 1.0))
    p99r = float(np.percentile(roi, 99.5))
    roi_disp = np.clip((roi - p1r) / max(p99r - p1r, 1e-9), 0.0, 1.0).astype(np.float32)

    return roi_disp, pore_mask.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_overlay_rgba(gray2d, pore2d, alpha):
    """Return float32 RGBA with cyan pore overlay on grayscale background."""
    rgba = np.stack([gray2d, gray2d, gray2d, np.ones_like(gray2d)], axis=-1).astype(np.float32)
    p = pore2d > 0
    rgba[p, 0] = 0.0  * alpha + rgba[p, 0] * (1.0 - alpha)
    rgba[p, 1] = 0.75 * alpha + rgba[p, 1] * (1.0 - alpha)
    rgba[p, 2] = 1.0  * alpha + rgba[p, 2] * (1.0 - alpha)
    return np.clip(rgba, 0.0, 1.0)


def _canvas_to_rgb(fig):
    """Extract matplotlib figure as uint8 RGB numpy array."""
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return buf.reshape(h, w, 4)[:, :, :3].copy()
    except AttributeError:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3).copy()


def _build_figure(title):
    """Create a black-background figure with imshow, title, and slice counter."""
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.02, right=0.98)
    ax.axis("off")

    # placeholder image (will be updated per frame)
    im = ax.imshow(np.zeros((10, 10)), cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest", aspect="equal")

    ttl = fig.text(0.5, 0.96, title,
                   ha="center", va="top", color="white",
                   fontsize=22, fontweight="bold", fontfamily="DejaVu Sans")
    cnt = fig.text(0.5, 0.03, "",
                   ha="center", va="bottom", color="#BBBBBB",
                   fontsize=15, fontfamily="DejaVu Sans")
    return fig, ax, im, ttl, cnt


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER ALL FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def render_all_frames(roi_disp, pore_mask):
    nz = roi_disp.shape[0]
    print(f"\nRendering {nz} frames per animation ...")

    fig_r, _, im_r, ttl_r, cnt_r = _build_figure("Raw CT ROI")
    fig_o, _, im_o, ttl_o, cnt_o = _build_figure("Segmented pore space")

    raw_frames = []
    ov_frames  = []
    t_start    = time.time()

    for z in range(nz):
        if z == 0 or (z + 1) % 30 == 0 or z == nz - 1:
            elapsed = time.time() - t_start
            print(f"  Frame {z + 1:3d} / {nz}  ({elapsed:.1f}s)")

        label = f"Slice  {z + 1:3d} / {nz}"

        # raw frame
        im_r.set_data(roi_disp[z])
        cnt_r.set_text(label)
        raw_frames.append(_canvas_to_rgb(fig_r))

        # overlay frame
        ov = _make_overlay_rgba(roi_disp[z], pore_mask[z], PORE_ALPHA)
        im_o.set_data(ov)
        cnt_o.set_text(label)
        ov_frames.append(_canvas_to_rgb(fig_o))

    plt.close(fig_r)
    plt.close(fig_o)

    h, w = raw_frames[0].shape[:2]
    print(f"  Frame dimensions: {w} x {h} px  ({len(raw_frames)} frames)")
    return raw_frames, ov_frames


# ─────────────────────────────────────────────────────────────────────────────
#  WRITE MP4
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_even_dims(frames):
    """Crop to even pixel dimensions (required by libx264)."""
    h, w = frames[0].shape[:2]
    h2 = h - (h % 2);  w2 = w - (w % 2)
    if h2 == h and w2 == w:
        return frames
    return [f[:h2, :w2] for f in frames]


def write_mp4(frames, path):
    path = Path(path)
    frames = _ensure_even_dims(frames)

    # Attempt 1: imageio with ffmpeg
    try:
        import imageio
        imageio.mimwrite(str(path), frames, fps=FPS, quality=8, macro_block_size=None)
        print(f"  Saved {path.name}  ({len(frames)} frames, {FPS} fps)  [imageio]")
        return
    except Exception as e1:
        pass

    # Attempt 2: imageio v3 plugin API
    try:
        import imageio
        import imageio_ffmpeg  # noqa: F401
        writer = imageio.get_writer(str(path), fps=FPS, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  Saved {path.name}  [imageio v3]")
        return
    except Exception as e2:
        pass

    # Attempt 3: matplotlib FFMpegWriter
    try:
        import matplotlib.animation as manim
        h, w = frames[0].shape[:2]
        fig2, ax2 = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        fig2.patch.set_facecolor("black")
        fig2.subplots_adjust(0, 0, 1, 1)
        ax2.axis("off")
        im2 = ax2.imshow(frames[0], interpolation="nearest", aspect="auto")

        def _upd(i):
            im2.set_data(frames[i])
            return [im2]

        ani = manim.FuncAnimation(fig2, _upd, frames=len(frames),
                                   blit=True, interval=1000 // FPS)
        writer = manim.FFMpegWriter(
            fps=FPS, bitrate=4000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        )
        ani.save(str(path), writer=writer, dpi=100)
        plt.close(fig2)
        print(f"  Saved {path.name}  [FFMpegWriter]")
        return
    except Exception as e3:
        pass

    print(f"  ERROR: Could not write {path.name}. Install ffmpeg or imageio[ffmpeg].")
    print(f"    pip install imageio[ffmpeg]")


# ─────────────────────────────────────────────────────────────────────────────
#  WRITE GIF
# ─────────────────────────────────────────────────────────────────────────────

def write_gif(frames, path):
    path = Path(path)
    subset    = frames[::GIF_EVERY_N]
    gif_fps   = max(1, FPS // GIF_EVERY_N)
    duration  = int(1000 / gif_fps)   # ms per frame

    # Try Pillow directly (most reliable for GIF, no ffmpeg needed)
    try:
        from PIL import Image as PILImage
        pil_frames = []
        for f in subset:
            img = PILImage.fromarray(f).convert("RGB")
            new_w = max(4, int(f.shape[1] * GIF_SCALE))
            new_h = max(4, int(f.shape[0] * GIF_SCALE))
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            pil_frames.append(img.convert("P", palette=PILImage.ADAPTIVE, colors=128))

        pil_frames[0].save(
            str(path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        print(f"  Saved {path.name}  ({len(pil_frames)} frames, {gif_fps} fps, "
              f"{int(GIF_SCALE*100)}% size)  [Pillow]")
        return
    except ImportError:
        pass

    # Fallback: imageio
    try:
        import imageio
        from PIL import Image as PILImage
        small = []
        for f in subset:
            img = PILImage.fromarray(f)
            new_w = max(4, int(f.shape[1] * GIF_SCALE))
            new_h = max(4, int(f.shape[0] * GIF_SCALE))
            small.append(np.array(img.resize((new_w, new_h), PILImage.LANCZOS)))
        imageio.mimsave(str(path), small, fps=gif_fps, loop=0)
        print(f"  Saved {path.name}  [imageio]")
    except Exception as e:
        print(f"  GIF skipped ({e})")


# ─────────────────────────────────────────────────────────────────────────────
#  STILL FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def save_still_frames(raw_frames, ov_frames):
    n = len(raw_frames)
    indices = [0, n // 2, n - 1]
    suffixes = ["start", "middle", "end"]

    for idx, suf in zip(indices, suffixes):
        p = OUTPUT_DIR / f"raw_frame_{suf}.png"
        plt.imsave(str(p), raw_frames[idx])
        print(f"  Saved {p.name}")

    for idx, suf in zip(indices, suffixes):
        p = OUTPUT_DIR / f"overlay_frame_{suf}.png"
        plt.imsave(str(p), ov_frames[idx])
        print(f"  Saved {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Cordatum Shell - Animation Generator")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")

    roi_disp, pore_mask = load_and_segment()
    raw_frames, ov_frames = render_all_frames(roi_disp, pore_mask)

    print("\nWriting MP4 videos ...")
    write_mp4(raw_frames, OUTPUT_DIR / "cordatum_roi_raw_flythrough.mp4")
    write_mp4(ov_frames,  OUTPUT_DIR / "cordatum_roi_pore_overlay_flythrough.mp4")

    if SAVE_GIF:
        print("\nWriting GIFs ...")
        write_gif(raw_frames, OUTPUT_DIR / "cordatum_roi_raw_flythrough.gif")
        write_gif(ov_frames,  OUTPUT_DIR / "cordatum_roi_pore_overlay_flythrough.gif")

    print("\nSaving still frames ...")
    save_still_frames(raw_frames, ov_frames)

    print(f"\n{'=' * 60}")
    print(f"  Done in {time.time() - t0:.1f} s")
    print(f"  All outputs -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
