#!/usr/bin/env python3
"""Adaptive window pull for bracketed HDR.

Automatically identifies blown highlights in the reference frame, selects the
best underexposed frame for detail recovery, and blends with adaptive feathering.
No neural network flow needed — designed for tripod-mounted brackets.

Uses SAFNet-style weight functions for the base merge, with a selective window
pull that substitutes underexposed frame data in highlight regions.

Outputs:
  - windowpull_result.tif    — 32-bit float, full HDR, RGB
  - windowpull_preview.png   — 16-bit tonemapped preview

Usage:
  python3 safnet_windowpull.py <NEF1> <NEF2> ... <NEFn> --output-dir <dir>
"""

import argparse
import json
import math
import os
import subprocess
import sys
import numpy as np
import cv2
import rawpy


def get_exif(nef_path):
    """Extract exposure metadata from NEF."""
    result = subprocess.run(
        ["exiftool", "-ExposureTime", "-ExposureCompensation", "-n", "-json", nef_path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)[0]
    return {
        "path": nef_path,
        "exposure_time": float(data["ExposureTime"]),
        "ev_comp": float(data.get("ExposureCompensation", 0)),
    }


def develop_nef(nef_path):
    """Demosaic NEF to sRGB float32 in [0, 1], RGB order, full resolution."""
    raw = rawpy.imread(nef_path)
    rgb = raw.postprocess(
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=16,
        no_auto_bright=True,
        gamma=(2.2, 4.5),
    )
    # Ensure dimensions are multiples of 16 (for potential future SAFNet use)
    h, w = rgb.shape[:2]
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    if new_h != h or new_w != w:
        rgb = rgb[:new_h, :new_w, :]
    return (rgb / 65535.0).astype(np.float32)


def compute_adaptive_mask(ref_ldr, feather_frac=0.005):
    """Compute a smooth highlight mask from the reference frame.

    Identifies near-saturated pixels (any channel > adaptive threshold),
    then creates a feathered mask using distance transform.

    Args:
        ref_ldr: Reference frame as float32 RGB in [0, 1]
        feather_frac: Feather radius as fraction of image diagonal

    Returns:
        mask: float32 array in [0, 1], same H x W as ref_ldr
              1.0 = fully blown (use underexposed), 0.0 = normal (use standard merge)
    """
    h, w = ref_ldr.shape[:2]
    diagonal = math.sqrt(h * h + w * w)
    feather_px = max(3, int(diagonal * feather_frac))
    # Ensure odd kernel size
    feather_px = feather_px | 1

    # Per-channel max — blown if ANY channel is near saturation
    max_ch = np.max(ref_ldr, axis=2)

    # Adaptive core threshold: use Otsu on the top 20% of brightness
    # to find the natural break between "bright interior" and "blown highlights"
    bright_pixels = max_ch[max_ch > 0.5]
    if len(bright_pixels) < 100:
        # Very few bright pixels — no highlights to pull
        print("  No significant highlights detected in reference.")
        return np.zeros((h, w), dtype=np.float32)

    # Otsu's threshold on the bright region
    bright_uint8 = (bright_pixels * 255).clip(0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(bright_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core_thresh = otsu_val / 255.0

    # Clamp to reasonable range — don't go below 0.7 (would affect too much)
    # or above 0.98 (would miss useful regions)
    core_thresh = np.clip(core_thresh, 0.7, 0.98)
    print(f"  Adaptive highlight threshold: {core_thresh:.3f} (Otsu on bright pixels)")

    # Core blown region: pixels above threshold
    core = (max_ch >= core_thresh).astype(np.uint8)

    # Erode slightly to remove isolated bright pixels (specular noise)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(core, erode_kernel, iterations=1)

    # Distance transform for smooth feathering
    # Distance from each non-core pixel to nearest core pixel
    inv_core = 1 - core
    dist = cv2.distanceTransform(inv_core, cv2.DIST_L2, 5)

    # Create feathered mask: 1.0 at core, fading to 0 over feather_px
    mask = np.zeros_like(max_ch)
    mask[core == 1] = 1.0
    transition = (core == 0) & (dist < feather_px)
    mask[transition] = 1.0 - dist[transition] / feather_px

    # Additional Gaussian smooth for natural edges
    blur_size = max(3, feather_px // 2) | 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_size / 3)

    pct_core = (core > 0).mean() * 100
    pct_feathered = (mask > 0.01).mean() * 100
    print(f"  Core blown pixels: {pct_core:.1f}%, feathered region: {pct_feathered:.1f}%")
    print(f"  Feather radius: {feather_px}px ({feather_frac*100:.1f}% of diagonal)")

    return mask.astype(np.float32)


def select_pull_frame(frames, ref_idx, mask):
    """Select the best underexposed frame for the window pull.

    Picks the frame where the blown region (mask > 0.5) has the best
    exposure — median brightness closest to 0.4 (well-exposed, slight
    safety margin from midtone).

    Args:
        frames: list of (ldr_float32_rgb, exif_dict)
        ref_idx: index of reference frame
        mask: highlight mask from compute_adaptive_mask

    Returns:
        best_idx: index of the best pull frame
    """
    blown_region = mask > 0.5
    if not blown_region.any():
        # Fallback: use darkest frame
        exposures = [f[1]["exposure_time"] for f in frames]
        return int(np.argmin(exposures))

    best_idx = None
    best_score = float("inf")

    for i, (ldr, exif) in enumerate(frames):
        if i == ref_idx:
            continue
        if exif["exposure_time"] >= frames[ref_idx][1]["exposure_time"]:
            continue  # Only consider darker frames

        # Median brightness of this frame in the blown region
        lum = 0.2126 * ldr[:, :, 0] + 0.7152 * ldr[:, :, 1] + 0.0722 * ldr[:, :, 2]
        median_bright = np.median(lum[blown_region])

        # Score: distance from ideal exposure (0.4 = well-exposed)
        score = abs(median_bright - 0.4)
        print(f"    Frame {i} (EV {exif['ev_comp']:+.0f}, {exif['exposure_time']:.4f}s): "
              f"median in blown region = {median_bright:.3f}, score = {score:.3f}")

        if score < best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        # No darker frames — use darkest available
        exposures = [f[1]["exposure_time"] for f in frames]
        best_idx = int(np.argmin(exposures))

    return best_idx


def safnet_weight_merge(frames, ref_idx):
    """SAFNet-style weighted merge using luminance-based weight functions.

    No flow, no neural network — just the weight formula from SAFNet's utils.py,
    generalized to N exposures.

    Returns:
        hdr: float32 RGB, linearized HDR image
    """
    ref_ldr = frames[ref_idx][0]
    ref_lum = 0.2126 * ref_ldr[:, :, 0] + 0.7152 * ref_ldr[:, :, 1] + 0.0722 * ref_ldr[:, :, 2]

    exposures = [f[1]["exposure_time"] for f in frames]
    min_exp = min(exposures)
    exp_ratios = [et / min_exp for et in exposures]

    h, w = ref_ldr.shape[:2]
    sum_img = np.zeros((h, w, 3), dtype=np.float64)
    sum_w = np.zeros((h, w, 1), dtype=np.float64)

    for i, (ldr, exif) in enumerate(frames):
        # Linearize
        lin = (ldr.astype(np.float64) ** 2.2) / exp_ratios[i]

        # Well-exposedness weight: Gaussian centered at 0.5
        lum = 0.2126 * ldr[:, :, 0] + 0.7152 * ldr[:, :, 1] + 0.0722 * ldr[:, :, 2]
        w_exp = np.exp(-0.5 * ((lum - 0.5) / 0.2) ** 2)

        # Saturation penalty: near-0 and near-1 pixels get low weight
        min_ch = np.min(ldr, axis=2)
        max_ch = np.max(ldr, axis=2)
        w_sat = (1 - np.exp(-0.5 * ((max_ch - 1.0) / 0.1) ** 2)) * \
                (1 - np.exp(-0.5 * ((min_ch - 0.0) / 0.1) ** 2))

        weight = (w_exp * w_sat)[:, :, np.newaxis]
        sum_img += weight * lin
        sum_w += weight

    hdr = (sum_img / (sum_w + 1e-12)).astype(np.float32)
    return hdr


def apply_window_pull(hdr_base, pull_frame_ldr, pull_exp_ratio, mask, strength=1.0):
    """Apply selective window pull to the base HDR merge.

    In highlighted regions (mask > 0), substitute the pull frame's linearized
    values WITHOUT full irradiance normalization — intentionally compressing
    the highlight DR to show exterior detail.

    Args:
        hdr_base: Base HDR merge (float32 RGB)
        pull_frame_ldr: Pull frame's LDR values (float32 RGB, [0,1])
        pull_exp_ratio: Pull frame's exposure ratio relative to darkest
        mask: Highlight mask (float32, [0,1])
        strength: How much to compress highlights (1.0 = full pull, 0.5 = half)

    Returns:
        hdr_pulled: HDR image with window pull applied
    """
    # Linearize the pull frame
    pull_lin_full = (pull_frame_ldr.astype(np.float64) ** 2.2) / pull_exp_ratio

    # "Pulled" version: linearize but with reduced exposure compensation
    # This compresses the highlight dynamic range
    # strength=1.0: no exposure compensation (darkest possible)
    # strength=0.0: full compensation (same as standard merge)
    effective_ratio = pull_exp_ratio ** (1.0 - strength)
    pull_lin_compressed = (pull_frame_ldr.astype(np.float64) ** 2.2) / effective_ratio

    mask_3ch = mask[:, :, np.newaxis].astype(np.float64)
    hdr_pulled = hdr_base.astype(np.float64) * (1 - mask_3ch) + pull_lin_compressed * mask_3ch

    return hdr_pulled.astype(np.float32)


def tonemap_mu(hdr, mu=5000):
    """Mu-law tonemapping."""
    return (np.log(1 + mu * np.clip(hdr, 0, None)) / math.log(1 + mu)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Adaptive window pull for bracketed HDR")
    parser.add_argument("nefs", nargs="+", help="NEF files (any order, auto-sorted by exposure)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--pull-strength", type=float, default=0.8,
                        help="Window pull strength: 0=none, 1=maximum (default: 0.8)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Read EXIF and sort by exposure ---
    print("Reading EXIF metadata...")
    exifs = [get_exif(p) for p in args.nefs]
    exifs.sort(key=lambda x: x["exposure_time"])
    for i, e in enumerate(exifs):
        print(f"  [{i}] {os.path.basename(e['path'])}: "
              f"{e['exposure_time']:.4f}s, EV {e['ev_comp']:+.0f}")

    # --- Identify reference (middle exposure) ---
    ref_idx = len(exifs) // 2
    print(f"\nReference frame: [{ref_idx}] {os.path.basename(exifs[ref_idx]['path'])}")

    # --- Develop all NEFs at full resolution ---
    print("\nDeveloping RAW files at full resolution...")
    frames = []
    for i, e in enumerate(exifs):
        print(f"  Developing [{i}] {os.path.basename(e['path'])}...", end=" ", flush=True)
        ldr = develop_nef(e["path"])
        print(f"{ldr.shape[1]}x{ldr.shape[0]}")
        frames.append((ldr, e))

    # --- Compute adaptive highlight mask from reference ---
    print("\nComputing adaptive highlight mask...")
    ref_ldr = frames[ref_idx][0]
    mask = compute_adaptive_mask(ref_ldr)

    # Save mask for diagnostics (but only the mask, not multiple variants)
    mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "windowpull_mask.png"), mask_uint8)

    # --- Select best pull frame ---
    print("\nSelecting best underexposed frame for window pull...")
    pull_idx = select_pull_frame(frames, ref_idx, mask)
    print(f"  Selected: [{pull_idx}] {os.path.basename(frames[pull_idx][1]['path'])}")

    # --- Base merge (SAFNet-style weights, no flow) ---
    print("\nComputing base HDR merge...")
    hdr_base = safnet_weight_merge(frames, ref_idx)
    print(f"  Base HDR: range [{hdr_base.min():.6f}, {hdr_base.max():.6f}]")

    # --- Apply window pull ---
    print(f"\nApplying window pull (strength={args.pull_strength})...")
    min_exp = min(e["exposure_time"] for _, e in frames)
    pull_exp_ratio = frames[pull_idx][1]["exposure_time"] / min_exp

    hdr_pulled = apply_window_pull(
        hdr_base, frames[pull_idx][0], pull_exp_ratio, mask,
        strength=args.pull_strength
    )
    print(f"  Pulled HDR: range [{hdr_pulled.min():.6f}, {hdr_pulled.max():.6f}]")

    # --- Save outputs ---
    print("\nSaving outputs...")

    # 32-bit float TIFF (lossless, full precision)
    # OpenCV writes BGR, so convert
    tif_path = os.path.join(args.output_dir, "windowpull_result.tif")
    cv2.imwrite(tif_path, hdr_pulled[:, :, ::-1])
    size_mb = os.path.getsize(tif_path) / (1024 * 1024)
    print(f"  {tif_path} ({size_mb:.1f} MB, 32-bit float RGB)")

    # 16-bit tonemapped preview
    preview = tonemap_mu(hdr_pulled, mu=5000)
    preview_16 = (np.clip(preview, 0, 1) * 65535).astype(np.uint16)
    preview_path = os.path.join(args.output_dir, "windowpull_preview.png")
    cv2.imwrite(preview_path, preview_16[:, :, ::-1])
    preview_mb = os.path.getsize(preview_path) / (1024 * 1024)
    print(f"  {preview_path} ({preview_mb:.1f} MB, 16-bit tonemapped)")

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
