#!/usr/bin/env python3
"""Convert a 3-exposure NEF bracket into SAFNet input format.

Creates a directory with:
  - ldr_img_1.tif (underexposed, 16-bit)
  - ldr_img_2.tif (middle/reference, 16-bit)
  - ldr_img_3.tif (overexposed, 16-bit)
  - exposure.txt (log2 exposure times, one per line)

Usage:
  python3 safnet_prepare_inputs.py <under.NEF> <mid.NEF> <over.NEF> <output_dir> [--max-dim N]
"""

import argparse
import os
import sys
import numpy as np

try:
    import rawpy
    import cv2
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install rawpy opencv-python-headless")
    sys.exit(1)


def get_exposure_time(nef_path):
    """Extract exposure time in seconds from RAW EXIF."""
    import subprocess
    result = subprocess.run(
        ["exiftool", "-ExposureTime", "-n", "-json", nef_path],
        capture_output=True, text=True
    )
    import json
    data = json.loads(result.stdout)
    return float(data[0]["ExposureTime"])


def develop_nef(nef_path, max_dim=None):
    """Demosaic NEF to sRGB 16-bit image using rawpy.

    Returns (H, W, 3) uint16 array in RGB order.
    """
    raw = rawpy.imread(nef_path)
    # Develop with camera white balance, sRGB output, 16-bit
    rgb = raw.postprocess(
        use_camera_wb=True,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=16,
        no_auto_bright=True,
        gamma=(2.2, 4.5),  # Standard sRGB-like gamma
    )

    if max_dim is not None:
        h, w = rgb.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            # Ensure dimensions are multiples of 16 (SAFNet requirement)
            new_h = (new_h // 16) * 16
            new_w = (new_w // 16) * 16
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  Resized {w}x{h} -> {new_w}x{new_h}")

    return rgb


def main():
    parser = argparse.ArgumentParser(description="Prepare NEF bracket for SAFNet")
    parser.add_argument("under", help="Underexposed NEF (shortest exposure)")
    parser.add_argument("mid", help="Middle/reference NEF")
    parser.add_argument("over", help="Overexposed NEF (longest exposure)")
    parser.add_argument("output_dir", help="Output directory for SAFNet input")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Max dimension (pixels). Use 1500 for fast testing.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    nef_paths = [args.under, args.mid, args.over]
    names = ["ldr_img_1.tif", "ldr_img_2.tif", "ldr_img_3.tif"]
    labels = ["under", "mid", "over"]

    # Get exposure times
    exposure_times = []
    for path, label in zip(nef_paths, labels):
        et = get_exposure_time(path)
        exposure_times.append(et)
        print(f"  {label}: {os.path.basename(path)} -> {et}s")

    # Write exposure.txt (log2 of exposure times)
    log2_exposures = [np.log2(et) for et in exposure_times]
    expo_path = os.path.join(args.output_dir, "exposure.txt")
    with open(expo_path, "w") as f:
        for val in log2_exposures:
            f.write(f"{val:.6f}\n")
    print(f"  exposure.txt: {log2_exposures}")

    # Develop and save TIFFs
    for path, name, label in zip(nef_paths, names, labels):
        print(f"  Developing {label}...", end=" ", flush=True)
        rgb = develop_nef(path, max_dim=args.max_dim)
        # OpenCV expects BGR for writing
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, name)
        cv2.imwrite(out_path, bgr)
        print(f"saved {name} ({rgb.shape[1]}x{rgb.shape[0]})")

    print(f"\nReady: {args.output_dir}")


if __name__ == "__main__":
    main()
