#!/usr/bin/env python3
"""
Detect pink/green ghost artifacts in HDR-merged DNG files.

Ghost artifacts in HDR merges appear as color fringes (pink/magenta and green)
at exposure layer boundaries where different exposures contribute conflicting
Bayer channel values.

Detection uses two complementary signals:
1. Saturated pink/green pixel fraction — ghost fringes are highly saturated
   colors that stand out from natural scene content
2. Chrominance fringe energy — Sobel gradient of the LAB a* channel at
   saturated pink/green pixels, detecting sharp color transitions typical
   of ghost boundaries

Usage:
    python3 detect_ghost_artifacts.py /path/to/merged/ [--threshold 8] [--jobs 4]
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import rawpy


def analyze_dng(dng_path: str) -> dict:
    """Analyze a single DNG for pink/green ghost artifacts."""
    try:
        raw = rawpy.imread(dng_path)
        rgb = raw.postprocess(
            half_size=True,
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
        )
        raw.close()
    except Exception as e:
        return {"file": dng_path, "error": str(e)}

    img8 = (rgb.astype(np.float32) / 65535.0 * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    L_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]  # green(-) ↔ magenta(+), 128=neutral
    sat = hsv[:, :, 1]   # 0-255
    hue = hsv[:, :, 0]   # 0-179 in OpenCV

    valid = (L_ch > 15) & (L_ch < 240)
    total = float(valid.sum()) if valid.sum() > 0 else 1.0

    # Chrominance gradient magnitude (Sobel of a* channel)
    gx = cv2.Sobel(a_ch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a_ch, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Strong chrominance edges in valid luminance range
    strong_edge = (grad_mag > 15) & valid
    high_sat = sat > 80
    very_high_sat = sat > 140

    # Hue masks
    pink_hue = (hue >= 150) | (hue <= 10)   # ~300-360 and 0-20 degrees
    green_hue = (hue >= 35) & (hue <= 85)   # ~70-170 degrees

    # Fringe pixels: strong chrominance edge + saturated + specific hue
    pink_fringe = strong_edge & high_sat & pink_hue
    green_fringe = strong_edge & high_sat & green_hue

    # Intensity-weighted fringe energy
    pink_fringe_energy = (
        float(grad_mag[pink_fringe].sum()) / total if pink_fringe.any() else 0
    )
    green_fringe_energy = (
        float(grad_mag[green_fringe].sum()) / total if green_fringe.any() else 0
    )

    # Saturated pink/green pixel fraction (proven discriminator)
    pink_sat_frac = float((pink_hue & very_high_sat & valid).sum()) / total
    green_sat_frac = float((green_hue & very_high_sat & valid).sum()) / total

    # Composite ghost score
    ghost_score = (
        (pink_fringe_energy + green_fringe_energy) * 5
        + (pink_sat_frac + green_sat_frac) * 500
    )

    return {
        "file": os.path.basename(dng_path),
        "ghost_score": round(ghost_score, 2),
        "pink_sat_frac": round(pink_sat_frac, 6),
        "green_sat_frac": round(green_sat_frac, 6),
        "pink_fringe_energy": round(pink_fringe_energy, 4),
        "green_fringe_energy": round(green_fringe_energy, 4),
    }


def parse_dng_name(dng_name: str) -> tuple:
    """Parse DNG filename to extract first file base and last file number.

    HDRMerge names outputs as: {first_basename_no_ext}-{last_number_suffix}.dng
    """
    stem = Path(dng_name).stem
    m = re.match(r"^(.+)-(\d+)$", stem)
    if m:
        return m.group(1), m.group(2)
    return stem, None


def find_source_nefs(dng_name: str, nef_dir: str) -> list:
    """Find source NEF files for a given DNG output."""
    first_base, last_num = parse_dng_name(dng_name)

    if last_num is None:
        candidates = [
            os.path.join(nef_dir, first_base + ext)
            for ext in (".NEF", ".nef")
        ]
        return [c for c in candidates if os.path.exists(c)]

    m = re.search(r"(\d+)(?:-\d+)?$", first_base)
    if not m:
        return []
    first_num = int(m.group(1))
    last_num_int = int(last_num)

    nef_files = sorted(
        f for f in os.listdir(nef_dir) if f.upper().endswith(".NEF")
    )

    result = []
    for nef in nef_files:
        stem = Path(nef).stem
        nm = re.search(r"(\d+)(?:-\d+)?$", stem)
        if nm:
            num = int(nm.group(1))
            if first_num <= num <= last_num_int:
                result.append(os.path.join(nef_dir, nef))

    # Ensure first file is at position 0
    first_nef_name = first_base + ".NEF"
    for i, r in enumerate(result):
        if os.path.basename(r) == first_nef_name and i != 0:
            result.insert(0, result.pop(i))
            break

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect pink/green ghost artifacts in merged DNG files"
    )
    parser.add_argument("merged_dir", help="Directory containing merged DNG files")
    parser.add_argument(
        "--nef-dir",
        help="Directory with source NEF files (default: parent of merged_dir)",
    )
    parser.add_argument(
        "--threshold", type=float, default=8.0,
        help="Ghost score threshold for flagging (default: 8.0)",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=4,
        help="Parallel analysis jobs (default: 4)",
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file for full results",
    )
    parser.add_argument(
        "--hdrmerge-bin",
        default=os.path.expanduser(
            "~/Development/hdrmerge/build/hdrmerge.app/Contents/MacOS/hdrmerge"
        ),
    )
    parser.add_argument("--deghost-sigma", type=float, default=3.0)
    parser.add_argument("--deghost-iterations", type=int, default=3)
    parser.add_argument(
        "--reprocess-dir",
        help="Output dir for re-processed files (default: merged-deghost/)",
    )
    args = parser.parse_args()

    merged_dir = os.path.abspath(args.merged_dir)
    nef_dir = (
        os.path.abspath(args.nef_dir) if args.nef_dir
        else os.path.dirname(merged_dir)
    )
    if not args.reprocess_dir:
        args.reprocess_dir = os.path.join(
            os.path.dirname(merged_dir), "merged-deghost"
        )

    dng_files = sorted(
        os.path.join(merged_dir, f)
        for f in os.listdir(merged_dir)
        if f.lower().endswith(".dng")
    )

    if not dng_files:
        print(f"No DNG files found in {merged_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Analyzing {len(dng_files)} DNG files ({args.jobs} workers)...",
        file=sys.stderr,
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(analyze_dng, f): f for f in dng_files}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            results.append(result)
            if "error" not in result:
                score = result["ghost_score"]
                flag = " *** FLAGGED" if score >= args.threshold else ""
                print(
                    f"  [{done}/{len(dng_files)}] {result['file']}: "
                    f"score={score:.2f}{flag}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [{done}/{len(dng_files)}] {result['file']}: "
                    f"ERROR - {result['error']}",
                    file=sys.stderr,
                )

    results.sort(key=lambda r: r.get("ghost_score", -1), reverse=True)
    flagged = [r for r in results if r.get("ghost_score", 0) >= args.threshold]
    errors = [r for r in results if "error" in r]

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results: {args.output}", file=sys.stderr)

    # Summary
    print(f"\n{'='*70}", file=sys.stderr)
    print(
        f"SUMMARY: {len(flagged)} flagged / {len(results)} total "
        f"(threshold={args.threshold})",
        file=sys.stderr,
    )
    print(f"{'='*70}", file=sys.stderr)
    if errors:
        print(f"Errors: {len(errors)}", file=sys.stderr)

    if flagged:
        print(
            f"\n{'File':<35} {'Score':>7} {'PinkSat':>9} {'GreenSat':>9} "
            f"{'PinkFrng':>9} {'GreenFrng':>9}",
            file=sys.stderr,
        )
        print(
            f"{'-'*35} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9}",
            file=sys.stderr,
        )
        for r in flagged:
            print(
                f"{r['file']:<35} {r['ghost_score']:>7.2f} "
                f"{r['pink_sat_frac']:>9.6f} {r['green_sat_frac']:>9.6f} "
                f"{r['pink_fringe_energy']:>9.4f} {r['green_fringe_energy']:>9.4f}",
                file=sys.stderr,
            )

    # Generate re-processing script
    if flagged:
        script_lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            f"# Re-processing script for {len(flagged)} ghost-artifact files",
            f"# Detection threshold: {args.threshold}",
            f"# Deghost: sigma={args.deghost_sigma}, "
            f"iterations={args.deghost_iterations}, mode=robust",
            "",
            f'HDRMERGE="{args.hdrmerge_bin}"',
            f'OUTPUT_DIR="{args.reprocess_dir}"',
            "",
            'mkdir -p "$OUTPUT_DIR"',
            "",
        ]

        reprocess_count = 0
        skipped = []
        for r in flagged:
            dng_name = r["file"]
            source_nefs = find_source_nefs(dng_name, nef_dir)

            if len(source_nefs) < 3:
                skipped.append(
                    f"# SKIP {dng_name}: {len(source_nefs)} NEFs "
                    f"(need >= 3 for deghosting)"
                )
                continue

            reprocess_count += 1
            nef_list = " \\\n    ".join(f'"{n}"' for n in source_nefs)
            script_lines.extend([
                f"# {dng_name} (score: {r['ghost_score']:.2f})",
                f'"$HDRMERGE" -a \\',
                f"    --deghost {args.deghost_sigma} \\",
                f"    --deghost-mode robust \\",
                f"    --deghost-iterations {args.deghost_iterations} \\",
                f'    -O "$OUTPUT_DIR" \\',
                f"    {nef_list}",
                "",
            ])

        if skipped:
            script_lines.extend(skipped)
            script_lines.append("")

        script_lines.append(
            f'echo "Done: {reprocess_count} files re-processed into $OUTPUT_DIR"'
        )

        script_path = os.path.join(
            os.path.dirname(merged_dir), "reprocess_ghosts.sh"
        )
        with open(script_path, "w") as f:
            f.write("\n".join(script_lines) + "\n")
        os.chmod(script_path, 0o755)

        print(f"\nRe-processing script: {script_path}", file=sys.stderr)
        print(f"Files to re-process: {reprocess_count}", file=sys.stderr)
        if skipped:
            print(f"Skipped (< 3 NEFs): {len(skipped)}", file=sys.stderr)
        print(f"\nRun:  {script_path}", file=sys.stderr)

    # Score distribution
    scores = sorted(
        r.get("ghost_score", 0) for r in results if "error" not in r
    )
    if scores:
        print(f"\n{'='*70}", file=sys.stderr)
        print("SCORE DISTRIBUTION", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        for p in [50, 75, 90, 95, 99]:
            print(f"  P{p:2d}: {np.percentile(scores, p):.2f}", file=sys.stderr)
        print(f"  Max: {max(scores):.2f}  Min: {min(scores):.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
