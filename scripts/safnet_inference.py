#!/usr/bin/env python3
"""Run SAFNet inference on a prepared bracket and save HDR + diagnostic masks.

Outputs:
  - hdr_merged.hdr       — merged HDR (before refinement)
  - hdr_refined.hdr      — refined HDR (final output)
  - mask0.png            — selection mask for underexposed frame (0=discard, 255=use)
  - mask2.png            — selection mask for overexposed frame
  - mask0_color.png      — colorized mask overlay on reference
  - mask2_color.png      — colorized mask overlay on reference
  - flow0_vis.png        — optical flow visualization (under -> ref)
  - flow2_vis.png        — optical flow visualization (over -> ref)
  - weights_vis.png      — final per-frame weight map (R=under, G=mid, B=over)

Usage:
  python3 safnet_inference.py <input_dir> <output_dir> [--safnet-root /path/to/SAFNet]
"""

import argparse
import os
import sys
import math
import numpy as np
import cv2
import torch


def get_device():
    """Pick the best available device.

    MPS doesn't support grid_sample with border padding (used by SAFNet's warp),
    so we fall back to CPU on Apple Silicon. The model is tiny (1.12M params) so
    CPU is fine for inference.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def read_ldr(img_path):
    """Read 16-bit TIFF as float32 in [0, 1], RGB order."""
    img = np.asarray(cv2.imread(img_path, -1)[:, :, ::-1])
    img = (img / 2**16).clip(0, 1).astype(np.float32)
    return img


def read_expos(txt_path):
    """Read exposure.txt and return normalized exposure ratios."""
    vals = np.loadtxt(txt_path)
    expos = np.power(2, vals - vals.min()).astype(np.float32)
    return expos


def flow_to_color(flow, max_flow=None):
    """Convert optical flow to HSV color visualization."""
    flow_u = flow[0]
    flow_v = flow[1]
    mag = np.sqrt(flow_u**2 + flow_v**2)
    if max_flow is None:
        max_flow = mag.max() + 1e-6
    ang = np.arctan2(flow_v, flow_u)

    hsv = np.zeros((*flow_u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def mask_overlay(ref_img, mask, color=(0, 0, 255)):
    """Overlay mask on reference image. mask in [0,1]."""
    ref_uint8 = (ref_img * 255).clip(0, 255).astype(np.uint8)
    overlay = ref_uint8.copy()
    color_layer = np.full_like(ref_uint8, color)
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    overlay = (ref_uint8 * (1 - mask_3ch * 0.5) + color_layer * mask_3ch * 0.5).clip(0, 255).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Run SAFNet inference with diagnostic output")
    parser.add_argument("input_dir", help="Directory with ldr_img_*.tif and exposure.txt")
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument("--safnet-root", default="/Users/stefanbaxter/Development/SAFNet",
                        help="Path to SAFNet repository")
    parser.add_argument("--scale-factor", type=float, default=0.5,
                        help="Internal processing scale (default 0.5)")
    args = parser.parse_args()

    # Add SAFNet to Python path
    sys.path.insert(0, args.safnet_root)

    from models.SAFNet import SAFNet
    from utils import merge_hdr, range_compressor

    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load inputs
    print("Loading inputs...")
    img0 = read_ldr(os.path.join(args.input_dir, "ldr_img_1.tif"))
    img1 = read_ldr(os.path.join(args.input_dir, "ldr_img_2.tif"))
    img2 = read_ldr(os.path.join(args.input_dir, "ldr_img_3.tif"))
    expos = read_expos(os.path.join(args.input_dir, "exposure.txt"))
    print(f"  Image size: {img1.shape[1]}x{img1.shape[0]}")
    print(f"  Exposures (normalized): {expos}")

    # Linearize (gamma 2.2 inverse)
    imgs_ldr = [img0, img1, img2]
    imgs_lin = [(ldr ** 2.2) / e for ldr, e in zip(imgs_ldr, expos)]

    # Build 6-channel input tensors: [linear, ldr]
    def make_input(lin, ldr):
        combined = np.concatenate([lin, ldr], axis=2)  # (H, W, 6)
        return torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).to(device)

    img0_c = make_input(imgs_lin[0], imgs_ldr[0])
    img1_c = make_input(imgs_lin[1], imgs_ldr[1])
    img2_c = make_input(imgs_lin[2], imgs_ldr[2])

    # Load model
    print("Loading SAFNet model...")
    model = SAFNet().to(device).eval()
    ckpt_path = os.path.join(args.safnet_root, "checkpoints", "SAFNet_siggraph17.pth")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Run inference — get flow, masks, and HDR outputs
    print("Running inference...")
    with torch.no_grad():
        # Step 1: Get flow and selection masks
        flow0, flow2, mask0, mask2 = model.forward_flow_mask(
            img0_c, img1_c, img2_c, scale_factor=args.scale_factor
        )

        # Step 2: Full forward pass for HDR outputs
        img_hdr_m, img_hdr_r = model(img0_c, img1_c, img2_c,
                                      scale_factor=args.scale_factor, refine=True)

    print("  Done!")

    # === Save outputs ===

    # HDR images
    hdr_m = img_hdr_m[0].cpu().permute(1, 2, 0).numpy()
    hdr_r = img_hdr_r[0].cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(args.output_dir, "hdr_merged.hdr"), hdr_m[:, :, ::-1])
    cv2.imwrite(os.path.join(args.output_dir, "hdr_refined.hdr"), hdr_r[:, :, ::-1])
    print("  Saved hdr_merged.hdr, hdr_refined.hdr")

    # Tonemapped previews (mu-law)
    for name, hdr in [("merged", hdr_m), ("refined", hdr_r)]:
        tm = np.log(1 + 5000 * hdr) / math.log(1 + 5000)
        tm_uint8 = (tm * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, f"preview_{name}.png"), tm_uint8[:, :, ::-1])
    print("  Saved preview_merged.png, preview_refined.png")

    # Selection masks (the key diagnostic)
    m0 = mask0[0, 0].cpu().numpy()
    m2 = mask2[0, 0].cpu().numpy()
    cv2.imwrite(os.path.join(args.output_dir, "mask0_under.png"),
                (m0 * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, "mask2_over.png"),
                (m2 * 255).clip(0, 255).astype(np.uint8))
    print("  Saved mask0_under.png (bright=use underexposed), mask2_over.png (bright=use overexposed)")

    # Colorized mask overlays on reference
    ref_ldr = img1[:, :, ::-1]  # BGR for display
    overlay0 = mask_overlay(img1, m0, color=(0, 100, 255))  # Orange = under
    overlay2 = mask_overlay(img1, m2, color=(255, 100, 0))  # Blue = over
    cv2.imwrite(os.path.join(args.output_dir, "mask0_overlay.png"), overlay0[:, :, ::-1])
    cv2.imwrite(os.path.join(args.output_dir, "mask2_overlay.png"), overlay2[:, :, ::-1])
    print("  Saved mask0_overlay.png, mask2_overlay.png")

    # Flow visualizations
    f0 = flow0[0].cpu().numpy()
    f2 = flow2[0].cpu().numpy()
    max_flow = max(np.sqrt(f0[0]**2 + f0[1]**2).max(),
                   np.sqrt(f2[0]**2 + f2[1]**2).max()) + 1e-6
    cv2.imwrite(os.path.join(args.output_dir, "flow0_vis.png"), flow_to_color(f0, max_flow))
    cv2.imwrite(os.path.join(args.output_dir, "flow2_vis.png"), flow_to_color(f2, max_flow))
    print(f"  Saved flow0_vis.png, flow2_vis.png (max flow: {max_flow:.1f} px)")

    # Weight map visualization (R=under, G=mid, B=over)
    from utils import weight_3expo_low_tog17, weight_3expo_mid_tog17, weight_3expo_high_tog17
    ref_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    w_low = weight_3expo_low_tog17(ref_t) * mask0
    w_mid = weight_3expo_mid_tog17(ref_t) + weight_3expo_low_tog17(ref_t) * (1.0 - mask0) + weight_3expo_high_tog17(ref_t) * (1.0 - mask2)
    w_high = weight_3expo_high_tog17(ref_t) * mask2

    # Normalize weights
    w_sum = w_low + w_mid + w_high + 1e-9
    w_low_n = (w_low / w_sum).mean(dim=1)[0].cpu().numpy()
    w_mid_n = (w_mid / w_sum).mean(dim=1)[0].cpu().numpy()
    w_high_n = (w_high / w_sum).mean(dim=1)[0].cpu().numpy()

    weight_vis = np.stack([
        (w_high_n * 255).clip(0, 255),  # B = overexposed
        (w_mid_n * 255).clip(0, 255),   # G = reference
        (w_low_n * 255).clip(0, 255),   # R = underexposed
    ], axis=-1).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, "weights_vis.png"), weight_vis)
    print("  Saved weights_vis.png (R=under, G=mid/ref, B=over)")

    # Print mask statistics
    print(f"\n--- Selection Mask Statistics ---")
    print(f"  mask0 (under): mean={m0.mean():.3f}, >0.5: {(m0 > 0.5).mean()*100:.1f}% of pixels")
    print(f"  mask2 (over):  mean={m2.mean():.3f}, >0.5: {(m2 > 0.5).mean()*100:.1f}% of pixels")
    print(f"  Pixels where reference dominates (both masks < 0.3): {((m0 < 0.3) & (m2 < 0.3)).mean()*100:.1f}%")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
