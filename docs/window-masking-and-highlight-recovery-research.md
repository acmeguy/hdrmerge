# Window Masking, Highlight Identification & Exposure Fusion Research

> Research compiled 2026-02-23. Covers techniques for identifying and recovering
> blown highlights (especially windows) in bracketed HDR imaging, and modern
> alternatives to classical exposure fusion.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background Concepts](#2-background-concepts)
   - 2.1 [Luminosity Masks](#21-luminosity-masks)
   - 2.2 [Mertens Exposure Fusion (2007)](#22-mertens-exposure-fusion-2007)
   - 2.3 [Window Pull Technique](#23-window-pull-technique)
3. [How HDRMerge Handles This Today](#3-how-hdrmerge-handles-this-today)
4. [Non-Generative / Deterministic Approaches](#4-non-generative--deterministic-approaches)
   - 4.1 [Extended Exposure Fusion (EEF) — Deep Dive](#41-extended-exposure-fusion-eef--deep-dive)
   - 4.2 [Improved Weight Functions (AWE + 3D Gradient)](#42-improved-weight-functions-awe--3d-gradient)
   - 4.3 [Luminance HDR](#43-luminance-hdr)
5. [CNN Regression Models (Non-Generative Neural)](#5-cnn-regression-models-non-generative-neural)
   - 5.1 [SAFNet — Selective Alignment Fusion Network — Deep Dive](#51-safnet--selective-alignment-fusion-network--deep-dive)
   - 5.2 [MEF-Net](#52-mef-net)
   - 5.3 [DeepFuse](#53-deepfuse)
6. [Generative / Diffusion-Based Approaches](#6-generative--diffusion-based-approaches)
   - 6.1 [UltraFusion](#61-ultrafusion)
   - 6.2 [Other Generative HDR Methods](#62-other-generative-hdr-methods)
7. [Window Segmentation Models](#7-window-segmentation-models)
   - 7.1 [SAM 2 / SAM 3 (Segment Anything)](#71-sam-2--sam-3-segment-anything)
   - 7.2 [BiRefNet](#72-birefnet)
   - 7.3 [Roboflow Windows Instance Segmentation](#73-roboflow-windows-instance-segmentation)
8. [Comparison Matrix](#8-comparison-matrix)
9. [Recommendations for HDRMerge](#9-recommendations-for-hdrmerge)
10. [References](#10-references)

---

## 1. Problem Statement

In bracketed HDR photography — especially interiors and architecture — windows
create an extreme dynamic range challenge. The scene outside a window can be
10–15 EV brighter than the interior. Standard HDR merging either:

- Blows out the windows entirely (favoring interior detail)
- Produces halos or artifacts at window edges from tone mapping
- Loses exterior detail if the bracket range doesn't cover it

The goal is to **identify** these bright regions (windows, skylights, direct
light sources) and **recover** detail from the shortest available exposure,
blending it seamlessly with the longer-exposure interior data.

This document surveys the state of the art in both **region identification**
(finding the windows) and **highlight recovery/fusion** (blending the right
exposure data into those regions).

---

## 2. Background Concepts

### 2.1 Luminosity Masks

Masks derived directly from an image's brightness values. A "Lights" mask
selects the brightest tones, "Darks" the darkest. They create smooth, feathered
selections based on luminance:

```
w(x, y) = f(L(x, y))
```

where `f` is a monotonic function mapping luminance to mask weight.

**Strengths:** Smooth transitions, no training data needed, works on any image.
**Weaknesses:** Purely luminance-based — a bright wall gets the same treatment as
a window. No semantic awareness.

**Reference:** [Greg Benz — HDR vs Luminosity Masks](https://gregbenzphotography.com/photography-tips/hdr-vs-luminosity-masks/)

### 2.2 Mertens Exposure Fusion (2007)

The classical algorithm for blending multiple exposures without tone mapping.
Computes three per-pixel quality measures for each exposure:

1. **Contrast (C):** Absolute value of Laplacian filter on grayscale — captures
   edges and texture.
2. **Saturation (S):** Standard deviation across R, G, B channels at each pixel
   — prefers vivid color.
3. **Well-exposedness (E):** Gaussian centered at 0.5:
   ```
   E(I) = exp(-0.5 * ((I - 0.5) / sigma)^2)
   ```
   Penalizes values near 0 (underexposed) or 1 (overexposed).

Combined weight per pixel:
```
W(x, y) = C^wc * S^ws * E^we
```
where `wc`, `ws`, `we` are tunable exponents.

Blending uses **Laplacian pyramid** decomposition of each exposure and
**Gaussian pyramid** decomposition of each weight map, fused at each scale level
to prevent seam artifacts.

**Known artifacts:**
- Out-of-range pixel values (values outside [0, 1])
- Low-frequency halos at strong exposure boundaries

**Reference:** [Mertens et al. 2007 (Stanford mirror)](https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf)
**IPOL implementation analysis:** https://www.ipol.im/pub/art/2018/230/

### 2.3 Window Pull Technique

A real estate photography workflow for recovering window detail:

1. **Capture:** Shoot bracketed exposures — one metered for interior (windows
   blown), one metered for exterior view through windows (interior dark). Often
   a flash frame is added aimed at 45 degrees to the window.

2. **Identify:** Select the darkest exposure (or a dedicated window-pull frame)
   that has exterior detail through the windows.

3. **Mask:** Create a mask isolating just the window regions:
   - Manual brush painting
   - Luminance range mask (select pixels above a brightness threshold)
   - **Darken blend mode:** `output = min(base, pull)` per-channel — naturally
     self-masking, only replaces truly blown regions
   - Feather/blur the mask edges to avoid hard transitions

4. **Blend:** Composite the window-pulled frame onto the HDR-merged base using
   the mask.

The darken blend mode approach is elegant because it requires no explicit mask
creation — only truly bright (blown) regions get replaced.

**Reference:** [RePhoto Journal — How to do a Window Pull](https://rephotojournal.com/how-to-do-a-window-pull/)

---

## 3. How HDRMerge Handles This Today

HDRMerge's `ImageStack::compose()` (`src/ImageStack.cpp:1381`) already implements
per-pixel exposure selection with saturation-aware weighting:

- **Per-channel saturation thresholds** (`satThreshPerCh[4]`) define where each
  Bayer channel clips.
- **Rolloff weighting** begins at 90% of the saturation threshold, creating a
  smooth transition zone rather than a hard cutoff.
- **Bayer-block rolloff consistency** uses the most conservative (minimum)
  threshold across all channels to prevent color fringe at exposure transitions
  after demosaicing.
- **Feathered mask** (`BoxBlur` of the exposure selection map) softens spatial
  transitions between exposures.

This is conceptually similar to a luminosity mask + window pull: saturated pixels
in longer exposures get replaced by shorter-exposure data. Key differences from a
dedicated window-pull approach:

| Current HDRMerge | Window-Pull Enhancement |
|---|---|
| Per-pixel saturation threshold | Region-aware (spatial coherence) |
| Hard rolloff at ~0.9 × satThreshold | Smooth luminosity-weighted blend |
| No semantic awareness | Could identify window *regions* vs. specular highlights |
| Single merge pass | Could allow separate treatment of highlight regions |

---

## 4. Non-Generative / Deterministic Approaches

### 4.1 Extended Exposure Fusion (EEF) — Deep Dive

**Authors:** Charles Hessel & Jean-Michel Morel
**Venue:** WACV 2020 + IPOL 2019 (peer-reviewed, reproducible)
**License:** BSD-style (IPOL)
**Language:** MATLAB/Octave (reference implementation), 91.5% MATLAB + 8.5% Shell
**Online demo:** https://ipolcore.ipol.im/demo/clientApp/demo.html?id=278

#### What Mertens Gets Wrong

Standard Mertens exposure fusion has two well-known artifacts:

1. **Out-of-range artifact:** The Laplacian pyramid blending can produce pixel
   values outside [0, 1] — especially at strong exposure boundaries like window
   edges. Clamping these creates visible discontinuities.

2. **Low-frequency halo:** The multi-scale blending propagates weight
   differences at coarse pyramid levels, creating broad halos around high-contrast
   transitions (exactly the window-to-interior boundary).

Both artifacts are most visible precisely where HDR scenes have the highest
dynamic range — window edges, bright sky transitions, direct light sources.

#### The EEF Solution: Remapping to Restrained Dynamic Range

The core innovation is a **remapping function** `g` that transforms each input
bracket into multiple images with *restrained* (narrower) dynamic range:

**Step 1 — Generate extended sequence:**
For each of the N input brackets, generate M new images using the remapping
function, expanding the sequence from N to M x N images:

```
M = ceil(1 / beta)
```

where **beta** (default 0.3) controls the width of the restrained range. With
beta=0.3, each input bracket produces ceil(1/0.3) = 4 remapped versions.

For each bracket n and remapping index k:
```
seq(n, k) = clamp(g(input_n, k), 0, 1)
```

The remapping function `g` centers each version on a different tonal range
(k/M, (k+1)/M, etc.) and uses a decay that falls off like 1/x² outside the
restrained range. The decay is controlled by **lambda** (default 0.125). A
smooth decay avoids creating false edges that would propagate into the final
image.

**Why this helps highlights:** A window region that is fully clipped (value=1.0)
in the long exposure gets remapped into M versions — the versions centered on
lower tonal ranges effectively "ignore" the blown region (it falls far outside
their restrained range and gets near-zero weight), while the versions centered
on high values still capture whatever structure exists near the clip point. When
combined with the short exposure's remapped versions (which capture the exterior
detail), the Laplacian pyramid has a much richer set of well-behaved inputs to
blend — no single image needs to span the full dynamic range.

**Step 2 — Compute weights:**
The same Mertens quality measures (contrast, saturation, well-exposedness) are
applied, but now on the restrained-range images. The well-exposedness Gaussian
naturally gives near-zero weight to values outside the restrained range, so each
remapped image only contributes where its tonal range is valid.

The `improve` flag (default: on) uses IPOL-improved weights that further
suppress contributions from outside the restrained range, eliminating the
out-of-range artifact.

**Step 3 — Laplacian pyramid blending:**
Standard multi-scale blending on the extended M x N sequence. Because each
image has a restrained range, the Laplacian pyramid coefficients are smaller
and better-behaved — the low-frequency halo is suppressed because no single
image forces a large coarse-scale correction.

#### Configurable Parameters

| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| `beta` | (0, 1] | 0.3 | Restrained dynamic range width. Lower = more images, finer control. |
| `Wsat` | % | 1 | Maximum white-saturated pixels allowed before discarding. |
| `Bsat` | % | 1 | Maximum black-saturated pixels allowed before discarding. |
| `nScales` | int | 0 (auto) | Laplacian pyramid depth. -1 = autoMin, -2 = autoMax. |
| `improve` | 0/1 | 1 | Use IPOL-improved weights (recommended) vs. WACV-original. |
| `lambda` | float | 0.125 | Remapping decay speed. Controls smoothness of tonal transitions. |

#### Usage

```bash
# Requires GNU Octave 4.0+ with image package
pkg install -forge image

# Run EEF on a bracketed sequence
octave -W -qf runeef.m --beta 0.3 --improve 1 bracket_dark.jpg bracket_mid.jpg bracket_bright.jpg
```

Outputs: fused PNG, weight maps (for visualization), remapping function plots.
Also includes an optional registration (alignment) script with homography
estimation.

#### Relevance to HDRMerge

EEF operates on tone-mapped 8-bit images, not linear RAW. However, the core
ideas are transferable to `compose()`:

1. **Restrained-range remapping** — Analogous to applying the saturation rolloff
   at multiple threshold levels rather than a single 0.9 x satThreshold. Each
   "virtual threshold" would produce a weight contribution, blended via the
   existing feather/blur mechanism.

2. **Smooth decay functions** — The 1/x² decay with lambda control is more
   principled than the current linear rolloff in the saturation transition zone.
   Could replace the linear ramp between `satRolloffPerCh` and `satThreshPerCh`.

3. **Per-scale weight normalization** — The pyramid-based weight normalization
   is what eliminates halos. HDRMerge's current BoxBlur feathering operates at a
   single scale; a multi-scale approach could reduce any remaining halo artifacts
   at exposure transitions.

| Resource | Link |
|---|---|
| WACV 2020 paper | https://openaccess.thecvf.com/content_WACV_2020/papers/Hessel_An_Extended_Exposure_Fusion_and_its_Application_to_Single_Image_WACV_2020_paper.pdf |
| IPOL article (peer-reviewed, reproducible) | https://www.ipol.im/pub/art/2019/278/ |
| Online demo (try it in browser) | https://ipolcore.ipol.im/demo/clientApp/demo.html?id=278 |
| GitHub — EEF implementation | https://github.com/chlsl/extended-exposure-fusion-ipol |
| GitHub — Simulated EF variant | https://github.com/chlsl/simulated-exposure-fusion-ipol |
| Demo archive (prior results) | https://ipolcore.ipol.im/demo/clientApp/archive.html?id=278 |

### 4.2 Improved Weight Functions (AWE + 3D Gradient)

**Venue:** Frontiers in Neurorobotics, 2022

Proposes two replacement quality metrics for Mertens' contrast/saturation/well-exposedness:

1. **Adaptive Well-Exposedness (AWE):** Adjusts the well-exposedness Gaussian
   per-exposure based on the actual exposure level, rather than using a fixed
   center at 0.5.
2. **3D Color Gradient:** Replaces the grayscale Laplacian contrast measure with
   a gradient computed across all three color channels simultaneously.

Results retain more detail in highlights and shadows compared to Mertens,
especially fine detail in highlight regions.

**Caveat:** No open-source code found. The paper describes the algorithm
completely enough to implement, but would require custom work.

| Resource | Link |
|---|---|
| Paper | https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846580/full |
| PMC full text | https://pmc.ncbi.nlm.nih.gov/articles/PMC8957254/ |

### 4.3 Luminance HDR

**License:** GPL
**Language:** C++ (Qt5)

Open-source HDR application with 15 tone-mapping operators (Mantiuk '06 & '08,
Fattal, Drago, Reinhard, etc.). Provides a complete HDR pipeline with:

- Debevec/Robertson HDR assembly
- Batch processing and CLI mode
- Cross-platform (Linux, macOS, Windows)

Not directly an exposure fusion tool, but useful as a reference implementation
for classical HDR algorithms.

| Resource | Link |
|---|---|
| GitHub | https://github.com/LuminanceHDR/LuminanceHDR |
| Website | https://luminancehdr.com/ |

---

## 5. CNN Regression Models (Non-Generative Neural)

These use convolutional neural networks to predict weight maps or directly
regress the fused output. **No diffusion, no hallucination** — the output is
derived from the input pixels only. Inference is deterministic given the same
input.

### 5.1 SAFNet — Selective Alignment Fusion Network — Deep Dive

**Authors:** Lingtong Kong, Bo Li, et al.
**Venue:** ECCV 2024
**License:** Academic research only (no commercial use)
**Language:** Python / PyTorch
**Training code:** Not released (company restriction); inference + weights available.

#### Core Philosophy: Discard, Don't Inpaint

The central insight is that **not all regions in non-reference frames are worth
precise alignment**. Specifically:

- Regions that are **overexposed** in a non-reference frame — worthless, discard
- Regions that are **underexposed** — worthless, discard
- Regions that **correspond to well-exposed texture already in the reference** —
  redundant, discard
- Regions with **valuable texture missing from the reference** — align and fuse

This is the opposite of UltraFusion's generative approach and closely mirrors
what HDRMerge already does: prefer the reference exposure, only pull from other
exposures where they add information the reference lacks.

#### Architecture Detail

**Input:** Three LDR exposures (under, middle/reference, over) as .tif files,
plus an exposure.txt with EV values. The middle exposure (L₂) is the reference.

**Preprocessing — Gamma linearization:**
```
H_i = L_i^gamma / t_i    (gamma = 2.2)
```
Each frame is concatenated as a 6-channel tensor: [LDR, linearized].

**Stage 1 — Pyramid Encoder:**
- 4-level feature pyramid
- Each level: two 3x3 convolutions (strides 2 and 1), followed by PReLU
- 8 convolution layers total, 40 feature channels at all scales
- Compact design: only 1.12M parameters total

**Stage 2 — Coarse-to-Fine Decoder (the key contribution):**

Iterates from coarsest (level 4) to finest (level 1). At each level k, the
decoder jointly predicts:

- **Optical flow** F₂→₁ and F₂→₃ (motion from reference to each non-reference)
- **Selection masks** M₁ and M₃ (per-pixel probability of using each non-ref)

The selection masks are single-channel tensors with sigmoid output (values in
[0, 1]). The decoder uses:
- 5 x 3x3 convolutions + 1 x 4x4 deconvolution per block
- Group convolutions (group=3) with channel shuffle for efficiency
- Non-reference features are backward-warped using flow from coarser levels

**The joint refinement is mutually beneficial:**
- Better selection masks → flow estimation focuses on useful regions only
- Better flow → warped features help identify which regions have valuable texture

**Stage 3 — Merging with Selection-Weighted Reweighting:**

The selection masks modify the fusion weights. Given base weights Λ₁, Λ₂, Λ₃
(from exposure times/linearization):

```
W₁ = Λ₁ * M₁              (non-ref 1: kept only where mask says "valuable")
W₃ = Λ₃ * M₃              (non-ref 3: kept only where mask says "valuable")
W₂ = Λ₂ + Λ₁*(1-M₁) + Λ₃*(1-M₃)   (reference absorbs discarded weight)
```

This is elegant: weight that would have gone to a discarded non-reference region
is **transferred to the reference frame**. The reference always gets at least its
base weight, plus any weight from regions the other frames can't contribute to.

The merged HDR image: `H_m = W₁*H₁_warped + W₂*H₂ + W₃*H₃_warped`

**Stage 4 — Detail Refine Module:**
- Three independent 2-layer feature extractors (one per LDR input + fused result)
- Aligns non-reference features using the predicted optical flow
- 5 dilated residual blocks + 1 convolution for residual detail estimation
- Input includes flow, selection masks, and merged HDR for guidance

#### Loss Functions

```
Total loss:  L = L_r + 0.1 * L_m
```

**Refined output loss (L_r):** L1 + perceptual loss on mu-law tonemapped domain:
```
L_r = L1(T(H_r), T(H_gt)) + 0.01 * L_perceptual(T(H_r), T(H_gt))
```
where T is mu-law tonemapping with mu=5000.

**Merged image auxiliary loss (L_m):** L1 + census loss (7x7 patches):
```
L_m = L1(T(H_m), T(H_gt)) + L_census(T(H_m), T(H_gt))
```

The census loss enforces structural consistency via local binary pattern
matching — robust to brightness differences.

#### Performance Numbers

On 1500x1000 resolution (NVIDIA A30 GPU):

| Model | Time | Parameters | FLOPs |
|---|---|---|---|
| **SAFNet** | **0.151s** | **1.12M** | **0.976T** |
| SAFNet-S (small) | 0.049s | 0.33M | 0.198T |
| SCTNet (prior SOTA) | 3.466s | 1.67M | — |
| HDR-Transformer | 2.673s | 1.12M | — |

SAFNet is **23x faster** than SCTNet and **18x faster** than HDR-Transformer.

**Quality (Kalantari 2017 test set):**

| Metric | SAFNet | vs. FlexHDR | vs. SCTNet |
|---|---|---|---|
| PSNR-l (dB) | 43.18 | +0.58 | +0.89 |
| SSIM-l | 0.9917 | +0.0015 | +0.0030 |
| PSNR-mu | 44.66 | +0.31 | +0.17 |
| HDR-VDP2 | 66.93 | +0.37 | +0.28 |

UltraFusion (CVPR 2025) outperforms SAFNet on perceptual metrics (MUSIQ 68.82
vs. 61.70) but at much higher computational cost and with generative
hallucination tradeoffs.

#### How It Handles Highlight/Window Regions

The selection masks **implicitly learn** to discard overexposed regions. The
paper observes that "motion estimation in regions with distinct texture is much
easier than in saturated areas" — so the network learns that saturated (blown)
regions yield unreliable flow and automatically assigns them low selection
probability, transferring their weight to the reference frame.

This means window regions in an overexposed bracket get M≈0 (discarded), and
their weight flows to the reference frame's W₂ term. The detail for those
regions comes from whichever exposure has the reference role, or from the
underexposed bracket if it has high M and good flow.

#### Running Inference

```bash
git clone https://github.com/ltkong218/SAFNet.git
cd SAFNet
pip install torch thop pynvml

# Pretrained weights are in checkpoints/
python eval_SAFNet_siggraph17.py
```

**Input format:** Three .tif LDR images + exposure.txt. Can run on CPU (set
device accordingly), though much slower.

#### Relevance to HDRMerge

SAFNet's selection-mask-weighted reweighting formula is directly applicable:

1. **The reweighting concept** — Weight from "useless" non-reference regions
   flows back to the reference. HDRMerge's `compose()` already does a version of
   this (saturated pixels fall back to shorter exposures), but SAFNet's approach
   is continuous (sigmoid probability) rather than threshold-based.

2. **Spatial coherence** — SAFNet's masks are spatially coherent (nearby pixels
   tend to have similar selection probabilities) because the CNN operates on
   local neighborhoods. HDRMerge's per-pixel threshold + BoxBlur feathering
   approximates this but is less sophisticated.

3. **Joint mask-flow refinement** — The idea that "where to look" and "how to
   align" should be solved together is relevant for HDRMerge's sub-pixel
   alignment + exposure selection pipeline.

4. **What to extract without the neural network** — Even without running SAFNet,
   the reweighting formula `W_ref = Λ_ref + Σ Λ_i * (1 - M_i)` is a useful
   design pattern. If HDRMerge computed a continuous "usefulness" score per pixel
   per exposure (based on distance from saturation, local contrast, etc.), the
   same reweighting could improve exposure transitions.

| Resource | Link |
|---|---|
| Paper (arXiv) | https://arxiv.org/abs/2407.16308 |
| Paper (ECCV PDF) | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03782.pdf |
| Code + weights | https://github.com/ltkong218/SAFNet |

### 5.2 MEF-Net

**Venue:** IEEE TIP, 2019
**Language:** Python

Predicts weight maps at low resolution using a fully convolutional network, then
upsamples with guided filtering for full-resolution fusion.

**Key features:**
- Trained end-to-end with perceptual **MEF-SSIM** loss (no ground truth HDR
  images needed for training).
- Very fast: 10–1000x faster than classical methods.
- Lightweight architecture suitable for edge deployment.

| Resource | Link |
|---|---|
| Paper | https://kedema.org/paper/19_TIP_MEF-Net.pdf |
| Code | https://github.com/makedede/MEFNet |

### 5.3 DeepFuse

**Venue:** ICCV 2017
**Language:** Python

First unsupervised deep learning method for multi-exposure fusion. Uses a
no-reference quality metric as loss function — no paired training data required.
Handles extreme exposure pairs.

Older but proven; established the paradigm that SAFNet and MEF-Net build on.

| Resource | Link |
|---|---|
| Paper | https://ar5iv.labs.arxiv.org/html/1712.07384 |

---

## 6. Generative / Diffusion-Based Approaches

These use generative models (typically Stable Diffusion) and can **hallucinate
detail** that wasn't present in any input frame. Powerful but non-deterministic
and not suitable for linear RAW pipelines.

### 6.1 UltraFusion

**Authors:** Zixuan Chen et al. (Shanghai AI Lab, ZJU, CUHK)
**Venue:** CVPR 2025 (Highlight)
**License:** GPL-3.0

The state-of-the-art for multi-exposure fusion. Handles up to **9 stops** of
exposure difference.

**Core idea:** Reframes exposure fusion as **guided inpainting** — the
overexposed image is the "damaged" input, and the underexposed image provides
soft guidance for what blown-out regions should look like. Built on Stable
Diffusion V2.1.

**Architecture (3 stages):**

1. **Pre-alignment via RAFT optical flow** — Bidirectional motion estimation
   between exposures. Forward-backward consistency check creates an occlusion
   mask for regions where alignment failed.

2. **Decompose-and-Fuse Control Branch (DFCB)** — Decomposes the underexposed
   guide into:
   - Structure: normalized luminance (SSIM-like structural component)
   - Color: UV chrominance channels (YUV space)

   Injected into the SD U-Net via multi-scale cross-attention as *soft guidance*
   rather than hard pixel constraints. This makes it robust to misalignment and
   lighting variation.

3. **Fidelity Control Branch (FCB)** — Direct shortcut connections to the VAE
   decoder to preserve fine texture lost in latent-space compression.

**Training:**
- Synthetic pairs from Vimeo-90K (motion) + SICE (multi-exposure)
- DFCB: diffusion reconstruction loss
- FCB: L1 reconstruction loss

**Limitations:**
- ~3.3 seconds per 512x512 on RTX 4090 (not real-time)
- Large motion + occlusion degrades to single-image HDR
- **Operates on tone-mapped 8-bit imagery, not linear RAW** — cannot be a
  drop-in replacement for HDRMerge's linear DNG pipeline
- Generative: can hallucinate detail not present in any input

**Status:** HuggingFace demo is currently broken (ModuleNotFoundError). Must run
locally with CUDA GPU or via Google Colab.

| Resource | Link |
|---|---|
| Paper (arXiv HTML) | https://arxiv.org/html/2501.11515v3 |
| Paper (CVPR PDF) | https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_UltraFusion_Ultra_High_Dynamic_Imaging_using_Exposure_Fusion_CVPR_2025_paper.pdf |
| Code + training | https://github.com/OpenImagingLab/UltraFusion |
| Pre-trained weights | HuggingFace: `zxchen00/UltraFusion` |
| Dataset (UltraFusion100) | Google Drive (linked in repo) |
| Project page | https://openimaginglab.github.io/UltraFusion/ |

### 6.2 Other Generative HDR Methods

| Project | Year | Venue | Notes | Code |
|---|---|---|---|---|
| HDRFlow | 2024 | CVPR | Real-time HDR video, flow-based alignment | https://github.com/OpenImagingLab/HDRFlow |
| SelfHDR | 2024 | ICLR | Self-supervised, no paired training data | https://github.com/cszhilu1998/SelfHDR |
| SCTNet | 2023 | ICCV | Alignment-free deghosting, semantic transformer | https://github.com/Zongwei97/SCTNet |
| Joint-HDRDN | 2023 | CVPR | Joint denoising + fusion, mobile HDR dataset | https://github.com/shuaizhengliu/Joint-HDRDN |
| Deep-HdrReconstruction | 2020 | SIGGRAPH | Masked features for saturated regions (single image) | https://github.com/marcelsan/Deep-HdrReconstruction |
| VinAI single_image_hdr | 2023 | WACV | Virtual multi-exposure from single image | https://github.com/VinAIResearch/single_image_hdr |

---

## 7. Window Segmentation Models

These identify window **regions** semantically, independent of luminance. Useful
as a preprocessing step to generate masks for region-specific exposure selection.

### 7.1 SAM 2 / SAM 3 (Segment Anything)

**Author:** Meta AI
**License:** Apache 2.0

Zero-shot segmentation — prompt with a point, box, or text concept on a window
and it segments it. No fine-tuning needed.

- **SAM 2** (July 2024): Real-time image and video segmentation. ViT-B/L/H
  encoders. Weights publicly available.
- **SAM 3** (Nov 2025): Adds concept-based prompting — can segment by text
  description ("window") without any spatial prompt.

**Practical use:** Run as a preprocessing step to generate a binary window mask,
then use that mask to apply different exposure selection logic in HDRMerge's
`compose()` (e.g., always prefer the shortest non-clipped exposure inside
windows, with aggressive feathering at edges).

| Resource | Link |
|---|---|
| SAM 2 code + weights | https://github.com/facebookresearch/segment-anything-2 |
| SAM 3 announcement | https://ai.meta.com/blog/segment-anything-model-3/ |
| Original SAM | https://github.com/facebookresearch/segment-anything |

### 7.2 BiRefNet

**Venue:** CAAI AIR 2024
**License:** Open source (HuggingFace)

Bilateral Reference Network for high-resolution dichotomous image segmentation.
Good at separating foreground from background with clean edges. Multiple model
sizes available on HuggingFace, plus ONNX exports.

Could segment bright window regions from dark interiors as a binary
foreground/background task.

| Resource | Link |
|---|---|
| Code + weights | https://github.com/ZhengPeng7/BiRefNet |

### 7.3 Roboflow Windows Instance Segmentation

A YOLOv7 model trained specifically on 1,345 labeled window images. Directly
detects and segments windows as objects.

**Strengths:** Purpose-built for the exact task. Fast inference. Free tier
available.
**Weaknesses:** Small training set (1,345 images). May not generalize well to all
window types, lighting conditions, or architectural styles.

| Resource | Link |
|---|---|
| Dataset + model | https://universe.roboflow.com/roboflow-universe-projects/windows-instance-segmentation |

---

## 8. Comparison Matrix

| Method | Type | GPU Required | Deterministic | Open Weights | Operates on RAW/Linear | Highlight Quality | Complexity |
|---|---|---|---|---|---|---|---|
| **EEF** | Classical | No | Yes | N/A | Adaptable | Good | Low |
| **AWE + 3D Gradient** | Classical | No | Yes | N/A | Adaptable | Good | Low (no code) |
| **Luminance HDR** | Classical | No | Yes | N/A | Yes (HDR pipeline) | Good | Medium |
| **SAFNet** | CNN regression | Optional | Yes* | Yes | No (8-bit) | Very good | Medium |
| **MEF-Net** | CNN regression | Optional | Yes* | Yes | No (8-bit) | Good | Low |
| **DeepFuse** | CNN regression | Optional | Yes* | Partial | No (8-bit) | Good | Low |
| **UltraFusion** | Generative (SD) | Yes (CUDA) | No | Yes | No (8-bit) | Excellent | High |
| **SAM 2/3** | Segmentation | Optional | Yes* | Yes (Apache 2.0) | N/A (mask only) | N/A | Medium |
| **BiRefNet** | Segmentation | Optional | Yes* | Yes | N/A (mask only) | N/A | Medium |
| **Roboflow Windows** | Segmentation | Optional | Yes* | Yes | N/A (mask only) | N/A | Low |

\* Deterministic at inference time given the same input and weights.

---

## 9. Recommendations for HDRMerge

### Near-term: Study EEF's weight improvements

Extended Exposure Fusion is the most directly applicable research. It's
classical, deterministic, has C++ code, and specifically targets the highlight
artifacts that Mertens (and by extension HDRMerge's current approach) suffers
from. The "extended exposure sequence" concept — generating restrained-dynamic-range
variants of each bracket before fusion — could inform improvements to the
saturation rolloff weighting in `compose()`.

### Medium-term: Evaluate SAFNet's selection masks

SAFNet's "discard bad regions" philosophy aligns closely with HDRMerge's existing
approach. Running SAFNet on a few test brackets would show what a learned
selection mask looks like compared to HDRMerge's threshold-based mask. If the
learned masks are meaningfully better (especially at window boundaries), it would
motivate implementing similar spatial-coherence heuristics in `compose()` without
requiring a neural network at runtime.

### Optional: Window segmentation as preprocessing

If specific window-region treatment is desired (e.g., always use shortest
exposure in window areas, different feathering radius), SAM 2/3 could generate
per-image window masks as a preprocessing step. This would be a separate tool
invoked before HDRMerge, producing a mask image that `compose()` could
optionally consume. This is the most architecturally clean separation of concerns
but adds workflow complexity.

### Not recommended for HDRMerge integration

UltraFusion and other generative approaches operate on tone-mapped 8-bit
imagery, produce non-deterministic output, and can hallucinate detail. They are
conceptually interesting but architecturally incompatible with HDRMerge's linear
RAW DNG pipeline where bit-accurate, deterministic output is paramount.

---

## 10. References

1. Mertens, T., Kautz, J., & Van Reeth, F. (2007). "Exposure Fusion." *Pacific Graphics.*
   https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf

2. Hessel, C. & Morel, J.-M. (2019). "Extended Exposure Fusion." *IPOL.*
   https://www.ipol.im/pub/art/2019/278/

3. Hessel, C. & Morel, J.-M. (2020). "An Extended Exposure Fusion and its Application to Single Image Contrast Enhancement." *WACV.*
   https://openaccess.thecvf.com/content_WACV_2020/papers/Hessel_An_Extended_Exposure_Fusion_and_its_Application_to_Single_Image_WACV_2020_paper.pdf

4. Kong, L. et al. (2024). "SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging." *ECCV.*
   https://arxiv.org/abs/2407.16308

5. Chen, Z. et al. (2025). "UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion." *CVPR.*
   https://arxiv.org/abs/2501.11515

6. Ma, K., Li, H., Yong, H., Wang, Z., Meng, D., & Zhang, L. (2019). "Deep Guided Learning for Fast Multi-Exposure Image Fusion." *IEEE TIP.*
   https://kedema.org/paper/19_TIP_MEF-Net.pdf

7. Ram Prabhakar, K., Srikar, V., & Babu, R.V. (2017). "DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs."
   https://ar5iv.labs.arxiv.org/html/1712.07384

8. Li, Y. et al. (2022). "Multi-Exposure Image Fusion Algorithm Based on Improved Weight Function." *Frontiers in Neurorobotics.*
   https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846580/full

9. Kirillov, A. et al. (2023). "Segment Anything." *Meta AI.*
   https://github.com/facebookresearch/segment-anything

10. Zheng, P. et al. (2024). "BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation."
    https://github.com/ZhengPeng7/BiRefNet

11. Awesome HDR Imaging paper collection:
    https://github.com/rebeccaeexu/Awesome-High-Dynamic-Range-Imaging
