# Modern HDR Merge Techniques & Libraries — Research Findings

**Date**: 2026-02-20
**Context**: HDRMerge codebase analysis and modernization research for Apple Silicon (M3/M4)

---

## Table of Contents

1. [Noise-Optimal Merge Weighting](#1-noise-optimal-merge-weighting)
2. [Ghost Detection & Deghosting](#2-ghost-detection--deghosting)
3. [Alignment Improvements Beyond MTB](#3-alignment-improvements-beyond-mtb)
4. [Compression Improvements](#4-compression-improvements)
5. [DNG Format Updates](#5-dng-format-updates)
6. [LibRaw Upgrade Path](#6-libraw-upgrade-path)
7. [Apple Silicon Optimization](#7-apple-silicon-optimization)
8. [Exposure Fusion vs HDR Merge](#8-exposure-fusion-vs-hdr-merge)
9. [Bit Depth & Precision](#9-bit-depth--precision)
10. [Bayer-Domain (Pre-Demosaic) Merge Validation](#10-bayer-domain-pre-demosaic-merge-validation)
11. [Multi-Exposure Super-Resolution](#11-multi-exposure-super-resolution)
12. [Deep Learning HDR Methods](#12-deep-learning-hdr-methods)
13. [Priority Ranking](#13-priority-ranking)
14. [References](#14-references)

---

## 1. Noise-Optimal Merge Weighting

**Status**: HDRMerge currently does binary pixel selection — for each pixel, it scans from brightest to darkest exposure and uses the first non-saturated value. This discards valid data from overlapping well-exposed regions.

### The Problem

When multiple exposures have valid (non-saturated, non-clipped) data at a pixel position, binary selection throws away sqrt(N) potential SNR improvement. For a 3-bracket set where 2 frames share well-exposed mid-tones, weighted averaging provides ~1.4x (3 dB) SNR improvement over using the single best frame.

### Hanji et al. (ECCV 2020) — Poisson Noise Estimator

The most practically relevant paper for our workflow. A **Poisson noise estimator** achieves near-MLE-optimal merge quality *without* camera-specific calibration.

- For linear raw data, weight each exposure's pixel by `w_k = t_k / raw_k` (exposure time / signal level)
- This is the inverse of Poisson noise variance
- Validated across 4 cameras (smartphone to full-frame mirrorless)
- Comparable to full MLE without requiring camera noise parameter databases
- Code: https://github.com/gfxdisp/HDRutils

### Granados et al. (CVPR 2010) — Theoretically Optimal Weight

When read noise is known, the optimal weight is:

```
w_k = t_k² / (t_k × raw_k + σ_read²)
```

Where `t_k` is exposure time, `raw_k` is the pixel value, and `σ_read²` is read noise variance. The MLE estimator using this weight is within ~1% of the Cramer-Rao Lower Bound (CRLB) — essentially information-theoretically optimal.

The Poisson-only approximation (Hanji) is within ~1% of this and needs no calibration.

### Aguerrebere et al. (SIAM J. Imaging Sciences, 2014)

Comprehensive comparison of HDR merge algorithms against the CRLB:

- **MLE and Robertson et al.** produce the least noisy estimates, within ~10% of the CRLB
- **Debevec & Malik hat function** weighting is suboptimal compared to variance-based MLE
- Even with saturated pixels, proper statistical treatment (integrating saturated pixel information rather than discarding) improves estimation

### Camera Noise Model

Based on Cao et al. (CVPR 2023) and Wischow et al. (2024), the noise variance at a pixel with signal level s (in electrons) is:

```
Var(pixel) = s + σ_read² + σ_dark² × t_exposure
```

Where:
- `s` = photon signal (Poisson contribution, dominates in well-exposed regions)
- `σ_read²` = read noise variance (ISO-dependent, dominates in shadows)
- `σ_dark² × t` = dark current (negligible for short exposures on modern sensors)

### Google HDR+ Pipeline (Reference Implementation)

The gold standard for burst merge noise reduction (Hasinoff et al., SIGGRAPH Asia 2016):

1. Tile-based alignment (16x16 tiles, hierarchical)
2. Per-tile pairwise similarity comparison to reference
3. **Wiener filter in frequency domain**: per-tile, per-frequency-bin filtering parameterized by camera noise model
4. All processing on raw Bayer data; demosaicing after merge

The merge logic for overlapping well-exposed regions is directly transferable to an exposure-bracket workflow.

### Practical Implementation

```
For each output pixel (x, y):
  merged_radiance = Σ(w_k × radiance_k) / Σ(w_k)

  where:
    radiance_k = (raw_k - black_level) / t_k
    w_k = 1 / (radiance_k + σ_read² / gain²)   # Full model
    w_k = 1 / radiance_k                          # Poisson-only (no calibration needed)

  Only include exposures where raw_k is:
    - Above black_level + threshold (not clipped in shadows)
    - Below saturation_level - threshold (not saturated)
```

---

## 2. Ghost Detection & Deghosting

### Classical Approaches (Practical for C++ Implementation)

**Sigma Clipping on Radiance-Normalized Values**

Adapted from astrophotography stacking, this is the practical state of the art for a C++ tool:

1. Normalize each bracket to radiance: `radiance_k = (raw_k - black) / t_k`
2. Compute median radiance across brackets
3. Compute MAD (median absolute deviation) as robust sigma estimate
4. Reject pixels where `|radiance_k - median| > 3 × MAD`
5. Weighted-average the remaining pixels

Catches moving objects, hot pixels, and flare present in only some exposures.

**Kappa-Sigma Clipping (Astrophotography Standard)**

Iterative variant:
1. Compute median and MAD
2. Reject beyond kappa × 1.4826 × MAD (1.4826 converts MAD to sigma-equivalent)
3. Recompute mean and standard deviation of remaining points
4. Iterate until convergence

**Variance/Consistency Checking**

Compare aligned pixel values across exposures after normalizing for exposure. Pixels deviating beyond a noise-dependent threshold are flagged as ghosted and excluded (fall back to reference exposure).

**Multi-Level MTB Ghost Maps (US Patent 9123141B2)**

Uses multiple MTB thresholds to detect ghost pixels more completely than single-level approaches.

### Commercial Tool Approaches

- **Lightroom**: Compares bracketed exposures to identify movement areas; selects single exposure for ghost regions. Offers None/Low/Medium/High deghosting levels.
- **Capture One**: Layer-based blending with ghost reduction.

### Deep Learning Deghosting (2022-2025) — For Reference Only

These are impractical for our C++ desktop tool but represent the research frontier:

| Paper | Venue | Approach |
|-------|-------|----------|
| Ghost-free HDR with Context-aware Transformer | ECCV 2022 | Context-aware attention |
| SCTNet (Tel et al.) | ICCV 2023 | Alignment-free; spatial + channel attention |
| SAFNet | ECCV 2024 | Selective alignment fusion; 10x faster than prior SOTA |
| HDRFlow | CVPR 2024 | Real-time HDR video via optical flow; 40fps at 720p |
| PGHDR | JKSUCI 2025 | Progressive deformable alignment + quality-guided motion compensation |
| SelfHDR | ICLR 2024 | Self-supervised; no HDR ground truth needed |
| BracketIRE | ICLR 2025 | Unified denoise/deblur/HDR/SR from brackets |
| Conditional Diffusion for HDR | TCSVT 2023 | LDR features as diffusion condition |
| LEDiff | CVPR 2025 | Latent exposure diffusion |

---

## 3. Alignment Improvements Beyond MTB

### Current State: MTB (Ward 2003)

HDRMerge uses Median Threshold Bitmap alignment:
- Convert to grayscale, threshold at median, create binary bitmap
- Hierarchical pyramid shift search
- Very fast, robust to exposure differences (median is exposure-invariant)
- **Limitation**: Translation only — no rotation, scale, or lens distortion correction

### Tiered Alignment Strategy

| Tier | Method | Handles | Speed | Dependency |
|------|--------|---------|-------|------------|
| 0 (current) | MTB | Translation | Very fast | None |
| 1 | ORB/AKAZE + RANSAC homography | Rotation, perspective, scale | Fast | OpenCV |
| 2 | Phase correlation refinement | Sub-pixel translation | Fast | FFT library |
| 3 | DIS optical flow | Local deformation, parallax | Medium | OpenCV |

### Feature-Based Alignment (Homography)

Comparative analysis (IEEE 2018):
- **ORB**: Best performer on brightness-varied images; fast; free
- **AKAZE**: Best for blur, rotation, perspective distortion; uses nonlinear diffusion filtering; free. pfstools switched to AKAZE.
- **SIFT**: Gold standard accuracy; now free in OpenCV 4.4+

**Best practice**: Compute homography from the pair of exposures closest in EV. Chain transformations. Use RANSAC with 3-5px reprojection threshold. Match features only in non-saturated overlap regions.

### Sub-Pixel Alignment

**Phase correlation**: Compute translation in frequency domain; parabolic peak fitting provides sub-pixel accuracy. Recent work (Springer 2022) proposes pyramid phase correlation with upsampling.

### Deep Learning Alignment (For Reference)

- **SuperPoint + LightGlue**: Learned features, more robust to illumination changes
- **Deformable convolution**: Used in SAFNet, PGHDR — warping + convolution
- **Cross-attention implicit alignment**: SCTNet bypasses explicit alignment entirely

---

## 4. Compression Improvements

### zlib-ng — Drop-In NEON-Optimized zlib

- **Repository**: https://github.com/zlib-ng/zlib-ng
- **Version**: 2.3.2 (late 2025)
- Build with `-DZLIB_COMPAT=ON` for drop-in replacement (same API, same headers)
- 2-3x faster compression/decompression on ARM64 (NEON slide hash, compare256, inflate chunk copy)
- Homebrew: `brew install zlib-ng`
- **Integration effort**: Zero code changes — build system only

### libdeflate — Fastest Whole-Buffer DEFLATE

- **Repository**: https://github.com/ebiggers/libdeflate
- **Version**: 1.25 (November 2025)
- 2.6x faster than stock zlib at default compression level
- Compression levels 1-12 (zlib: 1-9); levels 10-12 exceed zlib level 9 ratio
- ARM NEON optimized Adler-32 (~3x faster)
- **libtiff integration**: Since libtiff 4.2.0, libdeflate used automatically when available. 35-50% speed improvement on TIFF DEFLATE with zero code changes.
- **Limitation**: No streaming API (whole-buffer only)

### Comparison

| Library | Speed vs zlib | Ratio vs zlib | Drop-in? | ARM NEON | Effort |
|---------|---------------|---------------|----------|----------|--------|
| zlib (stock) | baseline | baseline | — | no | — |
| zlib-ng 2.3.2 | ~2-3x faster | identical | Yes (compat mode) | Yes | Very low |
| libdeflate 1.25 | ~2-2.6x faster | identical + extra levels | No (different API) | Yes | Low via libtiff |
| zstd 1.5.6 | ~3.4x faster | better | No | Yes | N/A (not valid in DNG) |

### Byte-Shuffle Preprocessing

Before DEFLATE, applying byte-shuffle to float data improves compression ratio significantly (what OpenEXR ZIP does internally):
- Reorder bytes so all MSBs together, all LSBs together
- Blosc2 benchmarks: 5-6x ratio improvement on some data
- HDRMerge's `encodeFPDeltaRow()` already does delta encoding (TIFF predictor 2); byte-shuffle is complementary

### Recommendation

1. **Easiest path**: Ensure libtiff is built with libdeflate support — free 35-50% faster tile compression
2. **Second**: Swap system zlib for zlib-ng compat mode — 2-3x faster direct `compress()` calls
3. **Both together**: libtiff tiles use libdeflate, remaining `compress()` calls use zlib-ng

---

## 5. DNG Format Updates

### DNG 1.6 (December 2021)
- Developed closely with Apple
- New opcode `WarpRectilinear2`
- Additional metadata tags

### DNG 1.7.0.0 (June 2023) — Major Update
- **JPEG XL (JXL) added as compression codec** for raw image data
- Lossless JXL: ~40% smaller files than DEFLATE
- Lossy JXL: up to 92% size reduction with perceptually lossless quality
- Used by Apple iPhone 16 Pro (ProRAW) and Samsung Galaxy S24 (Expert RAW)

### DNG 1.7.1.0 (September 2023)
- Minor refresh with additional JXL compression parameters

### DNG SDK 1.7.1 Build 2471 (January 2026)
- Latest SDK with DNG 1.7.x integration

### JPEG XL for Float Data — Not Ready Yet

- libjxl float compression is immature: poor ratios on float PFM data (43.7 bpp lossless)
- Subnormal FP16 values do not roundtrip losslessly
- At high effort levels, achieves ~2.2-2.4x ratio but compresses 30-100x slower than alternatives
- Ecosystem still catching up: darktable, RawTherapee have open issues for DNG 1.7

**Recommendation**: Wait 1-2 years for float path maturity and broader ecosystem support.

---

## 6. LibRaw Upgrade Path

| Version | Date | Key Changes |
|---------|------|-------------|
| **0.21.0** | Dec 2022 | Phase One IIQ-S v2, Canon CR3 filmrolls/CRM |
| **0.21.1** | — | Bug fixes |
| **0.21.2** | Dec 2023 | `LIBRAW_MAX_PROFILE_SIZE_MB` compile-time limit |
| **0.21.3** | Sep 2024 | Security: `LIBRAW_CALLOC_RAWSTORE`, 4-component JPEG DNG support |
| **0.21.4** | Apr 2025 | Maintenance |
| **0.22.0** | Jan 2026 | DNG 1.7/JXL reading, **64-bit file offsets (ABI break)**, 1,284 cameras, Nikon NEFX PixelShift, new `simplify_make_model()` API, `LibRaw_abstract_datastream` changes |

### Recommendations

- **Minimum**: Upgrade to LibRaw 0.21.3+ for security fixes and 4-component DNG support
- **Strategic**: LibRaw 0.22 for DNG 1.7 reading and broader camera support (requires ABI adaptation due to 64-bit offsets)

---

## 7. Apple Silicon Optimization

### Apple Accelerate Framework

Instead of writing raw NEON intrinsics, use Apple's frameworks for automatic ARM64+x86_64 optimization:

- **vImage**: Morphological operations (dilation = fattenMask), convolutions, format conversions, alpha compositing
- **vDSP**: Vector float operations, type conversions (`vImageConvert_PlanarFtoPlanar16F` for float-to-half)
- No `#ifdef __SSE2__` / `#ifdef __ARM_NEON__` branching needed — one code path for both architectures

### Current ARM64 Issues in HDRMerge

| Component | Issue | Impact |
|-----------|-------|--------|
| `fattenMask()` (ImageStack.cpp) | SSE2 path skipped on ARM64, scalar fallback | 8-10x slower |
| `compressFloats()` (DngFloatWriter.cpp) | F16C path skipped, scalar DNG_FloatToHalf() | 15-20% slower |
| Box blur (BoxBlur.cpp) | Scalar accumulation | 1.5-2x potential gain |

### Metal Compute (Longer Term)

- Metal 4 announced WWDC 2025: unified command encoder, neural rendering
- Metal Performance Shaders: pre-optimized compute shaders for image processing
- Processing HDR images with Metal: official Apple documentation available
- Useful for GPU-accelerated tile compression or image alignment

### Apple HDR / Gain Maps (ISO 21496-1)

- Apple "Adaptive HDR" using gain maps embedded in images
- libjxl v0.11.0 added Gain Map API
- Emerging standard for HDR display on Apple devices
- Potentially relevant for HDRMerge output in the future

---

## 8. Exposure Fusion vs HDR Merge

### Fundamental Distinction

- **HDR merge**: Recovers physically-linear radiance map; produces scene-referred output (DNG)
- **Exposure fusion** (Mertens 2007): Directly blends multi-exposure LDR images using quality measures; produces display-referred output

**For our RAW workflow**: Exposure fusion is fundamentally incompatible. We operate on linear Bayer data and output floating-point DNG. Mertens quality measures (contrast via Laplacian, saturation, Gaussian well-exposedness) are meaningless on linear raw data.

### Recent Exposure Fusion Advances (For Reference)

- **Extended Exposure Fusion** (Hessel, WACV 2020): Fixes out-of-range artifact and low-frequency halo
- **UltraFusion** (Chen et al., CVPR 2025 Highlight): First fusion handling 9-stop differences; guided inpainting via Stable Diffusion; display-referred output only

### Verdict

Exposure fusion remains a display-referred technique. For scene-referred linear DNG output, classical HDR merge with noise-aware weighting is the correct approach.

---

## 9. Bit Depth & Precision

### FP16 vs FP32 Comparison

| Property | FP16 | FP32 |
|----------|------|------|
| Mantissa bits | 10 | 23 |
| Precision in [0,1] | ~1/1024 | ~1/16,777,216 |
| Max value | 65,504 | ~3.4 x 10^38 |

### When FP32 Matters

- 14-bit sensor data has 16,384 levels; FP16's 10-bit mantissa is tight — loses ~4 bits in some intervals
- After merging 5 brackets spanning 10 EV, values span 1024:1 range
- **Intermediate computation**: FP32 strongly recommended to avoid accumulation errors
- **Output**: FP32 provides headroom for downstream raw processing (WB, exposure, shadow recovery)

### What Tools Use

- Lightroom: 16-bit float DNG output, 32-bit float internal processing
- Nik HDR Efex: 32-bit end-to-end
- HDRMerge (current): 32-bit float — **correct choice**

### Verdict

Retain 32-bit float. FP16 output saves ~50% file size but loses precision that matters for downstream processing. The file size cost is mitigated by DEFLATE compression.

---

## 10. Bayer-Domain (Pre-Demosaic) Merge Validation

### Advantages of Pre-Demosaic Merge (Our Architecture)

1. **Radiometric linearity preserved**: No non-linear transforms corrupt data before merge
2. **No interpolation artifacts**: Each pixel is a direct photosite measurement (demosaiced data is 2/3 interpolated)
3. **Color accuracy**: Per-channel radiometric relationship preserved
4. **DNG compatibility**: Merged Bayer mosaic writes directly as CFA-patterned DNG

### Empirical Evidence

A 2023 circumsolar radiometry study directly compared RAW-domain vs ISP-domain HDR:
- ISP-HDR exhibited **2x more near-saturation** within 0-4 degrees of the Sun
- **3-4x weaker circumsolar radial gradients** compared to RAW-HDR
- RAW-domain merge conclusively superior for preserving scene information

### CFA-Aware HDR Literature

- **Kang et al. (EURASIP 2014)**: Adaptive weighting function for Bayer-patterned HDR; ghost detection at mosaic level
- **Merging-ISP (2019)**: Neural network for joint demosaic/align/merge from CFA inputs; shows cascaded ISP suffers from error propagation

### Verdict

Our pre-demosaic Bayer-domain merge architecture is validated as superior by both theory and empirical studies.

---

## 11. Multi-Exposure Super-Resolution

### State of the Art

- **Lecouat et al. (SIGGRAPH 2022)**: Joint HDR + SR from raw bursts with exposure bracketing; up to 4x super-resolution
- **BracketIRE (ICLR 2025)**: Unified denoise/deblur/HDR/SR from brackets using Temporally Modulated Recurrent Network

### Practical Assessment

**Not applicable to our workflow**:
1. Requires sub-pixel misalignment (tripod brackets have minimal shift)
2. Requires 8+ frames (typical brackets are 3-5)
3. SR output doesn't correspond to valid Bayer mosaic — can't write as DNG
4. All implementations are deep-learning-based

Sub-pixel shifts in bracket sets are an artifact to correct, not exploit.

---

## 12. Deep Learning HDR Methods

### Alignment-Free Methods
- **SCTNet** (ICCV 2023): Semantics Consistent Transformer; spatial + channel attention — https://github.com/Zongwei97/SCTNet

### Fast Selective Methods
- **SAFNet** (ECCV 2024): Selective Alignment Fusion; order of magnitude faster than prior SOTA — https://github.com/ltkong218/SAFNet

### Real-Time Video HDR
- **HDRFlow** (CVPR 2024): First real-time HDR video via optical flow; 40fps at 720p — https://github.com/OpenImagingLab/HDRFlow

### Self-Supervised Methods
- **SelfHDR** (ICLR 2024): No HDR ground truth needed; competitive with supervised — https://github.com/cszhilu1998/SelfHDR

### Unified Bracket Processing
- **BracketIRE** (ICLR 2025): Joint denoise/deblur/HDR/SR — https://github.com/cszhilu1998/BracketIRE

### Display-Referred Fusion
- **UltraFusion** (CVPR 2025): 9-stop exposure fusion via guided inpainting — https://github.com/OpenImagingLab/UltraFusion

### Assessment

All require GPU inference, large models, and operate in sRGB/display domain. Not directly applicable to a lightweight C++ raw-to-DNG tool, but the algorithmic ideas (attention-based weighting, optical flow alignment) inform classical implementations.

---

## 13. Priority Ranking

| # | Enhancement | Quality/Performance Impact | Effort | Risk |
|---|------------|---------------------------|--------|------|
| 1 | **Noise-optimal weighted merge** (Poisson) | Major quality: sqrt(N) SNR gain | Medium (core algorithm) | Low |
| 2 | **Sigma-clipping ghost detection** | New capability | Medium | Low |
| 3 | **zlib-ng drop-in swap** | 2-3x faster compression | Very low (build only) | None |
| 4 | **NEON fattenMask via vImage** | 8-10x faster dilation | Low-medium | Low |
| 5 | **Feature-based alignment** (ORB/AKAZE) | Handheld bracket support | Medium | Low |
| 6 | **NEON float-to-half via vDSP** | 15-20% faster DNG write | Low | None |
| 7 | **LibRaw 0.21.3+ upgrade** | Security, new cameras | Low | Low |
| 8 | **Configurable zlib level** | User choice speed/size | Very low | None |
| 9 | **Streaming DNG writer** | 80% less peak memory | High | Medium |
| 10 | **DNG 1.7 / JPEG XL** | 40% smaller files | High | Medium (ecosystem) |

---

## 14. References

### Core HDR Merge Algorithms

- Hanji, P., Zhong, F., & Mantiuk, R. K. (2020). "Noise-Aware Merging of High Dynamic Range Image Stacks without Camera Calibration." ECCV 2020 Workshops. https://www.cl.cam.ac.uk/research/rainbow/projects/noise-aware-merging/ | Code: https://github.com/gfxdisp/HDRutils

- Granados, M., Ajdin, B., Wand, M., Theobalt, C., Seidel, H.-P., & Lensch, H. P. A. (2010). "Optimal HDR Reconstruction with Linear Digital Cameras." CVPR 2010. https://www.researchgate.net/publication/221364701_Optimal_HDR_Reconstruction_with_Linear_Digital_Cameras

- Aguerrebere, C., Delon, J., Gousseau, Y., & Musé, P. (2014). "Best Algorithms for HDR Image Generation: A Study of Performance Bounds." SIAM Journal on Imaging Sciences. https://hal.science/hal-00733853v1/file/best_hdr_algo_hal.pdf

- Debevec, P. E. & Malik, J. (1997). "Recovering High Dynamic Range Radiance Maps from Photographs." SIGGRAPH 1997. https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf

- Hasinoff, S. W. et al. (2016). "Burst Photography for High Dynamic Range and Low-Light Imaging on Mobile Cameras." SIGGRAPH Asia 2016 (Google HDR+). https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus.pdf

- Hasinoff, S. W. et al. (2010). "Noise-Optimal Capture for High Dynamic Range Photography." CVPR 2010. https://people.csail.mit.edu/hasinoff/hdrnoise/

### Noise Modeling

- Cao, Y. et al. (2023). "Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography." CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Physics-Guided_ISO-Dependent_Sensor_Noise_Modeling_for_Extreme_Low-Light_Photography_CVPR_2023_paper.html

- Wischow, M. et al. (2024). "Real-Time Noise Source Estimation from Image and Metadata." Advanced Intelligent Systems. https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300479

### Ghost Detection & Deghosting

- Chi, Z. et al. (2023). "HDR Imaging with Spatially Varying Signal-to-Noise Ratios." CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Chi_HDR_Imaging_With_Spatially_Varying_Signal-to-Noise_Ratios_CVPR_2023_paper.pdf

- Tel, S. et al. (2023). "Alignment-Free HDR Deghosting with Semantics Consistent Transformer" (SCTNet). ICCV 2023. https://github.com/Zongwei97/SCTNet

- Kong, L. et al. (2024). "Selective Alignment Fusion Network" (SAFNet). ECCV 2024. https://github.com/ltkong218/SAFNet | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03782.pdf

- Xu, G. et al. (2024). "HDRFlow: Real-Time HDR Video Reconstruction with Large Motions." CVPR 2024. https://github.com/OpenImagingLab/HDRFlow

- PGHDR (2025). "Progressive Deformable Feature Alignment with Quality-Guided Motion Compensation." JKSUCI. https://link.springer.com/article/10.1007/s44443-025-00230-z

- Zhang, Z. et al. (2024). "SelfHDR: Self-Supervised HDR Imaging." ICLR 2024. https://github.com/cszhilu1998/SelfHDR

- Zhang, Z. et al. (2025). "Exposure Bracketing Is All You Need" (BracketIRE). ICLR 2025. https://github.com/cszhilu1998/BracketIRE

- Yan, Q. et al. (2023). "SMAE: Few-Shot Learning for HDR Deghosting with Saturation-Aware Masked Autoencoders." CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_SMAE_Few-Shot_Learning_for_HDR_Deghosting_With_Saturation-Aware_Masked_Autoencoders_CVPR_2023_paper.pdf

- Wang, Y. et al. (2023). "Conditional Diffusion for HDR Deghosting." IEEE TCSVT 2023. https://dl.acm.org/doi/10.1109/TCSVT.2023.3326293

- Wang, Y. et al. (2025). "LEDiff: Latent Exposure Diffusion for HDR Generation." CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_LEDiff_Latent_Exposure_Diffusion_for_HDR_Generation_CVPR_2025_paper.pdf

### Alignment

- Ward, G. (2003). "Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Hand-Held Exposures" (MTB). http://www.anyhere.com/gward/papers/jgtpap2.pdf

- Gallo, O. et al. (2015). "Locally Non-Rigid Registration for Mobile HDR Photography." NVIDIA. https://research.nvidia.com/sites/default/files/pubs/2015-06_Locally-Non-rigid-Registration/GalloEtAl_LocallyNRR_2015.pdf

- Feature detector comparison for HDR (IEEE 2018). https://ieeexplore.ieee.org/document/8346440/

- Pyramid phase correlation with sub-pixel accuracy (Springer 2022). https://link.springer.com/article/10.1007/s11760-022-02158-7

### Exposure Fusion

- Mertens, T. et al. (2007). "Exposure Fusion." Pacific Graphics. https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf

- Hessel, C. (2020). "An Extended Exposure Fusion and Its Application to Single Image Contrast Enhancement." WACV 2020. https://openaccess.thecvf.com/content_WACV_2020/papers/Hessel_An_Extended_Exposure_Fusion_and_its_Application_to_Single_Image_WACV_2020_paper.pdf

- Chen, X. et al. (2025). "UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion." CVPR 2025 Highlight. https://github.com/OpenImagingLab/UltraFusion | https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_UltraFusion_Ultra_High_Dynamic_Imaging_using_Exposure_Fusion_CVPR_2025_paper.pdf

### Bayer-Domain HDR

- Kang, H. et al. (2014). "Bayer Patterned High Dynamic Range Image Reconstruction Using Adaptive Weighting Function." EURASIP Journal on Advances in Signal Processing. https://link.springer.com/article/10.1186/1687-6180-2014-76

- Merging-ISP (2019). "Multi-Exposure High Dynamic Range Image Signal Processing." arXiv:1911.04762. https://arxiv.org/abs/1911.04762

- RAW-domain vs ISP-domain HDR comparison (2023). Journal of Imaging. https://www.mdpi.com/2313-433X/11/12/442

### Multi-Exposure Super-Resolution

- Lecouat, B. et al. (2022). "High Dynamic Range and Super-Resolution from Raw Image Bursts." SIGGRAPH 2022. https://arxiv.org/abs/2207.14671

### Compression & Format

- zlib-ng: https://github.com/zlib-ng/zlib-ng
- libdeflate: https://github.com/ebiggers/libdeflate
- libjxl: https://github.com/libjxl/libjxl
- Blosc2: https://github.com/Blosc/c-blosc2
- Adobe DNG Specification: https://helpx.adobe.com/camera-raw/digital-negative.html
- DNG 1.6 (Library of Congress): https://www.loc.gov/preservation/digital/formats/fdd/fdd000628.shtml
- libjxl float compression issue: https://github.com/libjxl/libjxl/issues/1323
- Aras Pranckevicevius — Lossless Float Image Compression: https://aras-p.info/blog/2025/07/08/Lossless-Float-Image-Compression/
- Aras — EXR libdeflate: https://aras-p.info/blog/2021/08/09/EXR-libdeflate-is-great/
- Aras — Float Compression series (Blosc): https://aras-p.info/blog/2023/03/02/Float-Compression-8-Blosc/
- libtiff libdeflate integration: https://gitlab.com/libtiff/libtiff/-/merge_requests/158

### LibRaw

- LibRaw 0.22.0 Release: https://www.libraw.org/news/libraw-0-22-0-release
- LibRaw 0.21.3 Release: https://www.libraw.org/news/libraw-0-21-3-release
- LibRaw GitHub: https://github.com/LibRaw/LibRaw

### Apple Silicon & Frameworks

- Apple Accelerate: https://developer.apple.com/accelerate/
- Apple vImage: https://developer.apple.com/documentation/accelerate/vimage
- Processing HDR images with Metal: https://developer.apple.com/documentation/metal/processing-hdr-images-with-metal
- CIRAWFilter: https://developer.apple.com/documentation/coreimage/cirawfilter
- Support HDR images in your app (WWDC23): https://developer.apple.com/videos/play/wwdc2023/10181/

### Tone Mapping (For Reference)

- TMOz (2023): Perceptual tone mapping via CIECAM16. https://arxiv.org/abs/2309.16975
- iCAM06-m (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10007327/

### Open-Source HDR Tools

- jcelaya/hdrmerge: https://github.com/jcelaya/hdrmerge
- wjakob/hdrmerge (scientific): https://github.com/wjakob/hdrmerge
- Luminance HDR: https://github.com/LuminanceHDR/LuminanceHDR
- pfstools: https://sourceforge.net/projects/pfstools/
- OpenCV HDR module: https://docs.opencv.org/4.x/d2/df0/tutorial_py_hdr.html
- Awesome-High-Dynamic-Range-Imaging: https://github.com/rebeccaeexu/Awesome-High-Dynamic-Range-Imaging
- Awesome-HDR: https://github.com/ytZhang99/Awesome-HDR

### Sigma Clipping / Robust Statistics

- GNU Astronomy Utilities — Sigma Clipping: https://www.gnu.org/software/gnuastro/manual/html_node/Sigma-clipping.html
- Astropy Robust Statistics: https://docs.astropy.org/en/stable/stats/robust.html
- Kappa-Sigma Clipping empirical evidence: https://www.lightvortexastronomy.com/snr-increase-with-exposures-using-kappa-sigma-clipping-empirical-evidence.html
- Robust Chauvenet Outlier Rejection: https://arxiv.org/abs/1807.05276
- HDR+ Burst Denoising (IPOL analysis): https://www.ipol.im/pub/art/2021/336/article_lr.pdf
