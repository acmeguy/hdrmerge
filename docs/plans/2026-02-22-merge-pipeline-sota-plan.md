# HDR Merge Pipeline -- SOTA-Driven Implementation Plan

**Date**: 2026-02-22
**Last reviewed**: 2026-02-22 (deep-research pass against SOTA literature and current codebase)
**Scope**: Alignment, deghosting, noise modeling, response modeling, X-Trans interpolation
**Priority**: `P0` correctness/quality first, then `P1` modeling upgrades, then `P2` refinements

## Goals

1. Close known quality gaps in the merge path with highest return first.
2. Keep changes incrementally testable and reversible.
3. Add objective test vectors and acceptance metrics for each stage.

## Baseline (Current Code)

1. `--align-features` is exposed but stub-only; MTB fallback is always used. MTB is a 6-level pyramid, 9-offset search per scale, integer-pixel translation only (`src/Image.cpp:179-223`). Sub-pixel residuals are measured via parabolic SSD fit (`src/ImageStack.cpp:137-174`) but used only for diagnostics, not compose.
2. Deghosting is MAD-based per pixel (`src/ImageStack.cpp:869-924`): collects per-pixel exposure stack, computes median radiance, derives MAD-scaled sigma, then ghost confidence = max_deviation / (sigma * deghostSigma). A 3x3 separable box blur provides spatial coherence. Soft Gaussian weighting modulated by ghostMap is applied during compose (`src/ImageStack.cpp:1007-1032`). Requires numImages >= 3.
3. Noise profile: S is computed theoretically as `1/(max - cblack[c])` (`src/ImageStack.cpp:689-693`). O is estimated from optical black margin variance (`src/ImageStack.cpp:695-730`) with fallback to 1e-6. Both are scaled by 1/numImages for merged output. Shadow weighting blends inverse-variance weights (shadows) with Poisson weights (midtones) at a crossover point where shot noise = read noise (`src/ImageStack.cpp:959-979`).
4. Response function is pairwise chained darkest-to-brightest (`src/ImageStack.cpp:242-245`). Each pair is spline-fitted using ALGLIB penalized cubic spline (200 knots, smoothing=3) on the top 25% of ADU range (`src/Image.cpp:100-176`). Scalar linear fallback triggers only when spline has < max/8 data points.
5. X-Trans sub-pixel interpolation uses fixed `cfaStep=6` (`src/ImageStack.cpp:822`), Bayer uses `cfaStep=2`. Bilinear interpolation across same-color neighbors at that fixed distance (`src/ImageStack.cpp:742-774`).
6. DNG NoiseProfile written as tag 51041, DOUBLE format, 2 coefficients (S,O) per channel (`src/DngFloatWriter.cpp:440-442`).
7. **Pipeline ordering** (`src/ImageIO.cpp:170-207`): align (line 177) -> computeResponseFunctions (182) -> correctHotPixels (199) -> compose (204). Hot pixel correction currently runs *after* alignment and response fitting but *before* compose (which includes ghost map, noise estimation, and merge).

## Cross-Cutting Concern: Hot Pixel Correction Ordering

**Current ordering is suboptimal.** Hot pixel correction at `ImageIO.cpp:199` runs after alignment and response function fitting. Research confirms the standard ISP ordering is: hot pixel correction *first*, then alignment, then all downstream stages.

**Impact on each milestone:**
- **M1 (alignment):** Hot pixels cause persistent bit-flip errors in MTB at sparse regions. Impact is minor for MTB (global median is robust to isolated defects) but more significant for feature-based alignment where a cluster of hot pixels could generate false keypoints.
- **M4 (response):** Hot pixels inject anomalous ratio values into the response function fit, creating false nonlinearity artifacts in the spline path. Less impact on the scalar linear fit (robust regression can reject outliers), but the current spline path has no outlier rejection.
- **M2 (deghosting):** Hot pixels create consistent radiance outliers that either escape MAD detection (because all frames agree on the hot pixel value) or generate spurious ghost confidence where partial saturation breaks cross-frame consistency. Both corrupt the ghost map.
- **M3 (noise):** Hot pixels in optical black margins directly inflate O (read noise) via sample variance. A single hot pixel at 5000 ADU in a region with sigma ~10 can bias O by orders of magnitude. The current code uses sample variance, not MAD, making it especially vulnerable.

**Proposed action:** Move `correctHotPixels()` to run immediately after loading, before `align()`. This is a standalone change that benefits all milestones and can land independently.

**Caveat:** For the Nikon Z 9 at typical HDR bracket exposures (< 1 second, room temperature), the hot pixel population is very small (single digits to tens out of 45 million). The Z 9's BSI stacked CMOS sensor has on-chip dark current cancellation, and Nikon's pixel mapping routine handles most factory defects before the raw is written. The risk is real but low-severity for this specific camera. It becomes more relevant at high ISO, long exposures, or elevated sensor temperatures, and on other camera bodies with less sophisticated defect correction.

**References:** EMVA 1288 standard; ISP pipeline ordering in Stanford CS231M, eInfochips ISP survey, ST AN6211; Erdem et al. HDR deghosting STAR; MDPI Sensors 2023 (BSI CMOS hot pixel effects).

---

## Milestones

## `M1 (P0)` Implement True Feature Alignment

### Why

MTB-only alignment handles pure translation (2 DOF). For handheld brackets, rotation of even 0.5 degrees produces up to ~36px displacement error at the corners of a 45MP sensor (8256x5504). This manifests as visible blur in merged output away from the image center. MTB cannot correct rotation, scale, shear, or perspective. This is well-established: Ward 2003 explicitly describes a translation-only algorithm.

### Review Findings

**Confirmed accurate:**
- MTB limitation to translation-only is a real, documented constraint (Ward, Journal of Graphics Tools, Vol. 8 No. 2, 2003).
- AKAZE + ORB fallback is a sound detector/descriptor choice. AKAZE's nonlinear scale space preserves edges well across exposure changes (Tareen & Saleem, ICSPC 2018 comparative evaluation).
- Mutual nearest-neighbor + Lowe ratio test (0.75) is the standard, validated matching strategy (Lowe, IJCV 2004; Mishkin "How to Match" analysis).
- ECC refinement after feature-based coarse alignment is well-supported (Evangelidis & Psarakis, TPAMI 2008). ECC is inherently invariant to global affine intensity changes, making it well-suited for HDR brackets. Confirmed by Evangelidis 2017 refinement study and image stitching ECC study (2022).

**Issues identified and resolved:**

1. **Inconsistency in geometry model.** Original plan said "estimate homography with RANSAC" in one place and `estimateAffinePartial2D` in another. These are 8 DOF vs 4 DOF. **Resolution:** use 4 DOF as primary — it captures the dominant handheld motion (translation + rotation + uniform scale) with only 2 minimal-set points, making RANSAC converge faster. Production panorama tools (Hugin, PTGui, Adobe) use full homography as primary, but they deal with large viewpoint changes; for same-camera brackets with < 1 degree rotation, 4 DOF is well-constrained and less prone to degenerate solutions.

2. **Missing intermediate model escalation.** Original plan fell from 4 DOF directly to MTB (2 DOF translation-only). If 4 DOF fails, translation-only will certainly be worse. **Resolution:** progressive escalation 4 DOF -> 6 DOF (`estimateAffine2D`) -> 8 DOF (`findHomography`) -> MTB fallback.

3. **Feature detection on raw Bayer data is problematic.** The CFA pattern creates artificial high-frequency content that generates spurious keypoints. **Resolution:** detect on grayscale preview via 2x2 Bayer averaging, not raw mosaic.

4. **CLAHE normalization is critical.** Simple 8-bit normalization may not be sufficient across 2-4 EV exposure differences. CLAHE (Contrast Limited Adaptive Histogram Equalization) per exposure before detection is the correct approach. *(Note: the reverted uncommitted code already implemented this correctly.)*

5. **LoFTR/LightGlue are impractical for integration today.** LoFTR has no mature C++ inference path; LightGlue-ONNX (fabio-sim/LightGlue-ONNX) exists but targets NVIDIA GPU via TensorRT, not Apple Silicon. On CPU-only ONNX Runtime, inference is 0.5-2 seconds per pair. Both are overkill for same-camera small-displacement brackets where AKAZE works well. **Resolution:** keep as literature references only; do not plan implementation. **Revisit in 6 months** as CoreML backend for ONNX Runtime matures — for extreme exposure brackets (5+ EV) where AKAZE descriptor stability degrades, a learned matcher could help.

6. **Metrics: reprojection error should be primary.** MTF/edge acutance are appropriate but indirect — they can be confounded by exposure weighting, response curve errors, and sub-pixel interpolation artifacts. Reprojection error is the standard, direct measure of alignment quality. Acutance ROIs should be selected near image corners (where rotation error is maximized), not center.

7. **Consider MAGSAC++** (`cv::USAC_MAGSAC` in OpenCV 4.5+) instead of vanilla RANSAC for improved outlier handling (Barath et al., CVPR 2020).

### File-Level Changes

1. `src/ImageStack.cpp`
- Implement `align(bool useFeatures)` feature branch with OpenCV path:
  - Build grayscale preview via 2x2 Bayer averaging (not raw mosaic).
  - Apply CLAHE normalization per exposure before detection.
  - Detect/describe with AKAZE (fallback ORB if < 64 keypoints).
  - Match with KNN ratio test (0.75) + mutual nearest-neighbor cross-check.
  - Progressive geometry fit with RANSAC (prefer MAGSAC++ if OpenCV >= 4.5):
    1. `estimateAffinePartial2D` (4 DOF: translation + rotation + uniform scale).
    2. If reprojection error > 2px RMS, escalate to `estimateAffine2D` (6 DOF).
    3. If still poor, try `findHomography` (8 DOF).
    4. If all fail confidence checks, fall back to MTB.
  - Sanity checks: rotation < 2 deg, scale within 2%, inlier ratio >= 0.45, >= 16 inliers.
  - Keep MTB as fallback when feature confidence is poor.
- Optional ECC refinement: use `MOTION_EUCLIDEAN` (3 DOF) rather than pure translation, to also correct small rotation residuals. Initialize warpMatrix from the coarse feature-based result. Default termination criteria (50 iterations, epsilon 0.001) are appropriate given a good coarse alignment.
- Add logs with inlier count, reprojection error, model DOF used, and fallback reason.

2. `src/Image.hpp`
- Add optional per-image geometric transform metadata (model type, reprojection error, inlier ratio).

3. `src/CMakeLists.txt`
- Ensure OpenCV `core`, `features2d`, `calib3d`, `imgproc` are required when `HAVE_OPENCV` is set.

### Acceptance Criteria

1. On rotation/perspective test sets, edge sharpness improves versus MTB-only baseline, measured at image corners.
2. Mean reprojection error < 1.0 px on inliers for typical handheld brackets.
3. `--align-features` uses feature path when OpenCV is present and reports clear fallback when not.
4. No regression on tripod/static scenes (TV-A3).

### Test Vectors

1. `TV-A1`: Handheld 3-frame bracket with slight yaw/roll.
2. `TV-A2`: 5-frame bracket with mild perspective shift (foreground object near edge).
3. `TV-A3`: Tripod reference bracket (regression control).

### Metrics

1. **Primary:** Mean reprojection error (px), inlier ratio, model DOF used.
2. **Secondary:** Edge acutance in high-contrast slanted-edge ROI (corner ROIs, not center).
3. Runtime delta vs MTB path.

### Proposed Actions

- [x] ~~Resolve homography vs affinePartial2D inconsistency~~ (resolved: progressive escalation)
- [ ] Implement progressive 4->6->8 DOF model escalation
- [ ] Use CLAHE-normalized grayscale previews (Bayer-averaged, not raw mosaic)
- [ ] Evaluate MAGSAC++ availability in target OpenCV version
- [ ] Use `MOTION_EUCLIDEAN` for ECC refinement step
- [ ] Remove LoFTR/LightGlue from implementation scope (keep as references only; revisit in 6 months)

---

## `M2 (P0)` Reference-Guided Deghosting

### Why

Current pixelwise MAD deghost has no spatial coherence model: at motion boundaries, independent per-pixel accept/reject produces jagged, unstable edges. For coherent moving objects occupying > 50% of the pixel stack at a given position, the "moved" version becomes the statistical mode and the stationary background becomes the outlier — MAD then fails fundamentally. This is well-documented in stacking literature (MAD clipping requires < 50% outliers per stack) and the Tursun et al. (CGF 2015) deghosting survey, which categorizes pixelwise rejection as the simplest and weakest deghosting category.

### Review Findings

**Confirmed accurate:**
- MAD boundary instability and coherent-object failure are real, well-documented failure modes (Tursun 2015 survey; GNU Astronomy Utilities MAD clipping documentation).
- The proposed 6-step pipeline (reference selection -> exposure-normalized residuals -> motion confidence -> robust weighting -> hard fallback -> spatial regularization on confidence) closely matches classical SOTA approaches: Khan 2006 (probabilistic weights), Granados 2013 (noise-model-informed detection), Hu 2013 (reference selection + saturation).
- Spatial regularization on confidence (not radiance) is a technically sound choice that preserves texture while ensuring coherent ghost boundaries. Min et al. (2014) posed ghost map generation as a binary labeling problem with MRF spatial priors; the plan's continuous confidence formulation is more flexible.
- False-positive ghost detection in shadows is the central challenge for residual-based deghosting. Granados et al. (ACM TOG 2013) showed that "HDR deghosting can be significantly improved by modeling the noise distribution of the color values measured by the camera, which has been largely neglected in previous work."

**Issues identified and resolved:**

1. **M2 and M3 are tightly coupled.** The motion confidence threshold must be noise-aware: in shadows, sensor noise easily exceeds the residual threshold, causing false ghost masking. Without M3's calibrated noise model, M2 needs at minimum a simple Poisson + read noise approximation per exposure level. **Resolution:** implement an interim noise floor estimate within M2 (using existing OB-based O and theoretical S) even before M3 lands.

2. **Refinement iterations.** Khan et al. (2006) used iterative optimization of the probability map. A single-pass confidence map may miss difficult cases. **Resolution:** add at least one refinement pass. For difficult scenes (fine deformable motion like tree branches), 2-3 iterations may be needed — parameterize as `--deghost-iterations N` (default 1) rather than hardcoding.

3. **Huber vs Tukey distinction matters.** Huber loss never fully rejects a pixel (always contributes some weight). Tukey biweight completely excludes extreme outliers (zero weight beyond threshold). **Resolution:** use Tukey in high-confidence motion regions (ghostMap > 0.7) for clean exclusion; Huber in ambiguous regions (0.1 < ghostMap < 0.7) for graduated blending.

4. **Dynamic range tradeoff in reference fallback.** When hard-selecting the reference for strongly inconsistent pixels, HDR benefit is lost in those regions. This is inherent to all reference-fallback approaches (acknowledged in Hu 2013, Sen 2012). **Resolution:** explicitly document this tradeoff in acceptance criteria; log the percentage of pixels falling back to reference-only per merge.

5. **Citation corrections:**
   - "EHDR Fusion (ICCVW 2024)" — wrong year (it's ICCVW 2023), URL returns 404, and it's about event-camera + multi-exposure fusion (requires specialized hardware). Not relevant. **Removed.**
   - "DeepDuoHDR (TIP 2024)" — real paper (Alpay et al., IEEE TIP vol. 33 pp. 6592-6606, DOI: 10.1109/TIP.2024.3497838), but it's a deep learning method (attention + U-Net). **Reframed** as quality benchmark only, not implementation guide.

6. **Added classical references** that directly inform the proposed pipeline:
   - Khan et al. (ICIP 2006) — probabilistic ghost removal with iterative weight optimization.
   - Granados et al. (ACM TOG 2013) — noise-model-informed ghost detection, the key paper showing noise-aware thresholds are essential.
   - Hu et al. (CVPR 2013) — reference selection + saturation handling in ghost regions.
   - Sen et al. (ACM TOG 2012) — patch-based HDR, represents the quality ceiling for classical methods in Karaduzovic-Hadziabdic (2017) evaluation.
   - Min et al. (JIVP 2014) — MRF-based probabilistic ghost map with spatial priors.

7. **Test vector gaps.** Added three missing categories identified from the Tursun (2015) 9-category taxonomy and Karaduzovic-Hadziabdic (2017) evaluation dataset:
   - TV-G4: Static scene (regression control — was listed as acceptance criterion but not as a test vector).
   - TV-G5: Occlusion scene (object passing behind/in front of structure — reveals background that has no valid data in some frames).
   - TV-G6: Saturated motion (bright moving object — combined ghost + saturation problem, identified by Hu 2013 as uniquely difficult).

8. **Added HDR-VDP-2 metric.** Karaduzovic-Hadziabdic et al. (Computers & Graphics 2017) evaluated 13 deghosting methods across 36 scenes and found HDR-VDP-2 to be the most reliable predictor of subjective quality for deghosting artifacts.

### File-Level Changes

1. `src/ImageStack.cpp`
- Add reference-frame motion confidence map (prefer least-blurry mid exposure as reference; when best-exposed and least-blurry conflict, prefer least-blurry — motion blur in the reference propagates to all fallback regions).
- Replace/augment per-pixel MAD gating with:
  - motion mask from exposure-normalized residuals to reference,
  - noise-floor-aware threshold (even before M3: use existing theoretical S + OB-based O as interim estimate),
  - Tukey biweight for high-confidence motion regions (ghostMap > 0.7), Huber for ambiguous regions (0.1 < ghostMap < 0.7),
  - hard fallback to reference exposure where inconsistency remains high (log percentage of reference-only pixels),
  - configurable refinement iterations (default 1, parameterized via `--deghost-iterations`).
- Apply spatial regularization on confidence using guided/bilateral filter (edge-preserving, not simple box blur), to prevent bleeding between static and moving regions.
- Keep current `--deghost` sigma interface but map it to robust-threshold controls.

2. `src/LoadSaveOptions.hpp`
- Add deghost mode enum (`legacy`, `reference-robust`) with default to new mode after validation period.

3. `src/Launcher.cpp`
- Add CLI switches `--deghost-mode` and `--deghost-iterations N`.

### Acceptance Criteria

1. Moving-object scenes show fewer double edges and less texture tearing.
2. Static scenes remain visually unchanged (TV-G4 regression control).
3. No increase in false-positive masking on high-noise shadows — **quantified as false-positive rate in annotated static-shadow ROIs**.
4. Dynamic range loss in reference-fallback regions is bounded and documented (log % of reference-only pixels).

### Test Vectors

1. `TV-G1`: Person walking through frame in 3-bracket set (coherent rigid motion).
2. `TV-G2`: Tree branches/leaves moving in wind (fine deformable motion).
3. `TV-G3`: Water ripple scene (difficult stochastic motion).
4. `TV-G4`: Static scene (regression control — no motion).
5. `TV-G5`: Occlusion scene (object passing behind/in front of structure).
6. `TV-G6`: Saturated motion (bright moving object, e.g., headlights).

### Metrics

1. Ghost edge count in annotated ROIs.
2. LPIPS/SSIM in static ROI against best-exposed reference.
3. HDR-VDP-2 for overall perceptual quality assessment (Karaduzovic-Hadziabdic 2017).
4. False-positive rate in annotated static-shadow ROIs.
5. Visual QA checklist on motion boundaries.

### Proposed Actions

- [ ] Add interim noise floor estimate within M2 (theoretical S + OB-based O) before M3 lands
- [ ] Specify Huber (ambiguous) vs Tukey (high-confidence) usage by ghostMap threshold
- [ ] Parameterize refinement iterations (`--deghost-iterations`, default 1)
- [ ] Replace box blur spatial regularization with guided/bilateral filter
- [ ] Add TV-G4 (static regression), TV-G5 (occlusion), TV-G6 (saturated motion)
- [ ] Remove EHDR Fusion citation; reframe DeepDuoHDR as quality benchmark only
- [ ] Add Khan 2006, Granados 2013, Hu 2013, Sen 2012, Min 2014 to references

---

## `M3 (P1)` Calibrated Noise Model and Weighting

### Why

Current code computes S theoretically as `1/(max - cblack[c])` — this assumes ideal Poisson statistics with no analog gain nonlinearity, no quantization effects, and no PRNU. It works as a first-order approximation but ignores analog gain chain effects (which can make effective S differ from pure-Poisson by 10-30% depending on ISO and analog gain topology). O from optical black margins is a valid zero-signal anchor but is the sole source; OB margins can suffer from glow contamination, vary wildly in size across camera models, and the current code uses sample variance (not MAD), making it vulnerable to hot pixel outliers in the margins.

### Review Findings

**Confirmed accurate:**
- The affine heteroscedastic model `Var = S*signal + O` is physically correct and exactly matches the DNG NoiseProfile tag 51041 format (2 DOUBLEs per color plane, signal normalized to [0,1]). DNG spec defines `noise_variance(x) = S*x + O`.
- Black margin + signal-dependent regression is the established approach (EMVA 1288 photon transfer curve method; Android Camera2 `dng_noise_model.py`; darktable wavelet MAD noise profiling).
- All three cited papers are real and correctly characterized, with one correction: the CRLB derivation should be attributed primarily to Aguerrebere et al. (SIIMS 2014), not Granados et al. (CVPR 2010). Granados derives the MLE weight; Aguerrebere proves it achieves the CRLB asymptotically and analyzes sensitivity to calibration errors.

**Issues identified and resolved:**

1. **S estimation needs structure separation.** Simply computing variance of a patch conflates texture with noise. Three approaches ranked by robustness for HDR merge:
   - **(a) Temporal variance across aligned exposures** — naturally separates noise from structure since the scene is constant across frames. Theoretically cleanest for multi-frame data. **Caveat:** requires sub-pixel alignment accuracy; even 0.5px residual error on high-frequency texture will overestimate noise. Best in smooth/mid-frequency regions.
   - **(b) High-pass filter per tile** (Android Camera2 approach) — suppresses low-frequency scene structure before variance estimation. More robust to alignment imperfection. Established in Android's `dng_noise_model.py`.
   - **(c) Wavelet MAD** (darktable approach) — estimates noise in wavelet coefficients. Very robust but may require careful scale selection.
   **Resolution:** Implement both (a) and (b); cross-validate and use the method with lower residual for a given image. This guards against alignment-error bias in temporal variance and scene-structure leakage in spatial high-pass.

2. **DNG NoiseProfile for merged image is inherently approximate.** A single global (S, O) pair cannot capture spatially varying noise reduction from position-dependent exposure weighting. Highlights use one short exposure (noise ~ single frame); shadows use longest exposure(s) (noise reduced); midtones have multiple contributors (most noise reduction). **Resolution:** document this approximation. Compute effective (S_merged, O_merged) based on the weight-averaged effective number of exposures. Note that slightly overestimating noise is preferable (ACR/Lightroom can reduce NR, but users may not increase it).

3. **Dual-conversion-gain cameras (e.g., Nikon Z 9).** The Z 9's sensor switches conversion gain at certain ISO boundaries, causing discontinuities in both S and O. **Resolution:** the (S, O) monotonicity metric should test within each gain regime, not blindly across gain transitions.

4. **Added Hanji et al. (ECCV 2020)** — "Noise-Aware Merging of HDR Image Stacks without Camera Calibration." Shows that a simple Poisson-only weight `w_k = t_k / z_k` (dropping the O term entirely) achieves near-MLE performance without camera-specific calibration. Deviation from full MLE is largest in deep shadows where read noise dominates. This is a useful fallback when calibration is unreliable, and a validation reference — if the calibrated model performs worse than the uncalibrated Poisson weight, the calibration is broken.

5. **Optimal merge weight formula derivation.** The MLE inverse-variance weight is `w_k = t_k^2 / (S * z_k + O)` where z_k is the observed raw value (Granados CVPR 2010). The CRLB for the HDR irradiance estimate is `Var(L_hat) >= 1 / sum_k(t_k^2 / sigma_k^2(L))` (Aguerrebere SIIMS 2014). The MLE achieves within a few percent of the CRLB even with 2-3 exposures. The current code at `src/ImageStack.cpp:959-979` already implements a version of this for shadow weighting — the calibrated noise model improves the (S, O) parameters fed to this formula.

6. **Predicted-vs-measured validation.** Verify that the noise model's predicted variance matches actual measured variance (from OB regions or spatially flat scene patches) to within < 20% relative error. This catches calibration bugs that SNR or monotonicity tests might miss.

### File-Level Changes

1. `src/RawParameters.hpp`
- Add per-frame noise descriptors (read noise proxy, gain proxy, ISO normalization).

2. `src/RawParameters.cpp`
- Populate noise descriptors from available LibRaw/metadata fields.

3. `src/ImageStack.cpp`
- Replace `estimateNoiseProfile(...)` with calibrated estimator:
  - Use OB margins for O seed (with MAD, not sample variance, to reject glow contamination and hot pixel outliers).
  - For S estimation: implement both temporal variance (across aligned exposures at smooth regions) and spatial high-pass-filter-per-tile (Android Camera2 approach). Cross-validate; use the method with lower residual.
  - Compute channel-wise `(S, O)` with robustness guards.
  - Feed calibrated model into compose inverse-variance weighting: `w_k = t_k^2 / (S * z_k + O)` (Granados 2010).
- For DNG NoiseProfile output: compute effective (S_merged, O_merged) scaled by weight-based effective exposure count. Document that this is a global approximation for a spatially varying quantity. Err toward slight overestimation.
- Poisson-only fallback `w_k = t_k / z_k` (Hanji 2020) when calibration is unavailable or fails validation.

4. `src/DngFloatWriter.cpp`
- Keep current tag writing path; validate range/precision guards for `NOISEPROFILE`.

### Acceptance Criteria

1. Shadow noise decreases without midtone texture loss.
2. `NoiseProfile` values are stable across repeated runs and plausible across ISO changes.
3. Channel-wise (S, O) monotonic within each conversion-gain regime (not necessarily across gain transitions on dual-gain sensors like Z 9).
4. Predicted variance matches measured variance within 20% relative error.
5. Calibrated model performs at least as well as Poisson-only fallback (Hanji 2020) on SNR in shadow ROI. If not, the calibration is broken.
6. ACR/Lightroom default denoise behavior is more consistent for merged outputs.

### Test Vectors

1. `TV-N1`: ISO 100 static bracket (baseline clean sensor behavior).
2. `TV-N2`: ISO 1600 bracket with deep shadows.
3. `TV-N3`: Mixed lighting scene with dark textured regions.

### Metrics

1. SNR in dark ROI after merge (spatially flat region).
2. Noise power spectrum trend versus baseline.
3. Channel-wise `(S, O)` monotonicity with ISO (within gain regimes).
4. Predicted-vs-measured variance ratio per channel.
5. Comparison of calibrated merge SNR vs Poisson-only merge SNR (should be >= 1.0).

### Proposed Actions

- [ ] Implement both temporal variance and spatial high-pass S estimation; cross-validate
- [ ] Use MAD instead of sample variance for OB-based O estimation
- [ ] Document DNG NoiseProfile spatial-averaging approximation
- [ ] Account for dual-conversion-gain behavior in Z 9 monotonicity tests
- [ ] Add Poisson-only fallback and comparison metric (Hanji 2020)
- [ ] Attribute CRLB to Aguerrebere (2014), MLE weight to Granados (2010)

---

## `M4 (P1)` Response Modeling Strategy Split (Linear-First)

### Why

Modern CMOS RAW (including Nikon Z 9 lossless 14-bit NEF) is linear to well within 0.5% after black subtraction. The current pairwise-chained spline fitting introduces avoidable error propagation: each link in the chain inherits the previous spline's fitting artifacts, with cumulative error growing for darker images. For a 4-frame bracket, image 0's response depends on splines from images 1, 2, and 3 — a classic chain-rule error amplification.

### Review Findings

**Confirmed accurate:**
- Modern CMOS RAW linearity is well-established. Hasinoff (HDR+, SIGGRAPH Asia 2016), Hanji (ECCV 2020), and Granados (CVPR 2010) all assert linearity and derive merge weights directly from exposure time, with no response curve fitting.
- Robertson et al. (JEI 2003) explicitly states: "If the camera response is linear (as is approximately the case with RAW digital cameras), the response estimation step can be skipped and the linear exposure ratio from EXIF can be used directly."
- Debevec-Malik (SIGGRAPH 1997) was designed for 8-bit JPEG input (with baked-in gamma), not 14-bit linear RAW. For linear input, their smoothness regularizer forces the recovered curve toward linearity anyway, making the fitting unnecessary overhead.
- The current code's scalar linear fallback (`src/Image.cpp:159-175`) is already the correct approach for linear sensors — it fits a single-parameter exposure ratio via least squares. But it triggers only when the spline path fails (< max/8 data points), not as the primary strategy.

**Linearity caveats (minor, documented for completeness):**
- *Lossy compressed RAW:* Sony compressed 12-bit uses a nonlinear quantization curve. Nikon's lossless 14-bit NEF (the test corpus) is linear. The R^2 threshold for spline activation should be validated empirically against lossy-compressed test files to avoid false triggering from quantization noise.
- *Highlight-priority modes:* Some cameras apply analog knee compression before digitization. The Z 9 does not in standard modes.
- *Sub-3% full-well:* Photodiode enters a sublinear regime due to junction capacitance nonlinearity. The existing `v >= nv` guard already discards the darkest pixels.

**Issues identified and resolved:**

1. **Chain error propagation is structurally significant.** The spline for image i is fitted against radiance values computed by image i+1's already-fitted spline. For linear data, a 200-knot penalized cubic spline can introduce oscillatory artifacts (damped by the smoothing penalty but nonzero), and these propagate forward through the chain. **Resolution:** fit each image against the reference directly, not against its neighbor. This eliminates cumulative error.

2. **Top-25% sample restriction is a narrow band.** The code samples only `v >= max*0.75`, biasing toward near-saturation where analog highlight compression is most pronounced. For a 3-stop bracket, the overlap may be narrow enough to fall through to the scalar fallback anyway. **Resolution:** in linear mode, use the full unsaturated overlap range for the scalar fit.

3. **EXIF exposure ratios have practical errors (~1%).** Shutter speed quantization, aperture stepping precision, and flicker from artificial lighting can cause the true exposure ratio to deviate from EXIF by up to ~1%. The scalar linear fit corrects all of these using actual pixel data. **Resolution:** use EXIF as starting point, refine with scalar fit.

4. **R^2 threshold for spline activation.** The proposed R^2 < 0.995 threshold is a reasonable starting heuristic but needs empirical validation. A camera with lossy compressed RAW might show R^2 = 0.993 from quantization noise, not actual nonlinearity. **Resolution:** validate the threshold against the test corpus including at least one lossy-compressed camera if available. Document the threshold as configurable.

5. **Cross-frame radiance residual metric needs care.** Noise dominates at low signal in the overlap region; alignment residuals contaminate the metric on high-frequency texture. **Resolution:** restrict to spatially smooth overlap pixels (local variance below threshold); use inverse-variance weighting; report as histogram width vs Poisson noise floor. If the width is within the Poisson floor using EXIF ratios alone, no response estimation is needed — this serves as both metric and validation test.

### File-Level Changes

1. `src/Image.hpp`
- Add response mode enum (`linear_default`, `nonlinear_fit`).

2. `src/Image.cpp`
- In `computeResponseFunction(...)`:
  - **Primary path (linear_default):** scalar linear fit using full unsaturated overlap pixels, weighted by inverse variance. Each image fitted against reference directly (non-chained).
  - **Diagnostic:** compute R^2 of linear fit in overlap region; log deviation from EXIF ratio. If deviation > 2%, log a warning for investigation.
  - **Nonlinear path:** keep spline fitting behind explicit `--response-mode nonlinear` flag, gated by R^2 < threshold heuristic (default 0.995, configurable). When used, fit against reference directly (non-chained), not pairwise.
- Add numerical safeguards for low-sample and near-saturation regimes.

3. `src/ImageStack.cpp`
- Change chain direction: fit each image against reference directly, not sequentially against neighbor.

4. `src/LoadSaveOptions.hpp`
- Add `responseMode` option.

5. `src/Launcher.cpp`
- Add CLI switch `--response-mode linear|nonlinear`.

### Acceptance Criteria

1. Lower cross-frame radiance disagreement in overlap regions (measured in spatially smooth, unsaturated overlap pixels with inverse-variance weighting).
2. No highlight hue regressions.
3. Better stability on low-light brackets (no chain error amplification).
4. Overlap residual histogram width is within Poisson noise floor for linear cameras.

### Test Vectors

1. `TV-R1`: Normal daylight bracket (should be near-linear).
2. `TV-R2`: Very dark bracket set (low signal stress).
3. `TV-R3`: High highlight compression scene (near-saturation stress).

### Metrics

1. Overlap-region radiance residual histogram (width vs Poisson floor).
2. R^2 of linear fit per pair (diagnostic, should be > 0.995 for linear sensors).
3. Scalar fit deviation from EXIF ratio (diagnostic, should be < 2%).
4. Color error in highlight rolloff ROI.

### Proposed Actions

- [ ] Make scalar linear fit the primary path (currently only fallback for dark frames)
- [ ] Eliminate pairwise chaining: fit each image against reference directly
- [ ] Add R^2 linearity test to gate spline activation (threshold configurable, validate empirically)
- [ ] Use full unsaturated overlap range in linear mode (not just top 25%)
- [ ] Log EXIF ratio deviation as a diagnostic
- [ ] Keep spline behind explicit `--response-mode nonlinear` flag

---

## `M5 (P2)` Pattern-True X-Trans Sub-Pixel Interpolation

### Why

The current code uses `cfaStep=6` for X-Trans — a fixed distance to the nearest same-color neighbor in any direction. The X-Trans 6x6 repeat pattern (`src/CFAPattern.hpp:76`, stored as `xtrans[6][6]`) has non-uniform same-color spacing that varies by position and color within the tile. A fixed step of 6 always reaches a same-color pixel (the pattern repeats at period 6), but for green pixels this is 3-6x further than necessary. For Bayer (`cfaStep=2`) the fixed step is always correct since the pattern is regular.

### Review Findings

**The original claim that cfaStep=6 "can mis-sample by reaching a different-color pixel" was incorrect.** Because the X-Trans pattern has period 6, stepping exactly 6 in any cardinal direction always lands on the same color. The step is *correct*, just unnecessarily far for green pixels. The actual problem is interpolation quality degradation from using distant neighbors.

**Exact same-color distances computed for the standard X-Trans III/IV pattern:**

Across all 36 positions x 4 cardinal directions (144 direction-slots total):

| Distance | Count | Percent | Which colors |
|----------|-------|---------|--------------|
| 1        | 48    | 33.3%   | Green only   |
| 2        | 48    | 33.3%   | Green only   |
| 6        | 48    | 33.3%   | Red + Blue   |

- **Green pixels (24/36 positions, 96/144 direction-slots):** Nearest same-color is always at distance 1 or 2 (never 6). Each green pixel has two directions at distance 1 and two at distance 2. Using cfaStep=6 skips a neighbor that is 3-6x closer.
- **Red pixels (6/36 positions, 24/144 direction-slots):** Nearest same-color is always exactly 6 in all 4 directions. cfaStep=6 is already optimal for red.
- **Blue pixels (6/36 positions, 24/144 direction-slots):** Same as red — always exactly 6. cfaStep=6 is already optimal for blue.

**Conclusion:** cfaStep=6 is only suboptimal for green pixels (67% of all direction-slots), where it causes unnecessary over-smoothing by interpolating from a neighbor 6 pixels away when one exists at distance 1-2. For R/B pixels, no improvement is possible in cardinal directions.

**The scope is small and low-risk.** This affects only X-Trans cameras (Fuji) and only when `--sub-pixel` is enabled. No impact on the Bayer path. No Fuji RAF test files are currently available on connected volumes.

### File-Level Changes

1. `src/CFAPattern.hpp`
- Expose same-color neighborhood query API: given (x, y) and direction (+x, -x, +y, -y), return the distance to the nearest same-color pixel in that direction by scanning the periodic pattern.

2. `src/ImageStack.cpp`
- Replace `cfaStep` constant in `interpolateCFA(...)` with per-pixel, per-direction lookup from the CFA pattern.
- Keep fast Bayer path unchanged (cfaStep=2 is always correct).

3. `test/testImageStack.cpp`
- Add X-Trans-specific interpolation regression tests verifying same-color neighbor lookup for all 36 positions in the 6x6 tile, in all 4 cardinal directions.

### Acceptance Criteria

1. Reduced over-smoothing in X-Trans green channel sub-pixel interpolation (distance 1-2 neighbors used instead of distance 6).
2. No runtime regression >10% for Bayer workflows.
3. Precomputed 6x6x4 lookup table matches exact computed distances (48 slots at d=1, 48 at d=2, 48 at d=6).

### Test Vectors

1. `TV-X1`: Fuji X-Trans handheld bracket with fine foliage.
2. `TV-X2`: X-Trans architecture scene with repetitive detail.

*(Note: no X-Trans test files currently available. Acquiring Fuji RAF test brackets is a prerequisite for validation.)*

### Metrics

1. Chroma edge error in demosaiced output.
2. Runtime split by Bayer vs X-Trans datasets.

### Proposed Actions

- [ ] Implement per-pixel same-color neighbor lookup in CFAPattern
- [ ] Add unit test covering all 36 X-Trans tile positions x 4 directions
- [ ] Acquire Fuji X-Trans RAF test brackets for validation

---

## Cross-Cutting Test Harness

## Dataset Layout

Create fixed test corpus under `test/hdr_pipeline_vectors/`:

1. `alignment/TV-A1 ... TV-A3`
2. `deghost/TV-G1 ... TV-G6`
3. `noise/TV-N1 ... TV-N3`
4. `response/TV-R1 ... TV-R3`
5. `xtrans/TV-X1 ... TV-X2`

Each vector includes:

1. `inputs/` raw files
2. `roi.json` annotated evaluation regions
3. `expected.json` metric thresholds
4. optional `notes.md`

## CLI Regression Script

Add `scripts/benchmark-merge-pipeline.sh`:

1. builds target configuration,
2. runs all vectors with controlled flags,
3. writes `artifacts/<date>/metrics.csv`,
4. compares against baseline thresholds.

## Suggested command template

```bash
build/hdrmerge \
  --single \
  --align-features \
  --deghost 3.0 \
  --deghost-mode reference-robust \
  --response-mode linear \
  --clip-percentile 99.9 \
  -O artifacts/out \
  test/hdr_pipeline_vectors/<group>/<case>/inputs
```

*(Note: `--batch` is now the default when files are provided on the CLI.)*

## Rollout Strategy

1. Land each milestone behind a feature flag first.
2. Keep legacy path for one release cycle.
3. Flip defaults only after vector thresholds pass on full corpus.

## Recommended Order

0. **Hot pixel reorder** — move `correctHotPixels()` before `align()` in `ImageIO.cpp`. Standalone prerequisite, benefits all milestones.
1. `M4` linear-first response mode — **lowest risk, highest immediate quality gain, no new dependencies. Eliminates chain error propagation that currently affects every merge.**
2. `M3` calibrated noise model — **feeds into both M2 (deghost thresholds) and compose weighting. Should land before M2 so deghosting can be noise-aware from day one.**
3. `M1` feature alignment — requires OpenCV dependency; well-defined scope.
4. `M2` reference-guided deghosting — benefits from M3 noise model being available.
5. `M5` X-Trans interpolation refinement — smallest scope, lowest priority, blocked on acquiring X-Trans test data.

**Rationale for reorder:** The original M1-first order assumed alignment was the top quality gap. However, the chain error propagation in M4 affects every single merge (not just handheld), and M3's noise model is a prerequisite for M2's correctness in shadows. Doing M4 then M3 first gives the most quality improvement with the least risk, and ensures M2 lands on a solid foundation.

## Risks

1. OpenCV dependency variance across environments for `M1`.
2. Over-aggressive motion masking in `M2` if thresholds are not noise-aware — **mitigated by landing M3 before M2**.
3. Noise model S estimation sensitive to alignment quality for temporal-variance method in `M3` — **mitigated by cross-validating with spatial high-pass method**.
4. Response-mode compatibility regressions in `M4` for unusual RAW encodings (lossy compressed, highlight-priority modes) — **mitigated by configurable R^2 threshold and empirical validation**.
5. X-Trans CFA neighbor lookup edge cases in `M5` — **mitigated by exhaustive unit test over all 36x4 positions**.
6. Hot pixel correction reordering could theoretically affect alignment on images where hot pixels happened to aid MTB convergence — vanishingly unlikely; MTB uses global median, isolated defects have negligible effect.

## Definition of Done

1. All milestone acceptance criteria pass.
2. Full vector corpus passes threshold checks in CI.
3. No critical regressions in manual QA across Lightroom, darktable, and RawTherapee.

## Detailed Technical Notes

## Hot Pixel Correction Reorder

Move `correctHotPixels()` in `src/ImageIO.cpp` from line 199 (after response computation) to immediately after loading, before `align()` at line 177. The existing hot pixel detection algorithm (ratio-based spatial sigma-clipping against same-color Bayer neighbors across exposures) is sound and does not need changes — only the call site ordering.

## `M1` Alignment Method Details

1. Build grayscale preview from raw via 2x2 Bayer averaging; apply CLAHE normalization.
2. Use AKAZE as primary detector/descriptor, ORB fallback (if < 64 keypoints).
3. Matching:
- KNN ratio test (Lowe, threshold 0.75) forward and backward.
- Mutual consistency filtering (cross-check).
4. Geometry fit (progressive):
- `estimateAffinePartial2D` first (4 DOF: translation + rotation + uniform scale).
- Escalate to `estimateAffine2D` (6 DOF) if reprojection error > 2px RMS.
- Escalate to `findHomography` (8 DOF) if still poor.
- Fall back to MTB if all models fail confidence checks.
- Prefer MAGSAC++ (`cv::USAC_MAGSAC`) over vanilla RANSAC if OpenCV >= 4.5.
5. Refinement:
- ECC with `MOTION_EUCLIDEAN` (3 DOF) to refine translation + rotation residuals.

## `M2` Deghosting Method Details

1. Choose one reference exposure (default middle EV index; override by blur score; when best-exposed and least-blurry conflict, prefer least-blurry — motion blur in reference propagates to all fallback regions).
2. Compute exposure-normalized residual per bracket against reference.
3. Build motion confidence map from robust residual statistics, with noise-floor-aware threshold (from M3 calibrated model, or interim theoretical S + OB-based O if M3 not yet landed).
4. Apply Tukey biweight in high-confidence motion regions (ghostMap > 0.7); Huber in ambiguous regions (0.1 < ghostMap < 0.7).
5. For strongly inconsistent pixels, hard-select reference exposure (log % of reference-only pixels; acknowledge dynamic range loss).
6. Run configurable refinement iterations on confidence map (default 1; parameterize for difficult scenes).
7. Apply edge-preserving spatial regularization (guided/bilateral filter) on confidence, not on final radiance.

## `M3` Noise Model Details

1. Fit per-channel heteroscedastic model: `Var = S*signal + O`.
2. Combine:
- optical-black margin stats for `O` seed (using MAD, not sample variance, to resist glow contamination and hot pixel outliers),
- for `S` estimation: implement both temporal variance across aligned exposures (at smooth regions) and spatial high-pass-filter-per-tile (Android Camera2 approach); cross-validate and use the method with lower residual,
- ISO/gain-aware priors from metadata when available.
3. Use fitted variance model for merge weights: `w_k = t_k^2 / (S * z_k + O)` (MLE inverse-variance, Granados 2010).
4. For DNG NoiseProfile output: compute effective (S_merged, O_merged) scaled by weight-based effective exposure count. Document that this is a global approximation for a spatially varying quantity. Err toward slight overestimation (ACR/Lightroom use NoiseProfile for default NR; users can reduce but may not think to increase).
5. Fallback: if calibration is unavailable or fails validation, use Poisson-only weight `w_k = t_k / z_k` (Hanji 2020).

## `M4` Response Model Details

1. Primary path: scalar linear fit per pair, each image fitted against reference directly (non-chained). Uses inverse-variance-weighted overlap pixels across the full unsaturated overlap range (not just top 25%).
2. Diagnostic: compute R^2 of linear fit; log deviation from EXIF ratio.
3. Nonlinear path enabled only when R^2 < threshold (default 0.995, configurable) or via explicit `--response-mode nonlinear` flag. Validate threshold empirically against lossy-compressed test files.
4. Guards:
- minimum sample count,
- saturation-proximity exclusion,
- low-light degeneracy fallback.
5. When nonlinear path is used, fit against reference directly (non-chained) to prevent error accumulation.

## `M5` X-Trans Interpolation Details

1. Replace fixed cfaStep=6 with per-pixel, per-direction same-color neighbor lookup from the 6x6 CFA pattern. cfaStep=6 is always *correct* (same color guaranteed at period 6) but unnecessarily far for green pixels (67% of direction-slots have same-color at distance 1 or 2). For R/B pixels, cfaStep=6 is already optimal — no same-color cardinal neighbor exists at any shorter distance.
2. The lookup can be precomputed as a static 6x6x4 table (36 positions x 4 directions = 144 entries, values are 1, 2, or 6). No runtime scanning needed.
3. Keep current Bayer fast path unchanged and branch only for X-Trans.

## SOTA Reference Mapping

1. `M1`:
- Ward MTB baseline (translation-only): Ward 2003.
- Feature matching + robust geometric fit: OpenCV AKAZE/ORB docs and RANSAC APIs.
- AKAZE vs ORB comparison: Tareen & Saleem (ICSPC 2018).
- Feature matching best practices: Mishkin "How to Match" analysis; Lowe (IJCV 2004).
- MAGSAC++: Barath et al. (CVPR 2020).
- ECC refinement: Evangelidis & Psarakis (TPAMI 2008).
- Learned matchers (literature reference only, revisit in 6 months): LoFTR (CVPR 2021), LightGlue (ICCV 2023).
2. `M2`:
- Classical deghosting survey: Tursun et al. (CGF 2015).
- Objective evaluation: Karaduzovic-Hadziabdic et al. (C&G 2017).
- Probabilistic ghost removal: Khan et al. (ICIP 2006).
- Noise-model-informed detection: Granados et al. (ACM TOG 2013).
- Saturation + motion handling: Hu et al. (CVPR 2013).
- Patch-based HDR (quality ceiling): Sen et al. (ACM TOG 2012).
- MRF ghost map: Min et al. (JIVP 2014).
- Deep learning quality benchmark: DeepDuoHDR (IEEE TIP 2024).
3. `M3`:
- Noise-model-driven merge precedent: HDR+ (SIGGRAPH Asia 2016).
- MLE optimal weighting: Granados et al. (CVPR 2010).
- CRLB performance bounds: Aguerrebere et al. (SIIMS 2014).
- Calibration-free near-optimal merge: Hanji et al. (ECCV 2020).
- Sensor noise standard: EMVA 1288.
- Android noise model calibration: Camera2 `dng_noise_model.py`.
4. `M4`:
- HDR response/merge foundations: Debevec-Malik (SIGGRAPH 1997), Robertson et al. (JEI 2003).
- Robertson's explicit note that response estimation can be skipped for linear RAW cameras.
5. `M5`:
- CFA-consistent processing requirement from raw-domain literature and DNG pipeline practices.

## References (Primary Sources)

1. Adobe Digital Negative (DNG) specification landing page:
- https://helpx.adobe.com/camera-raw/digital-negative.html
2. Adobe DNG 1.7.1.0 specification (tags, compression, metadata):
- https://helpx.adobe.com/content/dam/help/en/camera-raw/digital-negative/jcr_content/root/content/flex/items/contentmenu/contentmenuitem_1644459277/body/content/contentmenuitem_1376527252/content/data/dng_spec_1_7_1_0.pdf
3. Ward, G. (2003), MTB alignment, Journal of Graphics Tools Vol. 8 No. 2:
- https://www.cg.tuwien.ac.at/courses/Visualisierung2/HallOfFame/2005/ElKoura/ward03.pdf
4. Lowe, D. (2004), Distinctive Image Features from Scale-Invariant Keypoints, IJCV:
- https://doi.org/10.1023/B:VISI.0000029664.99615.94
5. OpenCV feature and motion estimation APIs:
- AKAZE: https://docs.opencv.org/4.x/d8/d30/classcv_1_1AKAZE.html
- ECC / findTransformECC: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
- estimateAffinePartial2D: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
6. Evangelidis, G. D. & Psarakis, E. Z. (TPAMI 2008), ECC image alignment:
- https://doi.org/10.1109/TPAMI.2008.113
7. Tareen, S.A.K. & Saleem, Z. (ICSPC 2018), AKAZE/ORB comparison:
- https://doi.org/10.1109/ICSPC.2018.8526772
8. Barath, D. et al. (CVPR 2020), MAGSAC++:
- https://doi.org/10.1109/CVPR42600.2020.00138
9. LoFTR (CVPR 2021) — literature reference only:
- https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html
10. LightGlue (ICCV 2023) — literature reference only:
- https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html
11. DeepDuoHDR (IEEE TIP 2024) — quality benchmark only:
- https://doi.org/10.1109/TIP.2024.3497838
12. Tursun et al. (CGF 2015), deghosting survey:
- https://doi.org/10.1111/cgf.12593
13. Karaduzovic-Hadziabdic et al. (C&G 2017), deghosting evaluation:
- https://doi.org/10.1016/j.cag.2017.05.019
14. Khan et al. (ICIP 2006), probabilistic ghost removal:
- https://doi.org/10.1109/ICIP.2006.312441
15. Granados et al. (ACM TOG 2013), noise-aware ghost detection:
- https://doi.org/10.1145/2508363.2508410
16. Hu et al. (CVPR 2013), HDR deghosting with saturation:
- https://doi.org/10.1109/CVPR.2013.163
17. Sen et al. (ACM TOG 2012), robust patch-based HDR:
- https://doi.org/10.1145/2366145.2366222
18. Min et al. (JIVP 2014), probabilistic motion pixel detection:
- https://doi.org/10.1186/1687-5281-2014-42
19. HDR+ burst merge pipeline (SIGGRAPH Asia 2016):
- https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/
20. Granados et al. (CVPR 2010), MLE optimal HDR reconstruction:
- https://vcai.mpi-inf.mpg.de/projects/opthdr/
21. Aguerrebere et al. (SIAM J. Imaging Sciences, 2014), CRLB performance bounds:
- https://doi.org/10.1137/130938203
22. Hanji et al. (ECCV 2020), noise-aware merging without calibration:
- https://www.cl.cam.ac.uk/~rkm38/pdfs/hanji2020_noise_aware_HDR_merging.pdf
23. Debevec and Malik (SIGGRAPH 1997):
- https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf
24. Robertson et al. (Journal of Electronic Imaging, 2003):
- https://doi.org/10.1117/1.1557695
25. EMVA 1288 standard (sensor noise characterization):
- https://www.emva.org/standards-technology/emva-1288/

## Internal Research Sources

1. `docs/research-modern-hdr-techniques.md`
2. `docs/hdr-merge-improvements-phase2-research.md`
3. `docs/exposure-normalization-and-dng-metadata.md`
