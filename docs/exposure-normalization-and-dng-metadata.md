# HDR Exposure Normalization & DNG Metadata — Research and Proposed Fixes

**Date**: 2026-02-21
**Context**: Investigating why merged HDR images appear too dark when scenes contain blown-out highlights (e.g., windows), and what can be done within the DNG output pipeline to fix it.
**Related**: [`docs/research-modern-hdr-techniques.md`](research-modern-hdr-techniques.md), [`docs/implementation-plan.md`](implementation-plan.md)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Academic Background](#3-academic-background)
4. [DNG Metadata for HDR](#4-dng-metadata-for-hdr)
5. [Proposed Fix 1: Percentile-Based Normalization](#5-proposed-fix-1-percentile-based-normalization)
6. [Proposed Fix 2: BaselineExposure Tag](#6-proposed-fix-2-baselineexposure-tag)
7. [Proposed Fix 3: NoiseProfile Tag](#7-proposed-fix-3-noiseprofile-tag)
8. [What NOT to Do: Tone Mapping in the Merge](#8-what-not-to-do-tone-mapping-in-the-merge)
9. [Implementation Plan](#9-implementation-plan)
10. [References](#10-references)

---

## 1. Problem Statement

When merging exposure brackets where one region of the scene is significantly brighter than the rest (e.g., a window in an interior shot, a bright sky behind a shaded foreground), the merged DNG output appears **uniformly too dark**. The blown-out area is handled correctly (data is taken from shorter exposures), but the brightness of that area dominates the output scaling, compressing the rest of the image into the bottom fraction of the output range.

### Concrete example

Consider an interior shot with a window:
- Room interior: ~200 radiance units (well-exposed across multiple brackets)
- Window area: ~20,000 radiance units (short exposure provides valid data)
- Ratio: 100:1

The current normalization maps 20,000 to `params.max` (e.g., 16383 for 14-bit). The room interior maps to ~164, using only ~1% of the available range. When opened in a raw processor, the image appears nearly black except for the window.

The per-pixel merge itself is correct — the problem is entirely in how the merged radiance values are scaled into the output range.

---

## 2. Root Cause Analysis

### The merge pipeline (working correctly)

`ImageStack::compose()` (lines 586–663 in `src/ImageStack.cpp`) performs a Poisson-optimal weighted merge:

1. **Weight assignment**: Each exposure gets base weight `1/relativeExposure` (proportional to exposure time — longer exposures captured more photons, lower relative noise).

2. **Saturation rolloff**: As raw values approach `satThreshold`, weights smoothly reduce via quadratic falloff `w *= t²` where `t = (satThreshold - raw) / range`. This prevents hard edges at saturation boundaries.

3. **Ghost detection** (optional): MAD-based sigma clipping rejects outlier exposures when 3+ are available.

4. **Weighted average**: `output = sum(w_k * radiance_k) / sum(w_k)` — standard MLE estimator for Poisson noise.

This is all sound. The merged radiance values faithfully represent the scene.

### The normalization (the problem)

Lines 674–681:

```cpp
float mult = (params.max - params.maxBlack) / maxVal;
#pragma omp parallel for
for (size_t y = 0; y < params.rawHeight; ++y) {
    for (size_t x = 0; x < params.rawWidth; ++x) {
        dst(x, y) *= mult;
        dst(x, y) += params.blackAt(x - params.leftMargin, y - params.topMargin);
    }
}
```

`maxVal` is the **global maximum** radiance across all pixels, computed via OpenMP reduction on line 579. The entire image is divided by this single value.

**Why this causes darkening**: A single bright pixel (or small bright region) determines `maxVal`. The `mult` factor becomes tiny, compressing the entire output range. The bright area maps correctly to near-white, but everything else maps to near-black.

This is equivalent to a global tone curve anchored at the scene maximum — the same problem Reinhard et al. (2002) identified and solved with percentile-based key estimation.

### Missing DNG metadata

`DngFloatWriter::createMainIFD()` and `createRawIFD()` write no exposure-related metadata tags:

- No `BaselineExposure` (tag 50730) — raw processors don't know where "normal" brightness should be
- No `BaselineNoise` (tag 50731) — raw processors can't calibrate noise reduction for the merged result
- No `NoiseProfile` (tag 51041) — same issue

For floating-point DNGs (which HDRMerge produces), the absence of `BaselineExposure` means raw processors assume 0.0 EV, rendering the data at face value. Since the normalization already compressed everything dark, the image opens dark with no automatic compensation.

---

## 3. Academic Background

### Reinhard et al. (2002) — Scene Key and the Percentile Problem

Reinhard's global tone mapping operator computes the "key" of a scene (its overall brightness) from the log-average luminance:

```
key = exp( (1/N) * sum(log(L_i + epsilon)) )
```

He showed that when extreme highlights are included in this computation, the key is inflated, causing the tone curve to shift and darken the rest of the image. His solution: **exclude the top and bottom percentiles** when computing the key value.

This directly applies to our normalization problem. Using the global maximum (`maxVal`) is the most extreme version of this mistake — anchoring the entire scale to a single pixel.

### Debevec & Malik (1997) — HDR Radiance Maps

The original HDR merge paper uses a triangle weight function `w(z) = min(z, Z_max - z)` that peaks at mid-range values. Saturated pixels get zero weight. The merged output is a physical radiance map in arbitrary units — the paper explicitly notes that **display mapping is a separate step**.

HDRMerge's approach (Poisson-optimal weights, saturation rolloff) is better-motivated than Debevec's triangle function, but the principle is the same: the merge produces scene-referred radiance, and mapping to display (or output) range is a distinct concern.

### Mertens, Kautz & Van Reeth (2007) — Exposure Fusion

Exposure fusion bypasses HDR entirely by directly blending LDR exposures using quality-guided weight maps (contrast, saturation, well-exposedness). It uses **Laplacian pyramid blending** for seamless transitions.

The well-exposedness weight uses a Gaussian centered at 0.5:

```
E = exp(-(I - 0.5)² / (2 * 0.2²))
```

A pixel at 1.0 (blown) gets weight ~0.044 per channel, ~0.00009 combined. Blown pixels are effectively excluded without affecting the rest of the image.

This is a **display-referred** technique — not appropriate for linear DNG output. But it demonstrates the principle: blown areas should be handled locally, not allowed to affect global scaling.

### Durand & Dorsey (2002) — Bilateral Filter Tone Mapping

Decomposes the log-luminance image into base (large-scale) and detail (local contrast) layers using a bilateral filter. Only the base layer is compressed. This preserves local contrast while reducing global dynamic range.

Again a display technique, but relevant insight: the problem is fundamentally about **global vs. local** treatment of brightness. A global normalization factor lets one region dictate the entire image's appearance.

### Fattal, Lischinski & Werman (2002) — Gradient Domain Compression

Attenuates large gradients (big luminance jumps like window edges) while preserving small gradients (texture, local detail). Reconstructs via Poisson equation. Produces natural-looking results without halos.

Not applicable to raw DNG output, but reinforces that the window-darkening problem is well-studied and well-solved in the literature.

---

## 4. DNG Metadata for HDR

The DNG specification defines several tags relevant to HDR-merged floating-point output. None are currently written by HDRMerge.

### BaselineExposure (Tag 50730)

**Type**: SRATIONAL, Count: 1, Default: 0.0

From the DNG spec: "Camera models vary in the trade-off they make between highlight headroom and shadow noise. Some cameras allow a significant amount of highlight headroom, which requires a positive BaselineExposure value to move the zero point."

**Effect**: Specifies by how much (in EV/stops) to shift the zero point of the raw processor's exposure slider. A value of +2.0 means "render this file 2 stops brighter than the raw data would suggest by default."

**For HDR-merged DNGs**: This is the primary mechanism for controlling default rendering brightness. If you store the full extended dynamic range linearly (with most scene content concentrated in the lower values), `BaselineExposure` tells the processor where "normal" should be.

**How Lightroom's HDR merge uses it**: Adobe stores the full dynamic range in a float DNG and sets `BaselineExposure` to compensate, so the image opens at a natural brightness while preserving full highlight headroom for the user to adjust.

**Key considerations**:
- Values are scene-dependent, not just camera-dependent
- Aggressive values can cause color shifts at high exposure levels (documented by DiglLoyd, 2024)
- RawTherapee initially ignored `BaselineExposure` entirely, causing float HDR DNGs to appear "extremely dark" — exactly our symptom (GitHub issue #2809)

### BaselineNoise (Tag 50731)

**Type**: RATIONAL, Count: 1, Default: 1.0

Specifies the relative noise level of the camera model. Values below 1.0 indicate cleaner-than-typical output. Raw processors use this to calibrate noise reduction strength.

For HDR-merged DNGs, this should be reduced (e.g., 0.5–0.7) since the merge reduces noise through weighted averaging of multiple exposures. A 3-bracket merge with substantial overlap can achieve ~1.7x (sqrt(3)) SNR improvement.

### NoiseProfile (Tag 51041)

**Type**: DOUBLE, Count: 2 * ColorPlanes

Specifies the noise model parameters `(a, b)` for each color plane:

```
variance(signal) = a + b * signal
```

Where `a` represents read noise and `b` represents photon shot noise. For HDR-merged images, both parameters should decrease relative to a single exposure.

This is more precise than `BaselineNoise` and is preferred by modern raw processors. Computing it requires knowledge of the source camera's noise model (available from the input NEFs' metadata or from databases like photonstophotos.net).

### WhiteLevel (Tag 50717)

Currently set to `params->max` (line 305 in `DngFloatWriter.cpp`). For floating-point DNGs the convention is that WhiteLevel represents the maximum non-clipped value. The current value is correct for the integer-range output the normalization produces, but would need revisiting if normalization changes.

---

## 5. Proposed Fix 1: Percentile-Based Normalization

### Concept

Replace the global maximum normalization with a percentile-based approach. Instead of:

```cpp
float mult = (params.max - params.maxBlack) / maxVal;
```

Use:

```cpp
float mult = (params.max - params.maxBlack) / percentile_999;
```

Where `percentile_999` is the 99.9th percentile of the merged radiance distribution. Values above this percentile clip to white.

### Why 99.9th percentile

- **99th percentile** (too aggressive): Clips 1% of pixels. In a 45MP image, that's 450,000 pixels — could lose legitimate highlight detail in non-trivial areas.
- **99.9th percentile** (recommended): Clips 0.1% of pixels (45,000 in a 45MP image). This is almost always isolated specular highlights, light sources, or the very center of a blown window — areas where clipping is acceptable or even desirable.
- **99.99th percentile** (too conservative): Only clips 4,500 pixels. May not be aggressive enough to prevent darkening from large bright regions like windows that span thousands of pixels.

The 99.9th percentile is the standard choice in scientific imaging (astronomy, microscopy) and matches what tools like `dcraw`, `RawTherapee`, and `darktable` use internally for auto-exposure.

### Implementation approach

Computing exact percentiles requires sorting all pixel values (O(N log N)) or building a histogram. For a ~45MP image, a histogram approach is efficient:

```cpp
// After the compose loop, before normalization:
// Build histogram of merged radiance values (quantized to e.g. 65536 bins)
const int numBins = 65536;
std::vector<size_t> histogram(numBins, 0);
float binScale = (numBins - 1) / maxVal;
for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
        int bin = std::min((int)(dst(x, y) * binScale), numBins - 1);
        histogram[bin]++;
    }
}

// Find 99.9th percentile
size_t totalPixels = width * height;
size_t target = (size_t)(totalPixels * 0.999);
size_t cumulative = 0;
int percentileBin = numBins - 1;
for (int i = 0; i < numBins; ++i) {
    cumulative += histogram[i];
    if (cumulative >= target) {
        percentileBin = i;
        break;
    }
}
float percentile999 = (percentileBin + 0.5f) / binScale;
float mult = (params.max - params.maxBlack) / percentile999;
```

### Tradeoffs

| Pro | Con |
|-----|-----|
| Simple, well-understood technique | Clips the top 0.1% of pixels to white |
| Directly prevents the darkening problem | Percentile choice is scene-dependent (99.9% is a good default but not universal) |
| Preserves linearity of the output data | Very uniformly bright scenes (no outliers) may see negligible change |
| No additional DNG metadata needed | Slightly more computation (histogram pass) — negligible vs. merge time |
| Compatible with all raw processors | |

### Impact on quality

For the window-in-interior scenario: the window pixels above the 99.9th percentile clip to white, while the room interior now uses the full output range. The improvement is dramatic — the image goes from appearing nearly black to properly exposed.

For scenes without extreme highlights: the 99.9th percentile will be very close to the global maximum, so the behavior is nearly identical to the current normalization. No regression.

---

## 6. Proposed Fix 2: BaselineExposure Tag

### Concept

Write the DNG `BaselineExposure` tag (50730) to tell raw processors how many stops to shift the default exposure. This compensates for the fact that the merged data concentrates most scene content in the lower portion of the output range (to preserve highlight headroom).

### Computing the value

The goal is to find how many stops the "typical" scene content sits below the midpoint of the output range:

```cpp
// Compute median or geometric mean of the merged radiance
// (after normalization, before writing DNG)
double logSum = 0.0;
size_t count = 0;
for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
        float v = dst(x, y) - params.blackAt(...);  // above-black value
        if (v > 0) {
            logSum += std::log2((double)v);
            count++;
        }
    }
}
double geometricMean = std::exp2(logSum / count);

// Target: geometric mean should sit at ~18% of output range (photographic mid-gray)
double targetMidgray = 0.18 * (params.max - params.maxBlack);
double baselineExposure = std::log2(targetMidgray / geometricMean);

// Clamp to reasonable range
baselineExposure = std::max(-5.0, std::min(5.0, baselineExposure));
```

### Writing the tag

In `DngFloatWriter::createMainIFD()`:

```cpp
// BaselineExposure — shift raw processor default exposure
int32_t bleRational[2] = {
    (int32_t)std::round(baselineExposure * 100),
    100
};
mainIFD.addEntry(50730, IFD::SRATIONAL, 1, bleRational);
```

### Interaction with Fix 1

These two fixes are complementary:

- **Fix 1 alone**: Prevents extreme darkening. The image opens at a reasonable brightness, but the raw processor's default exposure (0 EV) may still be slightly off for high-DR scenes.
- **Fix 2 alone**: Tells the raw processor to boost brightness, but the underlying data is still compressed by the global max normalization. The processor can boost it, but may amplify quantization artifacts in low-precision formats (16-bit float).
- **Fix 1 + Fix 2 together**: The normalization uses the full output range effectively (Fix 1), and the raw processor opens at exactly the right brightness (Fix 2). This is the ideal combination.

### Tradeoffs

| Pro | Con |
|-----|-----|
| Raw processor opens at correct brightness | Requires computation of scene statistics (geometric mean) |
| User can still adjust exposure freely | Some raw processors may not honor `BaselineExposure` (rare) |
| Full highlight headroom preserved in float data | Value is scene-dependent — may not be perfect for all scenes |
| Standard DNG mechanism — this is what it's designed for | RawTherapee only added support recently (issue #2809) |
| No clipping of any pixel values | |

### Compatibility

- **Adobe Camera Raw / Lightroom**: Full support. This is how Adobe's own HDR merge works.
- **RawTherapee**: Supported since ~2018 (after issue #2809 was resolved).
- **darktable**: Supported.
- **Capture One**: Supported for DNG files.
- **Processors that ignore it**: The image will appear darker than intended but the data is all there — the user can manually boost exposure.

---

## 7. Proposed Fix 3: NoiseProfile Tag

### Concept

Write the DNG `NoiseProfile` tag (51041) to inform raw processors that the merged result has lower noise than a single exposure. This improves downstream noise reduction: the processor applies less aggressive NR, preserving detail.

### Computing the value

The noise model for a single exposure is:

```
variance(signal) = a + b * signal
```

Where `a` is read noise variance and `b` is photon shot noise coefficient. For a weighted merge of N exposures:

```
variance_merged ≈ variance_single / N_effective
```

Where `N_effective` is the effective number of exposures contributing at a given signal level. For Poisson-optimal weighting with significant overlap, `N_effective ≈ N` in well-exposed regions and `≈ 1` in regions where only one exposure has valid data.

A conservative estimate:

```cpp
// N_eff is between 1 and numImages, depending on scene content.
// Use sqrt(numImages) as a conservative middle ground for the overall profile.
double nEff = std::sqrt((double)numImages);

// Source noise parameters (from input NEF metadata or camera database)
double a_source = readNoiseVariance;  // e.g., from Exif or photonstophotos.net
double b_source = shotNoiseCoeff;     // typically ~1.0 for normalized data

double a_merged = a_source / nEff;
double b_merged = b_source / nEff;
```

### Writing the tag

In `DngFloatWriter::createMainIFD()`:

```cpp
// NoiseProfile — per color plane (a, b) pairs
double noiseProfile[2 * params->colors];
for (int c = 0; c < params->colors; ++c) {
    noiseProfile[2*c]     = a_merged;  // read noise component
    noiseProfile[2*c + 1] = b_merged;  // shot noise component
}
mainIFD.addEntry(51041, IFD::DOUBLE, 2 * params->colors, noiseProfile);
```

### Simpler alternative: BaselineNoise

If computing per-channel noise model parameters is too complex, the simpler `BaselineNoise` tag (50731) can be used:

```cpp
// BaselineNoise — relative noise level (1.0 = normal, <1.0 = cleaner)
uint32_t baselineNoise[2] = {
    (uint32_t)std::round(1000.0 / std::sqrt((double)numImages)),
    1000
};
mainIFD.addEntry(50731, IFD::RATIONAL, 1, baselineNoise);
```

For a 3-bracket merge: `BaselineNoise = 1/sqrt(3) ≈ 0.577`
For a 5-bracket merge: `BaselineNoise = 1/sqrt(5) ≈ 0.447`

### Tradeoffs

| Pro | Con |
|-----|-----|
| Raw processors apply appropriate NR strength | Noise reduction varies by signal level — a single factor is approximate |
| Preserves detail that would otherwise be smoothed away | `N_effective` varies spatially (shadows vs. highlights) |
| Simple to implement (especially `BaselineNoise`) | Not all processors use `NoiseProfile`/`BaselineNoise` |
| Correct behavior — merged images genuinely have less noise | Computing accurate per-channel parameters requires camera noise model data |

---

## 8. What NOT to Do: Tone Mapping in the Merge

It may be tempting to add local tone mapping (bilateral filter, gradient domain compression, exposure fusion) to solve the darkening problem. **This would be wrong for HDRMerge's purpose.**

### Why not

1. **HDRMerge produces scene-referred linear DNG data.** The entire point is to let the user's raw processor (Lightroom, RawTherapee, darktable, Capture One) handle the creative rendering. Baking in tone mapping removes this flexibility.

2. **Tone mapping is a lossy, irreversible transformation.** Once applied, the user cannot recover the original linear relationship between scene luminance values. This conflicts with the project's quality-first philosophy.

3. **CFA-domain complications.** HDRMerge works on raw Bayer mosaic data (pre-demosaic). Most tone mapping algorithms assume demosaiced RGB or luminance images. Applying them to CFA data would require demosaicing first, tone mapping, then re-mosaicing — introducing artifacts at every step.

4. **DNG readers expect linear data.** The DNG specification for CFA images assumes the pixel values are proportional to scene radiance (after black subtraction and before white balance). Tone-mapped values would confuse the color pipeline.

### The correct separation of concerns

```
HDRMerge's job:       Scene radiance estimation (linear, noise-optimal)
                      + proper normalization and DNG metadata

Raw processor's job:  Tone mapping, local contrast, highlight recovery,
                      shadow lifting, color grading, noise reduction
```

The fixes proposed in this document (percentile normalization + BaselineExposure + NoiseProfile) stay firmly on HDRMerge's side of this boundary. They ensure the raw processor receives well-scaled data with correct metadata, enabling it to do its job properly.

---

## 9. Implementation Plan

### Priority and ordering

| Step | Fix | Effort | Impact | Risk |
|------|-----|--------|--------|------|
| 1 | Percentile normalization | Small (1 file, ~30 lines) | High — directly fixes darkening | Very low — fallback is current behavior |
| 2 | BaselineExposure tag | Small (1 file, ~15 lines) | Medium — improves default rendering | Very low — ignored by processors that don't support it |
| 3 | BaselineNoise tag | Small (1 file, ~5 lines) | Low-medium — improves NR behavior | None — purely additive metadata |
| 4 | NoiseProfile tag (optional) | Medium (needs noise model) | Low-medium — more precise NR calibration | Low — requires per-camera noise data |

### Step 1: Percentile normalization

**File**: `src/ImageStack.cpp`, `compose()` function

**Change**: After the compose loop (line 670), add a histogram pass to compute the 99.9th percentile, then use it instead of `maxVal` for the normalization factor. Keep `maxVal` available for logging/diagnostics.

**Testing**: Compare output DNGs from the 3-bracket and 5-bracket test sets (interior scenes with windows ideally) opened in Lightroom/RawTherapee. Verify:
- Scenes with extreme highlights are no longer crushed dark
- Scenes without extreme highlights produce near-identical output (regression check)
- Pixel values above the percentile clip cleanly to white without banding

**CLI option**: Consider adding `--clip-percentile <0.0-1.0>` (default 0.999) for user control. A value of 1.0 gives current behavior (no clipping).

### Step 2: BaselineExposure tag

**File**: `src/DngFloatWriter.cpp`, `createMainIFD()`

**Change**: Accept a `baselineExposure` value (computed in `compose()` or passed through), add the SRATIONAL tag 50730 to the main IFD.

**Computation site**: Either in `ImageStack::compose()` (has access to pixel data) or as a separate pass. The geometric mean of above-black pixel values, compared to 18% gray target, gives the EV shift.

**Testing**: Open output DNGs in Lightroom and RawTherapee. Verify the default exposure slider position corresponds to a natural-looking rendering. Compare with manually adjusting exposure on current (no-tag) output.

### Step 3: BaselineNoise tag

**File**: `src/DngFloatWriter.cpp`, `createMainIFD()`

**Change**: Compute `1.0 / sqrt(numImages)` and write as tag 50731. Requires passing the number of merged exposures to the writer.

**Testing**: Open in Lightroom, compare noise reduction behavior between single-exposure DNG and merged DNG. The merged version should show less aggressive NR at default settings.

### Relationship to existing implementation plan

These fixes correspond to **Phase 7: Format Evolution** in [`docs/implementation-plan.md`](implementation-plan.md), specifically Steps 15–16 (DNG format updates). They can be implemented independently of the streaming DNG writer work (Step 14) and the other optimizations in Phases 1–6.

However, if the streaming writer (`opt/step14-streaming-dng-writer` branch) modifies the normalization or DNG writing pipeline, these changes should be coordinated to avoid conflicts.

---

## 10. References

### Core papers

- Debevec, P. E. & Malik, J. (1997). "Recovering High Dynamic Range Radiance Maps from Photographs." SIGGRAPH 1997. https://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf

- Mertens, T., Kautz, J. & Van Reeth, F. (2007). "Exposure Fusion." Pacific Graphics 2007. https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf

- Reinhard, E., Stark, M., Shirley, P. & Ferwerda, J. (2002). "Photographic Tone Reproduction for Digital Images." SIGGRAPH 2002.

- Durand, F. & Dorsey, J. (2002). "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images." SIGGRAPH 2002. https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf

- Fattal, R., Lischinski, D. & Werman, M. (2002). "Gradient Domain High Dynamic Range Compression." SIGGRAPH 2002. https://www.cs.huji.ac.il/~danix/hdr/hdrc.pdf

### Noise-optimal merging

- Hanji, P., Zhong, F. & Mantiuk, R. K. (2020). "Noise-Aware Merging of High Dynamic Range Image Stacks without Camera Calibration." ECCV 2020.

- Granados, M., Ajdin, B., Wand, M., Theobalt, C., Seidel, H.-P. & Lensch, H. P. A. (2010). "Optimal HDR Reconstruction with Linear Digital Cameras." CVPR 2010.

- Aguerrebere, C., Delon, J., Gousseau, Y. & Musé, P. (2014). "Best Algorithms for HDR Image Generation. A Study of Performance Bounds." SIAM J. Imaging Sciences.

### Improved weight functions

- Multi-Exposure Image Fusion Algorithm Based on Improved Weight Function (2022). PMC/Sensors. https://pmc.ncbi.nlm.nih.gov/articles/PMC8957254/

### DNG specification and compatibility

- Adobe DNG Specification 1.7.0.0. https://helpx.adobe.com/content/dam/help/en/photoshop/pdf/DNG_Spec_1_7_0_0.pdf

- RawTherapee floating-point DNG discussion. https://discuss.pixls.us/t/help-with-a-floating-point-dng-file-and-processing-it-in-rt/9522

- RawTherapee BaselineExposure support issue. https://github.com/Beep6581/RawTherapee/issues/2809

- RawDigger BaselineExposure analysis. https://www.rawdigger.com/howtouse/deriving-hidden-ble-compensation

- DiglLoyd BaselineExposure color shift analysis. https://diglloyd.com/blog/2024/20240220_0900-BaselineExposure.html
