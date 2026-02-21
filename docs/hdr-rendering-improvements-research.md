# HDR Rendering Improvements: Research & Rationale

## Problem Statement

HDRMerge produces linear, scene-referred floating-point DNGs from bracketed RAW
exposures. The merge preserves the full dynamic range, but when opened in Adobe
Camera Raw (ACR) / Lightroom, high-DR scenes (e.g. interior with bright windows)
don't render well by default. The information IS in the merged data — ACR just
doesn't receive the right guidance on how to render it.

Three metadata-level improvements address this. No pixel data changes required.

---

## 1. DefaultBlackRender = None (DNG tag 51110)

### Background
ACR applies automatic black-point subtraction by default when opening any DNG.
For HDR merges that already have carefully preserved shadow detail, this crushes
the lower end of the tonal range — exactly the detail the merge was meant to save.

### Solution
DNG Specification 1.4+ defines tag 51110 (`DefaultBlackRender`) with two values:
- `0` = Auto (default; ACR applies black-point subtraction)
- `1` = None (ACR skips black-point subtraction)

Setting this to `1` tells ACR to trust the black levels as-is.

### References
- Adobe DNG Specification 1.4, Section "DefaultBlackRender" (tag 51110)
- DNG SDK `dng_shared.h`: `fDefaultBlackRender`

---

## 2. Median-Based BaselineExposure

### Background
`BaselineExposure` (tag 50730) tells ACR how many EV stops to shift the default
rendering brightness. HDRMerge computes this from the merged pixel distribution
so ACR opens the image at a natural brightness.

### Previous approach: Geometric mean (= mean of log-luminance)
The geometric mean of all active-area pixel values serves as the "typical" scene
luminance. However, in high-DR scenes the distribution is heavily skewed: dark
pixels vastly outnumber bright ones (e.g. an interior room with small bright
windows). The geometric mean gets pulled toward shadows, making the default
rendering too bright.

### Improved approach: Median of log-luminance
The median is robust to distributional skew. For a uniform scene the median and
geometric mean nearly coincide; for a skewed HDR scene the median gives a more
perceptually natural "middle" brightness.

### Algorithm: Two-pass histogram-based median
1. **Pass 1** (OpenMP min/max reduction): Find `logMin`, `logMax` over all
   active-area above-black pixels, and count `totalPixels`.
2. **Pass 2** (per-thread histograms, critical merge): Build a 10000-bin
   histogram of log-luminance values, find the 50th-percentile bin, interpolate
   for sub-bin precision.
3. **Same formula**: `baselineExposureEV = log2(0.18 * range / medianValue)`,
   clamped to [-5, +5].

10000 bins give ~0.001 EV precision for a 14-stop scene — more than sufficient.

### References
- Debevec & Malik 1997: HDR radiance map assembly, exposure weighting
- Robertson et al. 2003: Robust median estimators for HDR reconstruction
- DNG Specification: `BaselineExposure` (tag 50730) — SRATIONAL, units of EV

---

## 3. Built-in Default HDR XMP Settings

### Background
ACR's default rendering settings assume a single-exposure RAW with normal DR.
HDR merges benefit from specific rendering defaults: extended slider ranges,
highlight recovery, shadow lifting, and a gentle tone curve.

Previously, users had to place a `default_profile.xmp` file next to the binary.
Hardcoding sensible defaults ensures every merge gets good rendering out of the
box, while still allowing user overrides.

### Priority chain (last writer wins)
1. `copyAllMetadata()` — from source RAW (lowest priority)
2. `injectDefaultHDRSettings()` — hardcoded HDR defaults (`setIfAbsent` pattern)
3. `injectACRProfile()` — from `-L` or `default_profile.xmp` (erase+replace)
4. `injectAdaptiveCurves()` — from `--auto-curves` (erase+replace, highest)

The `setIfAbsent` pattern means hardcoded defaults only fill gaps — they never
override values from the source RAW, a user-supplied profile, or auto-curves.

### Default values
| XMP Key                    | Value                                    | Purpose                         |
|---------------------------|------------------------------------------|---------------------------------|
| `crs:ProcessVersion`     | `"11.0"`                                 | PV2012+ extended slider range   |
| `crs:HDREditMode`        | `"1"`                                    | ACR HDR mode (+/-10 EV range)   |
| `crs:Highlights2012`     | `"-100"`                                 | Full highlight recovery         |
| `crs:Shadows2012`        | `"+100"`                                 | Full shadow lift                |
| `crs:Whites2012`         | `"-40"`                                  | Smoother white rolloff          |
| `crs:Blacks2012`         | `"+20"`                                  | Slight black lift               |
| `crs:ToneCurvePV2012`    | `(0,0)(64,70)(128,140)(192,200)(255,255)`| Gentle highlight shoulder       |
| `crs:ToneCurveName2012`  | `"Custom"`                               | Mark curve as custom            |

### References
- Adobe Camera Raw XMP namespace (`crs:*`): documented in XMP SDK
- Adobe "HDR Edit Mode" (ACR 9.0+): `HDREditMode=1` enables +/-10 EV exposure
- Thomas Knoll, Adobe Highlight Recovery Whitepaper: per-channel clipping
