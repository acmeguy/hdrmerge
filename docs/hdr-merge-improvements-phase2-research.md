# HDRMerge Phase 2 Improvements -- Research Notes

This document covers the technical background for four planned improvements:
per-image noise profiling (DNG tag 51041), forward color matrices (DNG tag 50964),
per-channel saturation thresholds, and sub-pixel alignment via parabolic SSD fitting.

---

## 1. NoiseProfile DNG Tag (51041)

### Specification

- **Tag:** 51041 (`NoiseProfile`)
- **Type:** DOUBLE
- **Count:** 2 x number_of_color_channels (typically 2 x 3 = 6 for RGB, or 2 x 4 = 8 if storing per-CFA-plane)

Each color channel `c` is described by a pair `(S_c, O_c)` such that the per-pixel
variance follows:

```
Var(pixel) = S_c * signal + O_c
```

where `signal` is the pixel value normalized to the range [0, 1].

### Deriving the Coefficients

**Shot noise coefficient S** (signal-dependent, Poisson):

```
S_c = 1.0 / (white_level - cblack[c])
```

This is the standard result for Poisson-distributed photon counts: variance equals
the mean, and normalizing the signal to [0,1] by dividing by the full-well capacity
`(white_level - cblack[c])` gives the per-unit-signal variance.

**Read noise coefficient O** (signal-independent, Gaussian):

Estimated from masked or optical-black pixel regions at the sensor margins. These
pixels receive no light, so their variance is purely read noise. The procedure:

1. Extract all masked pixels for channel `c`.
2. Compute their sample variance `var_black_c`.
3. Normalize: `O_c = var_black_c / (white_level - cblack[c])^2`.

### Scaling for Merged Exposures

When N exposures are merged with Poisson-optimal (inverse-variance) weighting, the
merged noise variance scales as:

```
S_merged = S / N
O_merged = O / N
```

Both coefficients decrease by a factor of N because:
- Poisson (shot) noise: averaging N independent samples reduces variance by 1/N.
- Gaussian (read) noise: same argument applies to independent read-noise draws.

This assumes equal-weight averaging. If exposures have unequal weights (e.g., due to
different ISO or exposure time), the effective N should be replaced by the sum of
squared weights divided by the square of the sum of weights (the standard
inverse-variance weighting result).

### Implementation Notes

Write the tag as an array of DOUBLE values in IFD0, ordered as:
`[S_0, O_0, S_1, O_1, ..., S_(n-1), O_(n-1)]`.

ACR, Lightroom, and darktable all read this tag and use it to drive noise reduction
strength. Providing accurate values avoids the need for these tools to estimate noise
from the image content itself (which is less reliable on HDR merges).

---

## 2. ForwardMatrix DNG Tag (50964)

### Specification

- **Tag:** 50964 (`ForwardMatrix1`)
- **Type:** SRATIONAL
- **Count:** 9 (a 3x3 matrix stored row-major)

The ForwardMatrix maps white-balanced camera-native colors to CIE XYZ in the
D50 Profile Connection Space (PCS):

```
XYZ_D50 = ForwardMatrix * diag(1/r_wb, 1/g_wb, 1/b_wb) * CameraRGB
```

### Relationship to ColorMatrix1 (Tag 50721)

The `ColorMatrix1` (tag 50721) maps in the opposite direction:

```
CameraRGB = ColorMatrix * XYZ
```

The ForwardMatrix is conceptually the inverse mapping. However, it is not simply
the matrix inverse of ColorMatrix because the ForwardMatrix also encodes a
normalization to the D50 white point.

### Computation

Given the 3x3 camera-to-XYZ matrix `camXyz` (which may already be available from
the raw file metadata or from dcraw/LibRaw color data):

1. **Invert** the `camXyz` matrix to get `xyzToCam`.
2. **Compute the raw ForwardMatrix** as the inverse: `FM_raw = inverse(xyzToCam) = camXyz`.
3. **Row-normalize** so that each row of `FM` sums to the corresponding component
   of the D50 white point `(X_D50, Y_D50, Z_D50) = (0.9642, 1.0000, 0.8249)`:

```cpp
// For each row i of FM:
double row_sum = FM[i][0] + FM[i][1] + FM[i][2];
double d50[3] = {0.9642, 1.0000, 0.8249};
for (int j = 0; j < 3; j++)
    FM[i][j] *= d50[i] / row_sum;
```

### Why It Matters

DNG readers that have both `ColorMatrix1` and `ForwardMatrix1` can produce more
accurate colors, particularly in saturated regions and under mixed lighting.
Without a ForwardMatrix, the reader must invert the ColorMatrix itself and may
apply a different normalization, leading to subtle hue shifts.

---

## 3. Per-Channel Saturation Theory

### The Problem with Global Thresholds

In a typical Bayer sensor, the three (or four) color channels do not clip at the
same ADU level. For example:

| Channel | Saturation (ADU) |
|---------|-------------------|
| Red     | 15000             |
| Green   | 16000             |
| Blue    | 15500             |

Using a single global threshold (the minimum across channels, here 15000) means
green and blue pixels that are still valid between 15000 and their respective
saturation points are discarded prematurely.

### Per-Channel Threshold Approach

Each channel `c` has its own saturation threshold `sat[c]`. During mask generation:

- A pixel in channel `c` is considered clipped if `value >= sat[c]`.
- The mask for blending/layer selection uses the **per-channel** test for
  determining which exposure contributes which channel value.

This recovers highlight color information that would otherwise be lost. Consider a
sunset scene: red may clip first, but green and blue still carry valid color
information for several hundred ADU above the global threshold.

### Conservative Global Mask for Layer Selection

While per-channel thresholds govern which pixel values are trusted, the **layer
selection mask** (which determines the exposure index assigned to each spatial
location) still uses the global (minimum) threshold. This is conservative:

```
global_threshold = min(sat[0], sat[1], ..., sat[n-1])
```

A pixel location is flagged as "needs longer exposure" only when the most
conservative channel clips. This prevents the layer-selection logic from creating
artifacts at color-channel boundaries in the Bayer pattern.

The per-channel thresholds are then applied during the final compositing step, where
individual channel values from different exposures can be blended independently.

---

## 4. Sub-Pixel Alignment via Parabolic SSD Fitting

### Integer-Pixel SSD Search (Existing)

The current alignment computes the Sum of Squared Differences (SSD) over a search
window of integer-pixel offsets and picks the minimum. This is accurate to +/-0.5 px.

### Parabolic Refinement

After finding the integer minimum at offset `(dx, dy)`, compute SSD at five points:

```
SSD_center = SSD(dx, dy)
SSD_left   = SSD(dx - 1, dy)
SSD_right  = SSD(dx + 1, dy)
SSD_up     = SSD(dx, dy - 1)
SSD_down   = SSD(dx, dy + 1)
```

Fit a parabola independently in X and Y:

```
For X:  y = a*x^2 + b*x + c
  Using points at x = {-1, 0, +1} with values {SSD_left, SSD_center, SSD_right}:
    a = (SSD_left + SSD_right) / 2 - SSD_center
    b = (SSD_right - SSD_left) / 2
    Fractional offset: delta_x = -b / (2 * a)

For Y:  same procedure with {SSD_up, SSD_center, SSD_down}
    a = (SSD_up + SSD_down) / 2 - SSD_center
    b = (SSD_down - SSD_up) / 2
    Fractional offset: delta_y = -b / (2 * a)
```

The sub-pixel offset is `(dx + delta_x, dy + delta_y)`.

### CFA-Aware Bilinear Interpolation

Applying the fractional shift requires interpolation. Naive bilinear interpolation
mixes values from different color channels in the CFA mosaic, causing color artifacts.

**Bayer pattern:** each color channel sits on a grid with spacing 2 in both X and Y.
Interpolation must only use neighbors of the same color, which are at distance 2:

```
For a red pixel at (x, y), the four neighbors for interpolation are:
  (x-2, y), (x+2, y), (x, y-2), (x, y+2)   -- or the 2x2 block of same-color
```

The bilinear weights are computed from the fractional offset scaled to the
same-color grid (divide by 2 for Bayer, by 6 for X-Trans 6x6 pattern).

**X-Trans pattern:** the minimum repeating unit is 6x6. Same-color neighbors are at
distance 6 in each axis. The interpolation kernel must respect the 6x6 periodicity.

### Skip Threshold

If `|delta_x| < 0.1` and `|delta_y| < 0.1`, skip the interpolation entirely. At
these offsets the improvement is below the noise floor for typical sensor resolutions
and the interpolation overhead is not justified.

### Expected Magnitudes

Typical handheld bracketing residuals after integer alignment: **0.1 -- 0.3 px**.
Tripod-mounted with shutter vibration: **0.05 -- 0.15 px**. Sub-pixel correction is
most beneficial in the handheld case.

---

## 5. References

1. **Adobe DNG Specification 1.7.1**
   Digital Negative (DNG) Specification, Adobe Systems, 2023.
   Tags 50721 (ColorMatrix1), 50964 (ForwardMatrix1), 51041 (NoiseProfile).

2. **Debevec, P. E. and Malik, J. (1997)**
   "Recovering High Dynamic Range Radiance Maps from Photographs."
   *Proceedings of ACM SIGGRAPH 1997*, pp. 369--378.

3. **Robertson, M. A., Borman, S., and Stevenson, R. L. (2003)**
   "Estimation-Theoretic Approach to Dynamic Range Enhancement using Multiple Exposures."
   *Journal of Electronic Imaging*, 12(2), pp. 219--228.

4. **Adobe DNG SDK**
   Reference implementation for reading and writing DNG files, including
   noise profile computation and forward matrix derivation.
   Available from Adobe's developer portal.
