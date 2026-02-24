# Cross-Bracket Neutral Area White Balance Detection

## Problem

Camera as-shot white balance (WB) is a single-point estimate made at capture time. For HDR exposure bracketing, we have multiple measurements of the same scene at different exposure levels. This redundancy can be exploited to identify genuinely neutral (achromatic) surfaces with high confidence and compute a more accurate WB from them.

## Approach

**Core insight**: A truly neutral (gray) surface maintains identical chromaticity ratios (R/G, B/G) across all exposure levels in linear raw data. By testing chromaticity stability across brackets AND requiring the surface to appear achromatic, we can identify high-confidence neutral references.

This is a novel combination of:
- Cross-exposure chromaticity invariance (physics-based)
- Progressive elimination (each bracket narrows candidates)
- Confidence-ranked WB computation from the strongest survivors

## Algorithm

### Phase 1: Candidate Selection (Reference Bracket)

Use the middle exposure bracket (best SNR, least clipping) as reference.

For each 2x2 Bayer quad position (x, y both even):
1. **Valid range check**: all 4 pixels must satisfy `noise_floor < value < saturation_threshold`
   - `noise_floor` = ~5% of sensor max (above black-subtracted noise)
   - `saturation_threshold` = ~90% of per-image saturation threshold
2. Compute raw channel values: `R`, `G_avg = (G1 + G2) / 2`, `B`
3. Compute raw ratios: `r = R / G_avg`, `b = B / G_avg`
4. **Neutrality test**: normalize using camera `preMul[]` (daylight WB) as rough reference:
   - `R' = R * preMul[R_ch]`, `G' = G_avg * preMul[G_ch]`, `B' = B * preMul[B_ch]`
   - `chroma_ratio = max(R', G', B') / min(R', G', B')`
   - Candidate if `chroma_ratio < T_neutral` (default: 1.15)
5. Store `(r, b)` as the reference chromaticity for this quad position

### Phase 2: Progressive Elimination (Subsequent Brackets)

Process remaining brackets ordered by exposure distance from reference (nearest first):

For each surviving candidate quad position:
1. If any pixel in the quad is outside valid range in this bracket -> **skip** (don't eliminate)
2. If the quad IS in valid range:
   - Compute `r_i`, `b_i` for this bracket
   - If `|r_i - r_ref| > T_stability` OR `|b_i - b_ref| > T_stability` -> **eliminate**
   - `T_stability` default: 0.05 (5% ratio deviation)
3. Requirement: a quad must be valid in >= 3 brackets AND survive all stability tests

### Phase 3: Survivor Ranking

Each surviving quad receives a confidence score:

```
score = num_valid_brackets * (1.0 / (1.0 + chroma_variance)) * (1.0 / (1.0 + ratio_spread))
```

Where:
- `num_valid_brackets`: how many brackets this quad was valid and stable in
- `chroma_variance`: cross-bracket variance of (r, b) — lower = more stable
- `ratio_spread`: `|r_mean - b_mean|` after preMul normalization — lower = more neutral

### Phase 4: WB Computation

1. Select top-scoring survivors (top 10% or minimum 50 quads)
2. Compute `median_r` and `median_b` from top survivors across all their valid brackets
3. New WB multipliers:
   - `camMul[R_channel] = 1.0 / median_r`
   - `camMul[G_channel] = 1.0`
   - `camMul[B_channel] = 1.0 / median_b`
   - `camMul[G2_channel] = camMul[G_channel]` (for 4-color CFA)
4. Normalize so minimum camMul = 1.0 (matches existing `adjustWhite()` convention)

### Fallback

If fewer than `MIN_SURVIVORS` (100) quads survive all phases, fall back to camera WB:
- Leave `camMul[]` unchanged
- Log a warning with the survivor count

## Integration

### Pipeline Placement

```
ImageIO::save():
    stack.computeNeutralWB(params)    <-- NEW: before compose
    stack.compose(params, ...)
    params_copy.adjustWhite(image)     <-- uses updated camMul
    DngFloatWriter::write(params)      <-- CameraNeutral reflects neutral-area WB
```

### New Method

`ImageStack::computeNeutralWB(RawParameters & params)`

- Iterates through `images[]` (individual brackets, sorted most to least exposed)
- Reference bracket: `images[size()/2]` (middle exposure)
- On success: modifies `params.camMul[]` in place
- On failure: leaves `camMul[]` unchanged, logs warning

### Affected Files

| File | Change |
|------|--------|
| `src/ImageStack.hpp` | Add `computeNeutralWB()` declaration |
| `src/ImageStack.cpp` | Implement `computeNeutralWB()` |
| `src/ImageIO.cpp` | Call `computeNeutralWB()` before compose in `save()` |

### No Changes To

- `DngFloatWriter.cpp` — already writes CameraNeutral from camMul
- `RawParameters.cpp` — adjustWhite() and autoWB() remain as fallback path
- CLI flags — always-on with graceful fallback

## Logging

The method logs:
- Initial candidate count from reference bracket
- Survivor count after each bracket pass
- Final survivor count and top confidence score
- Computed WB vs original camera WB (R/G, B/G ratios and % difference)
- Whether fallback was triggered

## Constants / Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_neutral` | 1.15 | Max chroma ratio for initial neutrality test |
| `T_stability` | 0.05 | Max ratio deviation across brackets |
| `MIN_VALID_BRACKETS` | 3 | Minimum brackets a quad must be valid in |
| `MIN_SURVIVORS` | 100 | Minimum survivors before fallback |
| `TOP_PERCENT` | 0.10 | Fraction of top-scoring survivors used for WB |
| `NOISE_FLOOR_FRAC` | 0.05 | Noise floor as fraction of sensor max |
| `SAT_FRAC` | 0.90 | Saturation threshold as fraction of per-image sat |

These are compile-time constants initially, tunable through testing.

## Prior Art

This approach fills a gap in published literature. Existing methods:
- **Single-image gray pixel detectors** (Yang/Gao CVPR 2015, Qian GI CVPR 2019) — no cross-exposure verification
- **DEF dual-exposure** (Arad et al. ECCV 2024) — uses chromatic differences but trains a neural network
- **NADE** (Optics Express 2025) — detects neutral areas in multi-exposure HDR but per-exposure independently

The combination of cross-exposure chromaticity stability as the primary neutral-surface identification signal is novel.
