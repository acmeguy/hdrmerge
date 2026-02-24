# Adaptive Highlight Pull — Design Document

**Date**: 2026-02-23
**Status**: Approved

## Problem

HDRMerge produces physically accurate HDR DNG files. In interior scenes with windows, this accuracy means window regions encode extremely high radiance values (10-20x brighter than the interior). Lightroom and other RAW processors struggle to recover detail from these extreme highlights even with the highlight slider at -100.

HDRMerge's current saturation rolloff begins at 90% of the clipping threshold. Pixels below 90% saturation get full weight from the longest exposure, even if they're very bright. The transition to shorter exposures happens too late and too abruptly for window/highlight regions.

## Solution

A two-phase enhancement to `compose()`:

1. **Highlight detection**: Build a feathered mask identifying near-saturated regions in the reference frame (longest exposure). Uses existing Bayer-domain infrastructure.
2. **Modified merge**: In masked regions, apply earlier rolloff (blend shorter exposures sooner) and radiance compression (deliberately reduce highlight brightness in the DNG).

When disabled (default), compose() behaves identically to today — zero overhead.

## Prior Art

This design was informed by testing SAFNet (ECCV 2024) on our brackets. Key finding: SAFNet's learned selection masks primarily handle motion rejection (not useful for tripod shots). The actual highlight recovery comes from luminance-based weight functions combined with the "window pull" technique from real estate photography — deliberately substituting darker exposure data in blown regions.

Our Python prototype demonstrated the technique works: adaptive Otsu threshold for detection, distance-transform feathering, direct pixel substitution. This design ports that concept to HDRMerge's Bayer-domain merge loop.

## New Parameters

| Parameter | Type | Default | CLI | Description |
|-----------|------|---------|-----|-------------|
| `highlightPull` | float | 0.0 | `--highlight-pull` | Pull strength [0, 1]. 0=disabled, 1=max compression |
| `highlightRolloff` | float | 0.9 | `--highlight-rolloff` | Rolloff start as fraction of saturation. Lower = earlier transition |

Both are added to `SaveOptions` and passed through to `compose()`.

## Detection Pass

Runs after ghost map computation, before the main merge loop. Operates on the **middle exposure** (index `numImages/2`), not the longest — the longest exposure has most of the scene near saturation, making it a poor discriminator between "bright interior" and "blown highlights."

```
detectIdx = numImages / 2
For each Bayer pixel (x, y):
    ch = CFA channel
    raw = images[detectIdx](x, y)
    brightness = raw / satThreshPerCh[ch]
    if brightness > highlightRolloff:
        highlightCore(x, y) = 1
    else:
        highlightCore(x, y) = 0
```

Then:
1. Morphological dilation with radius `featherRadius * 2` (CFA-aware)
2. Distance transform from core boundary
3. Normalize to [0, 1] float mask: 1.0 at core, linear falloff over feather distance

Reuses existing `fattenMask` / `BoxBlur` infrastructure.

## Modified Merge Loop

Inside the per-pixel loop, when `highlightPull > 0` and `highlightMask(x, y) > 0`:

### Earlier rolloff

The effective rolloff start interpolates between the default (0.9 * satThresh) and the user's lower threshold, modulated by the mask:

```cpp
double maskVal = highlightMask(x, y);
double effectiveRolloffFrac = 0.9 - maskVal * (0.9 - highlightRolloff);
double effectiveRolloff = effectiveRolloffFrac * satThreshPerCh[ch];
```

This means:
- Interior pixels (mask=0): rolloff at 90% (unchanged)
- Window core (mask=1): rolloff at `highlightRolloff` (e.g., 70%)
- Transition zone: smooth interpolation

### Radiance compression

After the weighted merge produces radiance `v`:

```cpp
if (highlightMask(x, y) > 0.0f) {
    double pullFactor = 1.0 - highlightPull * highlightMask(x, y);
    v *= pullFactor;
}
```

With `highlightPull=0.8`, mask=1.0: radiance reduced to 20% (~2.3 EV compression).

## Files to Modify

1. **`src/LoadSaveOptions.hpp`** — Add `highlightPull` (float, default 0.0) and `highlightRolloff` (float, default 0.9) to `SaveOptions`
2. **`src/ImageStack.hpp`** — Add `highlightPull` and `highlightRolloff` parameters to `compose()` signature
3. **`src/ImageStack.cpp`** — Detection pass + modified merge loop in `compose()`
4. **`src/Launcher.cpp`** — Parse `--highlight-pull` and `--highlight-rolloff` CLI flags
5. **`src/ImageIO.cpp`** — Pass new parameters through to `compose()` call

## Backwards Compatibility

- Default `highlightPull=0.0` means compose() is completely unchanged
- No new dependencies
- Existing CLI scripts work unmodified
- DNG output format is unchanged — the compression is baked into the Bayer values

## Performance

- Detection pass: ~O(width * height), parallelized with OpenMP, comparable to ghost map computation
- Per-pixel overhead in merge loop: one float multiply + compare per pixel (negligible)
- Estimated total overhead when enabled: 10-15% of compose time

## Testing Plan

1. Run on bracket 2 (kitchen with windows) with `--highlight-pull 0.8 --highlight-rolloff 0.7`
2. Compare DNG output in Lightroom with default settings vs. with pull enabled
3. Verify interior pixels are bit-identical when pull is disabled
4. Verify interior pixels are unchanged in highlight-pull mode (mask = 0 in interior)
5. Check window region in Lightroom: exterior detail should be recoverable with standard highlight slider
