# Adaptive Highlight Pull — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two CLI parameters (`--highlight-pull`, `--highlight-rolloff`) that enable highlight detection and radiance compression in `compose()`, recovering window/exterior detail in HDR DNG output.

**Architecture:** A feathered highlight mask is built from the reference frame (longest exposure) after ghost map computation. During the per-pixel merge loop, the mask modulates two behaviors: earlier rolloff to shorter exposures, and post-merge radiance compression. When disabled (default `highlightPull=0.0`), the code path is never entered.

**Tech Stack:** C++11, OpenMP, existing `Array2D<float>` and `BoxBlur` infrastructure. No new dependencies.

---

### Task 1: Add parameters to SaveOptions

**Files:**
- Modify: `src/LoadSaveOptions.hpp:51-74`

**Step 1: Add fields to SaveOptions struct**

Add `highlightPull` and `highlightRolloff` after the existing `subPixelAlign` field (line 68):

```cpp
// In struct SaveOptions, add two new members:
float highlightPull;
float highlightRolloff;
```

Update the constructor initializer list (line 69-73) to include defaults:

```cpp
SaveOptions() : bps(24), previewSize(0), saveMask(false), featherRadius(3),
    compressionLevel(6), deghostSigma(0.0f), deghostMode(DeghostMode::Robust),
    deghostIterations(1), clipPercentile(99.9),
    evShift(0.0),
    autoCurves(false), resizeLong(0), subPixelAlign(false),
    highlightPull(0.0f), highlightRolloff(0.9f) {}
```

**Step 2: Build to verify compilation**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Compiles cleanly (new fields unused so far — no warnings expected since this is a struct with constructor defaults).

**Step 3: Commit**

```bash
git add src/LoadSaveOptions.hpp
git commit -m "feat: add highlightPull/highlightRolloff to SaveOptions"
```

---

### Task 2: Add parameters to compose() signature

**Files:**
- Modify: `src/ImageStack.hpp:60-65`
- Modify: `src/ImageStack.cpp:1381` (signature only)

**Step 1: Update declaration in ImageStack.hpp**

Change the `compose()` declaration (line 60-65) to add two new parameters at the end:

```cpp
ComposeResult compose(const RawParameters & md, int featherRadius,
                       float deghostSigma = 0.0f,
                       DeghostMode deghostMode = DeghostMode::Robust,
                       int deghostIterations = 1,
                       double clipPercentile = 99.9,
                       bool subPixelAlign = false,
                       float highlightPull = 0.0f,
                       float highlightRolloff = 0.9f) const;
```

**Step 2: Update definition in ImageStack.cpp**

Change the `compose()` definition signature (line 1381) to match:

```cpp
ComposeResult ImageStack::compose(const RawParameters & params, int featherRadius, float deghostSigma, DeghostMode deghostMode, int deghostIterations, double clipPercentile, bool subPixelAlign, float highlightPull, float highlightRolloff) const {
```

**Step 3: Build to verify**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Compiles cleanly (new params have defaults, no callers need updating yet).

**Step 4: Commit**

```bash
git add src/ImageStack.hpp src/ImageStack.cpp
git commit -m "feat: add highlightPull/highlightRolloff params to compose()"
```

---

### Task 3: Pass parameters through from ImageIO::save()

**Files:**
- Modify: `src/ImageIO.cpp:203-208`

**Step 1: Update the compose() call in save()**

Change the call at lines 203-208 to pass the new parameters:

```cpp
ComposeResult composed = stack.compose(params, options.featherRadius,
                                         options.deghostSigma,
                                         options.deghostMode,
                                         options.deghostIterations,
                                         options.clipPercentile,
                                         options.subPixelAlign,
                                         options.highlightPull,
                                         options.highlightRolloff);
```

**Step 2: Build to verify**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Clean compile.

**Step 3: Commit**

```bash
git add src/ImageIO.cpp
git commit -m "feat: pass highlightPull/highlightRolloff through to compose()"
```

---

### Task 4: Parse CLI flags in Launcher

**Files:**
- Modify: `src/Launcher.cpp:354-557` (parseCommandLine)
- Modify: `src/Launcher.cpp:586-642` (showHelp)
- Modify: `src/Launcher.cpp:645-700` (checkGUI)

**Step 1: Add CLI parsing in parseCommandLine()**

Add these two blocks after the `--sub-pixel` handler (after line 529):

```cpp
} else if (string("--highlight-pull") == argv[i]) {
    if (++i < argc) {
        try {
            float val = stof(argv[i]);
            if (val >= 0.0f && val <= 1.0f) {
                saveOptions.highlightPull = val;
            } else {
                cerr << tr("--highlight-pull must be between 0 and 1.") << endl;
            }
        } catch (std::invalid_argument & e) {
            cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
        }
    }
} else if (string("--highlight-rolloff") == argv[i]) {
    if (++i < argc) {
        try {
            float val = stof(argv[i]);
            if (val >= 0.5f && val <= 0.95f) {
                saveOptions.highlightRolloff = val;
            } else {
                cerr << tr("--highlight-rolloff must be between 0.5 and 0.95.") << endl;
            }
        } catch (std::invalid_argument & e) {
            cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
        }
    }
```

**Step 2: Add help text in showHelp()**

Add after the `--sub-pixel` help line (after line 638):

```cpp
cout << "    " << "--highlight-pull S " << tr("Highlight pull strength [0, 1]. Compresses bright regions to recover window detail. 0=off (default).") << endl;
cout << "    " << "--highlight-rolloff F " << tr("Rolloff start as fraction of saturation [0.5, 0.95]. Lower = earlier transition to shorter exposures. Default 0.9.") << endl;
```

**Step 3: Add flag skipping in checkGUI()**

Add after the `--sub-pixel` handler in checkGUI() (after line 686):

```cpp
} else if (string("--highlight-pull") == argv[i]) {
    ++i; // skip the value
} else if (string("--highlight-rolloff") == argv[i]) {
    ++i; // skip the value
```

**Step 4: Build and verify CLI parsing**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
Expected: Clean compile.

Run: `build/hdrmerge.app/Contents/MacOS/hdrmerge --help 2>&1 | grep -i highlight`
Expected: Both `--highlight-pull` and `--highlight-rolloff` appear in help output.

**Step 5: Commit**

```bash
git add src/Launcher.cpp
git commit -m "feat: parse --highlight-pull and --highlight-rolloff CLI flags"
```

---

### Task 5: Implement highlight detection pass in compose()

This is the core detection logic. It runs after ghost map computation (after line 1636) and before the main merge loop (line 1638).

**Files:**
- Modify: `src/ImageStack.cpp:1636-1638`

**Step 1: Add the detection pass**

Insert the following code block between the ghost map section (line 1636) and the `float maxVal = 0.0;` line (line 1638):

```cpp
    // === Highlight detection pass ===
    // Build a feathered mask identifying near-saturated regions in the reference
    // frame (image 0, longest exposure). Used for earlier rolloff and radiance
    // compression when highlightPull > 0.
    Array2D<float> highlightMask;
    const bool doHighlightPull = highlightPull > 0.0f;
    if (doHighlightPull) {
        Log::debug("Highlight pull enabled: pull=", highlightPull,
                   " rolloff=", highlightRolloff);

        // Phase 1: Identify core highlight pixels in reference frame
        // A pixel is "highlight core" if its raw value exceeds highlightRolloff
        // fraction of its channel's saturation threshold.
        Array2D<uint8_t> highlightCore(width, height);
        #pragma omp parallel for
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                if (!images[0].contains(x, y)) {
                    highlightCore(x, y) = 0;
                    continue;
                }
                int ch = params.FC(x, y);
                double raw = static_cast<double>(images[0](x, y));
                double brightness = raw / satThreshPerCh[ch];
                highlightCore(x, y) = (brightness > highlightRolloff) ? 1 : 0;
            }
        }

        // Count core pixels for logging
        long long coreCount = 0;
        #pragma omp parallel for reduction(+:coreCount)
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                if (highlightCore(x, y)) coreCount++;
        double corePct = 100.0 * coreCount / (width * height);
        Log::debug("Highlight core: ", corePct, "% of pixels");

        if (coreCount == 0) {
            Log::debug("No highlight pixels detected, disabling highlight pull");
            // Leave highlightMask uninitialized — doHighlightPull check below
            // will use highlightMask.data() == nullptr as a secondary guard.
        } else {
            // Phase 2: Dilate + feather using BoxBlur
            // Convert core to float mask, then blur to create feathered falloff.
            // The blur radius is featherRadius * 2 (wider than exposure mask blur).
            highlightMask = Array2D<float>(width, height);
            int hlFeatherRadius = featherRadius * 2;

            // Seed mask: 1.0 at core pixels, 0.0 elsewhere
            #pragma omp parallel for
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x)
                    highlightMask(x, y) = highlightCore(x, y) ? 1.0f : 0.0f;

            // Iterative box blur for smooth feathering (3 passes approximates Gaussian)
            for (int pass = 0; pass < 3; ++pass) {
                // Horizontal pass
                Array2D<float> tmp(width, height);
                #pragma omp parallel for
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        float sum = 0.0f;
                        int count = 0;
                        int x0 = std::max(0, (int)x - hlFeatherRadius);
                        int x1 = std::min((int)width - 1, (int)x + hlFeatherRadius);
                        for (int nx = x0; nx <= x1; ++nx) {
                            sum += highlightMask(nx, y);
                            count++;
                        }
                        tmp(x, y) = sum / count;
                    }
                }
                // Vertical pass
                #pragma omp parallel for
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        float sum = 0.0f;
                        int count = 0;
                        int y0 = std::max(0, (int)y - hlFeatherRadius);
                        int y1 = std::min((int)height - 1, (int)y + hlFeatherRadius);
                        for (int ny = y0; ny <= y1; ++ny) {
                            sum += tmp(x, ny);
                            count++;
                        }
                        highlightMask(x, y) = sum / count;
                    }
                }
            }

            // Renormalize: ensure core pixels are 1.0, and clamp to [0, 1]
            float maskMax = 0.0f;
            #pragma omp parallel for reduction(max:maskMax)
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x)
                    if (highlightMask(x, y) > maskMax) maskMax = highlightMask(x, y);

            if (maskMax > 0.0f) {
                float invMax = 1.0f / maskMax;
                #pragma omp parallel for
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x) {
                        float v = highlightMask(x, y) * invMax;
                        if (v > 1.0f) v = 1.0f;
                        highlightMask(x, y) = v;
                    }
            }

            // Log feathered mask stats
            long long featheredCount = 0;
            #pragma omp parallel for reduction(+:featheredCount)
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x)
                    if (highlightMask(x, y) > 0.01f) featheredCount++;
            double featheredPct = 100.0 * featheredCount / (width * height);
            Log::debug("Highlight feathered region: ", featheredPct,
                       "% of pixels (feather radius=", hlFeatherRadius, ")");
        }
    }
```

**Step 2: Build to verify**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10`
Expected: Clean compile. The mask is computed but not yet used.

**Step 3: Smoke test — verify zero overhead when disabled**

Run a merge WITHOUT highlight pull to confirm identical behavior:

```bash
build/hdrmerge.app/Contents/MacOS/hdrmerge -v \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4730.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4731.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4732.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4733.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4734.NEF \
  -O merged-hlpull-test0
```
Expected: Output is produced, no errors. No "Highlight pull" messages in verbose output.

**Step 4: Test detection pass logging**

Run with highlight pull enabled:

```bash
build/hdrmerge.app/Contents/MacOS/hdrmerge -vv \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4730.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4731.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4732.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4733.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4734.NEF \
  --highlight-pull 0.8 --highlight-rolloff 0.7 \
  -O merged-hlpull-test1
```
Expected: Debug output shows "Highlight pull enabled", "Highlight core: X.X% of pixels", "Highlight feathered region: X.X% of pixels". Core should be a small percentage (roughly matching the window area in the kitchen scene).

**Step 5: Commit**

```bash
git add src/ImageStack.cpp
git commit -m "feat: implement highlight detection pass in compose()"
```

---

### Task 6: Implement modified merge loop (earlier rolloff + radiance compression)

This modifies the per-pixel merge to use the highlight mask. Two changes:
1. Earlier rolloff in the weight computation (lines 1681-1693)
2. Radiance compression after the weighted merge (line 1800)

**Files:**
- Modify: `src/ImageStack.cpp:1647-1815`

**Step 1: Add per-pixel highlight mask lookup at start of x-loop**

After the `int ch = params.FC(x, y);` line (line 1649), add:

```cpp
            // Highlight mask value for this pixel (0 = normal, >0 = highlight region)
            float hlMask = (doHighlightPull && highlightMask.data())
                ? highlightMask(x, y) : 0.0f;
```

**Step 2: Modify rolloff weight computation**

Replace the rolloff weight section (lines 1681-1693) with mask-aware version:

```cpp
                if (useBlockRolloff) {
                    double effectiveBlockRolloff = blockRolloff;
                    double effectiveBlockRange = blockRange;
                    if (hlMask > 0.0f) {
                        double effectiveFrac = 0.9 - hlMask * (0.9 - highlightRolloff);
                        effectiveBlockRolloff = effectiveFrac * blockThresh;
                        effectiveBlockRange = blockThresh - effectiveBlockRolloff;
                    }
                    if (raw >= blockThresh) {
                        w = 0.0;
                    } else if (raw > effectiveBlockRolloff) {
                        double t = (blockThresh - raw) / effectiveBlockRange;
                        w *= t * t;
                    }
                } else {
                    double effectiveRolloff = satRolloffPerCh[ch];
                    double effectiveRange = satRolloffRangePerCh[ch];
                    if (hlMask > 0.0f) {
                        double effectiveFrac = 0.9 - hlMask * (0.9 - highlightRolloff);
                        effectiveRolloff = effectiveFrac * satThreshPerCh[ch];
                        effectiveRange = satThreshPerCh[ch] - effectiveRolloff;
                    }
                    if (raw > effectiveRolloff) {
                        double t = (satThreshPerCh[ch] - raw) / effectiveRange;
                        w *= t * t;
                    }
                }
```

**Step 3: Add radiance compression after weighted merge**

After the line `v = weightedSum / totalWeight;` (line 1800), add:

```cpp
                // Highlight radiance compression: reduce brightness in highlight regions
                if (hlMask > 0.0f) {
                    double pullFactor = 1.0 - static_cast<double>(highlightPull) * hlMask;
                    v *= pullFactor;
                }
```

**Step 4: Build to verify**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -10`
Expected: Clean compile.

**Step 5: Full integration test — highlight pull on kitchen bracket**

Run:

```bash
build/hdrmerge.app/Contents/MacOS/hdrmerge -vv \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4730.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4731.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4732.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4733.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4734.NEF \
  --highlight-pull 0.8 --highlight-rolloff 0.7 \
  -O merged-hlpull-test2
```

Expected: DNG is produced. Debug output shows highlight mask stats. Compare this DNG to the one from test0 (no highlight pull) in Lightroom — window regions should have noticeably more recoverable detail.

**Step 6: Verify disabled mode is unchanged**

Compare test0 (no flags) output to a fresh run without flags. They should be identical:

```bash
# Run again without highlight pull to same output dir
build/hdrmerge.app/Contents/MacOS/hdrmerge \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4730.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4731.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4732.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4733.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4734.NEF \
  -o /Volumes/Oryggi/Eignamyndir/RAW/test/merged-hlpull-test3/merged.dng
```

The DNG from test0 and test3 should be bit-identical (same pixels, same metadata).

**Step 7: Commit**

```bash
git add src/ImageStack.cpp
git commit -m "feat: implement highlight-aware rolloff and radiance compression in compose()"
```

---

### Task 7: Visual validation in Lightroom

This is a manual testing task — no code changes.

**Step 1: Open both DNGs in Lightroom**

Open these two files side by side:
- `merged-hlpull-test0/merged.dng` (no highlight pull — baseline)
- `merged-hlpull-test2/merged.dng` (highlight pull 0.8, rolloff 0.7)

**Step 2: Verify interior quality**

With default Lightroom settings, compare interior regions (countertops, cabinets, walls). They should look identical — the highlight mask should be zero in interior regions, meaning those pixels are unchanged.

**Step 3: Verify window recovery**

Set Highlights slider to -100 in Lightroom on both images:
- Baseline: Window regions should still appear mostly white/blown
- Highlight pull: Window regions should show exterior detail (trees, sky, etc.)

**Step 4: Check for artifacts**

Look for:
- Color fringe at window/interior boundary (transition zone)
- Banding or posterization in the feathered region
- Any visible seam between pulled and non-pulled regions

If artifacts exist, note them for potential parameter tuning in a follow-up.

---

### Task 8: Test edge cases

**Step 1: Low-DR scene (bathroom bracket)**

```bash
build/hdrmerge.app/Contents/MacOS/hdrmerge -vv \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4622.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4623.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4624.NEF \
  --highlight-pull 0.8 --highlight-rolloff 0.7 \
  -O merged-hlpull-lowdr
```

Expected: Debug output should show very small core percentage (near 0%). With few or no highlight pixels, the feature should be effectively a no-op. DNG should look identical to a run without highlight pull.

**Step 2: Maximum strength test**

```bash
build/hdrmerge.app/Contents/MacOS/hdrmerge -vv \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4730.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4731.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4732.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4733.NEF \
  /Volumes/Oryggi/Eignamyndir/RAW/test/_EBX4734.NEF \
  --highlight-pull 1.0 --highlight-rolloff 0.5 \
  -O merged-hlpull-maxstr
```

Expected: DNG is produced without crashes. Window regions should be dramatically darker — possibly too dark, which is fine (confirms the range works). This is the extreme end of the parameter space.

**Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: complete adaptive highlight pull implementation

Adds --highlight-pull and --highlight-rolloff CLI parameters that enable
highlight detection and radiance compression in compose(). In highlight
regions (detected via reference frame brightness), the saturation rolloff
starts earlier and merged radiance is compressed, making window/exterior
detail recoverable in Lightroom.

Default highlightPull=0.0 means compose() is completely unchanged.
No new dependencies. Existing CLI scripts work unmodified."
```
