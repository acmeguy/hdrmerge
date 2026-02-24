# Cross-Bracket Neutral Area WB Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add automatic white balance computation by detecting chromatically stable neutral surfaces across exposure brackets.

**Architecture:** New method `ImageStack::computeNeutralWB()` runs a 4-phase algorithm (candidate selection, progressive elimination, ranking, WB computation) on the raw bracket images before composition. It modifies `RawParameters::camMul[]` in place, which flows through `adjustWhite()` and into the DNG CameraNeutral tag.

**Tech Stack:** C++11, OpenMP for parallelization, existing project types (Array2D, Image, RawParameters, CFAPattern)

**Design doc:** `docs/plans/2026-02-24-neutral-area-wb-design.md`

---

### Task 1: Add computeNeutralWB Declaration

**Files:**
- Modify: `src/ImageStack.hpp:60` (add declaration near `compose`)

**Step 1: Add method declaration**

In `src/ImageStack.hpp`, add after the `compose()` declaration (line 67):

```cpp
    bool computeNeutralWB(RawParameters & params) const;
```

Returns `true` if neutral WB was computed successfully, `false` if fallback was used.

**Step 2: Verify build**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`

Expected: Linker error about undefined `computeNeutralWB` (that's fine — implementation comes next)

**Step 3: Commit**

```bash
git add src/ImageStack.hpp
git commit -m "feat: declare computeNeutralWB in ImageStack"
```

---

### Task 2: Implement Phase 1 — Candidate Selection

**Files:**
- Modify: `src/ImageStack.cpp` (add implementation at end of file)

**Step 1: Add the struct and Phase 1 implementation**

At the end of `src/ImageStack.cpp` (before the closing namespace brace if any, or at end of file), add:

```cpp
namespace {
// Tuning constants for neutral WB detection
constexpr float NWB_T_NEUTRAL = 1.15f;         // max chroma ratio for neutrality test
constexpr float NWB_T_STABILITY = 0.05f;        // max ratio deviation across brackets
constexpr int   NWB_MIN_VALID_BRACKETS = 3;     // min brackets a quad must be valid in
constexpr int   NWB_MIN_SURVIVORS = 100;         // min survivors before fallback
constexpr float NWB_TOP_PERCENT = 0.10f;         // fraction of top survivors used for WB
constexpr float NWB_NOISE_FLOOR_FRAC = 0.05f;   // noise floor as fraction of sensor max
constexpr float NWB_SAT_FRAC = 0.90f;           // saturation threshold fraction
} // anon namespace

struct NeutralCandidate {
    size_t x, y;          // quad position (top-left corner, both even)
    float r_ref, b_ref;   // reference bracket R/G, B/G ratios
    float r_sum, b_sum;   // running sum of ratios across valid brackets
    float r_sq_sum, b_sq_sum; // running sum of squared ratios (for variance)
    int valid_count;       // number of brackets this quad was valid and stable in
    bool alive;            // still a candidate?
};

bool ImageStack::computeNeutralWB(RawParameters & params) const {
    Timer t("NeutralWB");

    if (images.size() < 2) {
        Log::debug("NeutralWB: need >= 2 brackets, have ", images.size(), ". Skipping.");
        return false;
    }

    const size_t numImages = images.size();
    const size_t refIdx = numImages / 2;  // middle bracket
    const Image & refImg = images[refIdx];

    // Compute per-image thresholds
    // Noise floor: fraction of the sensor max (already black-subtracted)
    const float noiseFloor = params.max * NWB_NOISE_FLOOR_FRAC;

    Log::debug("NeutralWB: ", numImages, " brackets, ref=", refIdx,
               ", noiseFloor=", noiseFloor, ", sensorMax=", params.max);

    // Determine channel positions in the 2x2 Bayer quad.
    // For a quad at (x, y) with x,y even:
    //   (x,   y)   = FC(x, y)
    //   (x+1, y)   = FC(x+1, y)
    //   (x,   y+1) = FC(x, y+1)
    //   (x+1, y+1) = FC(x+1, y+1)
    // We need to find which positions are R (channel 0), G (channel 1 or 3), B (channel 2).

    // --- Phase 1: Candidate selection from reference bracket ---
    std::vector<NeutralCandidate> candidates;
    const float satThreshRef = refImg.getMax() * NWB_SAT_FRAC;

    for (size_t y = 0; y + 1 < height; y += 2) {
        for (size_t x = 0; x + 1 < width; x += 2) {
            // Check all 4 pixels in the quad are within the reference image bounds
            if (!refImg.contains(x, y) || !refImg.contains(x+1, y) ||
                !refImg.contains(x, y+1) || !refImg.contains(x+1, y+1))
                continue;

            // Read the 4 raw values
            uint16_t v00 = refImg(x, y);
            uint16_t v10 = refImg(x+1, y);
            uint16_t v01 = refImg(x, y+1);
            uint16_t v11 = refImg(x+1, y+1);

            // Valid range check: all 4 above noise floor, all 4 below saturation
            if (v00 < noiseFloor || v10 < noiseFloor ||
                v01 < noiseFloor || v11 < noiseFloor)
                continue;
            if (v00 > satThreshRef || v10 > satThreshRef ||
                v01 > satThreshRef || v11 > satThreshRef)
                continue;

            // Map each position to its CFA channel
            int c00 = params.FC(x, y);
            int c10 = params.FC(x+1, y);
            int c01 = params.FC(x, y+1);
            int c11 = params.FC(x+1, y+1);

            // Extract R, G1, G2, B values based on CFA channel
            float chVal[4] = {0, 0, 0, 0};  // indexed by channel
            int   chCnt[4] = {0, 0, 0, 0};
            auto accum = [&](int ch, uint16_t val) {
                chVal[ch] += val;
                chCnt[ch]++;
            };
            accum(c00, v00);
            accum(c10, v10);
            accum(c01, v01);
            accum(c11, v11);

            // We need channels 0 (R), 1 (G), 2 (B). Channel 3 is G2 in RGBG.
            // Average greens: G_avg = (sum of channels 1 and 3) / (count of channels 1 and 3)
            float G_avg = (chVal[1] + chVal[3]) / (chCnt[1] + chCnt[3]);
            float R = chVal[0] / std::max(chCnt[0], 1);
            float B = chVal[2] / std::max(chCnt[2], 1);

            if (G_avg < 1.0f) continue;  // avoid division by zero

            float r = R / G_avg;
            float b = B / G_avg;

            // Neutrality test using preMul as rough daylight WB reference
            float Rn = R * params.preMul[0];
            float Gn = G_avg * params.preMul[1];
            float Bn = B * params.preMul[2];
            float maxC = std::max({Rn, Gn, Bn});
            float minC = std::min({Rn, Gn, Bn});
            if (minC < 1.0f) continue;
            float chromaRatio = maxC / minC;
            if (chromaRatio > NWB_T_NEUTRAL) continue;

            // This quad passes Phase 1
            NeutralCandidate cand;
            cand.x = x;
            cand.y = y;
            cand.r_ref = r;
            cand.b_ref = b;
            cand.r_sum = r;
            cand.b_sum = b;
            cand.r_sq_sum = r * r;
            cand.b_sq_sum = b * b;
            cand.valid_count = 1;
            cand.alive = true;
            candidates.push_back(cand);
        }
    }

    Log::debug("NeutralWB Phase 1: ", candidates.size(), " candidates from ref bracket ", refIdx);

    if (candidates.empty()) {
        Log::debug("NeutralWB: no candidates found in reference bracket. Using camera WB.");
        return false;
    }

    // --- Phase 2: Progressive elimination across other brackets ---
    // Process brackets ordered by exposure distance from reference (nearest first)
    std::vector<size_t> bracketOrder;
    for (size_t i = 0; i < numImages; i++) {
        if (i != refIdx) bracketOrder.push_back(i);
    }
    std::sort(bracketOrder.begin(), bracketOrder.end(), [&](size_t a, size_t b_idx) {
        return std::abs((int)a - (int)refIdx) < std::abs((int)b_idx - (int)refIdx);
    });

    for (size_t bi = 0; bi < bracketOrder.size(); bi++) {
        size_t imgIdx = bracketOrder[bi];
        const Image & img = images[imgIdx];
        const float satThresh = img.getMax() * NWB_SAT_FRAC;
        int eliminated = 0;

        for (auto & cand : candidates) {
            if (!cand.alive) continue;

            size_t x = cand.x, y = cand.y;

            // Check bounds in this (potentially shifted) bracket
            if (!img.contains(x, y) || !img.contains(x+1, y) ||
                !img.contains(x, y+1) || !img.contains(x+1, y+1))
                continue;  // skip, don't eliminate

            uint16_t v00 = img(x, y);
            uint16_t v10 = img(x+1, y);
            uint16_t v01 = img(x, y+1);
            uint16_t v11 = img(x+1, y+1);

            // Valid range check
            if (v00 < noiseFloor || v10 < noiseFloor ||
                v01 < noiseFloor || v11 < noiseFloor)
                continue;  // skip
            if (v00 > satThresh || v10 > satThresh ||
                v01 > satThresh || v11 > satThresh)
                continue;  // skip

            int c00 = params.FC(x, y);
            int c10 = params.FC(x+1, y);
            int c01 = params.FC(x, y+1);
            int c11 = params.FC(x+1, y+1);

            float chVal[4] = {0, 0, 0, 0};
            int   chCnt[4] = {0, 0, 0, 0};
            auto accum = [&](int ch, uint16_t val) {
                chVal[ch] += val;
                chCnt[ch]++;
            };
            accum(c00, v00);
            accum(c10, v10);
            accum(c01, v01);
            accum(c11, v11);

            float G_avg = (chVal[1] + chVal[3]) / (chCnt[1] + chCnt[3]);
            float R = chVal[0] / std::max(chCnt[0], 1);
            float B = chVal[2] / std::max(chCnt[2], 1);

            if (G_avg < 1.0f) continue;

            float r_i = R / G_avg;
            float b_i = B / G_avg;

            // Stability test against reference ratios
            if (std::abs(r_i - cand.r_ref) > NWB_T_STABILITY ||
                std::abs(b_i - cand.b_ref) > NWB_T_STABILITY) {
                cand.alive = false;
                eliminated++;
                continue;
            }

            // Accumulate for variance computation
            cand.r_sum += r_i;
            cand.b_sum += b_i;
            cand.r_sq_sum += r_i * r_i;
            cand.b_sq_sum += b_i * b_i;
            cand.valid_count++;
        }

        size_t alive = 0;
        for (const auto & c : candidates) if (c.alive) alive++;
        Log::debug("NeutralWB Phase 2: bracket ", imgIdx,
                   " -> eliminated ", eliminated, ", alive ", alive);
    }

    // Require minimum valid brackets
    for (auto & cand : candidates) {
        if (cand.alive && cand.valid_count < NWB_MIN_VALID_BRACKETS) {
            cand.alive = false;
        }
    }

    // Count final survivors
    std::vector<NeutralCandidate*> survivors;
    for (auto & cand : candidates) {
        if (cand.alive) survivors.push_back(&cand);
    }

    Log::debug("NeutralWB: ", survivors.size(), " survivors after all phases");

    if ((int)survivors.size() < NWB_MIN_SURVIVORS) {
        Log::debug("NeutralWB: only ", survivors.size(), " survivors (need ",
                   NWB_MIN_SURVIVORS, "). Using camera WB.");
        return false;
    }

    // --- Phase 3: Score survivors ---
    for (auto * cand : survivors) {
        float n = (float)cand->valid_count;
        float r_mean = cand->r_sum / n;
        float b_mean = cand->b_sum / n;
        float r_var = (cand->r_sq_sum / n) - (r_mean * r_mean);
        float b_var = (cand->b_sq_sum / n) - (b_mean * b_mean);
        float chroma_var = r_var + b_var;

        // Ratio spread: how far r and b are from each other after preMul normalization
        float rn = r_mean * params.preMul[0] / params.preMul[1];
        float bn = b_mean * params.preMul[2] / params.preMul[1];
        float ratio_spread = std::abs(rn - bn);

        // Store score in r_sq_sum (reuse field, no longer needed)
        cand->r_sq_sum = n * (1.0f / (1.0f + chroma_var * 1000.0f))
                           * (1.0f / (1.0f + ratio_spread));
    }

    // Sort by score descending
    std::sort(survivors.begin(), survivors.end(), [](const NeutralCandidate * a,
                                                      const NeutralCandidate * b) {
        return a->r_sq_sum > b->r_sq_sum;
    });

    // --- Phase 4: Compute WB from top survivors ---
    size_t topN = std::max((size_t)50, (size_t)(survivors.size() * NWB_TOP_PERCENT));
    topN = std::min(topN, survivors.size());

    // Collect all r, b values from top survivors for median computation
    std::vector<float> r_vals, b_vals;
    for (size_t i = 0; i < topN; i++) {
        float n = (float)survivors[i]->valid_count;
        r_vals.push_back(survivors[i]->r_sum / n);
        b_vals.push_back(survivors[i]->b_sum / n);
    }

    // Median
    std::sort(r_vals.begin(), r_vals.end());
    std::sort(b_vals.begin(), b_vals.end());
    float median_r = r_vals[r_vals.size() / 2];
    float median_b = b_vals[b_vals.size() / 2];

    // Compute new WB multipliers
    // camMul[c] is proportional to 1/neutral_ratio for that channel
    // R/G_avg = median_r for a neutral surface -> camMul[R] / camMul[G] = 1/median_r / 1 = 1/median_r
    float old_camMul[4];
    std::copy_n(params.camMul, 4, old_camMul);

    params.camMul[0] = 1.0f / median_r;  // R
    params.camMul[1] = 1.0f;             // G
    params.camMul[2] = 1.0f / median_b;  // B
    params.camMul[3] = 1.0f;             // G2

    Log::debug("NeutralWB: computed WB from ", topN, " top survivors (best score=",
               survivors[0]->r_sq_sum, ")");
    Log::debug("NeutralWB: median R/G=", median_r, ", B/G=", median_b);
    Log::debug("NeutralWB: old camMul: ", old_camMul[0], ' ', old_camMul[1],
               ' ', old_camMul[2], ' ', old_camMul[3]);
    Log::debug("NeutralWB: new camMul: ", params.camMul[0], ' ', params.camMul[1],
               ' ', params.camMul[2], ' ', params.camMul[3]);

    // Also clear hasAsShotNeutral since we're overriding with computed WB
    params.hasAsShotNeutral = false;

    return true;
}
```

**Step 2: Verify build compiles**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`

Expected: Clean compile (no linker errors since declaration + implementation now both exist, but `computeNeutralWB` is not called yet)

**Step 3: Commit**

```bash
git add src/ImageStack.cpp
git commit -m "feat: implement computeNeutralWB with all 4 phases"
```

---

### Task 3: Integrate into Save Pipeline

**Files:**
- Modify: `src/ImageIO.cpp:192-210` (add call before compose)

**Step 1: Add the call**

In `ImageIO::save()`, after `params.adjustWhite(...)` is called on line 202, we need to insert the neutral WB call. However, per the design, neutral WB should run BEFORE `adjustWhite()`. The key insight: `adjustWhite()` normalizes camMul (min = 1.0) and handles the `camMul[0] == 0` fallback. We want `computeNeutralWB()` to set camMul BEFORE that normalization.

Modify `src/ImageIO.cpp` around lines 197-202. The current code:

```cpp
    RawParameters params = *rawParameters.back();
    params.width = stack.getWidth();
    params.height = stack.getHeight();

    progress.advance(5, "Rendering image");
    params.adjustWhite(stack.getImage(stack.size() - 1));
```

Change to:

```cpp
    RawParameters params = *rawParameters.back();
    params.width = stack.getWidth();
    params.height = stack.getHeight();

    progress.advance(3, "Computing white balance");
    stack.computeNeutralWB(params);

    progress.advance(5, "Rendering image");
    params.adjustWhite(stack.getImage(stack.size() - 1));
```

This way:
1. `computeNeutralWB()` sets camMul from neutral area detection (or leaves it unchanged on fallback)
2. `adjustWhite()` normalizes camMul (min=1.0), handles 3-color sensors, etc.

**Step 2: Verify build + test**

Run: `cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) 2>&1 | tail -20`

Expected: Clean build.

Then test with real brackets:

```bash
cd /Users/stefanbaxter/Development/hdrmerge/build
./hdrmerge.app/Contents/MacOS/hdrmerge -v -o /Volumes/Oryggi/Eignamyndir/RAW/test/merged-neutralwb/ /Volumes/Oryggi/Eignamyndir/RAW/test/DSC_*.NEF
```

Expected output should include `NeutralWB:` log lines showing:
- Number of Phase 1 candidates
- Elimination progress per bracket
- Final survivor count
- Old vs new camMul values (or fallback message)

**Step 3: Commit**

```bash
git add src/ImageIO.cpp
git commit -m "feat: integrate neutral WB computation into save pipeline"
```

---

### Task 4: Test and Validate Output

**Files:**
- No code changes — manual validation

**Step 1: Run on test brackets with verbose logging**

```bash
cd /Users/stefanbaxter/Development/hdrmerge/build
./hdrmerge.app/Contents/MacOS/hdrmerge -v -o /Volumes/Oryggi/Eignamyndir/RAW/test/merged-neutralwb-v1/ /Volumes/Oryggi/Eignamyndir/RAW/test/DSC_*.NEF 2>&1 | tee /tmp/neutralwb-test.log
```

**Step 2: Inspect the log output**

Look at the NeutralWB lines. Key things to validate:
- Phase 1 candidate count is reasonable (thousands to hundreds of thousands for 45MP)
- Progressive elimination reduces candidates significantly
- Final survivors >= 100 (otherwise fallback triggered)
- New camMul values are reasonable (similar order of magnitude to old values)
- The R/G and B/G deviation from camera WB is reasonable (< 20% difference)

**Step 3: Compare DNG output**

Open the output DNG in a raw processor (Adobe Camera Raw, RawTherapee, etc.) and compare WB against the same scene processed with camera WB. The neutral-area WB should produce more accurate gray rendering on achromatic surfaces in the scene.

**Step 4: Test edge cases**

Run on a single-bracket input (should skip with "need >= 2 brackets"):
```bash
./hdrmerge.app/Contents/MacOS/hdrmerge -v -o /tmp/single-test/ /Volumes/Oryggi/Eignamyndir/RAW/test/DSC_0001.NEF 2>&1 | grep NeutralWB
```

---

### Task 5: Threshold Tuning (Iterative)

**Files:**
- Modify: `src/ImageStack.cpp` (adjust constants at top)

This task is iterative based on real-world results from Task 4:

**If too few survivors:** Loosen thresholds:
- Increase `NWB_T_NEUTRAL` from 1.15 to 1.20
- Increase `NWB_T_STABILITY` from 0.05 to 0.08
- Decrease `NWB_MIN_VALID_BRACKETS` from 3 to 2

**If too many noisy survivors:** Tighten thresholds:
- Decrease `NWB_T_NEUTRAL` to 1.10
- Decrease `NWB_T_STABILITY` to 0.03

**If WB result is way off camera WB:** The scene may not have neutral surfaces, or the neutrality test is selecting colored surfaces. Check the log for median_r and median_b values — they should be close to the inverse of the camera camMul ratios for a correctly white-balanced scene.

After each adjustment, rebuild and re-test:
```bash
cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu) && ./hdrmerge.app/Contents/MacOS/hdrmerge -v -o /Volumes/Oryggi/Eignamyndir/RAW/test/merged-neutralwb-v2/ /Volumes/Oryggi/Eignamyndir/RAW/test/DSC_*.NEF 2>&1 | grep NeutralWB
```

**Commit when satisfied:**

```bash
git add src/ImageStack.cpp
git commit -m "tune: adjust neutral WB detection thresholds"
```
