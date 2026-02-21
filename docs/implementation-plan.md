# HDRMerge Optimization — Complete Implementation Plan

**Date**: 2026-02-20
**Research basis**: [`docs/research-modern-hdr-techniques.md`](research-modern-hdr-techniques.md)
**Approach**: One change at a time, tested after each change. 17 steps across 7 phases.

---

## Table of Contents

- [Test Environment](#test-environment)
- [Phase 1: Build & Dependencies](#phase-1-build--dependencies-no-code-changes) (Steps 0-2)
- [Phase 2: Compression Pipeline](#phase-2-compression-pipeline) (Steps 3-5)
- [Phase 3: ARM64 SIMD](#phase-3-arm64-simd-optimization) (Steps 6-9)
- [Phase 4: Alignment](#phase-4-alignment-improvements) (Steps 10-11)
- [Phase 5: Core Algorithm](#phase-5-core-algorithm-quality-improvements) (Steps 12-13)
- [Phase 6: I/O Architecture](#phase-6-io-architecture) (Step 14)
- [Phase 7: Format Evolution](#phase-7-format-evolution-deferred) (Steps 15-16)
- [Metrics Log](#metrics-log)
- [Success Criteria](#success-criteria)
- [Git Strategy](#git-strategy)

---

## Test Environment

### Test Files

RAW source directory: `/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/`
Original (baseline) output: `/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/org/`

**Test set** (3 bracket groups, 13 NEF files):

| Set | Files | Brackets | Original DNG |
|-----|-------|----------|--------------|
| A | `_EBX4622.NEF` `_EBX4623.NEF` `_EBX4624.NEF` | 3 | `org/_EBX4622-4624.dng` |
| B | `_EBX4640.NEF` .. `_EBX4644.NEF` | 5 | `org/_EBX4640-4644.dng` |
| C | `_EBX4650.NEF` .. `_EBX4654.NEF` | 5 | `org/_EBX4650-4654.dng` |

**Why these**: Mix of 3-bracket and 5-bracket sets. The 5-bracket sets exercise noise-optimal merge and ghost detection more thoroughly (more overlapping well-exposed regions). Three groups is enough to catch regressions without wasting time on 220 merges.

### Output Directory

Each step writes output to its own directory for comparison:

```
/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/
  step00-baseline/
  step01-zlibng/
  step02-libraw-upgrade/
  step03-libdeflate/
  step04-byteshuffle/
  step05-compression-level/
  step06-neon-fatten/
  step07-neon-float2half/
  step08-neon-boxblur/
  step09-vimage-eval/
  step10-subpixel-align/
  step11-feature-align/
  step12-noise-optimal-merge/
  step13-ghost-detection/
  step14-streaming-dng/
```

### Build Command

```bash
cd /Users/stefanbaxter/Development/hdrmerge/build
cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Test Command (example for Set A)

```bash
BINARY=build/hdrmerge.app/Contents/MacOS/hdrmerge
SRC=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26
OUTDIR=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/stepNN

/usr/bin/time -l $BINARY -v -o "$OUTDIR/_EBX4622-4624.dng" \
  "$SRC/_EBX4622.NEF" "$SRC/_EBX4623.NEF" "$SRC/_EBX4624.NEF"
```

### Metrics Recorded Per Step

| Metric | How | Purpose |
|--------|-----|---------|
| **Wall time** | `/usr/bin/time -l` (real) | Overall performance |
| **Output file size** | `ls -la` (bytes) | Compression effectiveness |
| **Byte-level diff** | `cmp` | Detect any output change |
| **Pixel-level diff** | Decode + compare float arrays | Verify data integrity |
| **Peak memory** | `/usr/bin/time -l` (max RSS) | Memory usage |
| **DNG readability** | Open in Lightroom / exiftool | Format correctness |

---

## Phase 1: Build & Dependencies (No Code Changes)

### Step 0: Establish Baseline

**Goal**: Build the current codebase and run the test set. Record all metrics as reference.

**Changes**: None — build and run as-is.

**Current state**: Binary links against `/usr/lib/libz.1.dylib` (stock macOS zlib 1.2.12). fattenMask runs scalar on ARM64. compressFloats uses scalar `DNG_FloatToHalf`. Compose uses binary pixel selection.

**Verification**:
- [ ] Output should match `org/` byte-for-byte
- [ ] Record: wall time, file size, peak RSS per test set (A, B, C)
- [ ] Record: git commit hash of baseline

---

### Step 1: zlib-ng Drop-In Swap

**Research ref**: [Section 4 — Compression Improvements](research-modern-hdr-techniques.md#4-compression-improvements): "zlib-ng (v2.3.2) — NEON-optimized zlib fork, 2-3x faster on ARM64, zero code changes in compat mode"

**Goal**: Replace stock macOS zlib with zlib-ng for 2-3x faster DEFLATE in `DngFloatWriter`.

**Code to adapt**: `CMakeLists.txt` only (build system change)

**Implementation**:
1. `brew install zlib-ng`
2. Add to `CMakeLists.txt` before `find_package(ZLIB)`:
   ```cmake
   set(ZLIB_ROOT /opt/homebrew/opt/zlib-ng)
   ```
3. Rebuild and verify link target with `otool -L`.

**Risk**: None. zlib-ng compat mode is API/ABI identical.

**Verification**:
- [ ] `otool -L` shows zlib-ng, not `/usr/lib/libz.1.dylib`
- [ ] Output files **byte-identical** to step 0
- [ ] Wall time measurably faster (expect 1.5-2.5x on compression)
- [ ] Record: wall time, file sizes, peak memory

**Rollback**: Remove `ZLIB_ROOT` line from CMakeLists.txt.

---

### Step 2: LibRaw Upgrade

**Research ref**: [Section 6 — LibRaw Upgrade Path](research-modern-hdr-techniques.md#6-libraw-upgrade-path): "Minimum: Upgrade to LibRaw 0.21.3+ for security fixes and 4-component JPEG DNG support"

**Goal**: Ensure we're on LibRaw 0.21.3+ (currently linked: libraw_r.23.dylib via Homebrew).

**Code to adapt**: None if Homebrew LibRaw is already 0.21.3+. Verify version.

**Implementation**:
1. Check: `brew info libraw` — confirm version >= 0.21.3
2. If older: `brew upgrade libraw`
3. Rebuild HDRMerge — no code changes expected (API is compatible for 0.21.x series)
4. If upgrading to 0.22.x: check for 64-bit file offset ABI break in `src/RawParameters.cpp` and `src/Image.cpp` (any `LibRaw_abstract_datastream` usage)

**Key files** (only relevant if 0.22 ABI break):
- `src/RawParameters.cpp` — `RawParameters::fromLibRaw()` reads LibRaw structures
- `src/Image.cpp` — `Image::loadRawImage()` calls LibRaw API
- `src/ImageIO.cpp` — `ImageIO::load()` calls `LibRaw::open_file()`

**Risk**: Low for 0.21.x. Medium for 0.22 (ABI break).

**Verification**:
- [ ] Build succeeds with no warnings
- [ ] Output files **byte-identical** to step 1 (same LibRaw decoding)
- [ ] `exiftool` metadata matches
- [ ] Record: wall time, file sizes

**Rollback**: `brew switch libraw` to previous version, rebuild.

---

## Phase 2: Compression Pipeline

### Step 3: libdeflate Direct Integration

**Research ref**: [Section 4](research-modern-hdr-techniques.md#4-compression-improvements): "libdeflate (v1.25): 2.6x faster than stock zlib [...] ARM NEON optimized Adler-32. Already installed at `/opt/homebrew/Cellar/libdeflate/1.25/`"

**Goal**: Replace the `zlib::compress()` call in DngFloatWriter with libdeflate for faster per-tile compression.

**Code to adapt**:
- `CMakeLists.txt` — add `find_package` or manual detection for libdeflate, link library
- `src/DngFloatWriter.cpp` — replace `compress()` call in `writeRawData()` (around the OpenMP tile loop where `compress(cBuffer, &compressedLength, uBuffer, dstLen)` is called)

**Implementation**:
1. CMakeLists.txt: find and link libdeflate
   ```cmake
   find_library(LIBDEFLATE_LIBRARY deflate PATHS /opt/homebrew/lib)
   find_path(LIBDEFLATE_INCLUDE libdeflate.h PATHS /opt/homebrew/include)
   target_link_libraries(hdrmerge ${LIBDEFLATE_LIBRARY})
   target_include_directories(hdrmerge PRIVATE ${LIBDEFLATE_INCLUDE})
   ```
2. DngFloatWriter.cpp: add `#include <libdeflate.h>` at top
3. In `writeRawData()`, before the OpenMP parallel block, create per-thread compressors:
   ```cpp
   // Create one compressor per thread (libdeflate is not thread-safe per instance)
   int nThreads = omp_get_max_threads();
   std::vector<struct libdeflate_compressor*> compressors(nThreads);
   for (int i = 0; i < nThreads; i++)
       compressors[i] = libdeflate_alloc_compressor(6);
   ```
4. Replace the `compress()` call:
   ```cpp
   // Before:
   int err = compress(cBuffer, &compressedLength, uBuffer, dstLen);
   // After:
   int tid = omp_get_thread_num();
   size_t compressedLength = libdeflate_zlib_compress(
       compressors[tid], uBuffer, dstLen, cBuffer, dstLen);
   ```
5. After the loop, free compressors:
   ```cpp
   for (auto c : compressors) libdeflate_free_compressor(c);
   ```

**Risk**: Low. libdeflate produces standard zlib-format (RFC 1950) streams. Compressed bytes may differ from zlib but decompressed data is identical.

**Verification**:
- [ ] Output DNG opens correctly in Lightroom / exiftool
- [ ] Decompressed pixel data **bit-identical** to step 2
- [ ] File sizes within ~2% of step 2
- [ ] Wall time faster (expect 10-30% additional vs zlib-ng)
- [ ] Record: wall time, file sizes, peak memory

**Rollback**: Revert DngFloatWriter.cpp `compress()` call, remove libdeflate from CMakeLists.txt.

---

### Step 4: Byte-Shuffle Preprocessing Before DEFLATE

**Research ref**: [Section 4 — Byte-Shuffle Preprocessing](research-modern-hdr-techniques.md#4-compression-improvements): "Before DEFLATE, applying byte-shuffle to float data improves compression ratio significantly (what OpenEXR ZIP does internally). Blosc2 benchmarks: 5-6x ratio improvement on some data."

**Goal**: Add byte-level reordering of float data before delta encoding and DEFLATE, grouping MSBs and LSBs together for better entropy reduction.

**Code to adapt**:
- `src/DngFloatWriter.cpp` — modify `encodeFPDeltaRow()` which currently does delta encoding (TIFF Predictor 2). Add byte-shuffle as a preprocessing step before delta.

**Implementation**:
The current `encodeFPDeltaRow()` performs:
1. Byte reordering (endian) per pixel
2. Delta encoding across the row

The enhancement adds a byte-plane separation step before delta:
```cpp
// For 32-bit float data (bytesps=4), width pixels:
// Input:  [B0B1B2B3] [B0B1B2B3] [B0B1B2B3] ...
// After shuffle: [B0B0B0...] [B1B1B1...] [B2B2B2...] [B3B3B3...]
// Then delta encoding on each plane
```

This is equivalent to TIFF Predictor 3 (floating-point predictor) which is already defined in the TIFF/DNG spec. The key is to use the correct DNG tag value for `Predictor` in the IFD.

**Key consideration**: Verify DNG readers support Predictor 3. Adobe DNG SDK, Lightroom, darktable, and RawTherapee all support it. This is what OpenEXR ZIP does and is well-established.

**Risk**: Medium. Changes the compression format — DNG readers must support Predictor 3. Need to verify compatibility across all target applications.

**Verification**:
- [ ] Output DNG opens correctly in Lightroom, darktable, Adobe DNG Converter, RawTherapee
- [ ] Decompressed pixel data **bit-identical** to step 3
- [ ] File sizes **smaller** than step 3 (expect 10-30% improvement on float data)
- [ ] Wall time similar or slightly slower (shuffle adds a small pass, offset by better compression)
- [ ] Record: wall time, file sizes, peak memory

**Rollback**: Revert `encodeFPDeltaRow()` changes, restore Predictor tag to 2.

---

### Step 5: Configurable Compression Level

**Research ref**: [Section 4](research-modern-hdr-techniques.md#4-compression-improvements): "No CLI option to control compression level. Level 1 would be ~3x faster at the cost of ~20% larger files."

**Goal**: Add `-c LEVEL` CLI option (1-12 for libdeflate, 1=fastest, 6=default, 12=max).

**Code to adapt**:
- `src/LoadSaveOptions.hpp` — add `int compressionLevel = 6;` to `SaveOptions` struct
- `src/Launcher.cpp` — add `-c LEVEL` CLI parsing in the option loop (near the existing `-b`, `-r`, `-p` parsing)
- `src/DngFloatWriter.cpp` — pass `compressionLevel` to `libdeflate_alloc_compressor()`
- `src/DngPropertiesDialog.cpp` — optionally add compression level to GUI dialog

**Implementation**:
1. `LoadSaveOptions.hpp`: add field to SaveOptions
2. `Launcher.cpp`: in the CLI parsing block, add:
   ```cpp
   } else if (opt == "-c") {
       if (++i < argc) {
           saveOptions.compressionLevel = std::min(12, std::max(1, atoi(argv[i])));
       }
   ```
3. `DngFloatWriter.cpp`: use `saveOptions.compressionLevel` when creating compressors
4. Update help text in `Launcher.cpp` to document `-c`

**Risk**: None. Default behavior unchanged.

**Verification**:
- [ ] Default (`-c 6` or omitted): **byte-identical** to step 4
- [ ] `-c 1`: larger files, faster merge
- [ ] `-c 12`: smaller files, slower merge
- [ ] All levels produce valid DNG files
- [ ] Record: wall time and file size for levels 1, 6, 9, 12

**Rollback**: Remove `-c` parsing and field.

---

## Phase 3: ARM64 SIMD Optimization

### Step 6: NEON fattenMask

**Research ref**: [Section 7 — Apple Silicon Optimization](research-modern-hdr-techniques.md#7-apple-silicon-optimization): "fattenMask() in ImageStack.cpp has an SSE2 path [...] On ARM64, this falls back to scalar loops — estimated 8-10x slower."

**Goal**: Add ARM NEON code path for fattenMask using `vmaxq_u8` (16 bytes at a time), matching the existing SSE2 path's approach.

**Code to adapt**:
- `src/ImageStack.cpp` — the fattenMask function has three versions:
  - Scalar fallback (lines ~205-301, used when neither SSE2 nor NEON available)
  - SSE2 version (lines ~302-394, guarded by `#ifdef __SSE2__`)
  - **New**: NEON version, guarded by `#if defined(__ARM_NEON__) || defined(__ARM_NEON)`

**Implementation**:
1. Add `#include <arm_neon.h>` guarded by `#ifdef __ARM_NEON__`
2. Copy the SSE2 version structure and replace intrinsics:

| SSE2 | NEON | Purpose |
|------|------|---------|
| `__m128i` | `uint8x16_t` | 128-bit vector type |
| `_mm_loadu_si128` | `vld1q_u8` | Unaligned load 16 bytes |
| `_mm_storeu_si128` | `vst1q_u8` | Unaligned store 16 bytes |
| `_mm_max_epu8` | `vmaxq_u8` | Per-byte unsigned max |
| `_mm_set1_epi8(x)` | `vdupq_n_u8(x)` | Broadcast scalar to all lanes |
| `_mm_setzero_si128()` | `vdupq_n_u8(0)` | Zero vector |

3. Keep scalar remainder loop for width not divisible by 16.
4. Guard: `#elif defined(__ARM_NEON__) || defined(__ARM_NEON)`

**Risk**: None. `vmaxq_u8` is exact unsigned byte max — bit-identical to scalar and SSE2.

**Verification**:
- [ ] Output **byte-identical** to step 5
- [ ] Wall time for compose phase measurably faster (expect 3-8x on fattenMask)
- [ ] Run with `-vv` to observe timing breakdown
- [ ] Record: wall time, file sizes, peak memory

**Rollback**: Remove `#elif __ARM_NEON__` block.

---

### Step 7: NEON float-to-half Conversion

**Research ref**: [Section 7](research-modern-hdr-techniques.md#7-apple-silicon-optimization): "compressFloats() in DngFloatWriter.cpp has an SSE4.1/F16C path for float-to-half conversion. ARM64 gets scalar DNG_FloatToHalf() — 15-20% slower."

**Goal**: Add NEON hardware float-to-half conversion in `compressFloats()`.

**Code to adapt**:
- `src/DngFloatWriter.cpp` — the `compressFloats()` function, specifically the 16-bit output path which currently has:
  - F16C/SSE path: uses `_mm_cvtps_ph` to convert 8 floats at a time
  - Scalar fallback: `DNG_FloatToHalf()` one at a time

**Implementation**:
1. Add NEON path in compressFloats for `bytesps == 2`:
   ```cpp
   #elif defined(__ARM_NEON__) && defined(__ARM_FP16_FORMAT_IEEE)
   #include <arm_neon.h>
   // Process 4 floats at a time (NEON FP16 converts 4-wide)
   int i = 0;
   for (; i <= count - 4; i += 4) {
       float32x4_t f = vld1q_f32(reinterpret_cast<const float*>(&src[i * 4]));
       float16x4_t h = vcvt_f16_f32(f);
       vst1_f16(reinterpret_cast<__fp16*>(&dst[i * 2]), h);
   }
   // Scalar remainder
   for (; i < count; i++) {
       dst_16[i] = DNG_FloatToHalf(src_32[i]);
   }
   ```
2. Apple Silicon M1+ natively supports `__ARM_FP16_FORMAT_IEEE`

**Note**: Only exercised with `-b 16`. Default `-b 32` has no float conversion.

**Risk**: Low. Must verify `vcvt_f16_f32` produces identical results to `DNG_FloatToHalf()` for edge cases (NaN, Inf, denormals, zero).

**Verification**:
- [ ] `-b 32`: output **byte-identical** to step 6 (path not exercised)
- [ ] `-b 16`: pixel data **bit-identical** to step 6 `-b 16`
- [ ] Wall time improvement with `-b 16` (expect 10-15% faster write)
- [ ] Test edge cases: fully saturated and fully black pixels
- [ ] Record: wall time and file sizes for both `-b 32` and `-b 16`

**Rollback**: Remove NEON block from compressFloats.

---

### Step 8: NEON Box Blur

**Research ref**: [Section 7](research-modern-hdr-techniques.md#7-apple-silicon-optimization): "Box blur (BoxBlur.cpp) — Scalar accumulation — 1.5-2x potential gain."

**Goal**: Vectorize the box blur accumulation loops with NEON `vaddq_f32` / `vsubq_f32` for the vertical (transpose) blur pass.

**Code to adapt**:
- `src/BoxBlur.cpp` — two functions:
  - `boxBlurH()` — horizontal blur with sliding window average. Already parallelized per-row with OpenMP. The inner loop is a running sum — hard to vectorize (serial dependency).
  - `boxBlurT()` — transpose (vertical) blur. Processes **8 columns at a time** for cache efficiency. The 8-column inner loop accumulates `val[k]` for k=0..7. This is the vectorization target.

**Implementation**:
1. In `boxBlurT()`, the 8-column accumulation block (currently scalar `val[0]..val[7]`):
   ```cpp
   // Current scalar (8 independent accumulators):
   for (int k = 0; k < 8; k++) {
       val[k] += src[(ti + j) * w + i + k] - src[(li + j) * w + i + k];
       dst[(i + k) * h + j] = val[k] * iarr;
   }
   ```
   Replace with NEON (two `float32x4_t` vectors for 8 floats):
   ```cpp
   float32x4_t v0 = vld1q_f32(&val[0]);
   float32x4_t v1 = vld1q_f32(&val[4]);
   float32x4_t add0 = vld1q_f32(&src[(ti+j)*w + i]);
   float32x4_t add1 = vld1q_f32(&src[(ti+j)*w + i + 4]);
   float32x4_t sub0 = vld1q_f32(&src[(li+j)*w + i]);
   float32x4_t sub1 = vld1q_f32(&src[(li+j)*w + i + 4]);
   v0 = vaddq_f32(v0, vsubq_f32(add0, sub0));
   v1 = vaddq_f32(v1, vsubq_f32(add1, sub1));
   float32x4_t scale = vdupq_n_f32(iarr);
   vst1q_f32(&dst_tmp[0], vmulq_f32(v0, scale));
   vst1q_f32(&dst_tmp[4], vmulq_f32(v1, scale));
   // Then scatter to transposed output...
   ```
2. The scatter to transposed layout `dst[(i+k)*h + j]` prevents simple NEON stores — may need to keep scalar scatter or use `vst1q_lane_f32` for strided writes.

**Risk**: Low for correctness. Floating-point addition is not strictly associative, but the operations here are identical (same order) — just batched in SIMD lanes, each lane independent. Results should be bit-identical.

**Verification**:
- [ ] Output **byte-identical** to step 7
- [ ] Wall time for blur phase faster (expect 1.3-1.8x on boxBlurT)
- [ ] Record: wall time, file sizes, peak memory

**Rollback**: Remove NEON block from boxBlurT.

---

### Step 9: Evaluate Apple Accelerate vImage

**Research ref**: [Section 7](research-modern-hdr-techniques.md#7-apple-silicon-optimization): "vImage: Morphological operations (dilation = fattenMask), convolutions, format conversions [...] No #ifdef branching needed — one code path for both architectures."

**Goal**: Evaluate whether Apple's vImage framework can replace our manual NEON intrinsics from steps 6-8, providing a single cross-architecture code path and potentially better performance (Apple tunes vImage per chip generation).

**Code to evaluate**:
- `src/ImageStack.cpp` fattenMask → `vImageMax_Planar8` (morphological dilation)
- `src/BoxBlur.cpp` → `vImageBoxConvolve_PlanarF` (box convolution)
- `src/DngFloatWriter.cpp` compressFloats → `vImageConvert_PlanarFtoPlanar16F` (float-to-half)

**Implementation**:
1. Create a test branch that replaces fattenMask's NEON/SSE2/scalar with vImage:
   ```cpp
   #include <Accelerate/Accelerate.h>
   vImage_Buffer srcBuf = { .data = maskData, .height = h, .width = w, .rowBytes = w };
   vImage_Buffer dstBuf = { .data = outData, .height = h, .width = w, .rowBytes = w };
   // Circular kernel for max dilation
   vImageMax_Planar8(&srcBuf, &dstBuf, NULL, 0, 0, radius*2+1, radius*2+1, kvImageNoFlags);
   ```
2. Benchmark against manual NEON from step 6
3. If vImage is faster or within 5%: adopt it (simpler code, auto-tuned per chip)
4. If vImage is significantly slower: keep manual NEON, document why

**Key consideration**: fattenMask uses a circular kernel (distance-based), not a rectangular kernel. `vImageMax_Planar8` uses rectangular kernels. May need to verify that a rectangular max approximation is acceptable, or implement circular via multiple passes.

**Risk**: Medium. vImage's rectangular max may not match the current circular max filter exactly — output could differ. This needs careful comparison.

**Verification**:
- [ ] Compare vImage output vs NEON output pixel-by-pixel
- [ ] If identical: adopt vImage, record performance delta
- [ ] If different: document the difference, decide whether circular vs rectangular matters
- [ ] Benchmark: compare wall time of vImage vs manual NEON for each operation
- [ ] Record: wall time comparison table

**Rollback**: Keep manual NEON from steps 6-8.

---

## Phase 4: Alignment Improvements

### Step 10: Sub-Pixel Phase Correlation Refinement

**Research ref**: [Section 3 — Sub-Pixel Alignment](research-modern-hdr-techniques.md#3-alignment-improvements-beyond-mtb): "Phase correlation: Compute translation in frequency domain; parabolic peak fitting provides sub-pixel accuracy. Recent work (Springer 2022) proposes pyramid phase correlation with upsampling."

**Goal**: After coarse MTB alignment (integer pixels), refine the translation to sub-pixel accuracy using phase correlation in the frequency domain.

**Code to adapt**:
- `src/Image.cpp` — `Image::alignWith()` currently returns integer `dx, dy` shifts from MTB
- `src/ImageStack.cpp` — `ImageStack::align()` applies the integer shifts
- **New code**: Add a sub-pixel refinement function called after MTB alignment

**Implementation**:
1. After MTB gives integer shift (dx, dy), extract small overlapping patches (~256x256) from both images centered on the aligned region
2. Compute 2D FFT of both patches (use ALGLIB FFT or Apple Accelerate `vDSP_fft2d_zip`)
3. Compute normalized cross-power spectrum: `R = F1 * conj(F2) / |F1 * conj(F2)|`
4. Inverse FFT of R gives correlation surface
5. Find peak location with sub-pixel accuracy via parabolic interpolation on the 3x3 neighborhood around the integer peak
6. Apply sub-pixel shift as a fractional displacement in `Array2D`

**Key consideration**: Sub-pixel shifts require interpolation when compositing. Currently all shifts are integer (displacing the Array2D). With sub-pixel shifts, the compose loop needs bilinear interpolation for the sub-pixel component. This adds complexity and a small performance cost.

**Alternative**: Use sub-pixel alignment only for quality metrics / reporting, keep integer shifts for compositing. This would detect slight misalignment without the interpolation cost.

**Dependencies**: FFT library. Options:
- ALGLIB (already linked): `fftr1d` / `fftc1d` — 1D only, would need to implement 2D via row/column passes
- Apple Accelerate `vDSP_fft2d_zip` — native 2D FFT, fast on Apple Silicon

**Risk**: Medium. Introduces floating-point shifts requiring interpolation in compose. Quality improvement may be subtle (most tripod shots are aligned to < 0.5px by MTB already).

**Verification**:
- [ ] On well-aligned tripod data: sub-pixel shift should be near-zero (< 0.3px), output nearly identical to step 9
- [ ] Measure alignment residual before and after refinement
- [ ] If sub-pixel shift applied: no visible interpolation artifacts
- [ ] Record: alignment residuals, wall time, file sizes

**Rollback**: Remove sub-pixel refinement, keep integer MTB alignment.

---

### Step 11: Feature-Based Alignment (ORB/AKAZE Homography)

**Research ref**: [Section 3 — Feature-Based Alignment](research-modern-hdr-techniques.md#3-alignment-improvements-beyond-mtb): "ORB: Best performer on brightness-varied images. AKAZE: Best for blur, rotation, perspective. pfstools switched to AKAZE. Compute homography from the pair of exposures closest in EV."

**Goal**: Add feature-based homography alignment as an optional mode for handheld brackets, handling rotation, perspective, and scale in addition to translation.

**Code to adapt**:
- `CMakeLists.txt` — add optional OpenCV dependency (`find_package(OpenCV COMPONENTS core features2d calib3d)`)
- `src/Image.cpp` — add `Image::alignWithHomography()` method
- `src/ImageStack.cpp` — add `ImageStack::alignFeatureBased()` calling the new method
- `src/Launcher.cpp` — add `--align-features` CLI flag (default remains MTB for backward compat)
- `src/Image.hpp` — add homography storage (`cv::Mat H` or float[9] matrix)

**Implementation**:
1. Add OpenCV as optional dependency in CMakeLists.txt:
   ```cmake
   find_package(OpenCV QUIET COMPONENTS core features2d calib3d)
   if(OpenCV_FOUND)
       add_definitions(-DHAVE_OPENCV)
       target_link_libraries(hdrmerge ${OpenCV_LIBS})
   endif()
   ```
2. In `Image::alignWithHomography()`:
   - Convert raw Bayer to grayscale (subsample green channel)
   - Detect AKAZE features in both reference and target exposure
   - Match features using BFMatcher with ratio test (Lowe's 0.7)
   - Filter matches to non-saturated regions only
   - Compute homography via `cv::findHomography(pts1, pts2, cv::RANSAC, 3.0)`
   - Store 3x3 homography matrix
3. In compose: warp pixel coordinates through inverse homography before sampling
4. Best practice: align each image to the reference (middle exposure) using the closest-EV pair for feature matching, then chain transformations

**Risk**: Medium. OpenCV is a large dependency (~50 MB). Feature matching between very different exposures is inherently noisy. RANSAC may fail on low-texture scenes.

**Verification**:
- [ ] With `--align-features` off: output **byte-identical** to step 10 (no regression)
- [ ] With `--align-features`: test on handheld brackets if available
- [ ] On tripod data: homography should be near-identity, output should be very close to MTB result
- [ ] No misalignment artifacts visible in output
- [ ] Record: number of features matched, RANSAC inlier ratio, wall time

**Rollback**: Disable `--align-features` flag, remove OpenCV usage (keep as optional dep).

---

## Phase 5: Core Algorithm (Quality Improvements)

### Step 12: Noise-Optimal Weighted Merge (Poisson Estimator)

**Research ref**: [Section 1 — Noise-Optimal Merge Weighting](research-modern-hdr-techniques.md#1-noise-optimal-merge-weighting): Hanji et al. (ECCV 2020): "A Poisson noise estimator achieves near-MLE-optimal merge quality without camera-specific calibration." Granados et al. (CVPR 2010): optimal weight `w_k = t_k² / (t_k × raw_k + σ_read²)`. Aguerrebere et al. (2014): MLE within ~10% of CRLB.

**This is the most significant quality change — it alters the core merge algorithm.**

**Goal**: Replace binary pixel selection in `ImageStack::compose()` with noise-variance-weighted averaging for overlapping well-exposed regions, yielding sqrt(N) SNR improvement.

**Code to adapt**:
- `src/ImageStack.cpp` — the compositing loop in `compose()` (the `#pragma omp parallel for` loop that iterates over y-coordinates and blends exposures)
- `src/ImageStack.hpp` — potentially add exposure time accessors if not already available
- `src/Image.hpp` / `src/RawParameters.hpp` — need exposure time per image for Poisson weighting

**Current algorithm** (in compose loop):
```
p = map(x, y)  // blurred mask value: fractional index into exposure stack
j = floor(p)   // primary exposure index
Blend pixel j and j+1 using fractional part of p
Apply response function
Handle saturation
```

**New algorithm** (Poisson-weighted merge):
```
For each pixel (x, y):
  weightedSum = 0
  totalWeight = 0
  For each exposure k (0..N-1):
    raw_k = images[k](x, y)
    if raw_k is valid (above black + margin, below saturation - margin):
      radiance_k = response(raw_k)  // or simply (raw_k - black) for linear sensors
      w_k = 1.0 / max(radiance_k, epsilon)  // Poisson inverse-variance
      weightedSum += w_k * radiance_k
      totalWeight += w_k

  if totalWeight > 0 and multiple valid exposures:
    merged = weightedSum / totalWeight
  else:
    // Fall back to current single-exposure selection
    merged = response(images[bestExposure](x, y))
```

**Key implementation considerations**:
- The existing mask/fatten/blur pipeline can remain for determining transition zones and feathering, but the actual pixel value computation changes
- The `reduction(max:maxVal)` OpenMP clause must be preserved for output scaling
- Epsilon = small value (e.g., 1e-6) to avoid division by zero in deep shadows
- Per-pixel loop over all N exposures is slightly more expensive than the current 2-exposure interpolation, but N is typically 3-5
- Exposure times accessible from `RawParameters::shutter` for each image

**Risk**: Medium. Changes pixel values — output intentionally differs from baseline.

**Verification**:
- [ ] Output **will differ** from previous steps — expected and correct
- [ ] DNG opens correctly in Lightroom / darktable / Adobe DNG Converter
- [ ] Visual: mid-tones **cleaner** (less noise), especially visible in 5-bracket Set B and C
- [ ] Highlights and deep shadows **unchanged** (only one valid exposure there)
- [ ] No banding, color shifts, or artifacts at exposure transition zones
- [ ] Export baseline and new DNG to TIFF in Lightroom, compare 100% crops
- [ ] Record: wall time (expect < 5% regression), file sizes

**Rollback**: Revert compose loop to binary pixel selection.

---

### Step 13: Sigma-Clipping Ghost Detection

**Research ref**: [Section 2 — Ghost Detection & Deghosting](research-modern-hdr-techniques.md#2-ghost-detection--deghosting): "Sigma Clipping on Radiance-Normalized Values — adapted from astrophotography stacking [...] catches moving objects, hot pixels, and flare." Also: "Kappa-Sigma Clipping [...] Iterative variant using MAD."

**Goal**: Add per-pixel outlier rejection before the weighted merge (step 12), detecting moving objects, hot pixels, and partial flare by comparing radiance-normalized values across exposures.

**Code to adapt**:
- `src/ImageStack.cpp` — add ghost detection inside the compose loop, before the weighted merge from step 12
- `src/Launcher.cpp` — add `--deghost` CLI flag with optional level (none/low/medium/high or sigma threshold)
- `src/LoadSaveOptions.hpp` — add `float deghostSigma = 0.0f;` (0 = off, positive = enabled with that sigma threshold, e.g., 3.0)

**Implementation**:
Integrate into the per-pixel compose loop from step 12:

```cpp
// After computing radiance_k for each valid exposure k:
if (numValid >= 3 && deghostSigma > 0) {
    // 1. Compute median of radiance values
    std::nth_element(radiances, radiances + numValid/2, radiances + numValid);
    float median = radiances[numValid / 2];

    // 2. Compute MAD (median absolute deviation)
    for (int k = 0; k < numValid; k++)
        absDevs[k] = fabs(radiances[k] - median);
    std::nth_element(absDevs, absDevs + numValid/2, absDevs + numValid);
    float mad = absDevs[numValid / 2] * 1.4826f;  // MAD to sigma

    // 3. Reject outliers
    for (int k = 0; k < numValid; k++) {
        if (fabs(radiances[k] - median) > deghostSigma * mad) {
            weights[k] = 0;  // Exclude this exposure at this pixel
        }
    }
    // 4. Proceed with weighted merge of remaining exposures
}
```

**Key considerations**:
- Minimum 3 valid exposures needed for sigma clipping to be meaningful
- For 3 exposures, MAD-based clipping is conservative (only rejects extreme outliers)
- For 5+ exposures, clipping is more effective
- Small arrays (3-5 elements) — `std::nth_element` is fine, no heap allocation needed
- Allocation-free: use stack arrays of max bracket size (typically <= 10)
- The `--deghost` flag defaults to off for backward compatibility

**Risk**: Medium. Adds computation per pixel. May incorrectly reject valid data in scenes with strong gradients across exposures (e.g., near saturation boundaries). The sigma threshold should be conservative (3.0 default).

**Verification**:
- [ ] With `--deghost` off: output **identical** to step 12
- [ ] With `--deghost 3.0`: output may differ slightly (outlier pixels use different weighting)
- [ ] On test data (static scene): minimal difference (no ghosts to detect)
- [ ] Hot pixel rejection: any known hot pixels should be cleaned
- [ ] No artifacts introduced by overly aggressive clipping
- [ ] Record: wall time (expect 5-10% overhead due to per-pixel median/MAD), file sizes

**Rollback**: Remove `--deghost` flag and clipping logic from compose loop.

---

## Phase 6: I/O Architecture

### Step 14: Streaming DNG Writer

**Research ref**: Codebase analysis finding: "DngFloatWriter::write() allocates the complete output file in a single std::unique_ptr<uint8_t[]>. For 61MP at 32-bit: ~800 MB–1 GB. With 4 concurrent jobs, peak memory can hit 6-8 GB."

**Goal**: Refactor DngFloatWriter to write tiles to disk as they are compressed, instead of buffering the entire DNG in memory. Reduce peak memory from ~800 MB to ~50-100 MB per job.

**Code to adapt**:
- `src/DngFloatWriter.cpp` — major refactor of `write()` method:
  - Currently: allocates `fileData` for entire output, builds all IFDs and tiles in memory, writes once
  - New: write IFD headers with placeholder offsets, stream compressed tiles to disk, seek back to patch tile offset table
- `src/DngFloatWriter.hpp` — interface may change (streaming vs monolithic)
- `src/TiffDirectory.cpp` / `src/TiffDirectory.hpp` — may need ability to serialize to file stream instead of memory buffer

**Implementation approach**:
1. Open output file at start of `write()`
2. Write main IFD + thumbnail (small, keep in memory)
3. Write raw tile offset/byte count arrays with placeholder zeros
4. For each tile (parallelized with OpenMP):
   - Compress tile to thread-local buffer
   - Write compressed data to file (sequential, protected by mutex)
   - Record actual offset and byte count
5. Seek back and patch the offset/byte count arrays
6. Write preview IFD at end of file

**Key challenges**:
- TIFF format requires knowing tile offsets before writing tile data (chicken-and-egg). Solution: write placeholder offsets, then seek back to patch them after all tiles written.
- OpenMP tile compression is parallel, but file writes must be sequential. Use a mutex or ordered write queue.
- This is a significant refactor touching the core output path.

**Risk**: High. Major refactor of DngFloatWriter. Must produce valid TIFF/DNG structure. Extensive testing required.

**Verification**:
- [ ] Output DNG opens correctly in Lightroom, darktable, exiftool, Adobe DNG Converter
- [ ] Decompressed pixel data **bit-identical** to step 13
- [ ] Peak RSS **significantly reduced** (target: < 200 MB vs ~800 MB baseline)
- [ ] File sizes identical to step 13 (same compression, same data)
- [ ] Wall time similar (slight overhead from seek-and-patch, offset by less memory pressure)
- [ ] Batch mode with `-j 4`: total memory stays reasonable (< 1 GB)
- [ ] Record: wall time, file sizes, peak memory

**Rollback**: Revert to monolithic in-memory DNG writer.

---

## Phase 7: Format Evolution (Deferred)

These steps are documented for completeness but should be deferred until ecosystem support matures.

### Step 15: DNG 1.7 / JPEG XL Output

**Research ref**: [Section 5 — DNG Format Updates](research-modern-hdr-techniques.md#5-dng-format-updates): "DNG 1.7.0.0 (June 2023): JPEG XL added as compression codec. Lossless JXL: ~40% smaller files than DEFLATE." Also: "JPEG XL for Float Data — Not Ready Yet: libjxl float compression is immature."

**Goal**: Add JPEG XL as an optional DNG compression codec.

**Code to adapt**:
- `CMakeLists.txt` — add optional libjxl dependency
- `src/DngFloatWriter.cpp` — add JXL compression path alongside DEFLATE
- `src/TiffDirectory.cpp` — DNG 1.7 tag values for JXL compression
- `src/Launcher.cpp` — add `--jxl` flag

**Why deferred**:
- libjxl float-data compression has poor ratios (43.7 bpp lossless on PFM data)
- Subnormal FP16 values don't roundtrip losslessly through JXL
- darktable (issue #17152), RawTherapee (issue #7131) have open issues for DNG 1.7 support
- Recommended: revisit when libjxl >= 1.0 stable and major raw processors support DNG 1.7 reading

**Prerequisite**: Step 14 (streaming writer) makes this easier since JXL tiles can also be streamed.

---

### Step 16: Gain Maps / ISO 21496-1

**Research ref**: [Section 7 — Apple HDR / Gain Maps](research-modern-hdr-techniques.md#7-apple-silicon-optimization): "Apple 'Adaptive HDR' using gain maps. ISO 21496-1 standard. libjxl v0.11.0 added Gain Map API. Emerging standard for HDR display on Apple devices."

**Goal**: Embed HDR gain maps in output DNG for automatic HDR display on Apple devices.

**Code to adapt**:
- `src/DngFloatWriter.cpp` — generate gain map from the HDR data
- `src/TiffDirectory.cpp` — additional IFD for gain map per ISO 21496-1

**Why deferred**:
- Standard is still at draft international phase
- No established open-source implementation for DNG gain maps
- Requires understanding of target display capabilities (SDR base + HDR headroom)
- Revisit when ISO 21496-1 is finalized and Apple's implementation is documented

---

## Metrics Log

Fill in after each step. Steps 0-9 should produce bit-identical output. Steps 10+ may differ.

### Steps 0-5: Build, Dependencies, Compression

| Step | Description | Set A time | Set A size | Set B time | Set B size | Set C time | Set C size | Identical to prev |
|------|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-------------------|
| 0 | Baseline | 5.86s | 117,113,806 | 6.53s | 106,710,316 | 6.32s | 129,106,446 | byte-identical to org/ |
| 1 | zlib-ng 2.3.3 | 5.51s | 116,870,558 | 6.16s | 106,320,644 | 6.27s | 128,617,186 | pixel-identical (timestamp + compressed bytes differ) |
| 2 | LibRaw 0.21.4 | — | — | — | — | — | — | no change (already >= 0.21.3) |
| 3 | libdeflate 1.25 | 5.61s | 116,822,792 | 6.55s | 106,663,814 | 6.41s | 128,919,166 | pixel-identical |
| 4 | Byte-shuffle | — | — | — | — | — | — | SKIPPED: already implemented (DNG FP predictor tag 34894) |
| 5 | Compression level | 4.56s (-c 6) | 116,822,792 | 5.81s (-c 6) | 106,663,814 | — | — | pixel-identical at all levels |

### Steps 6-9: ARM64 SIMD

| Step | Description | Set A time | Set B time | Set C time | Identical to prev | Speedup vs prev |
|------|-------------|-----------|-----------|-----------|-------------------|----------------|
| 6 | NEON fattenMask | | | | | |
| 7 | NEON float2half | | | | | |
| 8 | NEON box blur | | | | | |
| 9 | vImage eval | | | | see notes | |

### Steps 10-11: Alignment

| Step | Description | Set A align residual | Set B align residual | Identical to prev |
|------|-------------|---------------------|---------------------|-------------------|
| 10 | Sub-pixel phase corr | | | nearly |
| 11 | Feature homography | | | off by default |

### Steps 12-13: Algorithm

| Step | Description | Set A time | Set A size | Visual quality | Opens in LR |
|------|-------------|-----------|-----------|---------------|------------|
| 12 | Noise-optimal merge | | | | |
| 13 | Ghost detection | | | | |

### Step 14: I/O

| Step | Description | Set A time | Set A size | Peak RSS (MB) | Identical to prev |
|------|-------------|-----------|-----------|---------------|-------------------|
| 14 | Streaming DNG | | | | pixel-identical |

### Step 5 detail: Compression levels

| Level | Set A time | Set A size | Set B time | Set B size |
|-------|-----------|-----------|-----------|-----------|
| 1 | 5.90s | 118,448,148 | 6.30s | 108,429,272 |
| 6 | 4.56s | 116,822,792 | 5.81s | 106,663,814 |
| 9 | 4.85s | 116,494,028 | — | — |
| 12 | 7.16s | 115,379,750 | — | — |

---

## Success Criteria

### Performance (Phase 1-3)
- Steps 1-8 combined: **2-4x overall speedup** on the merge+write pipeline vs baseline
- Compression (steps 1-4): **2-3x faster** DEFLATE encoding
- SIMD (steps 6-8): **3-8x faster** fattenMask, measurable blur improvement

### Quality (Phase 5)
- Step 12: visually cleaner mid-tones in 5-bracket sets, sqrt(N) theoretical SNR gain
- Step 13: robust to hot pixels and flare, no artifacts on static scenes

### Memory (Phase 6)
- Step 14: peak RSS **reduced 70-80%** (< 200 MB vs ~800 MB for 61MP 32-bit output)

### Correctness (All phases)
- All output DNG files load in Adobe DNG Converter, Lightroom, darktable, RawTherapee
- All EXIF/metadata preserved
- Address Sanitizer clean on at least one full run
- No crashes or hangs on any test set

---

## Research Coverage

Every actionable finding from `research-modern-hdr-techniques.md` is mapped below:

| Research Section | Finding | Plan Step | Status |
|-----------------|---------|-----------|--------|
| Section 1 | Noise-optimal merge (Poisson) | Step 12 | Planned |
| Section 2 | Sigma-clipping ghost detection | Step 13 | Planned |
| Section 3 | Sub-pixel phase correlation | Step 10 | Planned |
| Section 3 | Feature-based alignment (ORB/AKAZE) | Step 11 | Planned |
| Section 4 | zlib-ng drop-in | Step 1 | **Done** (2dc75eb) |
| Section 4 | libdeflate integration | Step 3 | **Done** (df9cb99) |
| Section 4 | Byte-shuffle preprocessing | Step 4 | **Skipped** (already implemented as DNG FP predictor) |
| Section 4 | Configurable compression level | Step 5 | **Done** (9cd3485) |
| Section 5 | DNG 1.7 / JPEG XL | Step 15 | Deferred (ecosystem) |
| Section 6 | LibRaw upgrade | Step 2 | **Done** (0.21.4 verified) |
| Section 7 | NEON fattenMask | Step 6 | Planned |
| Section 7 | NEON float-to-half | Step 7 | Planned |
| Section 7 | NEON box blur | Step 8 | Planned |
| Section 7 | Apple Accelerate vImage | Step 9 | Planned (eval) |
| Section 7 | Metal compute | — | Deferred (large effort, low ROI vs CPU) |
| Section 7 | Gain maps / ISO 21496-1 | Step 16 | Deferred (standard not final) |
| Section 8 | Exposure fusion | — | Not applicable (confirmed) |
| Section 9 | Bit depth validation | — | Validated: 32-bit correct, no change |
| Section 10 | Bayer-domain merge | — | Validated: architecture correct, no change |
| Section 11 | Multi-exposure super-resolution | — | Not applicable (confirmed) |
| Section 12 | Deep learning HDR | — | Not applicable for C++ desktop tool |

---

## Git Strategy

Each step on its own branch, chained:

```
master (current uncommitted changes)
  └── opt/step01-zlibng
      └── opt/step02-libraw
          └── opt/step03-libdeflate
              └── opt/step04-byteshuffle
                  └── opt/step05-compression-level
                      └── opt/step06-neon-fatten
                          └── opt/step07-neon-float2half
                              └── opt/step08-neon-boxblur
                                  └── opt/step09-vimage-eval
                                      └── opt/step10-subpixel-align
                                          └── opt/step11-feature-align
                                              └── opt/step12-noise-optimal
                                                  └── opt/step13-ghost-detect
                                                      └── opt/step14-streaming-dng
```

After each step passes verification:
- Commit with descriptive message referencing the research section
- Tag with step number: `opt-step01`, `opt-step02`, etc.

After all steps pass:
- Squash-merge the chain back to master (or cherry-pick individual steps if some are rejected)
