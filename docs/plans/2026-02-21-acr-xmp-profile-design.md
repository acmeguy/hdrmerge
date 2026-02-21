# ACR XMP Profile & Adaptive Curves

## Problem

After HDR merging, every DNG requires the same 5 Lightroom adjustments (Exposure, Highlights, Shadows, Contrast, Blue Saturation). These could be embedded as Adobe Camera Raw (ACR) XMP metadata in the DNG so Lightroom applies them on import.

Additionally, per-image adaptive RGB tone curves could be generated via an ONNX neural network model, giving each image content-aware tone mapping without modifying the raw pixel data.

## Design

### CLI Interface

```
-L profile.xmp     Load ACR settings from a Lightroom .xmp preset file
--auto-curves       Run ONNX model to generate per-image adaptive RGB curves
```

Both flags are optional and independent. When combined, the profile's slider values and the model's tone curves coexist in the DNG's XMP.

### Profile Reading (-L)

Use exiv2 to open the `.xmp` file and extract all `crs:*` keys. Store them as key-value pairs. During metadata writing, inject these into the destination DNG's XmpData.

The reader is generic: it copies ALL `crs:` tags from the profile, not a hardcoded list. This means any ACR setting present in the preset (Clarity, Dehaze, HSL, etc.) flows through automatically.

Required tags for Lightroom to interpret values correctly:
- `crs:ProcessVersion` (e.g., "15.4")
- `crs:Version` (e.g., "16.1")

### Adaptive Curves (--auto-curves)

When enabled:

1. After `renderPreview()` produces the sRGB QImage, resize to 384x384
2. Convert to float32 CHW tensor normalized to [0,1]
3. Run ONNX Runtime inference with `free_xcittiny_wa14.onnx` model
4. Get 3x256 float curve LUTs (one per R, G, B channel)
5. Clip to [0,1], enforce monotonicity (sweep right-to-left)
6. Fit to ~20 control points per channel (greedy max-error insertion, threshold 0.005)
7. Write as `crs:ToneCurvePV2012Red`, `crs:ToneCurvePV2012Green`, `crs:ToneCurvePV2012Blue`
8. Set `crs:ToneCurveName2012` to "Custom"

If the profile also contains tone curve tags, the adaptive curves take precedence (per-image > fixed).

### Injection Point

In ExifTransfer, after `copyAllMetadata()`:

```
transferFile() / transfer()
  +-- copyAllMetadata(src, dst)       // existing: EXIF/IPTC/XMP from source NEF
  +-- injectACRProfile(dst, profile)  // NEW: crs: tags from .xmp profile
  +-- injectAdaptiveCurves(dst, curves) // NEW: crs:ToneCurve* from ONNX
```

Profile tags override any `crs:` tags from the source. Adaptive curve tags are written only if `--auto-curves` was used.

### Data Flow

```
SaveOptions gains:
  - QString acrProfilePath        (from -L flag)
  - bool autoCurves               (from --auto-curves flag)
  - AdaptiveCurves adaptiveCurves (populated after ONNX inference)

ImageIO::save():
  1. compose()            -> merged float data
  2. renderPreview()      -> sRGB QImage
  3. if autoCurves:       run ONNX on preview -> fill adaptiveCurves
  4. DngFloatWriter::write() -> DNG file with tiles
  5. Exif::transferFile() -> copy metadata + inject profile + inject curves
```

### Build System

ONNX Runtime is an optional dependency:

```cmake
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
  add_definitions(-DHAVE_ONNXRUNTIME)
  target_link_libraries(hdrmerge onnxruntime)
endif()
```

If ONNX Runtime is not installed, `--auto-curves` prints a warning and is ignored. The `-L` profile feature works regardless with no new dependencies.

### ACR XMP Tags Reference

Slider values (from profile):
- `crs:Exposure2012` — range -5.0 to +5.0
- `crs:Highlights2012` — range -100 to +100
- `crs:Shadows2012` — range -100 to +100
- `crs:Contrast2012` — range -100 to +100
- `crs:SaturationAdjustmentBlue` — range -100 to +100

Tone curves (from ONNX or profile):
- `crs:ToneCurvePV2012` — master curve, rdf:Seq of "input, output" pairs (0-255)
- `crs:ToneCurvePV2012Red` — red channel curve
- `crs:ToneCurvePV2012Green` — green channel curve
- `crs:ToneCurvePV2012Blue` — blue channel curve
- `crs:ToneCurveName2012` — "Linear" or "Custom"

### Model Details

- Model: AutoLevels `free_xcittiny_wa14.onnx` (XCiT-tiny-12 backbone)
- Input: [1, 3, 384, 384] float32, normalized [0,1], sRGB
- Output: [1, 3, 256] float32 — per-channel 256-entry curve LUTs
- Training: scanned film photos (not digital RAW — domain mismatch possible)
- Size: ~10-20MB
- Inference: ~100ms CPU on Apple Silicon

### Limitations

- The ONNX model was trained on scanned film, not HDR-merged digital landscapes. Curve quality on user's content is unproven.
- The sRGB preview used as model input may not perfectly represent Lightroom's rendering of the raw data.
- Per-channel RGB curves cannot express all of Lightroom's adjustments (Clarity, Dehaze, HSL are separate).
- ACR XMP values are absolute, not relative. No mechanism for "add to default."
