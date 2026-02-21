# ACR XMP Profile & Adaptive Curves Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Embed Adobe Camera Raw settings (from a Lightroom .xmp preset) and optional per-image adaptive RGB tone curves (from an ONNX model) into output DNG files.

**Architecture:** Extend `SaveOptions` with profile path and auto-curves flag. Read the .xmp profile via exiv2 in `ExifTransfer`. When `--auto-curves` is enabled, run ONNX inference on the sRGB preview in `ImageIO::save()` and pass the resulting curves through to `ExifTransfer`. All `crs:` tags are injected into the DNG's XmpData after source metadata is copied.

**Tech Stack:** C++11, exiv2 (existing), ONNX Runtime (new optional dependency), Qt5 (existing)

---

### Task 1: Add CLI flags and SaveOptions fields

**Files:**
- Modify: `src/LoadSaveOptions.hpp:46-57`
- Modify: `src/Launcher.cpp:272-401` (parseCommandLine)
- Modify: `src/Launcher.cpp:410-454` (showHelp)
- Modify: `src/Launcher.cpp:458-493` (checkGUI)

**Step 1: Add fields to SaveOptions**

In `src/LoadSaveOptions.hpp`, add to `SaveOptions`:

```cpp
struct SaveOptions {
    int bps;
    int previewSize;
    QString fileName;
    bool saveMask;
    QString maskFileName;
    int featherRadius;
    int compressionLevel;
    float deghostSigma;
    double clipPercentile;
    QString acrProfilePath;
    bool autoCurves;
    SaveOptions() : bps(32), previewSize(0), saveMask(false), featherRadius(3),
        compressionLevel(6), deghostSigma(0.0f), clipPercentile(99.9),
        autoCurves(false) {}
};
```

**Step 2: Parse -L and --auto-curves in Launcher::parseCommandLine()**

Add to the command-line parsing loop in `src/Launcher.cpp`:

```cpp
} else if (string("-L") == argv[i]) {
    if (++i < argc) {
        saveOptions.acrProfilePath = QString::fromLocal8Bit(argv[i]);
    }
} else if (string("--auto-curves") == argv[i]) {
    saveOptions.autoCurves = true;
```

**Step 3: Add help text in Launcher::showHelp()**

```cpp
cout << "    " << "-L PROFILE    " << tr("Apply ACR settings from a Lightroom .xmp preset to every output DNG.") << endl;
cout << "    " << "--auto-curves " << tr("Generate per-image adaptive RGB tone curves via ONNX model (requires onnxruntime).") << endl;
```

**Step 4: Update checkGUI() to treat -L as a non-GUI flag**

Add `-L` with value skip alongside existing flags:

```cpp
} else if (string("-L") == argv[i]) {
    ++i; // skip the value
} else if (string("--auto-curves") == argv[i]) {
    // flag only
```

**Step 5: Build and verify**

Run: `cd build && cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)`
Expected: Clean compile. Run `hdrmerge --help` and confirm new flags appear.

**Step 6: Commit**

```bash
git add src/LoadSaveOptions.hpp src/Launcher.cpp
git commit -m "Add -L and --auto-curves CLI flags for ACR XMP profile"
```

---

### Task 2: Read .xmp profile via exiv2

**Files:**
- Modify: `src/ExifTransfer.hpp:28-37`
- Modify: `src/ExifTransfer.cpp`

**Step 1: Add ACR profile types and function declarations**

In `src/ExifTransfer.hpp`:

```cpp
#ifndef _EXIFTRANSFER_HPP_
#define _EXIFTRANSFER_HPP_

#include <map>
#include <vector>
#include <array>
#include <QString>

namespace hdrmerge {

    // Per-channel adaptive curve: 3 channels, each with up to 20 control points (x, y) in [0,255]
    struct AdaptiveCurves {
        bool valid = false;
        // Each channel: vector of (input, output) pairs, integers 0-255
        std::vector<std::pair<int,int>> red, green, blue;
    };

    namespace Exif {
        // Existing
        void transfer(const QString & srcFile, const QString & dstFile,
                 const uint8_t * data, size_t dataSize);
        void transferFile(const QString & srcFile, const QString & tmpFile,
                 const QString & dstFile);

        // New: with ACR profile and adaptive curves injection
        void transferFile(const QString & srcFile, const QString & tmpFile,
                 const QString & dstFile,
                 const QString & acrProfilePath,
                 const AdaptiveCurves & curves);
    }

}

#endif
```

**Step 2: Implement profile reading and injection**

In `src/ExifTransfer.cpp`, add the new overload. The core logic:

```cpp
static void injectACRProfile(Exiv2::Image & dst, const QString & profilePath) {
    if (profilePath.isEmpty()) return;
    try {
        ExivImagePtr profile = Exiv2::ImageFactory::open(profilePath.toLocal8Bit().constData());
        profile->readMetadata();
        const Exiv2::XmpData & profileXmp = profile->xmpData();
        Exiv2::XmpData & dstXmp = dst.xmpData();
        for (const auto & datum : profileXmp) {
            // Only copy crs: (Camera Raw Settings) tags
            if (datum.groupName() == "crs") {
                auto key = Exiv2::XmpKey(datum.key());
                auto it = dstXmp.findKey(key);
                if (it != dstXmp.end()) {
                    *it = datum;  // Override existing
                } else {
                    dstXmp.add(datum);
                }
            }
        }
    } catch (Exiv2::Error & e) {
        std::cerr << "Error reading ACR profile " << profilePath.toLocal8Bit().constData()
                  << ": " << e.what() << std::endl;
    }
}

static void injectAdaptiveCurves(Exiv2::Image & dst, const AdaptiveCurves & curves) {
    if (!curves.valid) return;
    Exiv2::XmpData & xmp = dst.xmpData();

    // Helper: write a curve as crs:ToneCurvePV2012{channel}
    auto writeCurve = [&](const char * key, const std::vector<std::pair<int,int>> & pts) {
        Exiv2::XmpKey xmpKey(key);
        // Remove existing if present
        auto it = xmp.findKey(xmpKey);
        if (it != xmp.end()) xmp.erase(it);
        // Add as XmpSeq
        Exiv2::Value::AutoPtr val = Exiv2::Value::create(Exiv2::xmpSeq);
        for (auto & pt : pts) {
            val->read(std::to_string(pt.first) + ", " + std::to_string(pt.second));
        }
        xmp.add(xmpKey, val.get());
    };

    writeCurve("Xmp.crs.ToneCurvePV2012Red", curves.red);
    writeCurve("Xmp.crs.ToneCurvePV2012Green", curves.green);
    writeCurve("Xmp.crs.ToneCurvePV2012Blue", curves.blue);

    // Set master curve to linear (let per-channel curves do the work)
    {
        Exiv2::XmpKey key("Xmp.crs.ToneCurvePV2012");
        auto it = xmp.findKey(key);
        if (it != xmp.end()) xmp.erase(it);
        Exiv2::Value::AutoPtr val = Exiv2::Value::create(Exiv2::xmpSeq);
        val->read("0, 0");
        val->read("255, 255");
        xmp.add(key, val.get());
    }

    // Mark curves as custom
    xmp["Xmp.crs.ToneCurveName2012"] = "Custom";
}

void hdrmerge::Exif::transferFile(const QString & srcFile, const QString & tmpFile,
                                   const QString & dstFile,
                                   const QString & acrProfilePath,
                                   const AdaptiveCurves & curves) {
    ExivImagePtr dst, src;
    try {
        dst = Exiv2::ImageFactory::open(tmpFile.toLocal8Bit().constData());
        dst->readMetadata();
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error opening temp DNG: " << e.what() << std::endl;
        QFile::remove(tmpFile);
        return;
    }
    try {
        src = Exiv2::ImageFactory::open(srcFile.toLocal8Bit().constData());
        src->readMetadata();
        copyAllMetadata(src, dst);
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error: " << e.what() << std::endl;
        dst->exifData()["Exif.SubImage1.NewSubfileType"] = 0;
    }

    // Inject ACR profile (overrides source crs: tags)
    injectACRProfile(*dst, acrProfilePath);

    // Inject adaptive curves (overrides profile's tone curves if present)
    injectAdaptiveCurves(*dst, curves);

    try {
        dst->writeMetadata();
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error writing metadata: " << e.what() << std::endl;
        QFile::remove(tmpFile);
        return;
    }
    dst.reset();
    QFile::remove(dstFile);
    if (!QFile::rename(tmpFile, dstFile)) {
        std::cerr << "Failed to rename " << tmpFile.toLocal8Bit().constData()
                   << " to " << dstFile.toLocal8Bit().constData() << std::endl;
        QFile::remove(tmpFile);
    }
}
```

**Step 3: Build and verify**

Run: `make -j$(sysctl -n hw.ncpu)`
Expected: Clean compile.

**Step 4: Commit**

```bash
git add src/ExifTransfer.hpp src/ExifTransfer.cpp
git commit -m "Add ACR profile reading and adaptive curve injection via exiv2"
```

---

### Task 3: Thread profile and curves through the pipeline

**Files:**
- Modify: `src/DngFloatWriter.hpp:38-86`
- Modify: `src/DngFloatWriter.cpp:140-202` (write method)
- Modify: `src/ImageIO.cpp:187-212` (save method)

**Step 1: Add ACR options to DngFloatWriter**

In `src/DngFloatWriter.hpp`, add setters and fields:

```cpp
void setACRProfilePath(const QString & path) { acrProfilePath = path; }
void setAdaptiveCurves(const AdaptiveCurves & c) { adaptiveCurves = c; }
```

And private fields:

```cpp
QString acrProfilePath;
AdaptiveCurves adaptiveCurves;
```

Add `#include "ExifTransfer.hpp"` to the includes.

**Step 2: Pass options through DngFloatWriter::write() to Exif::transferFile()**

In `src/DngFloatWriter.cpp`, change the `Exif::transferFile()` call at line 201:

```cpp
// Transfer EXIF metadata from source NEF, write final DNG
Exif::transferFile(p.fileName, tempPath, dstFileName, acrProfilePath, adaptiveCurves);
```

**Step 3: Wire up in ImageIO::save()**

In `src/ImageIO.cpp`, after setting up the DngFloatWriter (around line 210):

```cpp
writer.setACRProfilePath(options.acrProfilePath);
// adaptiveCurves will be set later when ONNX support is added
writer.write(std::move(composed.image), params, options.fileName);
```

**Step 4: Build and test with a real .xmp profile**

Create a test profile at `/tmp/test-profile.xmp`:

```xml
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
   xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:ProcessVersion="15.4"
   crs:Version="16.1"
   crs:Exposure2012="+0.50"
   crs:Highlights2012="-60"
   crs:Shadows2012="+40"
   crs:Contrast2012="+20"
   crs:SaturationAdjustmentBlue="-15">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
```

Run: `hdrmerge -v -B -L /tmp/test-profile.xmp <3 test NEFs> -o /tmp/profile-test.dng`
Verify: `exiv2 -px /tmp/profile-test.dng | grep crs`
Expected: See Exposure2012, Highlights2012, Shadows2012, Contrast2012, SaturationAdjustmentBlue in output.

**Step 5: Commit**

```bash
git add src/DngFloatWriter.hpp src/DngFloatWriter.cpp src/ImageIO.cpp
git commit -m "Thread ACR profile through DngFloatWriter to ExifTransfer"
```

---

### Task 4: ONNX Runtime integration (optional dependency)

**Files:**
- Modify: `CMakeLists.txt:57-67`
- Create: `src/AdaptiveCurves.hpp`
- Create: `src/AdaptiveCurves.cpp`
- Modify: `src/ImageIO.cpp:187-212`

**Step 1: Add ONNX Runtime to CMakeLists.txt**

After the OpenCV block (line 67):

```cmake
# Optional: ONNX Runtime for adaptive tone curves (--auto-curves)
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS /opt/homebrew/lib)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /opt/homebrew/include/onnxruntime
          /opt/homebrew/include)
if(ONNXRUNTIME_LIBRARY AND ONNXRUNTIME_INCLUDE_DIR)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    add_definitions(-DHAVE_ONNXRUNTIME)
    include_directories("${ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(STATUS "ONNX Runtime not found: --auto-curves disabled")
endif()
```

Add to hdrmerge_libs and hdrmerge_sources:

```cmake
# In hdrmerge_libs, add conditionally:
if(ONNXRUNTIME_LIBRARY)
    set(hdrmerge_libs ${hdrmerge_libs} "${ONNXRUNTIME_LIBRARY}")
endif()

# In hdrmerge_sources, add:
src/AdaptiveCurves.cpp
```

**Step 2: Create AdaptiveCurves.hpp**

```cpp
#ifndef _ADAPTIVECURVES_HPP_
#define _ADAPTIVECURVES_HPP_

#include <QImage>
#include "ExifTransfer.hpp"  // for AdaptiveCurves struct

namespace hdrmerge {

// Run ONNX model on sRGB preview to predict per-channel RGB tone curves.
// Returns curves with valid=false if ONNX Runtime is unavailable or inference fails.
AdaptiveCurves predictAdaptiveCurves(const QImage & preview, const QString & modelPath);

// Default model path (next to the binary, or user-specified)
QString findOnnxModel();

}

#endif
```

**Step 3: Create AdaptiveCurves.cpp**

```cpp
#include "AdaptiveCurves.hpp"
#include "Log.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <QCoreApplication>
#include <QFileInfo>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

using namespace hdrmerge;

// Enforce monotonicity: sweep right-to-left, each value = min(self, right neighbor)
static void monotonize(std::vector<float> & curve) {
    for (int i = (int)curve.size() - 2; i >= 0; --i) {
        curve[i] = std::min(curve[i], curve[i + 1]);
    }
}

// Greedy spline fitting: pick up to maxPoints control points with max error < threshold
static std::vector<std::pair<int,int>> fitCurve(const std::vector<float> & curve, int maxPoints = 20, float maxError = 0.005f) {
    std::vector<std::pair<int,int>> points;
    points.push_back({0, (int)std::round(curve[0] * 255.0f)});
    points.push_back({255, (int)std::round(curve[255] * 255.0f)});

    for (int iter = 0; iter < maxPoints - 2; ++iter) {
        // Find point with maximum error
        float bestErr = 0;
        int bestIdx = -1;
        for (int i = 1; i < 255; ++i) {
            // Check if i is already a control point
            bool exists = false;
            for (auto & p : points) {
                if (p.first == i) { exists = true; break; }
            }
            if (exists) continue;

            // Linear interpolation between surrounding control points
            int leftIdx = 0, rightIdx = 255;
            float leftVal = points[0].second / 255.0f;
            float rightVal = points.back().second / 255.0f;
            for (auto & p : points) {
                if (p.first < i && p.first > leftIdx) { leftIdx = p.first; leftVal = p.second / 255.0f; }
                if (p.first > i && p.first < rightIdx) { rightIdx = p.first; rightVal = p.second / 255.0f; }
            }
            float t = (float)(i - leftIdx) / (float)(rightIdx - leftIdx);
            float interp = leftVal + t * (rightVal - leftVal);
            float err = std::abs(curve[i] - interp);
            if (err > bestErr) { bestErr = err; bestIdx = i; }
        }

        if (bestErr < maxError || bestIdx < 0) break;
        points.push_back({bestIdx, (int)std::round(curve[bestIdx] * 255.0f)});
        std::sort(points.begin(), points.end());
    }
    return points;
}

QString hdrmerge::findOnnxModel() {
    // Look next to the executable
    QString appDir = QCoreApplication::applicationDirPath();
    QString modelPath = appDir + "/free_xcittiny_wa14.onnx";
    if (QFileInfo::exists(modelPath)) return modelPath;
    // Look in ../share/hdrmerge/
    modelPath = appDir + "/../share/hdrmerge/free_xcittiny_wa14.onnx";
    if (QFileInfo::exists(modelPath)) return modelPath;
    return QString();
}

AdaptiveCurves hdrmerge::predictAdaptiveCurves(const QImage & preview, const QString & modelPath) {
    AdaptiveCurves result;

#ifndef HAVE_ONNXRUNTIME
    Log::progress("--auto-curves requires ONNX Runtime (not available in this build)");
    return result;
#else
    if (preview.isNull()) {
        Log::progress("Cannot generate adaptive curves: preview is null");
        return result;
    }
    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        Log::progress("Cannot generate adaptive curves: model file not found");
        Log::progress("  Expected: free_xcittiny_wa14.onnx next to hdrmerge binary");
        return result;
    }

    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hdrmerge-autocurves");
        Ort::SessionOptions opts;
        Ort::Session session(env, modelPath.toLocal8Bit().constData(), opts);

        // Prepare input: resize preview to 384x384, convert to CHW float32 [0,1]
        QImage scaled = preview.scaled(384, 384, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                               .convertToFormat(QImage::Format_RGB32);

        std::vector<float> inputData(1 * 3 * 384 * 384);
        for (int y = 0; y < 384; ++y) {
            const QRgb * line = (const QRgb *)scaled.constScanLine(y);
            for (int x = 0; x < 384; ++x) {
                QRgb px = line[x];
                inputData[0 * 384 * 384 + y * 384 + x] = qRed(px) / 255.0f;   // R
                inputData[1 * 384 * 384 + y * 384 + x] = qGreen(px) / 255.0f; // G
                inputData[2 * 384 * 384 + y * 384 + x] = qBlue(px) / 255.0f;  // B
            }
        }

        // Run inference
        std::array<int64_t, 4> inputShape = {1, 3, 384, 384};
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputData.data(), inputData.size(),
            inputShape.data(), inputShape.size());

        Ort::AllocatorWithDefaultOptions allocator;
        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto outputName = session.GetOutputNameAllocated(0, allocator);
        const char * inputNames[] = { inputName.get() };
        const char * outputNames[] = { outputName.get() };

        auto outputTensors = session.Run(Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1, outputNames, 1);

        // Extract curves: expect shape [1, 3, 256] or [1, 768]
        float * raw = outputTensors[0].GetTensorMutableData<float>();
        auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::vector<float> rCurve(256), gCurve(256), bCurve(256);
        for (int i = 0; i < 256; ++i) {
            rCurve[i] = std::max(0.0f, std::min(1.0f, raw[0 * 256 + i]));
            gCurve[i] = std::max(0.0f, std::min(1.0f, raw[1 * 256 + i]));
            bCurve[i] = std::max(0.0f, std::min(1.0f, raw[2 * 256 + i]));
        }

        // Post-process: enforce monotonicity
        monotonize(rCurve);
        monotonize(gCurve);
        monotonize(bCurve);

        // Fit to control points
        result.red = fitCurve(rCurve);
        result.green = fitCurve(gCurve);
        result.blue = fitCurve(bCurve);
        result.valid = true;

        Log::debug("Adaptive curves: R=", result.red.size(), " G=", result.green.size(),
                   " B=", result.blue.size(), " control points");

    } catch (const Ort::Exception & e) {
        Log::progress("ONNX Runtime error: ", e.what());
    } catch (const std::exception & e) {
        Log::progress("Adaptive curves error: ", e.what());
    }

    return result;
#endif
}
```

**Step 4: Wire ONNX inference into ImageIO::save()**

In `src/ImageIO.cpp`, after `renderPreview()` and before `DngFloatWriter::write()`:

```cpp
progress.advance(33, "Rendering preview");
QImage preview = renderPreview(composed.image, params, stack.getMaxExposure(), options.previewSize <= 1);

// Generate adaptive curves from preview if requested
AdaptiveCurves adaptiveCurves;
if (options.autoCurves) {
    progress.advance(40, "Generating adaptive curves");
    QString modelPath = findOnnxModel();
    adaptiveCurves = predictAdaptiveCurves(preview, modelPath);
}

progress.advance(66, "Writing output");
DngFloatWriter writer;
writer.setBitsPerSample(options.bps);
writer.setCompressionLevel(options.compressionLevel);
writer.setPreviewWidth((options.previewSize * stack.getWidth()) / 2);
writer.setPreview(preview);
writer.setBaselineExposure(composed.baselineExposureEV);
writer.setBaselineNoise(composed.numImages);
writer.setACRProfilePath(options.acrProfilePath);
writer.setAdaptiveCurves(adaptiveCurves);
writer.write(std::move(composed.image), params, options.fileName);
```

Add `#include "AdaptiveCurves.hpp"` to `src/ImageIO.cpp`.

**Step 5: Install ONNX Runtime and download model**

```bash
brew install onnxruntime
curl -L -o build/hdrmerge.app/Contents/MacOS/free_xcittiny_wa14.onnx \
    https://retroshine.eu/download/free_xcittiny_wa14.onnx
```

**Step 6: Build and test**

```bash
cd build && cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)
```

Expected: `Found ONNX Runtime:` in cmake output. Clean compile.

**Step 7: Test adaptive curves**

```bash
hdrmerge -vv --auto-curves <3 test NEFs> -o /tmp/curves-test.dng
exiv2 -px /tmp/curves-test.dng | grep ToneCurve
```

Expected: `ToneCurvePV2012Red`, `Green`, `Blue` with control point values.

**Step 8: Test profile + curves together**

```bash
hdrmerge -vv -L /tmp/test-profile.xmp --auto-curves <3 test NEFs> -o /tmp/both-test.dng
exiv2 -px /tmp/both-test.dng | grep crs
```

Expected: Both slider values (Exposure, Highlights, etc.) and tone curves present.

**Step 9: Commit**

```bash
git add CMakeLists.txt src/AdaptiveCurves.hpp src/AdaptiveCurves.cpp src/ImageIO.cpp
git commit -m "Add ONNX-based adaptive tone curves (--auto-curves)"
```

---

### Task 5: Verify in Lightroom

**Step 1: Open a DNG with profile only in Lightroom**

Import `/tmp/profile-test.dng` into Lightroom. Verify:
- Exposure slider shows +0.50
- Highlights shows -60
- Shadows shows +40
- Contrast shows +20
- Blue saturation shows -15

**Step 2: Open a DNG with profile + curves in Lightroom**

Import `/tmp/both-test.dng` into Lightroom. Verify:
- Same slider values as above
- Tone Curve panel shows "Custom" with per-channel R/G/B curves
- The image looks different from default (curves are being applied)

**Step 3: Open a DNG without any profile**

Confirm existing DNGs (no -L flag) are unaffected — Lightroom applies its normal defaults.
