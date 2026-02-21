# Streaming DNG Writer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the monolithic in-memory DNG output buffer with streaming file I/O, reducing peak memory by ~50%.

**Architecture:** Stream compressed tiles directly to a temp file using fwrite within the existing OpenMP parallel loop. After all tiles are written, seek back to patch the TIFF header and IFD structures. Then use a new file-based ExifTransfer path to copy EXIF/XMP/IPTC metadata from the source NEF and write the final output.

**Tech Stack:** C++11, stdio (fopen/fwrite/fseek/fclose), Exiv2 (file-based open), OpenMP, libdeflate, zlib

---

## Glossary

- **IFD**: Image File Directory — TIFF/DNG metadata structure containing tag entries
- **dataOffset**: byte offset where image data starts in the file (after all IFDs)
- **tile offset/byte count**: per-tile arrays telling TIFF readers where each compressed tile lives in the file
- **rawData**: `Array2D<float>` holding the merged HDR pixel data (input to writer)

## Constants

```
BINARY    = build/hdrmerge.app/Contents/MacOS/hdrmerge
SRC       = /Volumes/Oryggi/Eignamyndir/RAW/20.02.26
OUTDIR    = /Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng
REF       = /Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect
BUILD_CMD = cd /Users/stefanbaxter/Development/hdrmerge/build && cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)
```

---

### Task 1: Refactor ExifTransfer to extract shared metadata copy logic

The existing `ExifTransfer` class has `copyXMP()`, `copyIPTC()`, `copyEXIF()` as private methods. We need these for both the existing memory-based path and the new file-based path. Extract them into free functions that take `src` and `dst` Exiv2 image pointers.

**Files:**
- Modify: `src/ExifTransfer.cpp:32-189`
- Modify: `src/ExifTransfer.hpp:23-37`

**Step 1: Refactor ExifTransfer.cpp — extract copy functions**

Replace the `ExifTransfer` class with free functions. The class currently holds `src`, `dst`, `srcFile`, `dstFile`, `data`, `dataSize` as members. The copy methods only use `src` and `dst`. Extract them:

In `src/ExifTransfer.cpp`, replace the class and methods with:

```cpp
// Forward declarations for shared copy logic
#if EXIV2_TEST_VERSION(0,28,0)
using ExivImagePtr = Exiv2::Image::UniquePtr;
#else
using ExivImagePtr = Exiv2::Image::AutoPtr;
#endif

static bool excludeExifDatum(const Exiv2::Exifdatum & datum);
static void copyXMP(Exiv2::Image & src, Exiv2::Image & dst);
static void copyIPTC(Exiv2::Image & src, Exiv2::Image & dst);
static void copyEXIF(Exiv2::Image & src, Exiv2::Image & dst);

static void copyAllMetadata(ExivImagePtr & src, ExivImagePtr & dst) {
    copyXMP(*src, *dst);
    copyIPTC(*src, *dst);
    copyEXIF(*src, *dst);
}
```

Then change `copyXMP`, `copyIPTC`, `copyEXIF` from class methods to free functions taking `Exiv2::Image &` parameters. `excludeExifDatum` is already a static free function — keep it.

The existing `Exif::transfer()` function becomes:

```cpp
void hdrmerge::Exif::transfer(const QString & srcFile, const QString & dstFile,
                 const uint8_t * data, size_t dataSize) {
    ExivImagePtr dst, src;
    try {
#if EXIV2_TEST_VERSION(0,28,0)
        dst = Exiv2::ImageFactory::open(BasicIo::UniquePtr(new MemIo(data, dataSize)));
#else
        dst = Exiv2::ImageFactory::open(BasicIo::AutoPtr(new MemIo(data, dataSize)));
#endif
        dst->readMetadata();
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error: " << e.what() << std::endl;
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
    try {
        dst->writeMetadata();
        FileIo fileIo(dstFile.toLocal8Bit().constData());
        fileIo.open("wb");
        fileIo.write(dst->io());
        fileIo.close();
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error: " << e.what() << std::endl;
    }
}
```

Full implementations of the extracted functions:

```cpp
static void copyXMP(Exiv2::Image & src, Exiv2::Image & dst) {
    const Exiv2::XmpData & srcXmp = src.xmpData();
    Exiv2::XmpData & dstXmp = dst.xmpData();
    for (const auto & datum : srcXmp) {
        if (datum.groupName() != "tiff" && dstXmp.findKey(Exiv2::XmpKey(datum.key())) == dstXmp.end()) {
            dstXmp.add(datum);
        }
    }
}

static void copyIPTC(Exiv2::Image & src, Exiv2::Image & dst) {
    const Exiv2::IptcData & srcIptc = src.iptcData();
    Exiv2::IptcData & dstIptc = dst.iptcData();
    for (const auto & datum : srcIptc) {
        if (dstIptc.findKey(Exiv2::IptcKey(datum.key())) == dstIptc.end()) {
            dstIptc.add(datum);
        }
    }
}

static void copyEXIF(Exiv2::Image & src, Exiv2::Image & dst) {
    static const char * includeImageKeys[] = {
        "Exif.Image.Make",
        "Exif.Image.Model",
        "Exif.Image.Artist",
        "Exif.Image.Copyright",
        "Exif.Image.DNGPrivateData",
        "Exif.SubImage1.OpcodeList1",
        "Exif.SubImage1.OpcodeList2",
        "Exif.SubImage1.OpcodeList3"
    };

    const Exiv2::ExifData & srcExif = src.exifData();
    Exiv2::ExifData & dstExif = dst.exifData();

    for (const char * keyName : includeImageKeys) {
        auto iterator = srcExif.findKey(Exiv2::ExifKey(keyName));
        if (iterator != srcExif.end()) {
            dstExif[keyName] = *iterator;
        }
    }
    dstExif["Exif.SubImage1.NewSubfileType"] = 0u;

    for (const auto & datum : srcExif) {
        if (!excludeExifDatum(datum) && dstExif.findKey(Exiv2::ExifKey(datum.key())) == dstExif.end()) {
            dstExif.add(datum);
        }
    }
}
```

**Step 2: Build to verify the refactor compiles**

Run:
```bash
cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu)
```
Expected: Compiles with no errors. No behavioral change.

**Step 3: Smoke test — verify refactored ExifTransfer produces identical output**

Run:
```bash
mkdir -p /Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng
/usr/bin/time -l build/hdrmerge.app/Contents/MacOS/hdrmerge -v \
  -o "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4622.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4623.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4624.NEF"
```

Then verify metadata:
```bash
exiftool -G -a "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect/_EBX4622-4624.dng" > /tmp/s13.txt
exiftool -G -a "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng" > /tmp/s14.txt
diff /tmp/s13.txt /tmp/s14.txt
```
Expected: Identical metadata (possibly different timestamps for `[EXIF] ModifyDate`).

Verify pixel data:
```bash
cmp "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect/_EBX4622-4624.dng" \
    "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng"
```
Expected: Files differ only in timestamp bytes (the Exiv2 metadata transfer is unchanged, so structural bytes should match).

**Step 4: Commit the ExifTransfer refactor**

```bash
git add src/ExifTransfer.cpp src/ExifTransfer.hpp
git commit -m "Refactor ExifTransfer: extract shared metadata copy functions

Preparation for streaming DNG writer. No behavioral change."
```

---

### Task 2: Add file-based ExifTransfer path

Add `Exif::transferFile()` that opens a DNG from disk instead of from a memory buffer.

**Files:**
- Modify: `src/ExifTransfer.hpp:30-33` — add `transferFile` declaration
- Modify: `src/ExifTransfer.cpp` — add `transferFile` implementation

**Step 1: Add declaration to ExifTransfer.hpp**

Add to the `Exif` namespace:
```cpp
namespace Exif {
    void transfer(const QString & srcFile, const QString & dstFile,
             const uint8_t * data, size_t dataSize);
    void transferFile(const QString & srcFile, const QString & tmpFile,
             const QString & dstFile);
}
```

**Step 2: Add implementation to ExifTransfer.cpp**

Add after the existing `Exif::transfer()` function:

```cpp
void hdrmerge::Exif::transferFile(const QString & srcFile, const QString & tmpFile,
                                   const QString & dstFile) {
    ExivImagePtr dst, src;
    try {
        dst = Exiv2::ImageFactory::open(tmpFile.toLocal8Bit().constData());
        dst->readMetadata();
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error opening temp DNG: " << e.what() << std::endl;
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
    try {
        dst->writeMetadata();
        Exiv2::FileIo fileIo(dstFile.toLocal8Bit().constData());
        fileIo.open("wb");
        fileIo.write(dst->io());
        fileIo.close();
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error writing DNG: " << e.what() << std::endl;
    }
}
```

**Step 3: Build to verify it compiles**

Run:
```bash
cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu)
```
Expected: Compiles. `transferFile` is not called yet — just verifying it links.

**Step 4: Commit**

```bash
git add src/ExifTransfer.hpp src/ExifTransfer.cpp
git commit -m "Add file-based Exif::transferFile() for streaming DNG writer

Opens DNG from disk instead of memory buffer. Same metadata
copy logic as the existing transfer() function."
```

---

### Task 3: Rewrite DngFloatWriter to stream to file

This is the core change. Replace the monolithic `fileData` buffer with streaming file I/O.

**Files:**
- Modify: `src/DngFloatWriter.hpp:36-79` — remove fileData/pos/rawSize, add writePreviewsToFile/writeRawDataToFile
- Modify: `src/DngFloatWriter.cpp:131-617` — rewrite write(), writePreviews→writePreviewsToFile, writeRawData→writeRawDataToFile, remove rawSize

**Step 1: Update DngFloatWriter.hpp**

Remove these members:
- `std::unique_ptr<uint8_t[]> fileData;`
- `size_t pos;`

Remove these method declarations:
- `void writePreviews();`
- `void writeRawData();`
- `size_t rawSize();`

Add these method declarations:
- `void writePreviewsToFile(FILE * f, size_t dataOffset);`
- `void writeRawDataToFile(FILE * f);`

The full updated class should be:

```cpp
class DngFloatWriter {
public:
    DngFloatWriter() : previewWidth(0), bps(16), compressionLevel(6) {}

    void setPreviewWidth(size_t w) { previewWidth = w; }
    void setBitsPerSample(int b) { bps = b; }
    void setCompressionLevel(int level) { compressionLevel = level; }
    void setPreview(const QImage & p);
    void write(Array2D<float> && rawPixels, const RawParameters & p, const QString & dstFileName);

private:
    int previewWidth;
    int bps;
    int compressionLevel;
    const RawParameters * params;
    Array2D<float> rawData;
    IFD mainIFD, rawIFD, previewIFD;
    uint32_t width, height;
    uint32_t tileWidth, tileLength;
    uint32_t tilesAcross, tilesDown;
    QImage thumbnail;
    QImage preview;
    QByteArray jpegPreviewData;
    uint32_t subIFDoffsets[2];

    void createMainIFD();
    void createRawIFD();
    void calculateTiles();
    void writePreviewsToFile(FILE * f, size_t dataOffset);
    void writeRawDataToFile(FILE * f);
    void renderPreviews();
    void createPreviewIFD();
    size_t thumbSize();
    size_t previewSize();
};
```

**Step 2: Add `#include <cstdio>` and `#include <QFile>` to DngFloatWriter.cpp**

At the top of DngFloatWriter.cpp, add `#include <cstdio>` and `#include <QFile>` alongside the existing includes.

**Step 3: Rewrite `write()` method**

Replace `DngFloatWriter::write()` (lines 131-166) with:

```cpp
void DngFloatWriter::write(Array2D<float> && rawPixels, const RawParameters & p, const QString & dstFileName) {
    params = &p;
    rawData = std::move(rawPixels);
    width = rawData.getWidth();
    height = rawData.getHeight();

    renderPreviews();

    createMainIFD();
    subIFDoffsets[0] = 8 + mainIFD.length();
    createRawIFD();
    size_t dataOffset = subIFDoffsets[0] + rawIFD.length();
    if (previewWidth > 0) {
        createPreviewIFD();
        subIFDoffsets[1] = subIFDoffsets[0] + rawIFD.length();
        dataOffset += previewIFD.length();
    }
    mainIFD.setValue(SUBIFDS, (const void *)subIFDoffsets);

    // Open temp file for streaming
    QString tempPath = dstFileName + ".tmp";
    FILE * f = fopen(tempPath.toLocal8Bit().constData(), "w+b");
    if (!f) {
        std::cerr << "Failed to open temp file: " << tempPath.toLocal8Bit().constData() << std::endl;
        return;
    }

    {
        Timer t("Write output");

        // Write placeholder for headers + IFDs (will be patched later)
        std::vector<uint8_t> zeros(dataOffset, 0);
        fwrite(zeros.data(), 1, dataOffset, f);

        // Write previews directly to file
        writePreviewsToFile(f, dataOffset);

        // Write compressed tiles directly to file
        writeRawDataToFile(f);
    }

    // Release input pixel data (no longer needed)
    rawData = Array2D<float>();

    // Build final IFD buffer and patch file header
    std::vector<uint8_t> headerBuf(dataOffset);
    size_t headerPos = 0;
    TiffHeader().write(headerBuf.data(), headerPos);
    mainIFD.write(headerBuf.data(), headerPos, false);
    rawIFD.write(headerBuf.data(), headerPos, false);
    if (previewWidth > 0) {
        previewIFD.write(headerBuf.data(), headerPos, false);
    }

    fseek(f, 0, SEEK_SET);
    fwrite(headerBuf.data(), 1, dataOffset, f);
    fclose(f);

    // Transfer EXIF metadata from source NEF, write final DNG
    Exif::transferFile(p.fileName, tempPath, dstFileName);
    QFile::remove(tempPath);
}
```

**Step 4: Replace `writePreviews()` with `writePreviewsToFile()`**

Remove the old `writePreviews()` (lines 375-386). Add:

```cpp
void DngFloatWriter::writePreviewsToFile(FILE * f, size_t dataOffset) {
    size_t filePos = dataOffset;

    // Thumbnail
    size_t ts = thumbSize();
    mainIFD.setValue(STRIPBYTES, ts);
    mainIFD.setValue(STRIPOFFSETS, (uint32_t)filePos);
    fwrite(thumbnail.bits(), 1, ts, f);
    filePos += ts;

    // Preview JPEG
    if (previewWidth > 0) {
        size_t ps = previewSize();
        previewIFD.setValue(STRIPBYTES, ps);
        previewIFD.setValue(STRIPOFFSETS, (uint32_t)filePos);
        fwrite(jpegPreviewData.constData(), 1, ps, f);
    }
}
```

**Step 5: Replace `writeRawData()` with `writeRawDataToFile()`**

Remove the old `writeRawData()` (lines 536-617). Remove `rawSize()` (lines 527-533). Add:

```cpp
void DngFloatWriter::writeRawDataToFile(FILE * f) {
    size_t tileCount = tilesAcross * tilesDown;
    std::vector<uint32_t> tileOffsets(tileCount);
    std::vector<uint32_t> tileByteCounts(tileCount);
    int bytesps = bps >> 3;
    uLongf dstLen = tileWidth * tileLength * bytesps;

#ifdef HAVE_LIBDEFLATE
    int nThreads = 1;
    #ifdef _OPENMP
    nThreads = omp_get_max_threads();
    #endif
    std::vector<struct libdeflate_compressor*> compressors(nThreads);
    for (int i = 0; i < nThreads; i++)
        compressors[i] = libdeflate_alloc_compressor(compressionLevel);
    size_t cBufLen = libdeflate_zlib_compress_bound(compressors[0], dstLen);
#else
    size_t cBufLen = dstLen;
#endif

    #pragma omp parallel
    {
        Bytef * cBuffer = new Bytef[cBufLen];
        Bytef * uBuffer = new Bytef[dstLen];

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t y = 0; y < height; y += tileLength) {
            for (size_t x = 0; x < width; x += tileWidth) {
                size_t t = (y / tileLength) * tilesAcross + (x / tileWidth);
                size_t thisTileLength = y + tileLength > height ? height - y : tileLength;
                size_t thisTileWidth = x + tileWidth > width ? width - x : tileWidth;
                if (thisTileLength != tileLength || thisTileWidth != tileWidth) {
                    fill_n(uBuffer, dstLen, 0);
                }
                for (size_t row = 0; row < thisTileLength; ++row) {
                    Bytef * dst = uBuffer + row*tileWidth*bytesps;
                    Bytef * src = (Bytef *)&rawData(x, y+row);
                    compressFloats(src, thisTileWidth, bytesps);
                    encodeFPDeltaRow(src, dst, thisTileWidth, tileWidth, bytesps, 2);
                }
#ifdef HAVE_LIBDEFLATE
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif
                size_t compressedLength = libdeflate_zlib_compress(
                    compressors[tid], uBuffer, dstLen, cBuffer, cBufLen);
                tileByteCounts[t] = compressedLength;
                if (compressedLength == 0) {
                    std::cerr << "DNG Deflate: Failed compressing tile " << t << std::endl;
                }
#else
                uLongf compressedLength = dstLen;
                int err = compress(cBuffer, &compressedLength, uBuffer, dstLen);
                tileByteCounts[t] = compressedLength;
                if (err != Z_OK) {
                    std::cerr << "DNG Deflate: Failed compressing tile " << t << ", with error " << err << std::endl;
                }
#endif
                else {
                    #pragma omp critical
                    {
                        tileOffsets[t] = (uint32_t)ftell(f);
                        fwrite(cBuffer, 1, tileByteCounts[t], f);
                    }
                }
            }
        }

        delete [] cBuffer;
        delete [] uBuffer;
    }

#ifdef HAVE_LIBDEFLATE
    for (auto c : compressors) libdeflate_free_compressor(c);
#endif

    rawIFD.setValue(TILEOFFSETS, tileOffsets.data());
    rawIFD.setValue(TILEBYTES, tileByteCounts.data());
}
```

Key differences from the old `writeRawData()`:
- Uses `std::vector` instead of VLA for `tileOffsets`/`tileByteCounts` (safer, same perf)
- `ftell(f)` instead of `pos` for tile offset
- `fwrite()` instead of `std::copy_n` into `fileData`
- Fixed typo: `conpressedLength` → `compressedLength` in zlib fallback path

**Step 6: Build**

Run:
```bash
cd /Users/stefanbaxter/Development/hdrmerge/build && cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)
```
Expected: Compiles with no errors.

**Step 7: Run test Set A**

```bash
rm -f "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng"
/usr/bin/time -l build/hdrmerge.app/Contents/MacOS/hdrmerge -v \
  -o "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4622.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4623.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4624.NEF"
```
Expected: Produces a DNG file. Note wall time and peak RSS from `/usr/bin/time -l`.

**Step 8: Verify output exists and has reasonable size**

```bash
ls -la "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng"
ls -la "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect/_EBX4622-4624.dng"
```
Expected: Both files exist. Sizes should be close (within ~5% — compressed tile ordering may differ).

**Step 9: Commit the core streaming rewrite**

```bash
git add src/DngFloatWriter.hpp src/DngFloatWriter.cpp
git commit -m "Rewrite DngFloatWriter to stream tiles to file (Step 14)

Replace monolithic in-memory buffer with streaming file I/O.
Compressed tiles are fwritten directly in the OpenMP critical
section. Headers are patched via fseek after all tiles are
written. rawData is released before ExifTransfer runs.

Eliminates ~200 MB allocation per job."
```

---

### Task 4: Verify pixel-identical output

Compare step 14 output against step 13 reference at the pixel level.

**Files:** None (verification only)

**Step 1: Run all 3 test sets**

```bash
BINARY=build/hdrmerge.app/Contents/MacOS/hdrmerge
SRC=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26
OUT=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng

# Set A (3 brackets)
/usr/bin/time -l $BINARY -v \
  -o "$OUT/_EBX4622-4624.dng" \
  "$SRC/_EBX4622.NEF" "$SRC/_EBX4623.NEF" "$SRC/_EBX4624.NEF"

# Set B (5 brackets)
/usr/bin/time -l $BINARY -v \
  -o "$OUT/_EBX4640-4644.dng" \
  "$SRC/_EBX4640.NEF" "$SRC/_EBX4641.NEF" "$SRC/_EBX4642.NEF" \
  "$SRC/_EBX4643.NEF" "$SRC/_EBX4644.NEF"

# Set C (5 brackets)
/usr/bin/time -l $BINARY -v \
  -o "$OUT/_EBX4650-4654.dng" \
  "$SRC/_EBX4650.NEF" "$SRC/_EBX4651.NEF" "$SRC/_EBX4652.NEF" \
  "$SRC/_EBX4653.NEF" "$SRC/_EBX4654.NEF"
```

**Step 2: Compare file sizes**

```bash
REF=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect
OUT=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng
for f in _EBX4622-4624.dng _EBX4640-4644.dng _EBX4650-4654.dng; do
  echo "=== $f ==="
  ls -la "$REF/$f" | awk '{print "step13:", $5, "bytes"}'
  ls -la "$OUT/$f" | awk '{print "step14:", $5, "bytes"}'
done
```
Expected: Sizes close but not necessarily identical (compressed tile ordering may differ due to OpenMP scheduling; Exiv2 may serialize metadata slightly differently from file vs MemIo).

**Step 3: Compare EXIF metadata**

```bash
REF=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect
OUT=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng
for f in _EBX4622-4624.dng _EBX4640-4644.dng _EBX4650-4654.dng; do
  echo "=== $f ==="
  exiftool -G -a "$REF/$f" > /tmp/s13.txt
  exiftool -G -a "$OUT/$f" > /tmp/s14.txt
  diff /tmp/s13.txt /tmp/s14.txt || echo "DIFFERENCES FOUND"
done
```
Expected: Only differences should be `[EXIF] ModifyDate` (timestamp) and possibly `[EXIF] Software` if version string changed. All camera metadata, color matrices, white balance, orientation, etc. must be identical.

**Step 4: Compare pixel data using exiftool raw dump**

The DNG tile data may differ in byte order due to non-deterministic OpenMP scheduling, but the decoded pixel values must be identical. Use `dcraw` or `exiftool -b` to extract raw data:

```bash
REF=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step13-ghost-detect
OUT=/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng
for f in _EBX4622-4624.dng _EBX4640-4644.dng _EBX4650-4654.dng; do
  echo "=== $f pixel check ==="
  exiftool -b -PreviewImage "$REF/$f" > /tmp/ref_preview.jpg
  exiftool -b -PreviewImage "$OUT/$f" > /tmp/new_preview.jpg
  cmp /tmp/ref_preview.jpg /tmp/new_preview.jpg && echo "Preview: IDENTICAL" || echo "Preview: DIFFERS"
done
```

For true pixel-level comparison, the most reliable method is opening both in a raw processor and comparing. If `dcraw_emu` (from LibRaw) is available:

```bash
# If dcraw_emu is available:
# dcraw_emu -T -4 -o 0 "$REF/_EBX4622-4624.dng" -O /tmp/ref.tiff
# dcraw_emu -T -4 -o 0 "$OUT/_EBX4622-4624.dng" -O /tmp/new.tiff
# cmp /tmp/ref.tiff /tmp/new.tiff
```

**Step 5: Record results and note any differences**

If pixel data differs, investigate whether it's a tile ordering issue (acceptable) or actual data corruption (not acceptable). The TIFF tile offset array should correctly map each tile regardless of write order.

---

### Task 5: Measure memory improvement

**Step 1: Record peak RSS for all 3 test sets**

The `/usr/bin/time -l` output from Task 4 Step 1 includes `maximum resident set size`. Record these values.

Compare against step 13 baseline (re-run step 13 binary if needed):
```bash
# Build step 13 binary for comparison (checkout and build)
# Or use the recorded values from docs/implementation-plan.md metrics log
```

Expected: Peak RSS reduced by ~200 MB (the eliminated fileData buffer).

**Step 2: Update the metrics log**

Edit `docs/implementation-plan.md`, fill in the Step 14 row in the Metrics Log section:

```markdown
| 14 | Streaming DNG | <Set A time> | <Set A size> | <Peak RSS MB> | pixel-identical |
```

**Step 3: Commit metrics**

```bash
git add docs/implementation-plan.md
git commit -m "Record Step 14 metrics: streaming DNG writer

Peak RSS reduced from ~X MB to ~Y MB per job."
```

---

### Task 6: Clean up temp file on error and push

**Files:**
- Modify: `src/DngFloatWriter.cpp` — add error handling for fwrite/fseek failures

**Step 1: Add error cleanup to write()**

In the `write()` method, wrap the file operations with error checking. If any write fails, close and remove the temp file:

After `fopen`, add a scope guard pattern or manual cleanup. The simplest approach — check critical operations:

```cpp
// After fwrite of zeros:
if (fwrite(zeros.data(), 1, dataOffset, f) != dataOffset) {
    std::cerr << "Failed to write DNG header placeholder" << std::endl;
    fclose(f);
    QFile::remove(tempPath);
    return;
}
```

Apply similar checks to writePreviewsToFile and the final header patch fwrite. The tile writes in writeRawDataToFile already have error reporting via the existing `std::cerr` messages.

**Step 2: Build and verify**

```bash
cd /Users/stefanbaxter/Development/hdrmerge/build && make -j$(sysctl -n hw.ncpu)
```

**Step 3: Final smoke test — run Set A one more time**

```bash
rm -f "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng"
/usr/bin/time -l build/hdrmerge.app/Contents/MacOS/hdrmerge -v \
  -o "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/_EBX4622-4624.dng" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4622.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4623.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4624.NEF"
```

Verify no `.tmp` file remains:
```bash
ls /Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/step14-streaming-dng/*.tmp 2>/dev/null && echo "TEMP FILE LEFT BEHIND" || echo "Clean: no temp files"
```

**Step 4: Commit error handling and push**

```bash
git add src/DngFloatWriter.cpp
git commit -m "Add error handling for streaming DNG file I/O

Clean up temp file on write failure."
```

```bash
git push origin opt/step14-streaming-dng
```
