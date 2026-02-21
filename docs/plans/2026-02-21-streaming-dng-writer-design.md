# Streaming DNG Writer — Design Document

**Date**: 2026-02-21
**Step**: 14 of the optimization plan
**Branch**: `opt/step14-streaming-dng`

## Problem

`DngFloatWriter::write()` allocates the complete output file in a single `std::unique_ptr<uint8_t[]>` buffer. For 61MP at 32-bit float, this buffer is ~200 MB (worst-case uncompressed tile size + overhead). With 4 concurrent batch jobs, peak memory can hit 800 MB+ just for output buffers, on top of ~182 MB per job for input `rawData`.

Additionally, `ExifTransfer` requires the entire DNG in memory as `const uint8_t*`, doubling the memory pressure during the Exiv2 phase.

## Goal

Reduce peak memory per job from ~400 MB (rawData + fileData) to ~200 MB (rawData + small tile buffers) by streaming compressed tiles directly to disk and eliminating the monolithic output buffer.

## Architecture

### Current Flow

```
write():
  1. Create IFD structures (tiny, in memory)
  2. Calculate total size = IFDs + thumbnail + preview + rawSize()
  3. Allocate fileData buffer (full output size, ~200 MB)
  4. Write previews into buffer at data offset
  5. Compress tiles in parallel, write into buffer (omp critical)
  6. Rewind pos to 0, write headers + IFDs (they reference data offsets)
  7. Pass entire buffer to Exif::transfer(srcFile, dstFile, data, dataSize)
     → Exiv2 wraps in MemIo, copies EXIF from source NEF, writes to disk
```

### New Flow

```
write():
  1. Create IFD structures (same as today, tiny)
  2. Calculate dataOffset = 8 + mainIFD.length() + rawIFD.length() [+ previewIFD.length()]
  3. Open temp file (dstFileName + ".tmp")
  4. Write dataOffset bytes of zeros (placeholder for headers + IFDs)
  5. Write thumbnail data to file, record offset in mainIFD
  6. Write preview JPEG to file, record offset in previewIFD
  7. Compress tiles in parallel, fwrite in omp critical, record offsets/sizes
  8. Update rawIFD with tile offsets/byte counts
  9. Release rawData (frees ~182 MB)
  10. Build small IFD buffer (~3 KB), fseek to 0, fwrite headers + IFDs
  11. Close temp file
  12. Call Exif::transferFile(srcFile, tempFile, dstFile)
      → Exiv2 opens temp file from disk, copies EXIF, writes to dstFile
  13. Remove temp file
```

### Memory Comparison

| Phase | Current | Streaming |
|-------|---------|-----------|
| Tile compression | rawData (182 MB) + fileData (200 MB) = **382 MB** | rawData (182 MB) + thread buffers (N × 1 MB) = **~190 MB** |
| Exif transfer | fileData (200 MB) in Exiv2 MemIo | Exiv2 opens file from disk (~minimal from our side) |
| **Peak** | **~382 MB** | **~190 MB** |

## Detailed Changes

### 1. DngFloatWriter.hpp

Remove:
- `std::unique_ptr<uint8_t[]> fileData` member
- `size_t pos` member
- `size_t rawSize()` method (no longer needed for pre-allocation)

Add:
- `FILE* outputFile` member (used during write)

### 2. DngFloatWriter.cpp — `write()`

Replace the monolithic buffer allocation with file I/O:

```cpp
void DngFloatWriter::write(Array2D<float> && rawPixels, const RawParameters & p,
                           const QString & dstFileName) {
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

    // Open temp file
    QString tempPath = dstFileName + ".tmp";
    FILE * f = fopen(tempPath.toLocal8Bit().constData(), "wb");
    if (!f) { /* error handling */ }

    // Write placeholder for headers + IFDs
    std::vector<uint8_t> zeros(dataOffset, 0);
    fwrite(zeros.data(), 1, dataOffset, f);

    // Write previews directly to file
    writePreviewsToFile(f, dataOffset);

    // Write compressed tiles directly to file
    writeRawDataToFile(f);

    // Release input data (no longer needed)
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

    // Transfer EXIF from source NEF, write final output
    Exif::transferFile(p.fileName, tempPath, dstFileName);
    QFile::remove(tempPath);
}
```

### 3. DngFloatWriter.cpp — `writePreviewsToFile()`

Replaces `writePreviews()`. Writes thumbnail and preview JPEG directly to file:

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

### 4. DngFloatWriter.cpp — `writeRawDataToFile()`

Replaces `writeRawData()`. Same parallel compression, but fwrite instead of memcpy:

```cpp
void DngFloatWriter::writeRawDataToFile(FILE * f) {
    size_t tileCount = tilesAcross * tilesDown;
    uint32_t tileOffsets[tileCount];
    uint32_t tileBytes[tileCount];
    int bytesps = bps >> 3;
    uLongf dstLen = tileWidth * tileLength * bytesps;

    // ... libdeflate compressor setup (same as current) ...

    #pragma omp parallel
    {
        Bytef * cBuffer = new Bytef[cBufLen];
        Bytef * uBuffer = new Bytef[dstLen];

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t y = 0; y < height; y += tileLength) {
            for (size_t x = 0; x < width; x += tileWidth) {
                // ... compress tile (same as current) ...

                #pragma omp critical
                {
                    tileOffsets[t] = (uint32_t)ftell(f);
                    fwrite(cBuffer, 1, tileBytes[t], f);
                }
            }
        }
        delete [] cBuffer;
        delete [] uBuffer;
    }

    // ... free compressors ...
    rawIFD.setValue(TILEOFFSETS, tileOffsets);
    rawIFD.setValue(TILEBYTES, tileBytes);
}
```

### 5. ExifTransfer — New file-based overload

Add `Exif::transferFile()` that opens the DNG from disk instead of memory:

```cpp
// ExifTransfer.hpp
namespace Exif {
    void transfer(const QString & srcFile, const QString & dstFile,
                  const uint8_t * data, size_t dataSize);  // existing (keep for compat)
    void transferFile(const QString & srcFile, const QString & tmpFile,
                      const QString & dstFile);  // new
}

// ExifTransfer.cpp
void hdrmerge::Exif::transferFile(const QString & srcFile, const QString & tmpFile,
                                   const QString & dstFile) {
    ExifTransferFile exif(srcFile, tmpFile, dstFile);
    exif.copyMetadata();
}
```

The new `ExifTransferFile` class opens the temp DNG from disk:

```cpp
class ExifTransferFile {
public:
    ExifTransferFile(const QString & srcFile, const QString & tmpFile,
                     const QString & dstFile)
    : srcFile(srcFile), tmpFile(tmpFile), dstFile(dstFile) {}

    void copyMetadata() {
        try {
            dst = Exiv2::ImageFactory::open(tmpFile.toLocal8Bit().constData());
            dst->readMetadata();
        } catch (Exiv2::Error & e) { /* ... */ return; }
        try {
            src = Exiv2::ImageFactory::open(srcFile.toLocal8Bit().constData());
            src->readMetadata();
            copyXMP();
            copyIPTC();
            copyEXIF();
        } catch (Exiv2::Error & e) {
            dst->exifData()["Exif.SubImage1.NewSubfileType"] = 0;
        }
        try {
            dst->writeMetadata();
            Exiv2::FileIo fileIo(dstFile.toLocal8Bit().constData());
            fileIo.open("wb");
            fileIo.write(dst->io());
            fileIo.close();
        } catch (Exiv2::Error & e) { /* ... */ }
    }

private:
    // copyXMP(), copyIPTC(), copyEXIF() — identical to existing ExifTransfer
    // (factor out into shared helper or base class)
};
```

The metadata copy methods (copyXMP, copyIPTC, copyEXIF) are identical to the existing ones. To avoid duplication, factor the common logic into shared static/free functions.

### 6. Metadata Verification

All EXIF/XMP/IPTC transfer happens in the same ExifTransfer code paths. The only difference is how the DNG is opened (file vs MemIo). The metadata flow is:

1. Exiv2 opens temp DNG file → parses TIFF IFD structure
2. Exiv2 opens source NEF → reads all metadata
3. XMP: copies all non-tiff XMP tags from source
4. IPTC: copies all IPTC tags from source
5. EXIF: copies Make, Model, Artist, Copyright, DNGPrivateData, OpcodeList1/2/3, then all non-Image/SubImage/Thumbnail tags
6. Sets `Exif.SubImage1.NewSubfileType = 0` (Primary Image)
7. Writes modified DNG to final output path

This is byte-for-byte the same metadata transfer logic. The DNG structure (IFDs, tags, tile data) is identical — only the transport changes.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Exiv2 can't parse file-based DNG | Low (Exiv2 is designed for this) | Test with exiftool comparison against step 13 output |
| Tile order in file differs | None (TIFF readers use offset array) | Verify with multiple readers |
| fwrite/fseek errors | Low | Check return values, clean up temp file on error |
| Metadata loss | Low (same copy logic) | Diff exiftool output between step 13 and step 14 |

## Verification Plan

1. Build and run all 3 test sets (A, B, C)
2. **Pixel-identical**: decode both step 13 and step 14 DNG to float arrays, compare byte-for-byte
3. **Metadata-identical**: `exiftool -G -a step13.dng > /tmp/s13.txt && exiftool -G -a step14.dng > /tmp/s14.txt && diff /tmp/s13.txt /tmp/s14.txt`
4. **Peak RSS**: compare `/usr/bin/time -l` max resident set size
5. **File sizes**: should be very close (compressed tile ordering may cause minor differences)
6. **DNG validity**: open in Lightroom, darktable, Adobe DNG Converter
7. **Batch mode**: run with `-j 4` to verify total memory stays reasonable
