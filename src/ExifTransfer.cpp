/*
 *  HDRMerge - HDR exposure merging software.
 *  Copyright 2012 Javier Celaya
 *  jcelaya@gmail.com
 *
 *  This file is part of HDRMerge.
 *
 *  HDRMerge is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  HDRMerge is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HDRMerge. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <exiv2/exiv2.hpp>
#include <iostream>
#include <QFile>
#include "ExifTransfer.hpp"
using namespace hdrmerge;
using namespace Exiv2;
using namespace std;

#if EXIV2_TEST_VERSION(0,28,0)
using ExivImagePtr = Exiv2::Image::UniquePtr;
using ExivValuePtr = Exiv2::Value::UniquePtr;
#else
using ExivImagePtr = Exiv2::Image::AutoPtr;
using ExivValuePtr = Exiv2::Value::AutoPtr;
#endif

static bool excludeExifDatum(const Exiv2::Exifdatum & datum);
static void copyXMP(Exiv2::Image & src, Exiv2::Image & dst);
static void copyIPTC(Exiv2::Image & src, Exiv2::Image & dst);
static void copyEXIF(Exiv2::Image & src, Exiv2::Image & dst);
static void synthesizeLensXMP(Exiv2::Image & src, Exiv2::Image & dst);

static void copyAllMetadata(ExivImagePtr & src, ExivImagePtr & dst) {
    copyXMP(*src, *dst);
    copyIPTC(*src, *dst);
    copyEXIF(*src, *dst);
    synthesizeLensXMP(*src, *dst);
}


static void injectACRProfile(Exiv2::Image & dst, const QString & profilePath) {
    if (profilePath.isEmpty()) return;
    try {
        ExivImagePtr xmpFile = Exiv2::ImageFactory::open(profilePath.toLocal8Bit().constData());
        xmpFile->readMetadata();
        const Exiv2::XmpData & srcXmp = xmpFile->xmpData();
        Exiv2::XmpData & dstXmp = dst.xmpData();
        for (const auto & datum : srcXmp) {
            if (datum.groupName() == "crs") {
                auto it = dstXmp.findKey(Exiv2::XmpKey(datum.key()));
                if (it != dstXmp.end()) {
                    dstXmp.erase(it);
                }
                dstXmp.add(datum);
            }
        }
    } catch (Exiv2::Error & e) {
        std::cerr << "Exiv2 error reading ACR profile: " << e.what() << std::endl;
    }
}


static void injectDefaultHDRSettings(Exiv2::Image & dst) {
    Exiv2::XmpData & xmp = dst.xmpData();

    // setIfAbsent helper: only inject values not already present
    auto setIfAbsent = [&](const char * key, const std::string & value) {
        if (xmp.findKey(Exiv2::XmpKey(key)) == xmp.end()) {
            xmp[key] = value;
        }
    };

    setIfAbsent("Xmp.crs.ProcessVersion", "11.0");
    setIfAbsent("Xmp.crs.HDREditMode", "1");
    setIfAbsent("Xmp.crs.Highlights2012", "-100");
    setIfAbsent("Xmp.crs.Shadows2012", "+100");
    setIfAbsent("Xmp.crs.Whites2012", "-40");
    setIfAbsent("Xmp.crs.Blacks2012", "+20");

    // Tone curve (XMP seq) — only if not already present
    const char * curveKey = "Xmp.crs.ToneCurvePV2012";
    if (xmp.findKey(Exiv2::XmpKey(curveKey)) == xmp.end()) {
        ExivValuePtr val = Exiv2::Value::create(Exiv2::xmpSeq);
        val->read("0, 0");
        val->read("64, 70");
        val->read("128, 140");
        val->read("192, 200");
        val->read("255, 255");
        xmp.add(Exiv2::XmpKey(curveKey), val.get());
    }

    setIfAbsent("Xmp.crs.ToneCurveName2012", "Custom");
}


static void injectAdaptiveCurves(Exiv2::Image & dst, const hdrmerge::AdaptiveCurves & curves) {
    if (!curves.valid) return;
    Exiv2::XmpData & xmp = dst.xmpData();

    // Set master curve to linear
    {
        const char * key = "Xmp.crs.ToneCurvePV2012";
        auto it = xmp.findKey(Exiv2::XmpKey(key));
        if (it != xmp.end()) xmp.erase(it);
        ExivValuePtr val = Exiv2::Value::create(Exiv2::xmpSeq);
        val->read("0, 0");
        val->read("255, 255");
        xmp.add(Exiv2::XmpKey(key), val.get());
    }

    // Per-channel curves
    struct ChannelCurve {
        const char * key;
        const std::vector<std::pair<int,int>> * points;
    };
    ChannelCurve channels[] = {
        { "Xmp.crs.ToneCurvePV2012Red",   &curves.red },
        { "Xmp.crs.ToneCurvePV2012Green", &curves.green },
        { "Xmp.crs.ToneCurvePV2012Blue",  &curves.blue },
    };
    for (const auto & ch : channels) {
        auto it = xmp.findKey(Exiv2::XmpKey(ch.key));
        if (it != xmp.end()) xmp.erase(it);
        ExivValuePtr val = Exiv2::Value::create(Exiv2::xmpSeq);
        for (const auto & pt : *ch.points) {
            std::string s = std::to_string(pt.first) + ", " + std::to_string(pt.second);
            val->read(s);
        }
        xmp.add(Exiv2::XmpKey(ch.key), val.get());
    }

    // Mark as custom curve
    {
        const char * key = "Xmp.crs.ToneCurveName2012";
        auto it = xmp.findKey(Exiv2::XmpKey(key));
        if (it != xmp.end()) xmp.erase(it);
        ExivValuePtr val = Exiv2::Value::create(Exiv2::xmpText);
        val->read("Custom");
        xmp.add(Exiv2::XmpKey(key), val.get());
    }
}


static std::string trimString(const std::string & s) {
    const std::string whitespace(" \t\r\n\0", 5);
    size_t start = s.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(whitespace);
    return s.substr(start, end - start + 1);
}

static void synthesizeLensXMP(Exiv2::Image & src, Exiv2::Image & dst) {
    const Exiv2::ExifData & exif = src.exifData();
    Exiv2::XmpData & xmp = dst.xmpData();

    // Xmp.aux.Lens from Exif.Photo.LensModel
    auto lensModel = exif.findKey(Exiv2::ExifKey("Exif.Photo.LensModel"));
    if (lensModel != exif.end() && xmp.findKey(Exiv2::XmpKey("Xmp.aux.Lens")) == xmp.end()) {
        xmp["Xmp.aux.Lens"] = trimString(lensModel->toString());
    }

    // Xmp.aux.LensInfo from Exif.Photo.LensSpecification as XMP seq of rationals
    auto lensSpec = exif.findKey(Exiv2::ExifKey("Exif.Photo.LensSpecification"));
    if (lensSpec != exif.end() && lensSpec->count() >= 4
        && xmp.findKey(Exiv2::XmpKey("Xmp.aux.LensInfo")) == xmp.end()) {
        ExivValuePtr val = Exiv2::Value::create(Exiv2::xmpSeq);
        for (long i = 0; i < 4; ++i) {
            val->read(lensSpec->toString(i));
        }
        xmp.add(Exiv2::XmpKey("Xmp.aux.LensInfo"), val.get());
    }

    // Xmp.aux.SerialNumber from Exif.Photo.BodySerialNumber
    auto serial = exif.findKey(Exiv2::ExifKey("Exif.Photo.BodySerialNumber"));
    if (serial != exif.end() && xmp.findKey(Exiv2::XmpKey("Xmp.aux.SerialNumber")) == xmp.end()) {
        xmp["Xmp.aux.SerialNumber"] = trimString(serial->toString());
    }

    // Xmp.aux.LensSerialNumber from Exif.Photo.LensSerialNumber
    auto lensSerial = exif.findKey(Exiv2::ExifKey("Exif.Photo.LensSerialNumber"));
    if (lensSerial != exif.end() && xmp.findKey(Exiv2::XmpKey("Xmp.aux.LensSerialNumber")) == xmp.end()) {
        xmp["Xmp.aux.LensSerialNumber"] = trimString(lensSerial->toString());
    }

    // Xmp.exifEX.LensMake and LensModel
    auto lensMake = exif.findKey(Exiv2::ExifKey("Exif.Photo.LensMake"));
    if (lensMake != exif.end() && xmp.findKey(Exiv2::XmpKey("Xmp.exifEX.LensMake")) == xmp.end()) {
        xmp["Xmp.exifEX.LensMake"] = trimString(lensMake->toString());
    }
    if (lensModel != exif.end() && xmp.findKey(Exiv2::XmpKey("Xmp.exifEX.LensModel")) == xmp.end()) {
        xmp["Xmp.exifEX.LensModel"] = trimString(lensModel->toString());
    }
}


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


static bool excludeExifDatum(const Exifdatum & datum) {
    static const char * previewKeys[] {
        "Exif.OlympusCs.PreviewImageStart",
        "Exif.OlympusCs.PreviewImageLength",
        "Exif.Thumbnail.JPEGInterchangeFormat",
        "Exif.Thumbnail.JPEGInterchangeFormatLength",
        "Exif.NikonPreview.JPEGInterchangeFormat",
        "Exif.NikonPreview.JPEGInterchangeFormatLength",
        "Exif.Pentax.PreviewOffset",
        "Exif.Pentax.PreviewLength",
        "Exif.PentaxDng.PreviewOffset",
        "Exif.PentaxDng.PreviewLength",
        "Exif.Minolta.ThumbnailOffset",
        "Exif.Minolta.ThumbnailLength",
        "Exif.SonyMinolta.ThumbnailOffset",
        "Exif.SonyMinolta.ThumbnailLength",
        "Exif.Olympus.ThumbnailImage",
        "Exif.Olympus2.ThumbnailImage",
        "Exif.Minolta.Thumbnail",
        "Exif.PanasonicRaw.PreviewImage",
        "Exif.SamsungPreview.JPEGInterchangeFormat",
        "Exif.SamsungPreview.JPEGInterchangeFormatLength"
    };
    for (const char * pkey : previewKeys) {
        if (datum.key() == pkey) {
            return true;
        }
    }
    return
        datum.groupName().substr(0, 5) == "Thumb" ||
        datum.groupName().substr(0, 8) == "SubThumb" ||
        datum.groupName().substr(0, 5) == "Image" ||
        datum.groupName().substr(0, 8) == "SubImage";
}


static void copyEXIF(Exiv2::Image & src, Exiv2::Image & dst) {
    static const char * includeImageKeys[] = {
        // Correct Make and Model, from the input files
        // It is needed so that makernote tags are correctly copied
        "Exif.Image.Make",
        "Exif.Image.Model",
        "Exif.Image.Artist",
        "Exif.Image.Copyright",
        "Exif.Image.DNGPrivateData",
        // Opcodes generated by Adobe DNG converter
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
    // Now we set the SubImage1 file type to Primary Image
    // Exiv2 wouldn't modify SubImage1 tags if it was set before
    dstExif["Exif.SubImage1.NewSubfileType"] = 0u;

    for (const auto & datum : srcExif) {
        if (!excludeExifDatum(datum) && dstExif.findKey(Exiv2::ExifKey(datum.key())) == dstExif.end()) {
            dstExif.add(datum);
        }
    }
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
    // Inject hardcoded HDR defaults (setIfAbsent — won't override existing values)
    injectDefaultHDRSettings(*dst);
    // Inject ACR profile (overrides source metadata and defaults)
    injectACRProfile(*dst, acrProfilePath);
    // Inject adaptive curves (overrides profile's curves if present)
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
