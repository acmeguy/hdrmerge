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

#ifndef _DNGFLOATWRITER_HPP_
#define _DNGFLOATWRITER_HPP_

#include <cstdio>
#include <cmath>
#include <QString>
#include <QImage>
#include "config.h"
#include "Array2D.hpp"
#include "TiffDirectory.hpp"
#include "ExifTransfer.hpp"

namespace hdrmerge {

class RawParameters;

class DngFloatWriter {
public:
    DngFloatWriter() : previewWidth(0), bps(16), compressionLevel(6) {}

    void setPreviewWidth(size_t w) {
        previewWidth = w;
    }
    void setBitsPerSample(int b) {
        bps = b;
    }
    void setCompressionLevel(int level) {
        compressionLevel = level;
    }
    void setBaselineExposure(double ev) { baselineExposureEV = ev; }
    void setBaselineNoise(int numImages) {
        baselineNoiseRatio = (numImages > 1)
            ? 1.0 / std::sqrt(static_cast<double>(numImages))
            : 1.0;
    }
    void setNoiseProfile(const double * profile, int colors) {
        noiseProfileColors = colors;
        for (int i = 0; i < colors * 2; ++i)
            noiseProfileData[i] = profile[i];
    }
    void setACRProfilePath(const QString & path) { acrProfilePath = path; }
    void setAdaptiveCurves(const AdaptiveCurves & c) { adaptiveCurves = c; }
    void setPreview(const QImage & p);
    void write(Array2D<float> && rawPixels, const RawParameters & p, const QString & dstFileName);

private:
    int previewWidth;
    int bps;
    int compressionLevel;
    bool useJXL = false;
    double baselineExposureEV = 0.0;
    double baselineNoiseRatio = 1.0;
    double noiseProfileData[8] = {};
    int noiseProfileColors = 0;
    QString acrProfilePath;
    AdaptiveCurves adaptiveCurves;
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
    bool writePreviewsToFile(FILE * f, size_t dataOffset);
    bool writeRawDataToFile(FILE * f);
    void renderPreviews();
    void createPreviewIFD();
    size_t thumbSize();
    size_t previewSize();
};

} // namespace hdrmerge

#endif // _DNGFLOATWRITER_HPP_
