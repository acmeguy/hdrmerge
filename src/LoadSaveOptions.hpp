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

#ifndef _LOADSAVEOPTIONS_H_
#define _LOADSAVEOPTIONS_H_

#include <vector>
#include <QString>

namespace hdrmerge {

struct LoadOptions {
    std::vector<QString> fileNames;
    bool align;
    bool crop;
    bool useCustomWl;
    uint16_t customWl;
    bool batch;
    double batchGap;
    bool withSingles;
    bool alignFeatures;
    LoadOptions() : align(true), crop(true), useCustomWl(false), customWl(16383), batch(false), batchGap(2.0),
        withSingles(false), alignFeatures(false) {}
};


struct SaveOptions {
    int bps;
    int previewSize;
    QString fileName;
    bool saveMask;
    QString maskFileName;
    QString outputDir;
    int featherRadius;
    int compressionLevel;
    float deghostSigma;
    double clipPercentile;
    QString acrProfilePath;
    double evShift;
    float hotPixelSigma;
    bool autoCurves;
    int resizeLong;
    bool subPixelAlign;
    SaveOptions() : bps(24), previewSize(0), saveMask(false), featherRadius(3),
        compressionLevel(6), deghostSigma(0.0f), clipPercentile(99.9),
        evShift(0.0), hotPixelSigma(0.0f),
        autoCurves(false), resizeLong(0), subPixelAlign(false) {}
};

} // namespace hdrmerge

#endif // _LOADSAVEOPTIONS_H_
