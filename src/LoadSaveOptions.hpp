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

enum class ResponseMode { Linear, Nonlinear };
enum class DeghostMode { Legacy, Robust };

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
    float hotPixelSigma;
    ResponseMode responseMode;
    LoadOptions() : align(true), crop(true), useCustomWl(false), customWl(16383), batch(false), batchGap(2.0),
        withSingles(false), alignFeatures(false), hotPixelSigma(0.0f), responseMode(ResponseMode::Linear) {}
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
    DeghostMode deghostMode;
    int deghostIterations;
    double clipPercentile;
    QString acrProfilePath;
    double evShift;
    bool autoCurves;
    int resizeLong;
    bool subPixelAlign;
    float highlightPull;
    float highlightRolloff;
    float highlightKnee;
    float bilateralRangeSigma;
    int highlightMaskBlur;
    int highlightScaleBlur;
    float highlightBoostCap;
    SaveOptions() : bps(24), previewSize(0), saveMask(false), featherRadius(3),
        compressionLevel(6), deghostSigma(0.0f), deghostMode(DeghostMode::Robust),
        deghostIterations(1), clipPercentile(99.9),
        evShift(0.0),
        autoCurves(false), resizeLong(0), subPixelAlign(false),
        highlightPull(0.0f), highlightRolloff(0.9f), highlightKnee(2.0f),
        bilateralRangeSigma(0.5f), highlightMaskBlur(10),
        highlightScaleBlur(3), highlightBoostCap(4.0f) {}
};

} // namespace hdrmerge

#endif // _LOADSAVEOPTIONS_H_
