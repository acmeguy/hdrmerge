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

#include <cmath>
#include <vector>
#include <algorithm>
#include "Resizer.hpp"
#include "Log.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace hdrmerge {

static double sinc(double x) {
    if (std::abs(x) < 1e-8) return 1.0;
    double px = M_PI * x;
    return std::sin(px) / px;
}

static double lanczos3(double x) {
    if (std::abs(x) >= 3.0) return 0.0;
    return sinc(x) * sinc(x / 3.0);
}

// Resize a single-channel 2D image (one CFA sub-plane) using separable Lanczos-3.
// srcW x srcH -> dstW x dstH.  Double-precision accumulation, float output.
static std::vector<float> resizePlane(const std::vector<float> & src,
                                       size_t srcW, size_t srcH,
                                       size_t dstW, size_t dstH) {
    // Horizontal pass: src (srcW x srcH) -> tmp (dstW x srcH)
    std::vector<float> tmp(dstW * srcH);
    double xRatio = static_cast<double>(srcW) / dstW;
    double xFilterRadius = std::max(3.0, 3.0 * xRatio);
    int xFilterSize = static_cast<int>(std::ceil(xFilterRadius)) * 2 + 1;

    #pragma omp parallel for schedule(static)
    for (size_t y = 0; y < srcH; ++y) {
        for (size_t dx = 0; dx < dstW; ++dx) {
            double center = (dx + 0.5) * xRatio - 0.5;
            int left = static_cast<int>(std::floor(center - xFilterRadius));
            int right = static_cast<int>(std::ceil(center + xFilterRadius));
            (void)xFilterSize;

            double acc = 0.0;
            double wSum = 0.0;
            for (int sx = left; sx <= right; ++sx) {
                int csx = std::min(std::max(sx, 0), static_cast<int>(srcW) - 1);
                double dist = (sx - center) / xRatio;
                double w = lanczos3(dist);
                acc += w * src[y * srcW + csx];
                wSum += w;
            }
            tmp[y * dstW + dx] = static_cast<float>(acc / wSum);
        }
    }

    // Vertical pass: tmp (dstW x srcH) -> dst (dstW x dstH)
    std::vector<float> dst(dstW * dstH);
    double yRatio = static_cast<double>(srcH) / dstH;
    double yFilterRadius = std::max(3.0, 3.0 * yRatio);

    #pragma omp parallel for schedule(static)
    for (size_t x = 0; x < dstW; ++x) {
        for (size_t dy = 0; dy < dstH; ++dy) {
            double center = (dy + 0.5) * yRatio - 0.5;
            int top = static_cast<int>(std::floor(center - yFilterRadius));
            int bottom = static_cast<int>(std::ceil(center + yFilterRadius));

            double acc = 0.0;
            double wSum = 0.0;
            for (int sy = top; sy <= bottom; ++sy) {
                int csy = std::min(std::max(sy, 0), static_cast<int>(srcH) - 1);
                double dist = (sy - center) / yRatio;
                double w = lanczos3(dist);
                acc += w * tmp[csy * dstW + x];
                wSum += w;
            }
            dst[dy * dstW + x] = static_cast<float>(acc / wSum);
        }
    }

    return dst;
}


ResizeResult resizeCFA(Array2D<float> && input,
    size_t rawWidth, size_t rawHeight,
    size_t width, size_t height,
    size_t topMargin, size_t leftMargin,
    int targetLongEdge, const CFAPattern & cfa)
{
    size_t longEdge = std::max(width, height);

    // No upscaling: if target >= current, return unchanged
    if (targetLongEdge <= 0 || static_cast<size_t>(targetLongEdge) >= longEdge) {
        Log::msg(Log::DEBUG, "Resize: target ", targetLongEdge,
                 " >= current ", longEdge, ", skipping resize");
        ResizeResult r;
        r.image = std::move(input);
        r.rawWidth = rawWidth;
        r.rawHeight = rawHeight;
        r.width = width;
        r.height = height;
        r.topMargin = topMargin;
        r.leftMargin = leftMargin;
        return r;
    }

    int cfaRows = cfa.getRows();
    int cfaCols = cfa.getColumns();

    // Compute output active-area dimensions, rounded to CFA multiples
    double scale = static_cast<double>(targetLongEdge) / longEdge;
    size_t outWidth = static_cast<size_t>(width * scale);
    size_t outHeight = static_cast<size_t>(height * scale);
    // Round down to CFA multiples
    outWidth = (outWidth / cfaCols) * cfaCols;
    outHeight = (outHeight / cfaRows) * cfaRows;
    if (outWidth < static_cast<size_t>(cfaCols)) outWidth = cfaCols;
    if (outHeight < static_cast<size_t>(cfaRows)) outHeight = cfaRows;

    // Round margins to CFA multiples (preserve CFA phase)
    size_t outTopMargin = (static_cast<size_t>(topMargin * scale) / cfaRows) * cfaRows;
    size_t outLeftMargin = (static_cast<size_t>(leftMargin * scale) / cfaCols) * cfaCols;
    size_t outRawWidth = outWidth + outLeftMargin * 2;  // Approximate symmetry
    size_t outRawHeight = outHeight + outTopMargin * 2;
    // Ensure raw dimensions are CFA-aligned
    outRawWidth = (outRawWidth / cfaCols) * cfaCols;
    outRawHeight = (outRawHeight / cfaRows) * cfaRows;
    // Ensure raw dimensions encompass active area + margins
    if (outRawWidth < outWidth + outLeftMargin) outRawWidth = outWidth + outLeftMargin;
    if (outRawHeight < outHeight + outTopMargin) outRawHeight = outHeight + outTopMargin;
    outRawWidth = ((outRawWidth + cfaCols - 1) / cfaCols) * cfaCols;
    outRawHeight = ((outRawHeight + cfaRows - 1) / cfaRows) * cfaRows;

    Log::msg(Log::DEBUG, "Resize: ", width, "x", height, " -> ", outWidth, "x", outHeight,
             " (raw ", outRawWidth, "x", outRawHeight,
             ", margins ", outTopMargin, ",", outLeftMargin, ")");

    // Sub-image dimensions (input active area only)
    size_t subW = width / cfaCols;
    size_t subH = height / cfaRows;
    size_t outSubW = outWidth / cfaCols;
    size_t outSubH = outHeight / cfaRows;

    // Decompose: extract each CFA sub-plane from the active area
    int numPlanes = cfaRows * cfaCols;
    std::vector<std::vector<float>> subImages(numPlanes);
    for (int p = 0; p < numPlanes; ++p) {
        subImages[p].resize(subW * subH);
    }

    for (size_t sy = 0; sy < subH; ++sy) {
        for (size_t sx = 0; sx < subW; ++sx) {
            for (int cr = 0; cr < cfaRows; ++cr) {
                for (int cc = 0; cc < cfaCols; ++cc) {
                    int plane = cr * cfaCols + cc;
                    size_t srcY = topMargin + sy * cfaRows + cr;
                    size_t srcX = leftMargin + sx * cfaCols + cc;
                    subImages[plane][sy * subW + sx] = input[srcY * rawWidth + srcX];
                }
            }
        }
    }

    // Release input memory
    input = Array2D<float>();

    // Resize each sub-plane independently
    std::vector<std::vector<float>> resizedPlanes(numPlanes);
    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < numPlanes; ++p) {
        resizedPlanes[p] = resizePlane(subImages[p], subW, subH, outSubW, outSubH);
        subImages[p].clear();
        subImages[p].shrink_to_fit();
    }

    // Reinterleave into output CFA mosaic
    Array2D<float> output(outRawWidth, outRawHeight);
    // Zero-fill the entire output (margins will be zero)
    std::fill(output.begin(), output.end(), 0.0f);

    for (size_t sy = 0; sy < outSubH; ++sy) {
        for (size_t sx = 0; sx < outSubW; ++sx) {
            for (int cr = 0; cr < cfaRows; ++cr) {
                for (int cc = 0; cc < cfaCols; ++cc) {
                    int plane = cr * cfaCols + cc;
                    size_t dstY = outTopMargin + sy * cfaRows + cr;
                    size_t dstX = outLeftMargin + sx * cfaCols + cc;
                    output[dstY * outRawWidth + dstX] = resizedPlanes[plane][sy * outSubW + sx];
                }
            }
        }
    }

    ResizeResult result;
    result.image = std::move(output);
    result.rawWidth = outRawWidth;
    result.rawHeight = outRawHeight;
    result.width = outWidth;
    result.height = outHeight;
    result.topMargin = outTopMargin;
    result.leftMargin = outLeftMargin;
    return result;
}

} // namespace hdrmerge
