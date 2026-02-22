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

#include <algorithm>

#include "BoxBlur.hpp"
#include "ImageStack.hpp"
#include "Log.hpp"
#include "RawParameters.hpp"

#if defined(__SSE2__)
    #include <x86intrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif

using namespace std;
using namespace hdrmerge;


int ImageStack::addImage(Image && i) {
    if (images.empty()) {
        width = i.getWidth();
        height = i.getHeight();
    }
    images.push_back(std::move(i));
    int n = images.size() - 1;
    while (n > 0 && images[n] < images[n - 1]) {
        std::swap(images[n], images[n - 1]);
        --n;
    }
    return n;
}


void ImageStack::calculateSaturationLevel(const RawParameters & params, bool useCustomWl) {
    // Calculate max value of brightest image and assume it is saturated
    Image& brightest = images.front();

    std::vector<std::vector<size_t>> histograms(4, std::vector<size_t>(brightest.getMax() + 1));

    #pragma omp parallel
    {
        std::vector<std::vector<size_t>> histogramsThr(4, std::vector<size_t>(brightest.getMax() + 1));
        #pragma omp for schedule(dynamic,16) nowait
        for (size_t y = 0; y < height; ++y) {
            // get the color codes from x = 0 to 5, works for bayer and xtrans
            uint16_t fcrow[6];
            for (size_t i = 0; i < 6; ++i) {
                fcrow[i] = params.FC(i, y);
            }
            size_t x = 0;
            for (; x < width - 5; x+=6) {
                for(size_t j = 0; j < 6; ++j) {
                    uint16_t v = brightest(x + j, y);
                    ++histogramsThr[fcrow[j]][v];
                }
            }
            // remaining pixels
            for (size_t j = 0; x < width; ++x, ++j) {
                uint16_t v = brightest(x, y);
                ++histogramsThr[fcrow[j]][v];
            }
        }
        #pragma omp critical
        {
            for (int c = 0; c < 4; ++c) {
                for (std::vector<size_t>::size_type i = 0; i < histograms[c].size(); ++i) {
                    histograms[c][i] += histogramsThr[c][i];
                }
            }
        }
    }

    const size_t threshold = width * height / 10000;

    uint16_t maxPerColor[4] = {0, 0, 0, 0};

    for (int c = 0; c < 4; ++c) {
        for (int i = histograms[c].size() - 1; i >= 0; --i) {
            const size_t v = histograms[c][i];
            if (v > threshold) {
                maxPerColor[c] = i;
                break;
            }
        }
    }


    // Per-channel saturation thresholds
    for (int c = 0; c < 4; ++c) {
        uint16_t chMax = maxPerColor[c];
        uint16_t chThresh = params.max == 0 ? chMax : params.max;
        if (chMax > 0) chThresh = std::min(chThresh, chMax);
        if (!useCustomWl) chThresh *= 0.99;
        satThresholdPerChannel[c] = chThresh;
    }

    // Global threshold: most conservative (minimum across channels) for mask generation
    satThreshold = satThresholdPerChannel[0];
    for (int c = 1; c < 4; ++c) {
        satThreshold = std::min(satThreshold, satThresholdPerChannel[c]);
    }

    Log::debug("Using white levels: global=", satThreshold,
               " R=", satThresholdPerChannel[0], " G1=", satThresholdPerChannel[1],
               " G2=", satThresholdPerChannel[2], " B=", satThresholdPerChannel[3]);

    for (auto& i : images) {
        i.setSaturationThreshold(satThreshold);
    }
}


// Compute sub-pixel alignment residual between two integer-aligned images.
// Uses SSD (sum of squared differences) at 5 offsets with parabolic fitting.
// Only for diagnostic logging — compositing uses integer shifts.
static void measureSubPixelResidual(const Image & ref, const Image & img,
                                     size_t width, size_t height,
                                     double & fracDx, double & fracDy) {
    // Compute mean SSD at a given additional offset (ox, oy) relative to current alignment.
    // Subsample every 4th pixel for speed.
    auto computeSSD = [&](int ox, int oy) -> double {
        double ssd = 0;
        size_t count = 0;
        int margin = 2;
        for (size_t y = margin; y < height - margin; y += 4) {
            for (size_t x = margin; x < width - margin; x += 4) {
                int rx = (int)x, ry = (int)y;
                int ix = rx + ox, iy = ry + oy;
                if (ref.contains(rx, ry) && img.contains(ix, iy)) {
                    double d = (double)ref(rx, ry) - (double)img(ix, iy);
                    ssd += d * d;
                    ++count;
                }
            }
        }
        return count > 0 ? ssd / count : 1e18;
    };

    double c = computeSSD(0, 0);
    double l = computeSSD(-1, 0);
    double r = computeSSD(1, 0);
    double u = computeSSD(0, -1);
    double d = computeSSD(0, 1);

    // Parabolic interpolation: vertex of y = ax^2 + bx + c at x = -b/(2a)
    double denom_x = l + r - 2.0 * c;
    double denom_y = u + d - 2.0 * c;
    fracDx = (denom_x != 0.0) ? (l - r) / (2.0 * denom_x) : 0.0;
    fracDy = (denom_y != 0.0) ? (u - d) / (2.0 * denom_y) : 0.0;
    // Clamp to [-0.5, 0.5] — larger values indicate the parabola fit is unreliable
    fracDx = std::max(-0.5, std::min(0.5, fracDx));
    fracDy = std::max(-0.5, std::min(0.5, fracDy));
}

void ImageStack::align(bool useFeatures) {
    if (images.size() > 1) {
#ifdef HAVE_OPENCV
        if (useFeatures) {
            Log::progress("Feature-based alignment requested (OpenCV available)");
            // TODO: Implement AKAZE/ORB feature matching with homography estimation.
            // For now, fall through to MTB alignment.
            Log::progress("Feature-based alignment not yet implemented, using MTB");
        }
#else
        if (useFeatures) {
            Log::progress("Feature-based alignment requested but OpenCV not available, using MTB");
        }
#endif
        Timer t("Align");
        size_t errors[images.size()];
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < images.size(); ++i) {
            images[i].preScale();
        }
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < images.size() - 1; ++i) {
            errors[i] = images[i].alignWith(images[i + 1]);
        }
        for (size_t i = images.size() - 1; i > 0; --i) {
            images[i - 1].displace(images[i].getDeltaX(), images[i].getDeltaY());
            Log::debug("Image ", i - 1, " displaced to (", images[i - 1].getDeltaX(),
                       ", ", images[i - 1].getDeltaY(), ") with error ", errors[i - 1]);
        }
        for (auto & i : images) {
            i.releaseAlignData();
        }
        // Measure sub-pixel alignment residuals and store on each image
        for (size_t i = 0; i < images.size() - 1; ++i) {
            double fdx, fdy;
            measureSubPixelResidual(images[i + 1], images[i], width, height, fdx, fdy);
            images[i].setFracDx(fdx);
            images[i].setFracDy(fdy);
            Log::debug("Image ", i, " sub-pixel residual: (", fdx, ", ", fdy, ") px");
        }
        // Reference image (last in sorted order) has zero fractional offset
        images.back().setFracDx(0.0);
        images.back().setFracDy(0.0);
    }
}

void ImageStack::crop() {
    int dx = 0, dy = 0;
    for (auto & i : images) {
        int newDx = max(dx, i.getDeltaX());
        int bound = min(dx + width, i.getDeltaX() + i.getWidth());
        width = bound > newDx ? bound - newDx : 0;
        dx = newDx;
        int newDy = max(dy, i.getDeltaY());
        bound = min(dy + height, i.getDeltaY() + i.getHeight());
        height = bound > newDy ? bound - newDy : 0;
        dy = newDy;
    }
    for (auto & i : images) {
        i.displace(-dx, -dy);
    }
}


void ImageStack::computeResponseFunctions() {
    Timer t("Compute response functions");
    for (int i = images.size() - 2; i >= 0; --i) {
        images[i].computeResponseFunction(images[i + 1]);
    }
}


void ImageStack::generateMask() {
    Timer t("Generate mask");
    mask.resize(width, height);
    if(images.size() == 1) {
        // single image, fill in zero values
        std::fill_n(&mask[0], width*height, 0);
    } else {
        // multiple images, no need to prefill mask with zeroes. It will be filled correctly on the fly
        #pragma omp parallel for schedule(dynamic)
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t i = 0;
                while (i < images.size() - 1 &&
                    (!images[i].contains(x, y) ||
                    images[i].isSaturatedAround(x, y))) ++i;
                mask(x, y) = i;
            }
        }
    }
    // The mask can be used in compose to get the information about saturated pixels
    // but the mask can be modified in gui, so we have to make a copy to represent the original state
    origMask = mask;
}


double ImageStack::value(size_t x, size_t y) const {
    const Image & img = images[mask(x, y)];
    return img.exposureAt(x, y);
}

#if defined(__SSE2__)
// From The GIMP: app/paint-funcs/paint-funcs.c:fatten_region
// SSE version by Ingo Weyrich
static Array2D<uint8_t> fattenMask(const Array2D<uint8_t> & mask, int radius) {
    Timer t("Fatten mask (SSE version)");
    size_t width = mask.getWidth(), height = mask.getHeight();
    Array2D<uint8_t> result(width, height);

    int circArray[2 * radius + 1]; // holds the y coords of the filter's mask
    // compute_border(circArray, radius)
    for (int i = 0; i < radius * 2 + 1; i++) {
        double tmp;
        if (i > radius)
            tmp = (i - radius) - 0.5;
        else if (i < radius)
            tmp = (radius - i) - 0.5;
        else
            tmp = 0.0;
        circArray[i] = int(std::sqrt(radius*radius - tmp*tmp));
    }
    // offset the circ pointer by radius so the range of the array
    //     is [-radius] to [radius]
    int * circ = circArray + radius;

    const uint8_t * bufArray[height + 2*radius];
    for (int i = 0; i < radius; i++) {
        bufArray[i] = &mask[0];
    }
    for (size_t i = 0; i < height; i++) {
        bufArray[i + radius] = &mask[i * width];
    }
    for (int i = 0; i < radius; i++) {
        bufArray[i + height + radius] = &mask[(height - 1) * width];
    }
    // offset the buf pointer
    const uint8_t ** buf = bufArray + radius;

    #pragma omp parallel
    {
        uint8_t buffer[width * (radius + 1)];
        uint8_t *maxArray[radius+1];
        for (int i = 0; i <= radius; i++) {
            maxArray[i] = &buffer[i*width];
        }

        #pragma omp for schedule(dynamic,16)
        for (size_t y = 0; y < height; y++) {
            size_t x = 0;
            for (; x < width-15; x+=16) { // compute max array, use SSE to process 16 bytes at once
                __m128i lmax = _mm_loadu_si128((__m128i*)&buf[y][x]);
                if(radius<2) // max[0] is only used when radius < 2
                    _mm_storeu_si128((__m128i*)&maxArray[0][x],lmax);
                for (int i = 1; i <= radius; i++) {
                    lmax = _mm_max_epu8(_mm_loadu_si128((__m128i*)&buf[y + i][x]),lmax);
                    lmax = _mm_max_epu8(_mm_loadu_si128((__m128i*)&buf[y - i][x]),lmax);
                    _mm_storeu_si128((__m128i*)&maxArray[i][x],lmax);
                }
            }
            for (; x < width; x++) { // compute max array, remaining columns
                uint8_t lmax = buf[y][x];
                if(radius<2) // max[0] is only used when radius < 2
                    maxArray[0][x] = lmax;
                for (int i = 1; i <= radius; i++) {
                    lmax = std::max(std::max(lmax, buf[y + i][x]), buf[y - i][x]);
                    maxArray[i][x] = lmax;
                }
            }

            for (x = 0; (int)x < radius; x++) { // render scan line, first columns without SSE
                uint8_t last_max = maxArray[circ[radius]][x+radius];
                for (int i = radius - 1; i >= -(int)x; i--)
                    last_max = std::max(last_max,maxArray[circ[i]][x + i]);
                result(x, y) = last_max;
            }
            for (; x < width-15-radius+1; x += 16) { // render scan line, use SSE to process 16 bytes at once
                __m128i last_maxv = _mm_loadu_si128((__m128i*)&maxArray[circ[radius]][x+radius]);
                for (int i = radius - 1; i >= -radius; i--)
                    last_maxv = _mm_max_epu8(last_maxv,_mm_loadu_si128((__m128i*)&maxArray[circ[i]][x+i]));
                _mm_storeu_si128((__m128i*)&result(x,y),last_maxv);
            }

            for (; x < width; x++) { // render scan line, last columns without SSE
                int maxRadius = std::min(radius,(int)((int)width-1-(int)x));
                uint8_t last_max = maxArray[circ[maxRadius]][x+maxRadius];
                for (int i = maxRadius-1; i >= -radius; i--)
                    last_max = std::max(last_max,maxArray[circ[i]][x + i]);
                result(x, y) = last_max;
            }
        }
    }

    return result;
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
// NEON version: identical algorithm to SSE2, using ARM NEON intrinsics
static Array2D<uint8_t> fattenMask(const Array2D<uint8_t> & mask, int radius) {
    Timer t("Fatten mask (NEON version)");
    size_t width = mask.getWidth(), height = mask.getHeight();
    Array2D<uint8_t> result(width, height);

    int circArray[2 * radius + 1];
    for (int i = 0; i < radius * 2 + 1; i++) {
        double tmp;
        if (i > radius)
            tmp = (i - radius) - 0.5;
        else if (i < radius)
            tmp = (radius - i) - 0.5;
        else
            tmp = 0.0;
        circArray[i] = int(std::sqrt(radius*radius - tmp*tmp));
    }
    int * circ = circArray + radius;

    const uint8_t * bufArray[height + 2*radius];
    for (int i = 0; i < radius; i++) {
        bufArray[i] = &mask[0];
    }
    for (size_t i = 0; i < height; i++) {
        bufArray[i + radius] = &mask[i * width];
    }
    for (int i = 0; i < radius; i++) {
        bufArray[i + height + radius] = &mask[(height - 1) * width];
    }
    const uint8_t ** buf = bufArray + radius;

    #pragma omp parallel
    {
        uint8_t buffer[width * (radius + 1)];
        uint8_t *maxArray[radius+1];
        for (int i = 0; i <= radius; i++) {
            maxArray[i] = &buffer[i*width];
        }

        #pragma omp for schedule(dynamic,16)
        for (size_t y = 0; y < height; y++) {
            size_t x = 0;
            for (; x < width-15; x+=16) { // compute max array, 16 bytes at a time
                uint8x16_t lmax = vld1q_u8(&buf[y][x]);
                if(radius<2)
                    vst1q_u8(&maxArray[0][x], lmax);
                for (int i = 1; i <= radius; i++) {
                    lmax = vmaxq_u8(vld1q_u8(&buf[y + i][x]), lmax);
                    lmax = vmaxq_u8(vld1q_u8(&buf[y - i][x]), lmax);
                    vst1q_u8(&maxArray[i][x], lmax);
                }
            }
            for (; x < width; x++) { // remaining columns
                uint8_t lmax = buf[y][x];
                if(radius<2)
                    maxArray[0][x] = lmax;
                for (int i = 1; i <= radius; i++) {
                    lmax = std::max(std::max(lmax, buf[y + i][x]), buf[y - i][x]);
                    maxArray[i][x] = lmax;
                }
            }

            for (x = 0; (int)x < radius; x++) { // first columns scalar
                uint8_t last_max = maxArray[circ[radius]][x+radius];
                for (int i = radius - 1; i >= -(int)x; i--)
                    last_max = std::max(last_max,maxArray[circ[i]][x + i]);
                result(x, y) = last_max;
            }
            for (; x < width-15-radius+1; x += 16) { // render scan line, 16 bytes at a time
                uint8x16_t last_maxv = vld1q_u8(&maxArray[circ[radius]][x+radius]);
                for (int i = radius - 1; i >= -radius; i--)
                    last_maxv = vmaxq_u8(last_maxv, vld1q_u8(&maxArray[circ[i]][x+i]));
                vst1q_u8(&result(x,y), last_maxv);
            }

            for (; x < width; x++) { // last columns scalar
                int maxRadius = std::min(radius,(int)((int)width-1-(int)x));
                uint8_t last_max = maxArray[circ[maxRadius]][x+maxRadius];
                for (int i = maxRadius-1; i >= -radius; i--)
                    last_max = std::max(last_max,maxArray[circ[i]][x + i]);
                result(x, y) = last_max;
            }
        }
    }

    return result;
}
#else
// Scalar fallback
// From The GIMP: app/paint-funcs/paint-funcs.c:fatten_region
static Array2D<uint8_t> fattenMask(const Array2D<uint8_t> & mask, int radius) {
    Timer t("Fatten mask");
    size_t width = mask.getWidth(), height = mask.getHeight();
    Array2D<uint8_t> result(width, height);

    int circArray[2 * radius + 1]; // holds the y coords of the filter's mask
    // compute_border(circArray, radius)
    for (int i = 0; i < radius * 2 + 1; i++) {
        double tmp;
        if (i > radius)
            tmp = (i - radius) - 0.5;
        else if (i < radius)
            tmp = (radius - i) - 0.5;
        else
            tmp = 0.0;
        circArray[i] = int(std::sqrt(radius*radius - tmp*tmp));
    }
    // offset the circ pointer by radius so the range of the array
    //     is [-radius] to [radius]
    int * circ = circArray + radius;

    const uint8_t * bufArray[height + 2*radius];
    for (int i = 0; i < radius; i++) {
        bufArray[i] = &mask[0];
    }
    for (size_t i = 0; i < height; i++) {
        bufArray[i + radius] = &mask[i * width];
    }
    for (int i = 0; i < radius; i++) {
        bufArray[i + height + radius] = &mask[(height - 1) * width];
    }
    // offset the buf pointer
    const uint8_t ** buf = bufArray + radius;

    #pragma omp parallel
    {
        unique_ptr<uint8_t[]> buffer(new uint8_t[width * (radius + 1)]);
        unique_ptr<uint8_t *[]> maxArray;  // caches the largest values for each column
        maxArray.reset(new uint8_t *[width + 2 * radius]);
        for (int i = 0; i < radius; i++) {
            maxArray[i] = buffer.get();
        }
        for (size_t i = 0; i < width; i++) {
            maxArray[i + radius] = &buffer[(radius + 1) * i];
        }
        for (int i = 0; i < radius; i++) {
            maxArray[i + width + radius] = &buffer[(radius + 1) * (width - 1)];
        }
        // offset the max pointer
        uint8_t ** max = maxArray.get() + radius;

        #pragma omp for schedule(dynamic)
        for (size_t y = 0; y < height; y++) {
            uint8_t rowMax = 0;
            for (size_t x = 0; x < width; x++) { // compute max array
                max[x][0] = buf[y][x];
                for (int i = 1; i <= radius; i++) {
                    max[x][i] = std::max(std::max(max[x][i - 1], buf[y + i][x]), buf[y - i][x]);
                    rowMax = std::max(max[x][i], rowMax);
                }
            }

            uint8_t last_max = max[0][circ[-1]];
            int last_index = 1;
            for (size_t x = 0; x < width; x++) { // render scan line
                last_index--;
                if (last_index >= 0) {
                    if (last_max == rowMax) {
                        result(x, y) = rowMax;
                    } else {
                        last_max = 0;
                        for (int i = radius; i >= 0; i--)
                            if (last_max < max[x + i][circ[i]]) {
                                last_max = max[x + i][circ[i]];
                                last_index = i;
                            }
                        result(x, y) = last_max;
                    }
                } else {
                    last_index = radius;
                    last_max = max[x + radius][circ[radius]];

                    for (int i = radius - 1; i >= -radius; i--)
                        if (last_max < max[x + i][circ[i]]) {
                            last_max = max[x + i][circ[i]];
                            last_index = i;
                        }
                    result(x, y) = last_max;
                }
            }
        }
    }

    return result;
}
#endif

void ImageStack::correctHotPixels(const RawParameters & params, float sigma) {
    if (sigma <= 0.0f || images.size() < 2) return;

    const int numImages = (int)images.size();
    const int margin = 2;
    const int w = (int)width;
    const int h = (int)height;

    // Candidate neighbor offsets — distance 2 in cardinal and diagonal directions
    static const int offsets[][2] = {
        {-2, 0}, {2, 0}, {0, -2}, {0, 2},
        {-2, -2}, {-2, 2}, {2, -2}, {2, 2}
    };
    static const int numOffsets = 8;

    size_t corrected = 0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:corrected)
    for (int y = margin; y < h - margin; ++y) {
        std::vector<uint16_t> neighbors(numOffsets);
        std::vector<double> ratios(numImages);
        std::vector<double> absDevs(numImages);

        for (int x = margin; x < w - margin; ++x) {
            uint8_t color = params.FC(x, y);

            // Find same-color neighbors
            int nCount = 0;
            for (int k = 0; k < numOffsets; ++k) {
                int nx = x + offsets[k][0];
                int ny = y + offsets[k][1];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h &&
                    params.FC(nx, ny) == color) {
                    neighbors[nCount++] = k;
                }
            }
            if (nCount < 2) continue;

            // Collect ratio = pixel / localMedian for each non-saturated exposure
            int rCount = 0;
            for (int e = 0; e < numImages; ++e) {
                if (!images[e].contains(x, y)) continue;
                uint16_t val = images[e](x, y);
                if (val < 1 || images[e].isSaturated(val)) continue;

                // Compute local median of same-color neighbors in this exposure
                std::vector<uint16_t> nvals(nCount);
                int validN = 0;
                for (int n = 0; n < nCount; ++n) {
                    int nx = x + offsets[neighbors[n]][0];
                    int ny = y + offsets[neighbors[n]][1];
                    if (images[e].contains(nx, ny)) {
                        nvals[validN++] = images[e](nx, ny);
                    }
                }
                if (validN < 2) continue;

                std::nth_element(nvals.begin(), nvals.begin() + validN / 2,
                                 nvals.begin() + validN);
                double localMedian = nvals[validN / 2];
                // Skip low-signal exposures where ratios are noise-dominated
                if (localMedian < 100.0) continue;

                ratios[rCount++] = (double)val / localMedian;
            }

            if (rCount < 2) continue;

            // MAD outlier detection on ratios
            std::vector<double> sorted(ratios.begin(), ratios.begin() + rCount);
            std::nth_element(sorted.begin(), sorted.begin() + rCount / 2, sorted.end());
            double medianRatio = sorted[rCount / 2];

            for (int i = 0; i < rCount; i++)
                absDevs[i] = std::abs(ratios[i] - medianRatio);
            std::nth_element(absDevs.begin(), absDevs.begin() + rCount / 2,
                             absDevs.begin() + rCount);
            double mad = absDevs[rCount / 2] * 1.4826;

            if (mad <= 0.0) continue;
            double threshold = sigma * mad;

            // Check if ANY ratio is a significant outlier — if so, this pixel is hot.
            // Require both: exceeds MAD threshold AND deviates by >50% from median ratio.
            // The absolute ratio check prevents false positives from tiny MAD values.
            bool isHot = false;
            for (int i = 0; i < rCount; i++) {
                if (std::abs(ratios[i] - medianRatio) > threshold &&
                    std::abs(ratios[i] / medianRatio - 1.0) > 0.5) {
                    isHot = true;
                    break;
                }
            }
            if (!isHot) continue;

            // Replace this pixel in ALL exposures with same-color neighbor median
            for (int e = 0; e < numImages; ++e) {
                if (!images[e].contains(x, y)) continue;

                std::vector<uint16_t> nvals(nCount);
                int validN = 0;
                for (int n = 0; n < nCount; ++n) {
                    int nx = x + offsets[neighbors[n]][0];
                    int ny = y + offsets[neighbors[n]][1];
                    if (images[e].contains(nx, ny)) {
                        nvals[validN++] = images[e](nx, ny);
                    }
                }
                if (validN > 0) {
                    std::nth_element(nvals.begin(), nvals.begin() + validN / 2,
                                     nvals.begin() + validN);
                    images[e](x, y) = nvals[validN / 2];
                }
            }
            corrected++;
        }
    }
    Log::progress("Hot pixel correction: ", corrected, " pixels corrected (sigma=", sigma, ")");
}


// Estimate noise model parameters for DNG NoiseProfile tag (51041).
// For each channel c: variance = S[c] * signal + O[c], with signal normalized to [0,1].
// S = shot noise coefficient (Poisson), O = read noise (estimated from masked pixels).
// Both scaled by 1/numImages for the merged result.
static void estimateNoiseProfile(const std::vector<Image> & images,
                                  const RawParameters & params,
                                  int numImages, double noiseProfile[8]) {
    double range = static_cast<double>(params.max);
    if (range <= 0.0) return;

    for (int c = 0; c < params.colors; ++c) {
        // Shot noise coefficient: 1 / (max - cblack[c])
        double channelRange = range - params.cblack[c];
        if (channelRange <= 0.0) channelRange = range;
        double S = 1.0 / channelRange;

        // Read noise: estimate from masked/black pixel regions across all exposures
        double sumSq = 0.0;
        double sum = 0.0;
        size_t count = 0;

        for (const auto & img : images) {
            // Top margin pixels
            for (size_t y = 0; y < params.topMargin && y < img.getHeight(); ++y) {
                for (size_t x = 0; x < img.getWidth(); ++x) {
                    if (params.FC(x, y) == c && img.contains(x, y)) {
                        double v = static_cast<double>(img(x, y)) - params.cblack[c];
                        sum += v;
                        sumSq += v * v;
                        ++count;
                    }
                }
            }
            // Left margin pixels (below top margin to avoid double-counting)
            for (size_t y = params.topMargin; y < img.getHeight(); ++y) {
                for (size_t x = 0; x < params.leftMargin && x < img.getWidth(); ++x) {
                    if (params.FC(x, y) == c && img.contains(x, y)) {
                        double v = static_cast<double>(img(x, y)) - params.cblack[c];
                        sum += v;
                        sumSq += v * v;
                        ++count;
                    }
                }
            }
        }

        double O;
        if (count > 1) {
            O = (sumSq - sum * sum / count) / (count - 1) / (channelRange * channelRange);
        } else {
            O = 1e-6; // Conservative fallback
        }

        // Scale for merge: variance scales as 1/N
        noiseProfile[c * 2]     = S / numImages;
        noiseProfile[c * 2 + 1] = O / numImages;
    }
}


// CFA-aware bilinear interpolation: read a pixel with fractional offset,
// interpolating only from same-color neighbors on the CFA grid.
// For Bayer, same-color neighbors are at distance 2; for X-Trans, distance 6.
static inline double interpolateCFA(const Image & img, int x, int y,
                                     double fdx, double fdy,
                                     int step, size_t width, size_t height) {
    // Skip if fractional offset is negligible
    if (std::abs(fdx) < 0.1 && std::abs(fdy) < 0.1)
        return static_cast<double>(img(x, y));

    // Same-color neighbor positions (distance = step)
    int x0 = x, y0 = y;
    int x1 = (fdx >= 0) ? x + step : x - step;
    int y1 = (fdy >= 0) ? y + step : y - step;

    // Fractional weights (normalized to step distance)
    double wx = std::abs(fdx) / step;
    double wy = std::abs(fdy) / step;

    // Bounds check: fall back to nearest if any neighbor is out of bounds
    bool x1ok = x1 >= 0 && (size_t)x1 < width && img.contains(x1, y0);
    bool y1ok = y1 >= 0 && (size_t)y1 < height && img.contains(x0, y1);
    bool xyok = x1ok && y1ok && img.contains(x1, y1);

    double v00 = img(x0, y0);
    double v10 = x1ok ? (double)img(x1, y0) : v00;
    double v01 = y1ok ? (double)img(x0, y1) : v00;
    double v11 = xyok ? (double)img(x1, y1) : v00;

    double result = v00 * (1.0 - wx) * (1.0 - wy)
                  + v10 * wx * (1.0 - wy)
                  + v01 * (1.0 - wx) * wy
                  + v11 * wx * wy;

    return result;
}


ComposeResult ImageStack::compose(const RawParameters & params, int featherRadius, float deghostSigma, double clipPercentile, bool subPixelAlign) const {
    int imageMax = images.size() - 1;
    BoxBlur map(fattenMask(mask, featherRadius));
    measureTime("Blur", [&] () {
        map.blur(featherRadius);
    });
    Timer t("Compose");
    Array2D<float> dst(params.rawWidth, params.rawHeight);
    dst.displace(-(int)params.leftMargin, -(int)params.topMargin);
    dst.fillBorders(0.f);

    // Poisson-optimal merge: weight each exposure by 1/relativeExposure (∝ exposure time).
    // This is the MLE weight for Poisson noise — longer exposures captured more photons
    // and thus have lower relative noise, so they get higher weight.
    const int numImages = (int)images.size();
    std::vector<double> baseWeight(numImages);
    for (int k = 0; k < numImages; k++) {
        baseWeight[k] = 1.0 / images[k].getRelativeExposure();
    }

    // Per-channel saturation rolloff: each channel uses its own clipping threshold
    double satRolloffPerCh[4], satRolloffRangePerCh[4], satThreshPerCh[4];
    for (int c = 0; c < 4; ++c) {
        satThreshPerCh[c] = satThresholdPerChannel[c];
        satRolloffPerCh[c] = 0.9 * satThresholdPerChannel[c];
        satRolloffRangePerCh[c] = satThreshPerCh[c] - satRolloffPerCh[c];
    }

    // Bayer-block rolloff consistency: use the most conservative (minimum) threshold
    // across all channels for rolloff weight, preventing color fringe at exposure
    // transitions after demosaicing. Per-channel thresholds still used for hard rejection.
    bool useBlockRolloff = false;
    double blockThresh = 0, blockRolloff = 0, blockRange = 0;
    if (params.FC.getFilters() != 9) { // Bayer only, not X-Trans
        blockThresh = *std::min_element(satThreshPerCh, satThreshPerCh + 4);
        if (blockThresh < *std::max_element(satThreshPerCh, satThreshPerCh + 4)) {
            useBlockRolloff = true;
            blockRolloff = 0.9 * blockThresh;
            blockRange = blockThresh - blockRolloff;
            Log::debug("Bayer-block rolloff: min threshold=", blockThresh,
                       " rolloff=", blockRolloff);
        }
    }

    // CFA step for sub-pixel interpolation: distance to same-color neighbor
    const int cfaStep = (params.FC.getFilters() == 9) ? 6 : 2;
    if (subPixelAlign && numImages > 1) {
        Log::debug("Sub-pixel alignment enabled (CFA step=", cfaStep, ")");
    }

    const bool deghost = deghostSigma > 0.0f && numImages >= 3;
    if (deghost) {
        Log::debug("Ghost detection enabled: sigma=", deghostSigma);
    }

    // Estimate noise profile before compose for variance-based shadow weighting
    double noiseProfile[8] = {};
    estimateNoiseProfile(images, params, numImages, noiseProfile);

    // Pre-compute per-channel affine noise model coefficients (unscaled, per-exposure)
    // Sensor model: Var(Z) = a * Z + b, where Z is the raw ADU value
    double noiseA[4], noiseB[4];
    for (int c = 0; c < params.colors; ++c) {
        double channelRange = static_cast<double>(params.max) - params.cblack[c];
        if (channelRange <= 0.0) channelRange = static_cast<double>(params.max);
        noiseA[c] = noiseProfile[c * 2] * numImages * channelRange;
        noiseB[c] = noiseProfile[c * 2 + 1] * numImages * channelRange * channelRange;
    }

    // Shadow transition point: below this raw level, read noise dominates shot noise
    // and variance-based weighting improves SNR. Above it, Poisson weights are optimal.
    // Crossover is where shot noise = read noise: a * raw = b, so raw = b/a
    double shadowThresh[4], shadowBlendRange[4];
    for (int c = 0; c < 4; ++c) {
        if (noiseA[c] > 0.0) {
            shadowThresh[c] = noiseB[c] / noiseA[c];
            // Blend over 2x the crossover range for smooth transition
            shadowBlendRange[c] = shadowThresh[c];
            if (shadowBlendRange[c] < 1.0) shadowBlendRange[c] = 1.0;
        } else {
            shadowThresh[c] = 0.0;
            shadowBlendRange[c] = 1.0;
        }
    }

    // Pre-compute per-exposure relativeExposure for variance computation
    std::vector<double> relExp(numImages);
    for (int k = 0; k < numImages; k++) {
        relExp[k] = images[k].getRelativeExposure();
    }

    // Pre-compute spatial ghost confidence map for coherent deghosting
    Array2D<float> ghostMap;
    if (deghost) {
        ghostMap = Array2D<float>(width, height);
        #pragma omp parallel for schedule(dynamic,16)
        for (size_t y = 0; y < height; ++y) {
            std::vector<double> tmpRad(numImages), tmpDev(numImages), tmpSort(numImages);
            for (size_t x = 0; x < width; ++x) {
                int ch = params.FC(x, y);
                int nv = 0;
                for (int k = 0; k <= imageMax; k++) {
                    if (!images[k].contains(x, y)) continue;
                    uint16_t raw = images[k](x, y);
                    if (raw < 1 || raw >= satThreshPerCh[ch]) continue;
                    tmpRad[nv++] = images[k].exposureAt(x, y);
                }
                if (nv < 3) { ghostMap(x, y) = 0.0f; continue; }

                // MAD-based ghost confidence
                tmpSort.assign(tmpRad.begin(), tmpRad.begin() + nv);
                std::nth_element(tmpSort.begin(), tmpSort.begin() + nv/2, tmpSort.end());
                double median = tmpSort[nv/2];
                for (int i = 0; i < nv; i++)
                    tmpDev[i] = std::abs(tmpRad[i] - median);
                std::nth_element(tmpDev.begin(), tmpDev.begin() + nv/2, tmpDev.begin() + nv);
                double mad = tmpDev[nv/2] * 1.4826;

                double maxDev = *std::max_element(tmpDev.begin(), tmpDev.begin() + nv);
                float confidence = (mad > 0.0) ? static_cast<float>(maxDev / (deghostSigma * mad)) : 0.0f;
                ghostMap(x, y) = std::min(confidence, 1.0f);
            }
        }

        // Spatial filter: separable 3x3 box blur for ghost map coherence
        Array2D<float> tmpMap(width, height);
        #pragma omp parallel for
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float sum = ghostMap(x, y);
                int count = 1;
                if (x > 0) { sum += ghostMap(x-1, y); count++; }
                if (x + 1 < width) { sum += ghostMap(x+1, y); count++; }
                tmpMap(x, y) = sum / count;
            }
        }
        #pragma omp parallel for
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float sum = tmpMap(x, y);
                int count = 1;
                if (y > 0) { sum += tmpMap(x, y-1); count++; }
                if (y + 1 < height) { sum += tmpMap(x, y+1); count++; }
                ghostMap(x, y) = sum / count;
            }
        }
        Log::debug("Ghost map computed with spatial coherence filter");
    }

    float maxVal = 0.0;
    #pragma omp parallel for schedule(dynamic,16) reduction(max:maxVal)
    for (size_t y = 0; y < height; ++y) {
        std::vector<double> radiances(numImages);
        std::vector<double> weights(numImages);
        std::vector<double> absDevs(numImages);
        std::vector<double> sorted(numImages);

        for (size_t x = 0; x < width; ++x) {
            int numValid = 0;
            int ch = params.FC(x, y);

            // Collect valid exposures with their radiances and weights
            for (int k = 0; k <= imageMax; k++) {
                if (!images[k].contains(x, y)) continue;

                double raw;
                if (subPixelAlign && (images[k].getFracDx() != 0.0 || images[k].getFracDy() != 0.0)) {
                    raw = interpolateCFA(images[k], x, y,
                                         images[k].getFracDx(), images[k].getFracDy(),
                                         cfaStep, width, height);
                } else {
                    raw = static_cast<double>(images[k](x, y));
                }

                if (raw < 1) continue;
                if (raw >= satThreshPerCh[ch]) continue;

                // Blend between Poisson weight (1/relExp) and variance weight
                // (1/Var(radiance)) based on signal level. In shadows where read
                // noise dominates, variance weighting properly accounts for the
                // constant noise floor. In mid-tones/highlights, Poisson weight
                // is optimal and avoids response function edge cases.
                double w;
                if (raw < shadowThresh[ch] + shadowBlendRange[ch]) {
                    // Variance weight: w = 1 / (relExp^2 * (a*raw + b))
                    double pixelVar = noiseA[ch] * raw + noiseB[ch];
                    double variance = relExp[k] * relExp[k] * pixelVar;
                    double varWeight = 1.0 / variance;

                    if (raw <= shadowThresh[ch]) {
                        w = varWeight;
                    } else {
                        // Smooth blend from variance to Poisson weight
                        double t = (raw - shadowThresh[ch]) / shadowBlendRange[ch];
                        w = varWeight * (1.0 - t) + baseWeight[k] * t;
                    }
                } else {
                    w = baseWeight[k];
                }
                if (useBlockRolloff) {
                    if (raw >= blockThresh) {
                        w = 0.0;
                    } else if (raw > blockRolloff) {
                        double t = (blockThresh - raw) / blockRange;
                        w *= t * t;
                    }
                } else {
                    if (raw > satRolloffPerCh[ch]) {
                        double t = (satThreshPerCh[ch] - raw) / satRolloffRangePerCh[ch];
                        w *= t * t;
                    }
                }

                double radiance = (subPixelAlign && (images[k].getFracDx() != 0.0 || images[k].getFracDy() != 0.0))
                    ? images[k].exposureForRaw(raw)
                    : images[k].exposureAt(x, y);
                if (radiance <= 0.0) continue;

                radiances[numValid] = radiance;
                weights[numValid] = w;
                numValid++;
            }

            // Spatially coherent ghost detection: use pre-computed ghost map to
            // modulate per-pixel deghosting strength, preventing salt-and-pepper
            // artifacts at ghost boundaries
            if (deghost && numValid >= 3 && ghostMap(x, y) > 0.1f) {
                double ghostStrength = static_cast<double>(ghostMap(x, y));

                // Find median radiance (partial sort)
                sorted.assign(radiances.begin(), radiances.begin() + numValid);
                std::nth_element(sorted.begin(), sorted.begin() + numValid / 2, sorted.end());
                double median = sorted[numValid / 2];

                // Compute MAD
                for (int i = 0; i < numValid; i++)
                    absDevs[i] = std::abs(radiances[i] - median);
                std::nth_element(absDevs.begin(), absDevs.begin() + numValid / 2, absDevs.begin() + numValid);
                double mad = absDevs[numValid / 2] * 1.4826; // MAD to sigma

                // Soft Gaussian deghosting modulated by spatial ghost confidence.
                // ghostStrength=1.0: full deghosting (original behavior).
                // ghostStrength=0.0: no deghosting. Smooth spatial transitions.
                if (mad > 0.0) {
                    double threshold = deghostSigma * mad;
                    double invThreshSq = 1.0 / (threshold * threshold);
                    for (int i = 0; i < numValid; i++) {
                        double dev = std::abs(radiances[i] - median);
                        double gaussFactor = std::exp(-0.5 * dev * dev * invThreshSq);
                        weights[i] *= (1.0 - ghostStrength) + ghostStrength * gaussFactor;
                    }
                }
            }

            // Weighted merge of surviving exposures
            double weightedSum = 0.0;
            double totalWeight = 0.0;
            for (int i = 0; i < numValid; i++) {
                weightedSum += weights[i] * radiances[i];
                totalWeight += weights[i];
            }

            double v;
            if (totalWeight > 0.0) {
                v = weightedSum / totalWeight;
            } else {
                // All exposures saturated or unavailable — fall back to mask-based selection
                double p = map(x,y);
                p = p < 0.0 ? 0.0 : p;
                int j = (int)p;
                if (j > imageMax) j = imageMax;
                if (images[j].contains(x, y)) {
                    v = images[j].exposureAt(x, y);
                    if (j < (int)origMask(x,y)) {
                        v /= params.whiteMultAt(x, y);
                    }
                } else {
                    v = 0.0;
                }
            }

            dst(x, y) = v;
            if (v > maxVal) {
                maxVal = v;
            }
        }
    }

    dst.displace(params.leftMargin, params.topMargin);

    // Determine normalization value: percentile-based or global max
    float normVal = maxVal;
    if (clipPercentile < 100.0 && maxVal > 0.0f) {
        // Collect active-area pixel values for percentile computation
        std::vector<float> activePixels;
        activePixels.reserve(params.width * params.height);
        for (size_t y = params.topMargin; y < params.topMargin + params.height; ++y) {
            for (size_t x = params.leftMargin; x < params.leftMargin + params.width; ++x) {
                float v = dst(x, y);
                if (v > 0.0f) {
                    activePixels.push_back(v);
                }
            }
        }
        if (!activePixels.empty()) {
            size_t idx = static_cast<size_t>(
                (clipPercentile / 100.0) * (activePixels.size() - 1));
            std::nth_element(activePixels.begin(),
                             activePixels.begin() + idx,
                             activePixels.end());
            float pctVal = activePixels[idx];
            // Safety: if percentile value is implausibly low, fall back to maxVal
            if (pctVal >= 0.01f * maxVal) {
                normVal = pctVal;
            }
        }
    }

    // Scale to params.max and recover the black levels
    float mult = (params.max - params.maxBlack) / normVal;
    float clampMax = static_cast<float>(params.max - params.maxBlack);
    #pragma omp parallel for
    for (size_t y = 0; y < params.rawHeight; ++y) {
        for (size_t x = 0; x < params.rawWidth; ++x) {
            float v = dst(x, y) * mult;
            if (v > clampMax) v = clampMax;
            dst(x, y) = v + params.blackAt(x - params.leftMargin, y - params.topMargin);
        }
    }

    // Compute BaselineExposure from median of log-luminance (robust to HDR skew)
    double baselineExposureEV = 0.0;
    {
        // Pass 1: find log-luminance range and pixel count
        double logMin = 1e30, logMax = -1e30;
        size_t totalPixels = 0;
        #pragma omp parallel for reduction(min:logMin) reduction(max:logMax) reduction(+:totalPixels)
        for (size_t y = params.topMargin; y < params.topMargin + params.height; ++y) {
            for (size_t x = params.leftMargin; x < params.leftMargin + params.width; ++x) {
                float v = dst(x, y) - params.blackAt(x - params.leftMargin, y - params.topMargin);
                if (v > 0.0f) {
                    double lv = std::log(static_cast<double>(v));
                    if (lv < logMin) logMin = lv;
                    if (lv > logMax) logMax = lv;
                    totalPixels++;
                }
            }
        }

        if (totalPixels > 0 && logMax > logMin) {
            // Pass 2: histogram-based median via per-thread local histograms
            const int numBins = 10000;
            double binScale = (numBins - 1) / (logMax - logMin);
            std::vector<size_t> histogram(numBins, 0);

            #pragma omp parallel
            {
                std::vector<size_t> localHist(numBins, 0);
                #pragma omp for nowait
                for (size_t y = params.topMargin; y < params.topMargin + params.height; ++y) {
                    for (size_t x = params.leftMargin; x < params.leftMargin + params.width; ++x) {
                        float v = dst(x, y) - params.blackAt(x - params.leftMargin, y - params.topMargin);
                        if (v > 0.0f) {
                            double lv = std::log(static_cast<double>(v));
                            int bin = static_cast<int>((lv - logMin) * binScale);
                            if (bin < 0) bin = 0;
                            if (bin >= numBins) bin = numBins - 1;
                            localHist[bin]++;
                        }
                    }
                }
                #pragma omp critical
                {
                    for (int i = 0; i < numBins; i++)
                        histogram[i] += localHist[i];
                }
            }

            // Find 50th-percentile bin with linear interpolation
            size_t target = totalPixels / 2;
            size_t cumulative = 0;
            int medianBin = 0;
            for (int i = 0; i < numBins; i++) {
                cumulative += histogram[i];
                if (cumulative >= target) {
                    medianBin = i;
                    break;
                }
            }
            // Sub-bin interpolation: fraction within the median bin
            size_t prevCumulative = cumulative - histogram[medianBin];
            double frac = (histogram[medianBin] > 0)
                ? static_cast<double>(target - prevCumulative) / histogram[medianBin]
                : 0.5;
            double medianLog = logMin + (medianBin + frac) / binScale;
            double medianValue = std::exp(medianLog);

            double range = static_cast<double>(params.max - params.maxBlack);
            if (medianValue > 0.0 && range > 0.0) {
                baselineExposureEV = std::log2(0.18 * range / medianValue);
                if (baselineExposureEV < -5.0) baselineExposureEV = -5.0;
                if (baselineExposureEV > 5.0) baselineExposureEV = 5.0;
            }
        }
    }
    Log::debug("Normalization: clipPercentile=", clipPercentile,
               " normVal=", normVal, " maxVal=", maxVal,
               " BaselineExposure=", baselineExposureEV, " EV");

    ComposeResult result;
    result.image = std::move(dst);
    result.baselineExposureEV = baselineExposureEV;
    result.numImages = numImages;

    // Copy noise profile (already estimated before compose loop)
    std::copy(noiseProfile, noiseProfile + 8, result.noiseProfile);
    Log::debug("NoiseProfile: S0=", result.noiseProfile[0], " O0=", result.noiseProfile[1],
               " S1=", result.noiseProfile[2], " O1=", result.noiseProfile[3]);

    return result;
}
