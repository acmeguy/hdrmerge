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
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#endif
#include "BoxBlur.hpp"

namespace hdrmerge {

void BoxBlur::blur(size_t radius) {
    // From http://blog.ivank.net/fastest-gaussian-blur.html
    tmp.reset(new float[width*height]);
    size_t hr = std::round(radius*0.39);
    boxBlur(hr);
    boxBlur(hr);
    boxBlur(hr);
    tmp.reset();
}


void BoxBlur::boxBlur(size_t radius) {
    boxBlurH(radius);
    data.swap(tmp);
    boxBlurT(radius);
    data.swap(tmp);
}


void BoxBlur::boxBlurH(size_t r) {
    float iarr = 1.0 / (r+r+1);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < height; ++i) {
        size_t ti = i * width, li = ti, ri = ti + r;
        float val = data[li] * (r + 1);
        for (size_t j = 0; j < r; ++j) {
            val += data[li + j];
        }
        for (size_t j = 0; j <= r; ++j) {
            val += data[ri++] - data[li];
            tmp[ti++] = val*iarr;
        }
        for (size_t j = r + 1; j < width - r; ++j) {
            val += data[ri++] - data[li++];
            tmp[ti++] = val*iarr;
        }
        for (size_t j = width - r; j < width; ++j) {
            val += data[ri - 1] - data[li++];
            tmp[ti++] = val*iarr;
        }
    }
}


void BoxBlur::boxBlurT(size_t r) {
    float iarr = 1.0 / (r+r+1);
    const int numCols = 8; // process numCols columns at once for better usage of L1 cpu cache
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t viarr = vdupq_n_f32(iarr);
    #pragma omp parallel for schedule(dynamic,4)
    for (size_t i = 0; i < width-numCols+1; i+=numCols) {
        size_t ti = i, li = ti, ri = ti + r*width;
        float rpf = (float)(r + 1);
        float32x4_t val_lo = vmulq_n_f32(vld1q_f32(&data[li]), rpf);
        float32x4_t val_hi = vmulq_n_f32(vld1q_f32(&data[li+4]), rpf);
        for (size_t j = 0; j < r; ++j) {
            val_lo = vaddq_f32(val_lo, vld1q_f32(&data[li + j*width]));
            val_hi = vaddq_f32(val_hi, vld1q_f32(&data[li + j*width + 4]));
        }
        for (size_t j = 0; j <= r; ++j) {
            val_lo = vaddq_f32(val_lo, vsubq_f32(vld1q_f32(&data[ri]), vld1q_f32(&data[li])));
            val_hi = vaddq_f32(val_hi, vsubq_f32(vld1q_f32(&data[ri+4]), vld1q_f32(&data[li+4])));
            vst1q_f32(&tmp[ti], vmulq_f32(val_lo, viarr));
            vst1q_f32(&tmp[ti+4], vmulq_f32(val_hi, viarr));
            ri += width;
            ti += width;
        }
        for (size_t j = r + 1; j < height - r; ++j) {
            val_lo = vaddq_f32(val_lo, vsubq_f32(vld1q_f32(&data[ri]), vld1q_f32(&data[li])));
            val_hi = vaddq_f32(val_hi, vsubq_f32(vld1q_f32(&data[ri+4]), vld1q_f32(&data[li+4])));
            vst1q_f32(&tmp[ti], vmulq_f32(val_lo, viarr));
            vst1q_f32(&tmp[ti+4], vmulq_f32(val_hi, viarr));
            li += width;
            ri += width;
            ti += width;
        }
        for (size_t j = height - r; j < height; ++j) {
            val_lo = vaddq_f32(val_lo, vsubq_f32(vld1q_f32(&data[ri - width]), vld1q_f32(&data[li])));
            val_hi = vaddq_f32(val_hi, vsubq_f32(vld1q_f32(&data[ri - width + 4]), vld1q_f32(&data[li+4])));
            vst1q_f32(&tmp[ti], vmulq_f32(val_lo, viarr));
            vst1q_f32(&tmp[ti+4], vmulq_f32(val_hi, viarr));
            li += width;
            ti += width;
        }
    }
#else
    #pragma omp parallel for schedule(dynamic,4)
    for (size_t i = 0; i < width-numCols+1; i+=numCols) {
        size_t ti = i, li = ti, ri = ti + r*width;
        float val[numCols];
        for(size_t k=0;k<numCols;++k)
            val[k] = data[li+k] * (r + 1);
        for(size_t k=0;k<numCols;++k)
            for (size_t j = 0; j < r; ++j) {
                val[k] += data[li + j*width + k];
            }
        for (size_t j = 0; j <= r; ++j) {
            for(size_t k=0;k<numCols;++k) {
                val[k] += data[ri+k] - data[li+k];
                tmp[ti+k] = val[k]*iarr;
            }
            ri += width;
            ti += width;
        }
        for (size_t j = r + 1; j < height - r; ++j) {
            for(size_t k=0;k<numCols;++k) {
                val[k] += data[ri+k] - data[li+k];
                tmp[ti+k] = val[k]*iarr;
            }
            li += width;
            ri += width;
            ti += width;
        }
        for (size_t j = height - r; j < height; ++j) {
            for(size_t k=0;k<numCols;++k) {
                val[k] += data[ri - width + k] - data[li+ k];
                tmp[ti+k] = val[k]*iarr;
            }
            li += width;
            ti += width;
        }
    }
#endif
    // process the remaining columns
    for (size_t i = width - (width%numCols); i < width; ++i) {
        size_t ti = i, li = ti, ri = ti + r*width;
        float val = data[li] * (r + 1);
        for (size_t j = 0; j < r; ++j) {
            val += data[li + j*width];
        }
        for (size_t j = 0; j <= r; ++j) {
            val += data[ri] - data[li];
            tmp[ti] = val*iarr;
            ri += width;
            ti += width;
        }
        for (size_t j = r + 1; j < height - r; ++j) {
            val += data[ri] - data[li];
            tmp[ti] = val*iarr;
            li += width;
            ri += width;
            ti += width;
        }
        for (size_t j = height - r; j < height; ++j) {
            val += data[ri - width] - data[li];
            tmp[ti] = val*iarr;
            li += width;
            ti += width;
        }
    }

}


} // namespace hdrmerge
