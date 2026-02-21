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

#ifndef _RESIZER_HPP_
#define _RESIZER_HPP_

#include "Array2D.hpp"
#include "CFAPattern.hpp"

namespace hdrmerge {

struct ResizeResult {
    Array2D<float> image;
    size_t rawWidth, rawHeight;
    size_t width, height;
    size_t topMargin, leftMargin;
};

ResizeResult resizeCFA(Array2D<float> && input,
    size_t rawWidth, size_t rawHeight,
    size_t width, size_t height,
    size_t topMargin, size_t leftMargin,
    int targetLongEdge, const CFAPattern & cfa);

} // namespace hdrmerge

#endif // _RESIZER_HPP_
