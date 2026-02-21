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

#ifndef _RAWPARAMETERS_H_
#define _RAWPARAMETERS_H_

#include <QString>
#include "Array2D.hpp"
#include "CFAPattern.hpp"

class LibRaw;

namespace hdrmerge {

class RawParameters {
public:
    RawParameters();
    RawParameters(const QString & f) : RawParameters() {
        fileName = f;
    }
    virtual ~RawParameters() {}

    void fromLibRaw(LibRaw & rawData);

    bool isSameFormat(const RawParameters & r) const {
        return width == r.width && height == r.height && FC == r.FC && cdesc == r.cdesc;
    }
    double logExp() const;
    void dumpInfo() const;
    uint16_t blackAt(int x, int y) const {
        return cblack[FC(x, y)];
    }
    bool hasBlack() const {
        return black || cblack[0] || cblack[1] || cblack[2] || cblack[3];
    }
    float whiteMultAt(int x, int y) const {
        return camMul[FC(x, y)];
    }
    void adjustWhite(const Array2D<uint16_t> & image);
    void autoWB(const Array2D<uint16_t> & image);
    bool canAlign() const { return FC.canAlign(); }

    QString fileName;
    size_t width, height;
    size_t rawWidth, rawHeight, topMargin, leftMargin;
    std::string cdesc;
    CFAPattern FC;
    uint16_t max;
    uint16_t black;
    uint16_t maxBlack;
    uint16_t cblack[4];
    float preMul[4];
    float camMul[4];
    float camXyz[4][3];
    float rgbCam[3][4];
    float isoSpeed;
    float shutter;
    float aperture;
    std::string maker, model, description, dateTime;
    int colors;
    int flip;
    int tiffOrientation;

    // DNG dual-illuminant passthrough (only populated for DNG inputs)
    bool hasDualIlluminant = false;
    uint16_t illuminant1 = 0, illuminant2 = 0;
    float colorMatrix2[4][3] = {};
    float forwardMatrix1Dng[3][4] = {};
    float forwardMatrix2[3][4] = {};
    float calibration1[4][4] = {};
    float calibration2[4][4] = {};
    bool hasForwardMatrix1Dng = false;

    // AsShotNeutral from DNG (more accurate than computed 1/camMul)
    float asShotNeutral[4] = {};
    bool hasAsShotNeutral = false;

private:
    void adjustBlack();
    void calculateCamXyz();
    void loadCamXyzFromDng();
    void camXyzFromRgbCam();
};

} // namespace hdrmerge

#endif // _RAWPARAMETERS_H_
