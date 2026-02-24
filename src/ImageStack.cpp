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

#ifdef HAVE_OPENCV
    #include <opencv2/core.hpp>
    #include <opencv2/features2d.hpp>
    #include <opencv2/calib3d.hpp>
    #include <opencv2/imgproc.hpp>
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

#ifdef HAVE_OPENCV
// Build a grayscale 8-bit preview from raw Bayer data via 2x2 averaging.
// This avoids the CFA artifact problem when detecting features on raw mosaic data.
static cv::Mat bayerToGray8(const Image & img) {
    size_t w = img.getWidth(), h = img.getHeight();
    size_t gw = w / 2, gh = h / 2;
    cv::Mat gray(gh, gw, CV_8UC1);
    for (size_t y = 0; y < gh; ++y) {
        for (size_t x = 0; x < gw; ++x) {
            size_t bx = x * 2, by = y * 2;
            uint32_t sum = (uint32_t)img(bx, by) + img(bx + 1, by)
                         + img(bx, by + 1) + img(bx + 1, by + 1);
            // Scale 14-bit average to 8-bit
            gray.at<uint8_t>(y, x) = (uint8_t)std::min(255u, sum >> 8);
        }
    }
    return gray;
}

// Attempt feature-based alignment of img against ref. Returns true on success,
// filling dx, dy (integer) and fdx, fdy (fractional sub-pixel).
// On failure, returns false and the caller should fall back to MTB.
static bool featureAlign(const Image & img, const Image & ref,
                          int & outDx, int & outDy, double & outFdx, double & outFdy) {
    // Build grayscale previews (half resolution via 2x2 Bayer averaging)
    cv::Mat grayImg = bayerToGray8(img);
    cv::Mat grayRef = bayerToGray8(ref);

    // Apply CLAHE to normalize across exposure differences
    auto clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
    clahe->apply(grayImg, grayImg);
    clahe->apply(grayRef, grayRef);

    // Detect features with AKAZE
    auto akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> kpImg, kpRef;
    cv::Mat descImg, descRef;
    akaze->detectAndCompute(grayImg, cv::noArray(), kpImg, descImg);
    akaze->detectAndCompute(grayRef, cv::noArray(), kpRef, descRef);

    Log::debug("Features: img=", kpImg.size(), " ref=", kpRef.size());

    // ORB fallback if AKAZE finds too few features
    if (kpImg.size() < 64 || kpRef.size() < 64) {
        Log::debug("AKAZE insufficient, trying ORB fallback");
        auto orb = cv::ORB::create(2000);
        kpImg.clear(); kpRef.clear();
        orb->detectAndCompute(grayImg, cv::noArray(), kpImg, descImg);
        orb->detectAndCompute(grayRef, cv::noArray(), kpRef, descRef);
        Log::debug("ORB features: img=", kpImg.size(), " ref=", kpRef.size());
    }

    if (kpImg.size() < 16 || kpRef.size() < 16) {
        Log::debug("Feature alignment: too few keypoints, falling back to MTB");
        return false;
    }

    // KNN matching with Lowe ratio test (0.75) + cross-check
    auto matcher = cv::BFMatcher::create(
        descImg.type() == CV_8U ? cv::NORM_HAMMING : cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descImg, descRef, knnMatches, 2);

    // Ratio test
    std::vector<cv::DMatch> goodMatches;
    for (auto & m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) {
            goodMatches.push_back(m[0]);
        }
    }

    // Reverse match for cross-check
    std::vector<std::vector<cv::DMatch>> knnReverse;
    matcher->knnMatch(descRef, descImg, knnReverse, 2);
    std::vector<int> reverseMap(kpRef.size(), -1);
    for (auto & m : knnReverse) {
        if (m.size() >= 2 && m[0].distance < 0.75f * m[1].distance) {
            reverseMap[m[0].queryIdx] = m[0].trainIdx;
        }
    }

    // Keep only mutual nearest neighbors
    std::vector<cv::DMatch> mutualMatches;
    for (auto & m : goodMatches) {
        if (reverseMap[m.trainIdx] == m.queryIdx) {
            mutualMatches.push_back(m);
        }
    }

    Log::debug("Matches: ratio-test=", goodMatches.size(),
               " mutual=", mutualMatches.size());

    if (mutualMatches.size() < 16) {
        Log::debug("Feature alignment: too few matches (", mutualMatches.size(),
                 "), falling back to MTB");
        return false;
    }

    // Build point arrays (in half-resolution coordinates)
    std::vector<cv::Point2f> ptsImg, ptsRef;
    for (auto & m : mutualMatches) {
        ptsImg.push_back(kpImg[m.queryIdx].pt);
        ptsRef.push_back(kpRef[m.trainIdx].pt);
    }

    // Progressive geometry estimation: 4 DOF -> 6 DOF -> 8 DOF
    // Use MAGSAC++ (USAC_MAGSAC) for robust estimation
    cv::Mat transform;
    cv::Mat inliers;
    double reprErr = 1e9;
    int modelDOF = 0;
    std::string modelName;

    // Try 4 DOF (translation + rotation + uniform scale)
    // Note: estimateAffinePartial2D only supports RANSAC/LMEDS, not USAC_MAGSAC
    transform = cv::estimateAffinePartial2D(ptsImg, ptsRef, inliers,
                                             cv::RANSAC, 3.0, 2000, 0.99);
    if (!transform.empty()) {
        // Compute reprojection error on inliers
        double sumErr = 0;
        int nInliers = 0;
        for (size_t i = 0; i < ptsImg.size(); ++i) {
            if (inliers.at<uint8_t>(i)) {
                double px = transform.at<double>(0, 0) * ptsImg[i].x
                          + transform.at<double>(0, 1) * ptsImg[i].y
                          + transform.at<double>(0, 2);
                double py = transform.at<double>(1, 0) * ptsImg[i].x
                          + transform.at<double>(1, 1) * ptsImg[i].y
                          + transform.at<double>(1, 2);
                double dx = px - ptsRef[i].x;
                double dy = py - ptsRef[i].y;
                sumErr += dx * dx + dy * dy;
                nInliers++;
            }
        }
        reprErr = (nInliers > 0) ? std::sqrt(sumErr / nInliers) : 1e9;
        modelDOF = 4;
        modelName = "AffinePartial2D";

        double inlierRatio = (double)nInliers / ptsImg.size();
        Log::debug("4DOF: reprErr=", reprErr, "px inliers=", nInliers,
                   " ratio=", inlierRatio);

        // Escalate to 6 DOF if poor fit
        if (reprErr > 2.0 || inlierRatio < 0.45) {
            cv::Mat t6 = cv::estimateAffine2D(ptsImg, ptsRef, inliers,
                                               cv::RANSAC, 3.0, 2000, 0.99);
            if (!t6.empty()) {
                double sumErr6 = 0;
                int nInliers6 = 0;
                for (size_t i = 0; i < ptsImg.size(); ++i) {
                    if (inliers.at<uint8_t>(i)) {
                        double px = t6.at<double>(0, 0) * ptsImg[i].x
                                  + t6.at<double>(0, 1) * ptsImg[i].y
                                  + t6.at<double>(0, 2);
                        double py = t6.at<double>(1, 0) * ptsImg[i].x
                                  + t6.at<double>(1, 1) * ptsImg[i].y
                                  + t6.at<double>(1, 2);
                        double dx = px - ptsRef[i].x;
                        double dy = py - ptsRef[i].y;
                        sumErr6 += dx * dx + dy * dy;
                        nInliers6++;
                    }
                }
                double reprErr6 = (nInliers6 > 0) ? std::sqrt(sumErr6 / nInliers6) : 1e9;
                Log::debug("6DOF: reprErr=", reprErr6, "px inliers=", nInliers6);
                if (reprErr6 < reprErr) {
                    transform = t6;
                    reprErr = reprErr6;
                    nInliers = nInliers6;
                    modelDOF = 6;
                    modelName = "Affine2D";
                }
            }

            // Escalate to 8 DOF if still poor
            inlierRatio = (double)nInliers / ptsImg.size();
            if (reprErr > 2.0 || inlierRatio < 0.45) {
                cv::Mat H = cv::findHomography(ptsImg, ptsRef,
                                                cv::USAC_MAGSAC, 3.0, inliers, 2000, 0.99);
                if (!H.empty()) {
                    double sumErr8 = 0;
                    int nInliers8 = 0;
                    for (size_t i = 0; i < ptsImg.size(); ++i) {
                        if (inliers.at<uint8_t>(i)) {
                            double w = H.at<double>(2, 0) * ptsImg[i].x
                                     + H.at<double>(2, 1) * ptsImg[i].y
                                     + H.at<double>(2, 2);
                            double px = (H.at<double>(0, 0) * ptsImg[i].x
                                       + H.at<double>(0, 1) * ptsImg[i].y
                                       + H.at<double>(0, 2)) / w;
                            double py = (H.at<double>(1, 0) * ptsImg[i].x
                                       + H.at<double>(1, 1) * ptsImg[i].y
                                       + H.at<double>(1, 2)) / w;
                            double dx = px - ptsRef[i].x;
                            double dy = py - ptsRef[i].y;
                            sumErr8 += dx * dx + dy * dy;
                            nInliers8++;
                        }
                    }
                    double reprErr8 = (nInliers8 > 0) ? std::sqrt(sumErr8 / nInliers8) : 1e9;
                    Log::debug("8DOF: reprErr=", reprErr8, "px inliers=", nInliers8);
                    if (reprErr8 < reprErr) {
                        // Extract affine approximation from homography for dx/dy
                        transform = cv::Mat(2, 3, CV_64F);
                        transform.at<double>(0, 0) = H.at<double>(0, 0);
                        transform.at<double>(0, 1) = H.at<double>(0, 1);
                        transform.at<double>(0, 2) = H.at<double>(0, 2);
                        transform.at<double>(1, 0) = H.at<double>(1, 0);
                        transform.at<double>(1, 1) = H.at<double>(1, 1);
                        transform.at<double>(1, 2) = H.at<double>(1, 2);
                        reprErr = reprErr8;
                        nInliers = nInliers8;
                        modelDOF = 8;
                        modelName = "Homography";
                    }
                }
            }
        }

        // Sanity checks
        double inlierFinal = (double)nInliers / ptsImg.size();
        if (nInliers < 16 || inlierFinal < 0.45) {
            Log::debug("Feature alignment: poor inlier ratio (", inlierFinal,
                     "), falling back to MTB");
            return false;
        }

        // Extract rotation angle from transform matrix for diagnostic
        double a = transform.at<double>(0, 0);
        double b = transform.at<double>(0, 1);
        double rotation_deg = std::atan2(b, a) * 180.0 / M_PI;
        double scale = std::sqrt(a * a + b * b);

        if (std::abs(rotation_deg) > 2.0) {
            Log::debug("Feature alignment: rotation ", rotation_deg,
                     " deg exceeds 2 deg limit");
        }
        if (std::abs(scale - 1.0) > 0.02) {
            Log::debug("Feature alignment: scale ", scale,
                     " deviates > 2% from unity");
        }

        // Extract translation in full-resolution coordinates (preview is half-res)
        double txHalf = transform.at<double>(0, 2);
        double tyHalf = transform.at<double>(1, 2);
        double txFull = txHalf * 2.0;
        double tyFull = tyHalf * 2.0;

        // Split into integer and fractional parts.
        // CRITICAL: round to nearest EVEN integer to preserve Bayer CFA alignment.
        // Odd-pixel shifts swap R/B channels, causing pink/green color casts.
        outDx = (int)std::round(txFull / 2.0) * 2;
        outDy = (int)std::round(tyFull / 2.0) * 2;
        outFdx = txFull - outDx;
        outFdy = tyFull - outDy;

        Log::debug("Feature align: ", modelName, " (", modelDOF, "DOF) reprErr=",
                 reprErr, "px inliers=", nInliers, "/", ptsImg.size(),
                 " rot=", rotation_deg, "deg scale=", scale,
                 " tx=", txFull, " ty=", tyFull);
        return true;
    }

    Log::debug("Feature alignment: transform estimation failed, falling back to MTB");
    return false;
}
#endif // HAVE_OPENCV


void ImageStack::align(bool useFeatures) {
    if (images.size() > 1) {
#ifdef HAVE_OPENCV
        if (useFeatures) {
            Timer t("Feature Align");
            Log::debug("Feature-based alignment (AKAZE/ORB + MAGSAC++)");

            // Align each image against its neighbor (pairwise chain, like MTB)
            // then accumulate. This maximizes feature overlap between exposures.
            bool allSuccess = true;
            std::vector<int> featDx(images.size() - 1, 0);
            std::vector<int> featDy(images.size() - 1, 0);
            std::vector<double> featFdx(images.size() - 1, 0.0);
            std::vector<double> featFdy(images.size() - 1, 0.0);
            std::vector<bool> featOk(images.size() - 1, false);

            for (size_t i = 0; i < images.size() - 1; ++i) {
                int dx = 0, dy = 0;
                double fdx = 0, fdy = 0;
                if (featureAlign(images[i], images[i + 1], dx, dy, fdx, fdy)) {
                    featDx[i] = dx;
                    featDy[i] = dy;
                    featFdx[i] = fdx;
                    featFdy[i] = fdy;
                    featOk[i] = true;
                } else {
                    allSuccess = false;
                }
            }

            // For pairs where feature alignment failed, fall back to MTB
            if (!allSuccess) {
                // Prepare MTB pyramid data for failed pairs
                for (size_t i = 0; i < images.size(); ++i) {
                    images[i].preScale();
                }
                for (size_t i = 0; i < images.size() - 1; ++i) {
                    if (!featOk[i]) {
                        Log::debug("Image ", i, ": using MTB fallback");
                        images[i].alignWith(images[i + 1]);
                        featDx[i] = images[i].getDeltaX();
                        featDy[i] = images[i].getDeltaY();
                        featFdx[i] = 0.0;
                        featFdy[i] = 0.0;
                        // Reset dx/dy since we'll accumulate below
                        images[i].displace(-images[i].getDeltaX(), -images[i].getDeltaY());
                    }
                }
                for (auto & img : images) {
                    img.releaseAlignData();
                }
            }

            // Accumulate pairwise displacements (darkest image = reference = zero offset)
            for (size_t i = images.size() - 1; i > 0; --i) {
                int prevDx = images[i].getDeltaX();
                int prevDy = images[i].getDeltaY();
                images[i - 1].displace(prevDx + featDx[i - 1], prevDy + featDy[i - 1]);
                Log::debug("Image ", i - 1, " displaced to (",
                           images[i - 1].getDeltaX(), ", ",
                           images[i - 1].getDeltaY(), ")");
            }

            // Store fractional offsets (accumulated from pairwise)
            double accumFdx = 0.0, accumFdy = 0.0;
            for (int i = (int)images.size() - 2; i >= 0; --i) {
                accumFdx += featFdx[i];
                accumFdy += featFdy[i];
                images[i].setFracDx(accumFdx);
                images[i].setFracDy(accumFdy);
                Log::debug("Image ", i, " fractional offset: (",
                           accumFdx, ", ", accumFdy, ")");
            }
            images.back().setFracDx(0.0);
            images.back().setFracDy(0.0);
            return;
        }
#else
        if (useFeatures) {
            Log::debug("Feature-based alignment requested but OpenCV not available, using MTB");
        }
#endif
        // MTB alignment (default / fallback)
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


void ImageStack::computeResponseFunctions(bool linearMode) {
    Timer t("Compute response functions");
    // Fit each image against the reference (darkest) directly, not chained pairwise
    const Image & ref = images.back();
    for (int i = (int)images.size() - 2; i >= 0; --i) {
        images[i].computeResponseFunction(ref, linearMode);
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
    Log::debug("Hot pixel correction: ", corrected, " pixels corrected (sigma=", sigma, ")");
}


// Estimate noise model parameters for DNG NoiseProfile tag (51041).
// For each channel c: variance = S[c] * signal + O[c], with signal normalized to [0,1].
// S = shot noise coefficient (Poisson), O = read noise variance.
// Both estimated from actual pixel data using tile-based variance-vs-signal regression:
//   Var(raw) = S_raw * raw_signal + O_raw
// The 25th percentile of tile variances at each signal level provides robustness
// against texture contamination (smooth tiles dominate the low percentile).
// For cameras with optical black margins (topMargin/leftMargin > 0), O is refined
// using MAD of OB pixels. Both scaled by 1/numImages for the merged result.
static void estimateNoiseProfile(const std::vector<Image> & images,
                                  const RawParameters & params,
                                  int numImages, double noiseProfile[8]) {
    double range = static_cast<double>(params.max);
    if (range <= 0.0) return;

    for (int c = 0; c < params.colors; ++c) {
        double channelRange = range - params.cblack[c];
        if (channelRange <= 0.0) channelRange = range;

        // --- Tile-based variance-vs-signal regression from active pixels ---
        // Collect tiles of same-color pixels, compute per-tile mean and variance,
        // then fit the affine noise model Var = S_raw * signal + O_raw.
        const int tileStep = (params.FC.getFilters() == 9) ? 6 : 2;
        const int tileSameColorSize = 4;
        const int rawTileSize = tileSameColorSize * tileStep;

        struct TileStat { double mean; double variance; };
        std::vector<TileStat> tileStats;
        tileStats.reserve(10000);

        for (const auto & img : images) {
            size_t imgW = img.getWidth(), imgH = img.getHeight();
            for (size_t ty = 0; ty + rawTileSize <= imgH; ty += rawTileSize) {
                for (size_t tx = 0; tx + rawTileSize <= imgW; tx += rawTileSize) {
                    // Find starting position matching channel c
                    int sx = -1, sy = -1;
                    for (int dy = 0; dy < tileStep && sx < 0; ++dy) {
                        for (int dx = 0; dx < tileStep; ++dx) {
                            if (params.FC(tx + dx, ty + dy) == c) {
                                sx = tx + dx; sy = ty + dy;
                                break;
                            }
                        }
                    }
                    if (sx < 0) continue;

                    double sum = 0, sumSq = 0;
                    int cnt = 0;
                    for (int j = 0; j < tileSameColorSize; ++j) {
                        for (int i = 0; i < tileSameColorSize; ++i) {
                            int x = sx + i * tileStep;
                            int y = sy + j * tileStep;
                            if (x < 0 || (size_t)x >= imgW ||
                                y < 0 || (size_t)y >= imgH) continue;
                            if (!img.contains(x, y)) continue;
                            uint16_t raw = img(x, y);
                            if (raw < 1 || img.isSaturated(raw)) continue;
                            double v = static_cast<double>(raw);
                            sum += v; sumSq += v * v; cnt++;
                        }
                    }
                    if (cnt < 8) continue;
                    double mean = sum / cnt;
                    double variance = (sumSq - sum * sum / cnt) / (cnt - 1);
                    if (variance >= 0) tileStats.push_back({mean, variance});
                }
            }
        }

        double S_raw = 1.0; // Theoretical fallback: gain = 1 e/ADU
        double O_raw = 1.0; // Conservative fallback
        double S_spatial = -1.0; // From spatial tile regression
        double S_temporal = -1.0; // From inter-frame residual variance

        // --- Method 1: Spatial high-pass tile regression (Android Camera2 approach) ---
        if (tileStats.size() >= 100) {
            double maxMean = 0;
            for (auto & t : tileStats) if (t.mean > maxMean) maxMean = t.mean;

            if (maxMean > 100) {
                const int numBins = 32;
                double binWidth = maxMean / numBins;

                // Bin by mean signal, take 25th percentile variance per bin
                std::vector<double> binMeans, binP25Vars;
                for (int b = 0; b < numBins; ++b) {
                    double bLow = b * binWidth, bHigh = (b + 1) * binWidth;
                    std::vector<double> vars;
                    double sigSum = 0;
                    for (auto & t : tileStats) {
                        if (t.mean >= bLow && t.mean < bHigh) {
                            vars.push_back(t.variance);
                            sigSum += t.mean;
                        }
                    }
                    if (vars.size() < 10) continue;
                    size_t p25 = vars.size() / 4;
                    std::nth_element(vars.begin(), vars.begin() + p25, vars.end());
                    binMeans.push_back(sigSum / vars.size());
                    binP25Vars.push_back(vars[p25]);
                }

                if (binMeans.size() >= 3) {
                    int n = (int)binMeans.size();
                    double sX = 0, sY = 0, sXX = 0, sXY = 0;
                    for (int i = 0; i < n; ++i) {
                        sX += binMeans[i]; sY += binP25Vars[i];
                        sXX += binMeans[i] * binMeans[i];
                        sXY += binMeans[i] * binP25Vars[i];
                    }
                    double denom = n * sXX - sX * sX;
                    if (std::abs(denom) > 1e-10) {
                        double slope = (n * sXY - sX * sY) / denom;
                        double intercept = (sY - slope * sX) / n;
                        if (slope > 0 && slope < 10.0) {
                            S_spatial = slope;
                            S_raw = slope;
                        }
                        if (intercept > 0) {
                            O_raw = intercept;
                        }
                    }
                }
            }
        }

        // --- Method 2: Temporal variance across aligned exposures ---
        // For each pair of images, compute the exposure-normalized residual at
        // overlapping unsaturated same-color pixels, bin by signal level, and
        // regress to get S. This naturally separates noise from structure since
        // the scene is constant across frames. (Caveat: alignment residuals add
        // bias on high-frequency texture; the 25th percentile filter helps.)
        if (numImages >= 2) {
            const Image & ref = images.back(); // darkest = reference
            struct TemporalBin { double sumSig; double sumResidSq; int count; };
            const int tNumBins = 32;
            std::vector<TemporalBin> tBins(tNumBins);
            for (auto & b : tBins) { b.sumSig = 0; b.sumResidSq = 0; b.count = 0; }
            double refExp = ref.getRelativeExposure();
            double maxSig = 0;

            // Find starting pixel for channel c in CFA pattern
            int cStartX = -1, cStartY = -1;
            for (int dy = 0; dy < tileStep && cStartX < 0; ++dy) {
                for (int dx = 0; dx < tileStep; ++dx) {
                    if (params.FC(dx, dy) == c) {
                        cStartX = dx; cStartY = dy; break;
                    }
                }
            }
            if (cStartX < 0) cStartX = cStartY = 0;
            // Subsample: step by 2*tileStep (every other same-color pixel)
            const int tSubStep = tileStep * 2;

            // First pass: find max unsaturated signal in reference for binning
            for (size_t y = cStartY; y < ref.getHeight(); y += tSubStep) {
                for (size_t x = cStartX; x < ref.getWidth(); x += tSubStep) {
                    if (!ref.contains(x, y)) continue;
                    uint16_t rv = ref(x, y);
                    if (rv > 0 && !ref.isSaturated(rv) && rv > maxSig) maxSig = rv;
                }
            }

            if (maxSig > 100) {
                double tBinWidth = maxSig / tNumBins;

                // Collect residuals from each non-reference image vs reference
                for (int k = 0; k < numImages - 1; ++k) {
                    // raw_ref = raw_k * (slope_k / slope_ref) for the same scene radiance
                    double ratio = images[k].getRelativeExposure() / refExp;
                    for (size_t y = cStartY; y < ref.getHeight(); y += tSubStep) {
                        for (size_t x = cStartX; x < ref.getWidth(); x += tSubStep) {
                            if (!ref.contains(x, y) || !images[k].contains(x, y)) continue;
                            uint16_t rv = ref(x, y);
                            uint16_t kv = images[k](x, y);
                            if (rv < 1 || ref.isSaturated(rv)) continue;
                            if (kv < 1 || images[k].isSaturated(kv)) continue;

                            // Smoothness filter: skip high-gradient (textured) pixels
                            // where alignment residuals inflate temporal variance.
                            // Check same-color neighbors in reference at ±tileStep.
                            int xl = (int)x - tileStep, xr = (int)x + tileStep;
                            int yt = (int)y - tileStep, yb = (int)y + tileStep;
                            double maxGrad = 0;
                            if (xl >= 0 && ref.contains(xl, y))
                                maxGrad = std::max(maxGrad, std::abs((double)rv - ref(xl, y)));
                            if ((size_t)xr < ref.getWidth() && ref.contains(xr, y))
                                maxGrad = std::max(maxGrad, std::abs((double)rv - ref(xr, y)));
                            if (yt >= 0 && ref.contains(x, yt))
                                maxGrad = std::max(maxGrad, std::abs((double)rv - ref(x, yt)));
                            if ((size_t)yb < ref.getHeight() && ref.contains(x, yb))
                                maxGrad = std::max(maxGrad, std::abs((double)rv - ref(x, yb)));
                            // Skip if max gradient > 5% of signal (textured region)
                            if (maxGrad > 0.05 * rv) continue;

                            // Predicted reference value from image k
                            double predicted = kv * ratio;
                            double residual = (double)rv - predicted;
                            // Var(residual) = Var(rv) + ratio^2 * Var(kv)
                            //               = (S*rv + O) + ratio^2 * (S*kv + O)
                            // Effective signal for binning: rv
                            int bin = (int)(rv / tBinWidth);
                            if (bin < 0) bin = 0;
                            if (bin >= tNumBins) bin = tNumBins - 1;
                            // Store the residual^2 normalized by the combined variance denominator
                            // For regression: E[res^2] = S*(rv + ratio^2*kv) + O*(1 + ratio^2)
                            // We'll regress E[res^2] vs (rv + ratio^2*kv) to get S as slope
                            double combinedSig = (double)rv + ratio * ratio * (double)kv;
                            tBins[bin].sumSig += combinedSig;
                            tBins[bin].sumResidSq += residual * residual;
                            tBins[bin].count++;
                        }
                    }
                }

                // Fit: E[res^2] = S_temporal * combinedSig + intercept
                std::vector<double> tMeans, tVars;
                int totalSamples = 0;
                for (int b = 0; b < tNumBins; ++b) {
                    totalSamples += tBins[b].count;
                    if (tBins[b].count < 50) continue;
                    tMeans.push_back(tBins[b].sumSig / tBins[b].count);
                    tVars.push_back(tBins[b].sumResidSq / tBins[b].count);
                }

                if (tMeans.size() >= 3) {
                    int n = (int)tMeans.size();
                    double sX = 0, sY = 0, sXX = 0, sXY = 0;
                    for (int i = 0; i < n; ++i) {
                        sX += tMeans[i]; sY += tVars[i];
                        sXX += tMeans[i] * tMeans[i];
                        sXY += tMeans[i] * tVars[i];
                    }
                    double denom = n * sXX - sX * sX;
                    if (std::abs(denom) > 1e-10) {
                        double slope = (n * sXY - sX * sY) / denom;
                        Log::debug("Noise ch", c, " temporal: slope=", slope,
                                 " bins=", n, " samples=", totalSamples);
                        if (slope > 0 && slope < 10.0) {
                            S_temporal = slope;
                        }
                    }
                } else {
                    Log::debug("Noise ch", c, " temporal: insufficient bins (",
                             tMeans.size(), ") samples=", totalSamples,
                             " maxSig=", maxSig);
                }
            }
        }

        // --- Cross-validate: log both, prefer spatial unless temporal is clearly better ---
        // Spatial (tile-based) typically has 100x more samples and is inherently robust
        // to alignment errors. Temporal wins only when it's lower (indicating spatial has
        // texture contamination) AND the difference is modest (within 30%).
        if (S_spatial > 0 && S_temporal > 0) {
            double ratio = S_temporal / S_spatial;
            Log::debug("Noise ch", c, " cross-val: S_spatial=", S_spatial,
                     " S_temporal=", S_temporal, " ratio=", ratio);
            if (S_temporal < S_spatial && ratio >= 0.7) {
                // Temporal is lower but within 30%: likely reflects better
                // texture rejection. Use average of both estimates.
                S_raw = (S_spatial + S_temporal) / 2.0;
                Log::debug("Noise ch", c, ": using mean of spatial+temporal S=", S_raw);
            }
            // Otherwise keep spatial S_raw (already set) — either temporal is
            // higher (alignment residual inflation) or much lower (sampling bias)
        } else if (S_temporal > 0 && S_spatial <= 0) {
            S_raw = S_temporal;
        }

        // --- O: prefer MAD from OB margins if available ---
        // Note: Image stores only the active area. OB margins are accessible only
        // when topMargin/leftMargin > 0 and the Image coordinate space overlaps them.
        // For many cameras (e.g., Nikon Z 9) margins are 0 and we rely on the
        // regression intercept computed above.
        if (params.topMargin > 0 || params.leftMargin > 0) {
            std::vector<double> obValues;
            for (const auto & img : images) {
                for (size_t y = 0; y < params.topMargin && y < img.getHeight(); ++y) {
                    for (size_t x = 0; x < img.getWidth(); ++x) {
                        if (params.FC(x, y) == c && img.contains(x, y))
                            obValues.push_back(static_cast<double>(img(x, y)));
                    }
                }
                for (size_t y = params.topMargin; y < img.getHeight(); ++y) {
                    for (size_t x = 0; x < params.leftMargin && x < img.getWidth(); ++x) {
                        if (params.FC(x, y) == c && img.contains(x, y))
                            obValues.push_back(static_cast<double>(img(x, y)));
                    }
                }
            }
            if (obValues.size() > 10) {
                size_t n = obValues.size();
                std::nth_element(obValues.begin(), obValues.begin() + n/2, obValues.end());
                double median = obValues[n/2];
                for (auto & v : obValues) v = std::abs(v - median);
                std::nth_element(obValues.begin(), obValues.begin() + n/2, obValues.end());
                double mad = obValues[n/2] * 1.4826;
                O_raw = mad * mad;
            }
        }

        double S_dng = S_raw / channelRange;
        double O_dng = O_raw / (channelRange * channelRange);

        Log::debug("Noise ch", c, ": S=", S_dng, " O=", O_dng,
                 " (S_raw=", S_raw, " O_raw=", O_raw,
                 " tiles=", tileStats.size(), ")");

        // Scale for merge: variance scales as 1/N
        noiseProfile[c * 2]     = S_dng / numImages;
        noiseProfile[c * 2 + 1] = O_dng / numImages;
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


ComposeResult ImageStack::compose(const RawParameters & params, int featherRadius, float deghostSigma, DeghostMode deghostMode, int deghostIterations, double clipPercentile, bool subPixelAlign, float highlightPull, float highlightRolloff) const {
    int imageMax = images.size() - 1;
    BoxBlur map(fattenMask(mask, featherRadius));
    measureTime("Blur", [&] () {
        map.blur(featherRadius);
    });
    Timer t("Compose");
    Array2D<float> dst(params.rawWidth, params.rawHeight);
    dst.displace(-(int)params.leftMargin, -(int)params.topMargin);
    dst.fillBorders(0.f);

    const int numImages = (int)images.size();

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

    // Pre-compute per-exposure relativeExposure for MLE variance weighting
    std::vector<double> relExp(numImages);
    for (int k = 0; k < numImages; k++) {
        relExp[k] = images[k].getRelativeExposure();
    }

    // Pre-compute spatial ghost confidence map for coherent deghosting
    Array2D<float> ghostMap;
    int refIdx = 0; // reference exposure index for robust deghosting
    if (deghost) {
        ghostMap = Array2D<float>(width, height);

        if (deghostMode == DeghostMode::Robust) {
            // Reference-guided deghosting (Granados 2013 / Khan 2006 approach)
            // Select reference: middle exposure (best SNR in mid-tones, least motion blur)
            refIdx = numImages / 2;
            Log::debug("Robust deghost: ref=image[", refIdx, "] sigma=", deghostSigma,
                       " iterations=", deghostIterations);

            for (int iter = 0; iter < deghostIterations; ++iter) {
                // Compute per-pixel ghost confidence as max normalized residual vs reference
                #pragma omp parallel for schedule(dynamic,16)
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        int ch = params.FC(x, y);

                        // Get reference radiance
                        if (!images[refIdx].contains(x, y)) {
                            ghostMap(x, y) = 0.0f;
                            continue;
                        }
                        uint16_t refRaw = images[refIdx](x, y);
                        if (refRaw < 1 || refRaw >= satThreshPerCh[ch]) {
                            ghostMap(x, y) = 0.0f;
                            continue;
                        }
                        double refRad = images[refIdx].exposureAt(x, y);
                        if (refRad <= 0.0) { ghostMap(x, y) = 0.0f; continue; }

                        // Noise-aware threshold: sigma_noise at the reference signal level
                        double refVar = noiseA[ch] * refRaw + noiseB[ch];
                        if (refVar < noiseB[ch]) refVar = noiseB[ch];
                        if (refVar < 1.0) refVar = 1.0;
                        // Convert raw variance to radiance variance via response slope
                        double refSlope = relExp[refIdx];
                        double radVar = refVar * refSlope * refSlope;
                        double noiseSigma = std::sqrt(radVar);

                        // Max normalized residual across non-reference exposures
                        double maxNormRes = 0.0;
                        int nValid = 0;
                        for (int k = 0; k <= imageMax; k++) {
                            if (k == refIdx) continue;
                            if (!images[k].contains(x, y)) continue;
                            uint16_t raw = images[k](x, y);
                            if (raw < 1 || raw >= satThreshPerCh[ch]) continue;
                            double rad = images[k].exposureAt(x, y);
                            if (rad <= 0.0) continue;
                            double residual = std::abs(rad - refRad);
                            double normRes = (noiseSigma > 0.0) ? residual / noiseSigma : 0.0;
                            if (normRes > maxNormRes) maxNormRes = normRes;
                            nValid++;
                        }

                        if (nValid < 1) {
                            ghostMap(x, y) = 0.0f;
                        } else {
                            // Map normalized residual to confidence via sigmoid-like ramp
                            // Below deghostSigma: confidence = 0 (noise)
                            // Above 2*deghostSigma: confidence = 1 (definite ghost)
                            float conf = 0.0f;
                            if (maxNormRes > deghostSigma) {
                                conf = static_cast<float>(
                                    std::min(1.0, (maxNormRes - deghostSigma) / deghostSigma));
                            }
                            ghostMap(x, y) = conf;
                        }
                    }
                }

                // Edge-aware spatial filter: bilateral-like on CFA neighbors
                // Uses intensity similarity to reference to preserve ghost boundaries
                Array2D<float> tmpMap(width, height);
                const int filterR = 2; // radius in pixels
                const double spatialSigma = 1.5;
                const double rangeSigma = 0.1; // fraction of reference radiance

                #pragma omp parallel for schedule(dynamic,16)
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        if (!images[refIdx].contains(x, y)) {
                            tmpMap(x, y) = ghostMap(x, y);
                            continue;
                        }
                        double centerRad = images[refIdx].exposureAt(x, y);
                        if (centerRad <= 0.0) centerRad = 1.0;
                        double rangeScale = rangeSigma * centerRad;
                        if (rangeScale < 1.0) rangeScale = 1.0;
                        double invRange2 = 1.0 / (2.0 * rangeScale * rangeScale);
                        double invSpatial2 = 1.0 / (2.0 * spatialSigma * spatialSigma);

                        double wSum = 0.0, vSum = 0.0;
                        for (int dy = -filterR; dy <= filterR; ++dy) {
                            int ny = (int)y + dy;
                            if (ny < 0 || ny >= (int)height) continue;
                            for (int dx = -filterR; dx <= filterR; ++dx) {
                                int nx = (int)x + dx;
                                if (nx < 0 || nx >= (int)width) continue;
                                double spatW = std::exp(-(dx*dx + dy*dy) * invSpatial2);
                                double nRad = images[refIdx].contains(nx, ny)
                                    ? images[refIdx].exposureAt(nx, ny) : centerRad;
                                double rangeW = std::exp(-(nRad - centerRad) * (nRad - centerRad) * invRange2);
                                double w = spatW * rangeW;
                                wSum += w;
                                vSum += w * ghostMap(nx, ny);
                            }
                        }
                        tmpMap(x, y) = (wSum > 0.0) ? static_cast<float>(vSum / wSum) : ghostMap(x, y);
                    }
                }
                // Copy filtered result back
                #pragma omp parallel for
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x)
                        ghostMap(x, y) = tmpMap(x, y);

                if (deghostIterations > 1) {
                    // Count ghost pixels for logging
                    long long ghostCount = 0;
                    #pragma omp parallel for reduction(+:ghostCount)
                    for (size_t y = 0; y < height; ++y)
                        for (size_t x = 0; x < width; ++x)
                            if (ghostMap(x, y) > 0.1f) ghostCount++;
                    double ghostPct = 100.0 * ghostCount / (width * height);
                    Log::debug("Deghost iter ", iter + 1, ": ", ghostPct, "% ghost pixels");
                }
            }
            Log::debug("Reference-guided ghost map computed (bilateral filtered)");
        } else {
            // Legacy MAD-based ghost confidence
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
            Log::debug("Legacy ghost map computed (MAD + box blur)");
        }
    }

    // === Highlight detection pass ===
    // Build a feathered, graduated mask identifying near-saturated regions.
    // Uses the middle exposure for detection (longest is too saturated to
    // discriminate). Threshold is found automatically via Otsu's method on
    // the brightness histogram, or overridden by user's --highlight-rolloff.
    // The mask is graduated: brighter pixels get higher mask values, so the
    // pull effect is proportional to how blown each pixel is.
    Array2D<float> highlightMask;
    double hlThreshold = 0.9;  // effective Otsu threshold (set by detection pass)
    const bool doHighlightPull = highlightPull > 0.0f;
    if (doHighlightPull) {
        Log::debug("Highlight pull enabled: pull=", highlightPull,
                   " rolloff=", highlightRolloff);

        const int hlDetectIdx = numImages / 2;
        Log::debug("Highlight detection frame: image[", hlDetectIdx, "]");

        // Phase 1: Compute per-pixel brightness ratios and build histogram
        // for Otsu threshold finding
        Array2D<float> brightnessMap(width, height);
        const int histBins = 256;
        std::vector<long long> histogram(histBins, 0);
        long long brightCount = 0;

        #pragma omp parallel
        {
            std::vector<long long> localHist(histBins, 0);
            long long localBright = 0;
            #pragma omp for nowait
            for (size_t y = 0; y < height; ++y) {
                for (size_t x = 0; x < width; ++x) {
                    if (!images[hlDetectIdx].contains(x, y)) {
                        brightnessMap(x, y) = 0.0f;
                        continue;
                    }
                    int ch = params.FC(x, y);
                    double raw = static_cast<double>(images[hlDetectIdx](x, y));
                    float b = static_cast<float>(raw / satThreshPerCh[ch]);
                    if (b > 1.0f) b = 1.0f;
                    brightnessMap(x, y) = b;
                    if (b > 0.5f) {
                        int bin = static_cast<int>(b * (histBins - 1));
                        if (bin >= histBins) bin = histBins - 1;
                        localHist[bin]++;
                        localBright++;
                    }
                }
            }
            #pragma omp critical
            {
                for (int i = 0; i < histBins; i++)
                    histogram[i] += localHist[i];
                brightCount += localBright;
            }
        }

        if (brightCount < 100) {
            Log::debug("Too few bright pixels (", brightCount, "), disabling highlight pull");
        } else {
            // Phase 2: Otsu's method on bright pixels (>0.5) to find
            // the natural break between "bright interior" and "blown highlights"
            int otsuStart = histBins / 2;
            double otsuThreshold = highlightRolloff;

            double totalSum = 0.0, totalN = 0.0;
            for (int i = otsuStart; i < histBins; i++) {
                totalSum += (double)i * histogram[i];
                totalN += histogram[i];
            }

            if (totalN > 0) {
                double bestVariance = 0.0;
                int bestBin = otsuStart;
                double sumB = 0.0, nB = 0.0;
                for (int i = otsuStart; i < histBins - 1; i++) {
                    nB += histogram[i];
                    if (nB == 0) continue;
                    double nF = totalN - nB;
                    if (nF == 0) break;
                    sumB += (double)i * histogram[i];
                    double meanB = sumB / nB;
                    double meanF = (totalSum - sumB) / nF;
                    double variance = nB * nF * (meanB - meanF) * (meanB - meanF);
                    if (variance > bestVariance) {
                        bestVariance = variance;
                        bestBin = i;
                    }
                }
                double otsuVal = static_cast<double>(bestBin) / (histBins - 1);
                otsuVal = std::max(0.7, std::min(0.98, otsuVal));
                Log::debug("Otsu threshold: ", otsuVal, " (bin ", bestBin, "/", histBins, ")");

                // Use the lower of Otsu and user-specified rolloff
                otsuThreshold = std::min(otsuVal, static_cast<double>(highlightRolloff));
            }
            hlThreshold = otsuThreshold;
            Log::debug("Effective highlight threshold: ", otsuThreshold);

            // Phase 3: Build graduated core mask at 2x2 Bayer block resolution.
            // Using the MAX brightness in each 2x2 block ensures all CFA channels
            // at the same spatial location get identical mask values, preventing
            // color shifts from differential per-channel compression.
            double invRange = 1.0 / std::max(0.01, 1.0 - otsuThreshold);
            long long coreCount = 0;
            highlightMask = Array2D<float>(width, height);

            #pragma omp parallel for reduction(+:coreCount)
            for (size_t y = 0; y < height; y += 2) {
                for (size_t x = 0; x < width; x += 2) {
                    // Find max brightness across the 2x2 Bayer block
                    float maxB = brightnessMap(x, y);
                    if (x + 1 < width) maxB = std::max(maxB, brightnessMap(x + 1, y));
                    if (y + 1 < height) maxB = std::max(maxB, brightnessMap(x, y + 1));
                    if (x + 1 < width && y + 1 < height)
                        maxB = std::max(maxB, brightnessMap(x + 1, y + 1));

                    float m = 0.0f;
                    if (maxB > otsuThreshold) {
                        float t = static_cast<float>((maxB - otsuThreshold) * invRange);
                        if (t > 1.0f) t = 1.0f;
                        m = t * t * (3.0f - 2.0f * t);  // smoothstep: C1-continuous at endpoints
                        coreCount += 4;
                    }

                    // Apply uniform mask to all pixels in this block
                    highlightMask(x, y) = m;
                    if (x + 1 < width) highlightMask(x + 1, y) = m;
                    if (y + 1 < height) highlightMask(x, y + 1) = m;
                    if (x + 1 < width && y + 1 < height)
                        highlightMask(x + 1, y + 1) = m;
                }
            }

            double corePct = 100.0 * coreCount / (width * height);
            Log::debug("Highlight core: ", corePct, "% of pixels (graduated)");

            if (coreCount == 0) {
                Log::debug("No highlight pixels above threshold, disabling");
                highlightMask = Array2D<float>();
            } else {
                // Phase 4: Feather using iterative box blur (3 passes ~ Gaussian)
                int hlFeatherRadius = featherRadius * 2;

                for (int pass = 0; pass < 3; ++pass) {
                    Array2D<float> tmp(width, height);
                    #pragma omp parallel for
                    for (size_t y = 0; y < height; ++y) {
                        for (size_t x = 0; x < width; ++x) {
                            float sum = 0.0f;
                            int count = 0;
                            int x0 = std::max(0, (int)x - hlFeatherRadius);
                            int x1 = std::min((int)width - 1, (int)x + hlFeatherRadius);
                            for (int nx = x0; nx <= x1; ++nx) {
                                sum += highlightMask(nx, y);
                                count++;
                            }
                            tmp(x, y) = sum / count;
                        }
                    }
                    #pragma omp parallel for
                    for (size_t y = 0; y < height; ++y) {
                        for (size_t x = 0; x < width; ++x) {
                            float sum = 0.0f;
                            int count = 0;
                            int y0 = std::max(0, (int)y - hlFeatherRadius);
                            int y1 = std::min((int)height - 1, (int)y + hlFeatherRadius);
                            for (int ny = y0; ny <= y1; ++ny) {
                                sum += tmp(x, ny);
                                count++;
                            }
                            highlightMask(x, y) = sum / count;
                        }
                    }
                }

                // Clamp to [0, 1]
                #pragma omp parallel for
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x)
                        if (highlightMask(x, y) > 1.0f) highlightMask(x, y) = 1.0f;

                // Log stats
                long long featheredCount = 0;
                float maskMean = 0.0f;
                #pragma omp parallel for reduction(+:featheredCount) reduction(+:maskMean)
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x) {
                        if (highlightMask(x, y) > 0.01f) featheredCount++;
                        maskMean += highlightMask(x, y);
                    }
                double featheredPct = 100.0 * featheredCount / (width * height);
                maskMean /= (width * height);
                Log::debug("Highlight feathered region: ", featheredPct,
                           "% of pixels, mean mask=", maskMean,
                           " (feather radius=", hlFeatherRadius, ")");
            }
        }
    }

    float maxVal = 0.0;
    #pragma omp parallel for schedule(dynamic,16) reduction(max:maxVal)
    for (size_t y = 0; y < height; ++y) {
        std::vector<double> radiances(numImages);
        std::vector<double> weights(numImages);
        std::vector<int> exposureIdx(numImages);
        std::vector<double> absDevs(numImages);
        std::vector<double> sorted(numImages);

        for (size_t x = 0; x < width; ++x) {
            int numValid = 0;
            int ch = params.FC(x, y);

            // Highlight mask value for this pixel (0 = normal, >0 = highlight region)
            float hlMask = (doHighlightPull && highlightMask.size() > 0)
                ? highlightMask(x, y) : 0.0f;

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
                // When Bayer-block rolloff is active, use the block (minimum) threshold
                // for hard rejection so all CFA channels clip at the same level.
                // This prevents per-channel path divergence (merge vs fallback) that
                // causes color fringe at blown highlight boundaries.
                double hardThresh = useBlockRolloff ? blockThresh : satThreshPerCh[ch];
                if (raw >= hardThresh) continue;

                // Unified MLE weight: w = 1 / (relExp^2 * Var(raw))
                // where Var(raw) = a*raw + b (affine noise model).
                // This is the inverse-variance weight for the radiance estimate,
                // optimal across the full tonal range — shot-noise-dominated
                // highlights and read-noise-dominated shadows alike.
                double pixelVar = noiseA[ch] * raw + noiseB[ch];
                if (pixelVar < noiseB[ch]) pixelVar = noiseB[ch];
                if (pixelVar < 1.0) pixelVar = 1.0;
                double w = 1.0 / (relExp[k] * relExp[k] * pixelVar);
                if (useBlockRolloff) {
                    double effectiveBlockRolloff = blockRolloff;
                    double effectiveBlockRange = blockRange;
                    if (hlMask > 0.0f) {
                        double effectiveFrac = 0.9 - hlMask * (0.9 - highlightRolloff);
                        effectiveBlockRolloff = effectiveFrac * blockThresh;
                        effectiveBlockRange = blockThresh - effectiveBlockRolloff;
                    }
                    if (raw >= blockThresh) {
                        w = 0.0;
                    } else if (raw > effectiveBlockRolloff) {
                        double t = (blockThresh - raw) / effectiveBlockRange;
                        w *= t * t;
                    }
                } else {
                    double effectiveRolloff = satRolloffPerCh[ch];
                    double effectiveRange = satRolloffRangePerCh[ch];
                    if (hlMask > 0.0f) {
                        double effectiveFrac = 0.9 - hlMask * (0.9 - highlightRolloff);
                        effectiveRolloff = effectiveFrac * satThreshPerCh[ch];
                        effectiveRange = satThreshPerCh[ch] - effectiveRolloff;
                    }
                    if (raw > effectiveRolloff) {
                        double t = (satThreshPerCh[ch] - raw) / effectiveRange;
                        w *= t * t;
                    }
                }

                double radiance = (subPixelAlign && (images[k].getFracDx() != 0.0 || images[k].getFracDy() != 0.0))
                    ? images[k].exposureForRaw(raw)
                    : images[k].exposureAt(x, y);
                if (radiance <= 0.0) continue;

                radiances[numValid] = radiance;
                weights[numValid] = w;
                exposureIdx[numValid] = k;
                numValid++;
            }

            // Spatially coherent ghost detection: use pre-computed ghost map to
            // modulate per-pixel deghosting strength
            if (deghost && numValid >= 2 && ghostMap(x, y) > 0.1f) {
                double ghostStrength = static_cast<double>(ghostMap(x, y));

                if (deghostMode == DeghostMode::Robust) {
                    // Reference-guided: compare each exposure to reference radiance.
                    // Use Tukey biweight (ghostStrength > 0.7) for clean exclusion,
                    // Huber (0.1 < ghostStrength <= 0.7) for graduated blending.
                    double refRad = 0.0;
                    int refValidIdx = -1;
                    for (int i = 0; i < numValid; i++) {
                        if (exposureIdx[i] == refIdx) { refRad = radiances[i]; refValidIdx = i; break; }
                    }
                    if (refRad > 0.0) {
                        // Noise sigma at reference level
                        double refNoiseSigma = std::sqrt(noiseA[ch] * refRad / relExp[refIdx] + noiseB[ch])
                                               * relExp[refIdx];
                        if (refNoiseSigma < 1.0) refNoiseSigma = 1.0;
                        double threshold = deghostSigma * refNoiseSigma;

                        for (int i = 0; i < numValid; i++) {
                            if (i == refValidIdx) continue;
                            double dev = std::abs(radiances[i] - refRad);
                            double u = dev / threshold;

                            double factor;
                            if (ghostStrength > 0.7) {
                                // Tukey biweight: zero weight beyond threshold
                                if (u >= 1.0) {
                                    factor = 0.0;
                                } else {
                                    double t = 1.0 - u * u;
                                    factor = t * t;
                                }
                            } else {
                                // Huber: linear attenuation beyond threshold (never fully zero)
                                if (u <= 1.0) {
                                    factor = 1.0;
                                } else {
                                    factor = 1.0 / u;
                                }
                            }
                            weights[i] *= (1.0 - ghostStrength) + ghostStrength * factor;
                        }

                        // Hard fallback: if ghost confidence is very high and threshold
                        // exceeded, force reference-only to avoid ghosting artifacts
                        if (ghostStrength > 0.9 && refValidIdx >= 0) {
                            double totalNonRef = 0.0;
                            for (int i = 0; i < numValid; i++)
                                if (i != refValidIdx) totalNonRef += weights[i];
                            if (totalNonRef < weights[refValidIdx] * 0.05) {
                                // Non-reference weights negligible: go reference-only
                                for (int i = 0; i < numValid; i++)
                                    if (i != refValidIdx) weights[i] = 0.0;
                            }
                        }
                    }
                } else {
                    // Legacy MAD-based Gaussian deghosting
                    if (numValid >= 3) {
                        sorted.assign(radiances.begin(), radiances.begin() + numValid);
                        std::nth_element(sorted.begin(), sorted.begin() + numValid / 2, sorted.end());
                        double median = sorted[numValid / 2];

                        for (int i = 0; i < numValid; i++)
                            absDevs[i] = std::abs(radiances[i] - median);
                        std::nth_element(absDevs.begin(), absDevs.begin() + numValid / 2, absDevs.begin() + numValid);
                        double mad = absDevs[numValid / 2] * 1.4826;

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
                // All exposures saturated or unavailable — use darkest image.
                if (images[imageMax].contains(x, y)) {
                    v = images[imageMax].exposureAt(x, y);
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

    // Scale to params.max and recover the black levels.
    // Clamp highlights per 2x2 Bayer block to preserve CFA channel ratios;
    // independent per-pixel clamping destroys color balance in saturated
    // regions (all channels → same clampMax → magenta after WB).
    float mult = (params.max - params.maxBlack) / normVal;
    float clampMax = static_cast<float>(params.max - params.maxBlack);
    #pragma omp parallel for
    for (size_t y = 0; y < params.rawHeight; y += 2) {
        for (size_t x = 0; x < params.rawWidth; x += 2) {
            float sigs[2][2] = {};
            float maxSig = 0.0f;
            for (int dy = 0; dy < 2 && y + dy < params.rawHeight; ++dy)
                for (int dx = 0; dx < 2 && x + dx < params.rawWidth; ++dx) {
                    float v = dst(x + dx, y + dy) * mult;
                    sigs[dy][dx] = v;
                    if (v > maxSig) maxSig = v;
                }
            float blockScale = (maxSig > clampMax) ? clampMax / maxSig : 1.0f;
            for (int dy = 0; dy < 2 && y + dy < params.rawHeight; ++dy)
                for (int dx = 0; dx < 2 && x + dx < params.rawWidth; ++dx)
                    dst(x + dx, y + dy) = sigs[dy][dx] * blockScale
                        + params.blackAt(x + dx - params.leftMargin,
                                         y + dy - params.topMargin);
        }
    }

    // Compute BaselineExposure BEFORE highlight pull so we can use it
    // to determine the SDR-aware scale factor.
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

    // Bilateral base-detail highlight compression.
    // Decompose half-res log-luminance into base (smooth brightness) and detail
    // (texture/edges) using a bilateral filter. Compress only the base layer with
    // a Reinhard extended shoulder, then reconstruct by applying a uniform per-block
    // scale. This preserves window mullions, exterior scene detail, and surface
    // texture while compressing the large-scale brightness envelope into SDR range.
    if (doHighlightPull && highlightMask.size() > 0) {
        // Compute average white-balance multiplier across the 4 CFA positions
        // so the SDR ceiling matches the WB-corrected luminance domain.
        float avgWBMul = 0.25f * (params.camMul[0] + params.camMul[1]
                                 + params.camMul[2] + params.camMul[3]);
        float sdrCeiling = clampMax * std::pow(2.0f, -static_cast<float>(baselineExposureEV))
                           * avgWBMul;

        if (sdrCeiling < clampMax * avgWBMul && sdrCeiling > 0.0f) {
            Log::debug("Bilateral highlight compression: sdrCeiling=", sdrCeiling,
                       " clampMax=", clampMax, " avgWBMul=", avgWBMul);

            // Step A: Half-resolution log-luminance map from 2x2 Bayer blocks
            const int bw = static_cast<int>(params.width) / 2;
            const int bh = static_cast<int>(params.height) / 2;

            Array2D<float> logLum(bw, bh);
            #pragma omp parallel for schedule(dynamic, 16)
            for (int by = 0; by < bh; ++by) {
                for (int bx = 0; bx < bw; ++bx) {
                    int px = static_cast<int>(params.leftMargin) + bx * 2;
                    int py = static_cast<int>(params.topMargin) + by * 2;
                    float sum = 0.0f;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            float v = dst(px + dx, py + dy)
                                      - params.blackAt(bx * 2 + dx, by * 2 + dy);
                            if (v < 1.0f) v = 1.0f;
                            // White-balance-correct so the bilateral filter sees true
                            // scene luminance, not CFA-biased raw values.  Without this,
                            // the uniform per-block scale introduces a color cast after
                            // the DNG processor applies its own WB multipliers.
                            v *= params.whiteMultAt(bx * 2 + dx, by * 2 + dy);
                            sum += v;
                        }
                    logLum(bx, by) = std::log(sum * 0.25f);
                }
            }

            // Step B: Bilateral filter on log-luminance → base layer
            const int spatialSigma = std::max(featherRadius, 8);
            const float rangeSigma = 0.4f;  // ~1.5 stops in log-space
            const int bilateralR = static_cast<int>(std::ceil(2.0f * spatialSigma));
            const float invRangeSigma2 = 1.0f / (2.0f * rangeSigma * rangeSigma);
            const float invSpatialSigma2 = 1.0f / (2.0f * spatialSigma * spatialSigma);

            // Pre-compute spatial Gaussian kernel
            const int kernelSize = 2 * bilateralR + 1;
            std::vector<float> spatialKernel(kernelSize * kernelSize);
            for (int ky = -bilateralR; ky <= bilateralR; ++ky) {
                for (int kx = -bilateralR; kx <= bilateralR; ++kx) {
                    float d2 = static_cast<float>(kx * kx + ky * ky);
                    spatialKernel[(ky + bilateralR) * kernelSize + (kx + bilateralR)] =
                        std::exp(-d2 * invSpatialSigma2);
                }
            }

            Array2D<float> logBase(bw, bh);
            {
                Timer t("Bilateral filter");
                #pragma omp parallel for schedule(dynamic, 4)
                for (int by = 0; by < bh; ++by) {
                    for (int bx = 0; bx < bw; ++bx) {
                        float center = logLum(bx, by);
                        float wSum = 0.0f;
                        float vSum = 0.0f;

                        for (int ky = -bilateralR; ky <= bilateralR; ++ky) {
                            // Mirror boundary for y
                            int ny = by + ky;
                            if (ny < 0) ny = -ny;
                            if (ny >= bh) ny = 2 * bh - 2 - ny;

                            for (int kx = -bilateralR; kx <= bilateralR; ++kx) {
                                // Mirror boundary for x
                                int nx = bx + kx;
                                if (nx < 0) nx = -nx;
                                if (nx >= bw) nx = 2 * bw - 2 - nx;

                                float val = logLum(nx, ny);
                                float spatW = spatialKernel[(ky + bilateralR) * kernelSize
                                                            + (kx + bilateralR)];
                                float diff = val - center;
                                float rangeW = std::exp(-diff * diff * invRangeSigma2);
                                float w = spatW * rangeW;
                                wSum += w;
                                vSum += w * val;
                            }
                        }

                        logBase(bx, by) = (wSum > 0.0f) ? vSum / wSum : center;
                    }
                }
            }

            // Step C: Compress base layer + reconstruct
            // Reinhard extended: L_compressed = L * (1 + L/Lw²) / (1 + L)
            // where Lw = sdrCeiling (the SDR white point in signal space)
            const float Lw = sdrCeiling;
            const float Lw2 = Lw * Lw;
            long long compressedBlocks = 0;

            #pragma omp parallel for schedule(dynamic, 16) reduction(+:compressedBlocks)
            for (int by = 0; by < bh; ++by) {
                for (int bx = 0; bx < bw; ++bx) {
                    // Check highlight mask at block center (use top-left pixel of block)
                    int ax = bx * 2;
                    int ay = by * 2;
                    float hlMask = highlightMask(ax, ay);
                    if (hlMask <= 0.0f) continue;

                    // Extract base luminance
                    float L = std::exp(logBase(bx, by));

                    // Reinhard extended shoulder compression
                    float L_compressed = L * (1.0f + L / Lw2) / (1.0f + L / Lw);

                    // Blend with mask and pull strength
                    float pm = highlightPull * hlMask;
                    float epm = 1.0f - (1.0f - pm) * (1.0f - pm);  // quadratic ease-in
                    float L_final = L + (L_compressed - L) * epm;

                    // Compute scale factor from base layer compression
                    float scale = (L > 0.01f) ? L_final / L : 1.0f;
                    if (scale < 0.01f) scale = 0.01f;
                    if (scale > 1.0f) scale = 1.0f;

                    // Apply same scale to all 4 pixels in the 2x2 block
                    int px = static_cast<int>(params.leftMargin) + ax;
                    int py = static_cast<int>(params.topMargin) + ay;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            float black = params.blackAt(ax + dx, ay + dy);
                            float signal = dst(px + dx, py + dy) - black;
                            dst(px + dx, py + dy) = signal * scale + black;
                        }
                    ++compressedBlocks;
                }
            }

            Log::debug("Bilateral highlight compression: ", compressedBlocks,
                       " blocks compressed (spatialSigma=", spatialSigma,
                       " rangeSigma=", rangeSigma, " radius=", bilateralR, ")");
        } else {
            Log::debug("Highlights already below SDR white (sdrCeiling=",
                       sdrCeiling, "), skipping bilateral compression");
        }
    }

    // Ghost artifact scoring: detect color fringe via log-ratio gradients on 2x2 Bayer blocks
    double ghostScore = 0.0;
    if (params.FC.getFilters() != 9) { // Bayer only
        const int bw = static_cast<int>(params.width) / 2;
        const int bh = static_cast<int>(params.height) / 2;
        if (bw > 4 && bh > 4) {
            // Extract per-block log color ratios
            std::vector<float> logRG(bw * bh, 0.0f);
            std::vector<float> logBG(bw * bh, 0.0f);

            #pragma omp parallel for
            for (int by = 0; by < bh; ++by) {
                for (int bx = 0; bx < bw; ++bx) {
                    // Map block to active-area pixel coordinates
                    int px = static_cast<int>(params.leftMargin) + bx * 2;
                    int py = static_cast<int>(params.topMargin) + by * 2;
                    float channels[4] = {};  // indexed by CFA color
                    int counts[4] = {};
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            int ax = bx * 2 + dx;
                            int ay = by * 2 + dy;
                            uint8_t c = params.FC(ax, ay);
                            float v = dst(px + dx, py + dy)
                                      - params.blackAt(ax, ay);
                            if (v < 1.0f) v = 1.0f;
                            channels[c] += v;
                            counts[c]++;
                        }
                    }
                    // Average each channel
                    for (int c = 0; c < 4; ++c)
                        if (counts[c] > 0) channels[c] /= counts[c];

                    // RGGB: 0=R, 1=G, 2=B, 3=G2 — average the two greens
                    float G = (counts[1] > 0 ? channels[1] : 0.0f);
                    if (counts[3] > 0) G = (G + channels[3]) * 0.5f;
                    float R = channels[0];
                    float B = channels[2];

                    int idx = by * bw + bx;
                    if (G > 1.0f && R > 1.0f)
                        logRG[idx] = std::log2(R / G);
                    if (G > 1.0f && B > 1.0f)
                        logBG[idx] = std::log2(B / G);
                }
            }

            // Sobel gradient magnitude + local deviation from 5x5 mean
            long fringeCount = 0;
            long totalBlocks = 0;
            const float gradThresh = 0.4f;
            const float devThresh = 0.3f;

            #pragma omp parallel for reduction(+:fringeCount, totalBlocks)
            for (int by = 3; by < bh - 3; ++by) {
                for (int bx = 3; bx < bw - 3; ++bx) {
                    totalBlocks++;
                    // Sobel on logRG
                    float gxRG = -logRG[(by-1)*bw + bx-1] + logRG[(by-1)*bw + bx+1]
                                 -2*logRG[by*bw + bx-1]   + 2*logRG[by*bw + bx+1]
                                 -logRG[(by+1)*bw + bx-1]  + logRG[(by+1)*bw + bx+1];
                    float gyRG = -logRG[(by-1)*bw + bx-1] - 2*logRG[(by-1)*bw + bx]
                                 -logRG[(by-1)*bw + bx+1]
                                 +logRG[(by+1)*bw + bx-1] + 2*logRG[(by+1)*bw + bx]
                                 +logRG[(by+1)*bw + bx+1];
                    float gradRG = std::sqrt(gxRG*gxRG + gyRG*gyRG) * 0.125f;

                    // Sobel on logBG
                    float gxBG = -logBG[(by-1)*bw + bx-1] + logBG[(by-1)*bw + bx+1]
                                 -2*logBG[by*bw + bx-1]   + 2*logBG[by*bw + bx+1]
                                 -logBG[(by+1)*bw + bx-1]  + logBG[(by+1)*bw + bx+1];
                    float gyBG = -logBG[(by-1)*bw + bx-1] - 2*logBG[(by-1)*bw + bx]
                                 -logBG[(by-1)*bw + bx+1]
                                 +logBG[(by+1)*bw + bx-1] + 2*logBG[(by+1)*bw + bx]
                                 +logBG[(by+1)*bw + bx+1];
                    float gradBG = std::sqrt(gxBG*gxBG + gyBG*gyBG) * 0.125f;

                    float grad = std::max(gradRG, gradBG);
                    if (grad < gradThresh) continue;

                    // 5x5 local mean deviation
                    float sumRG = 0, sumBG = 0;
                    for (int dy = -2; dy <= 2; ++dy)
                        for (int dx = -2; dx <= 2; ++dx) {
                            int idx = (by + dy) * bw + (bx + dx);
                            sumRG += logRG[idx];
                            sumBG += logBG[idx];
                        }
                    float meanRG = sumRG / 25.0f;
                    float meanBG = sumBG / 25.0f;
                    int cidx = by * bw + bx;
                    float devRG = std::abs(logRG[cidx] - meanRG);
                    float devBG = std::abs(logBG[cidx] - meanBG);
                    float dev = std::max(devRG, devBG);

                    if (dev < devThresh) continue;

                    // Classify fringe hue
                    float rg = logRG[cidx], bg = logBG[cidx];
                    bool isFringe = (rg > 0.5f && bg < 0.3f)    // pink/magenta
                                 || (rg < -0.3f && bg < 0.3f)   // green
                                 || (rg > 0.3f && bg > 0.3f)    // purple
                                 || (rg > 0.3f && bg < -0.5f);  // yellow
                    if (isFringe)
                        fringeCount++;
                }
            }

            if (totalBlocks > 0)
                ghostScore = (static_cast<double>(fringeCount) / totalBlocks) * 1000.0;
        }
    }
    Log::debug("Ghost artifact score: ", ghostScore);

    ComposeResult result;
    result.image = std::move(dst);
    result.baselineExposureEV = baselineExposureEV;
    result.numImages = numImages;
    result.ghostScore = ghostScore;

    // Copy noise profile (already estimated before compose loop)
    std::copy(noiseProfile, noiseProfile + 8, result.noiseProfile);
    Log::debug("NoiseProfile: S0=", result.noiseProfile[0], " O0=", result.noiseProfile[1],
               " S1=", result.noiseProfile[2], " O1=", result.noiseProfile[3]);

    return result;
}
