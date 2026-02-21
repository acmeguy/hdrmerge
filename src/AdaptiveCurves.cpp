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

#include <iostream>
#include <cmath>
#include <algorithm>
#include <QCoreApplication>
#include <QFileInfo>
#include <QDir>
#include "AdaptiveCurves.hpp"
#include "Log.hpp"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace hdrmerge {

static const char * MODEL_FILENAME = "free_xcittiny_wa14.onnx";
static const int MODEL_SIZE = 384;

QString findOnnxModel() {
    // Look next to the binary
    QString appDir = QCoreApplication::applicationDirPath();
    QString candidate = appDir + "/" + MODEL_FILENAME;
    if (QFileInfo::exists(candidate)) return candidate;

    // Look in ../share/hdrmerge/
    candidate = appDir + "/../share/hdrmerge/" + MODEL_FILENAME;
    if (QFileInfo::exists(candidate)) return candidate;

    return QString();
}


#ifdef HAVE_ONNXRUNTIME

static std::vector<float> imageToTensor(const QImage & preview) {
    QImage scaled = preview.scaled(MODEL_SIZE, MODEL_SIZE, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                           .convertToFormat(QImage::Format_RGB32);
    std::vector<float> tensor(3 * MODEL_SIZE * MODEL_SIZE);
    for (int y = 0; y < MODEL_SIZE; ++y) {
        const QRgb * scanline = (const QRgb *)scaled.constScanLine(y);
        for (int x = 0; x < MODEL_SIZE; ++x) {
            QRgb pixel = scanline[x];
            int idx = y * MODEL_SIZE + x;
            tensor[0 * MODEL_SIZE * MODEL_SIZE + idx] = qRed(pixel) / 255.0f;
            tensor[1 * MODEL_SIZE * MODEL_SIZE + idx] = qGreen(pixel) / 255.0f;
            tensor[2 * MODEL_SIZE * MODEL_SIZE + idx] = qBlue(pixel) / 255.0f;
        }
    }
    return tensor;
}


static void enforceMonotonicity(std::vector<float> & lut) {
    // Right-to-left sweep: ensure non-decreasing
    for (int i = (int)lut.size() - 2; i >= 0; --i) {
        if (lut[i] > lut[i + 1]) {
            lut[i] = lut[i + 1];
        }
    }
}


static std::vector<std::pair<int,int>> fitSplineControlPoints(const std::vector<float> & lut, int maxPoints, float maxError) {
    // Greedy spline fit: start with endpoints, iteratively add the point with max error
    std::vector<std::pair<int,int>> points;
    points.push_back({0, std::max(0, std::min(255, (int)std::round(lut[0] * 255.0f)))});
    points.push_back({255, std::max(0, std::min(255, (int)std::round(lut[255] * 255.0f)))});

    for (int iter = 0; iter < maxPoints - 2; ++iter) {
        float worstErr = 0.0f;
        int worstIdx = -1;

        for (size_t seg = 0; seg < points.size() - 1; ++seg) {
            int x0 = points[seg].first;
            int y0 = points[seg].second;
            int x1 = points[seg + 1].first;
            int y1 = points[seg + 1].second;
            for (int x = x0 + 1; x < x1; ++x) {
                float t = (float)(x - x0) / (float)(x1 - x0);
                float expected = y0 + t * (y1 - y0);
                float actual = lut[x] * 255.0f;
                float err = std::abs(actual - expected);
                if (err > worstErr) {
                    worstErr = err;
                    worstIdx = x;
                }
            }
        }

        if (worstErr < maxError * 255.0f || worstIdx < 0) break;

        int y = std::max(0, std::min(255, (int)std::round(lut[worstIdx] * 255.0f)));
        // Insert in sorted order
        auto it = points.begin();
        while (it != points.end() && it->first < worstIdx) ++it;
        points.insert(it, {worstIdx, y});
    }

    return points;
}


AdaptiveCurves predictAdaptiveCurves(const QImage & preview, const QString & modelPath) {
    AdaptiveCurves result;
    if (modelPath.isEmpty()) {
        Log::msg(Log::PROGRESS, "ONNX model not found, skipping adaptive curves");
        return result;
    }
    if (preview.isNull()) {
        Log::msg(Log::PROGRESS, "No preview available for adaptive curves");
        return result;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hdrmerge");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        Ort::Session session(env, modelPath.toLocal8Bit().constData(), opts);

        // Prepare input tensor
        std::vector<float> inputData = imageToTensor(preview);
        std::array<int64_t, 4> inputShape = {1, 3, MODEL_SIZE, MODEL_SIZE};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

        // Query output name from the model (varies by export)
        Ort::AllocatorWithDefaultOptions allocator;
        auto outNamePtr = session.GetOutputNameAllocated(0, allocator);
        const char * outputName = outNamePtr.get();

        // Run inference
        const char * inputNames[] = {"input"};
        const char * outputNames[] = {outputName};
        auto outputTensors = session.Run(Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1, outputNames, 1);

        // Extract 3x256 LUT from output (shape [1, 768] = 3 channels x 256)
        float * outputData = outputTensors[0].GetTensorMutableData<float>();
        auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t totalElements = 1;
        for (auto s : shape) totalElements *= s;
        if (totalElements < 768) {
            Log::msg(Log::PROGRESS, "Unexpected ONNX output size (", totalElements,
                     " elements, need 768), skipping adaptive curves");
            return result;
        }

        int lutSize = 256;
        std::vector<float> luts[3];
        for (int c = 0; c < 3; ++c) {
            luts[c].resize(256);
            for (int i = 0; i < 256; ++i) {
                float v = outputData[c * lutSize + i];
                luts[c][i] = std::max(0.0f, std::min(1.0f, v));
            }
            enforceMonotonicity(luts[c]);
        }

        result.red = fitSplineControlPoints(luts[0], 20, 0.005f);
        result.green = fitSplineControlPoints(luts[1], 20, 0.005f);
        result.blue = fitSplineControlPoints(luts[2], 20, 0.005f);
        result.valid = true;

        Log::msg(Log::PROGRESS, "Adaptive curves: ", result.red.size(), "/",
                 result.green.size(), "/", result.blue.size(), " control points (R/G/B)");

    } catch (const Ort::Exception & e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    } catch (const std::exception & e) {
        std::cerr << "Adaptive curves error: " << e.what() << std::endl;
    }

    return result;
}

#else // !HAVE_ONNXRUNTIME

AdaptiveCurves predictAdaptiveCurves(const QImage & preview, const QString & modelPath) {
    (void)preview;
    (void)modelPath;
    Log::msg(Log::PROGRESS, "ONNX Runtime not available: --auto-curves requires building with ONNX Runtime support");
    return AdaptiveCurves();
}

#endif // HAVE_ONNXRUNTIME

} // namespace hdrmerge
