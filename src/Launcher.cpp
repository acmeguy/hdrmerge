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
#include <iomanip>
#include <set>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <QApplication>
#include <QTranslator>
#include <QLibraryInfo>
#include <QLocale>
#include <QDir>
#include <QFileInfo>
#include "Launcher.hpp"
#include "ImageIO.hpp"
#ifndef NO_GUI
#include "MainWindow.hpp"
#endif
#include "Log.hpp"
#include <libraw.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

namespace hdrmerge {

static const QStringList rawExtensions = {
    "*.nef", "*.cr2", "*.cr3", "*.arw", "*.raf", "*.orf", "*.rw2",
    "*.pef", "*.srw", "*.3fr", "*.mrw", "*.raw", "*.sr2", "*.mef", "*.nrw"
};

static void scanDirectory(const QString & path, std::vector<QString> & fileNames) {
    QDir dir(path);
    if (!dir.exists()) {
        cerr << "Directory does not exist: " << path << endl;
        return;
    }
    QFileInfoList entries = dir.entryInfoList(rawExtensions, QDir::Files, QDir::Name);
    for (const QFileInfo & entry : entries) {
        fileNames.push_back(entry.absoluteFilePath());
    }
    Log::debug("Found ", entries.size(), " raw files in ", path);
}

Launcher::Launcher(int argc, char * argv[]) : argc(argc), argv(argv), help(false), maxJobs(0) {
    Log::setOutputStream(cout);
    saveOptions.previewSize = 2;
}


int Launcher::startGUI() {
#ifndef NO_GUI
    // Create main window
    MainWindow mw;
    mw.preload(generalOptions.fileNames);
    mw.show();
    QMetaObject::invokeMethod(&mw, "loadImages", Qt::QueuedConnection);

    return QApplication::exec();
#else
    return 0;
#endif
}


struct CoutProgressIndicator : public ProgressIndicator {
    virtual void advance(int percent, const char * message, const char * arg) {
        if (arg) {
            Log::progress('[', setw(3), percent, "%] ", QCoreApplication::translate("LoadSave", message).arg(arg));
        } else {
            Log::progress('[', setw(3), percent, "%] ", QCoreApplication::translate("LoadSave", message));
        }
    }
};


list<LoadOptions> Launcher::getBracketedSets() {
    list<LoadOptions> result;
    list<pair<ImageIO::QDateInterval, QString>> dateNames;
    for (QString & name : generalOptions.fileNames) {
        ImageIO::QDateInterval interval = ImageIO::getImageCreationInterval(name);
        if (interval.start.isValid()) {
            dateNames.emplace_back(interval, name);
        } else {
            // We cannot get time information, process it alone
            result.push_back(generalOptions);
            result.back().fileNames.clear();
            result.back().fileNames.push_back(name);
        }
    }
    dateNames.sort();

    // Remove dual-card-slot duplicates (same timestamp + same EV)
    {
        auto it = dateNames.begin();
        while (it != dateNames.end()) {
            auto next = std::next(it);
            if (next != dateNames.end()
                && it->first.start == next->first.start
                && it->first.evThird() == next->first.evThird()) {
                Log::progress("Skipping duplicate: ", next->second,
                              " (same time+EV as ", it->second, ")");
                dateNames.erase(next);
            } else {
                ++it;
            }
        }
    }

    // Phase 1: Group by time gap
    using DateName = pair<ImageIO::QDateInterval, QString>;
    list<list<DateName>> timeGroups;
    ImageIO::QDateInterval lastInterval;
    for (auto & dn : dateNames) {
        if (lastInterval.start.isNull() || lastInterval.difference(dn.first) > generalOptions.batchGap) {
            timeGroups.emplace_back();
        }
        timeGroups.back().push_back(dn);
        lastInterval = dn.first;
    }

    // Phase 2: Subdivide by EV pattern (detect repeated exposure values)
    for (auto & tg : timeGroups) {
        result.push_back(generalOptions);
        result.back().fileNames.clear();
        std::set<int> seenEVs;
        for (auto & dn : tg) {
            int evKey = dn.first.evThird();
            if (seenEVs.count(evKey) && result.back().fileNames.size() >= 2) {
                // EV repeat — start new bracket set
                result.push_back(generalOptions);
                result.back().fileNames.clear();
                seenEVs.clear();
            }
            seenEVs.insert(evKey);
            result.back().fileNames.push_back(dn.second);
        }
    }

    int setNum = 0;
    for (auto & i : result) {
        Log::progressN("Set ", setNum++, ":");
        for (auto & j : i.fileNames) {
            Log::progressN(" ", j);
        }
        Log::progress();
    }
    return result;
}


static QString applyOutputDir(const QString & fileName, const QString & outputDir) {
    if (outputDir.isEmpty()) return fileName;
    QDir outDir(outputDir);
    if (outDir.isRelative())
        outDir.setPath(QFileInfo(fileName).absolutePath() + "/" + outputDir);
    outDir.mkpath(".");
    return outDir.absolutePath() + "/" + QFileInfo(fileName).fileName();
}


int Launcher::automaticMerge() {
    auto tr = [&] (const char * text) { return QCoreApplication::translate("LoadSave", text); };
    list<LoadOptions> optionsSet;
    if (generalOptions.batch) {
        optionsSet = getBracketedSets();
    } else {
        optionsSet.push_back(generalOptions);
    }

    auto startTime = std::chrono::steady_clock::now();

    // For a single set, run sequentially (no thread overhead)
    if (optionsSet.size() <= 1 || maxJobs <= 1) {
        ImageIO io;
        int result = 0;
        for (LoadOptions & options : optionsSet) {
            if (!options.withSingles && options.fileNames.size() == 1) {
                Log::progress(tr("Skipping single image %1").arg(options.fileNames.front()));
                continue;
            }
            auto setStart = std::chrono::steady_clock::now();
            CoutProgressIndicator progress;
            int numImages = options.fileNames.size();
            int loadResult = io.load(options, progress);
            if (loadResult < numImages * 2) {
                int format = loadResult & 1;
                int i = loadResult >> 1;
                if (format) {
                    cerr << tr("Error loading %1, it has a different format.").arg(options.fileNames[i]) << endl;
                } else {
                    cerr << tr("Error loading %1, file not found.").arg(options.fileNames[i]) << endl;
                }
                result = 1;
                continue;
            }
            SaveOptions setOptions = saveOptions;
            if (!setOptions.fileName.isEmpty()) {
                setOptions.fileName = io.replaceArguments(setOptions.fileName, "");
                int extPos = setOptions.fileName.lastIndexOf('.');
                if (extPos > setOptions.fileName.length() || setOptions.fileName.mid(extPos) != ".dng") {
                    setOptions.fileName += ".dng";
                }
            } else {
                setOptions.fileName = io.buildOutputFileName();
            }
            setOptions.fileName = applyOutputDir(setOptions.fileName, setOptions.outputDir);
            Log::progress(tr("Writing result to %1").arg(setOptions.fileName));
            double ghostScore = io.save(setOptions, progress);
            double setElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - setStart).count();
            Log::msg(Log::INFO, setOptions.fileName, "  ", setElapsed, "s  ghost=", ghostScore);

            // Auto-reprocess with deghosting if ghost score exceeds threshold
            const double ghostThreshold = 5.0;
            const float autoDeghostSigma = 3.0f;
            if (ghostScore > ghostThreshold && setOptions.deghostSigma <= 0.0f) {
                Log::progress("Ghost score ", ghostScore, " > ", ghostThreshold,
                              ", reprocessing with deghosting (sigma=", autoDeghostSigma, ")");
                setOptions.deghostSigma = autoDeghostSigma;
                double dgScore = io.save(setOptions, progress);
                double dgElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - setStart).count();
                Log::msg(Log::INFO, setOptions.fileName, "  ", dgElapsed, "s  ghost=", dgScore, " (deghosted)");
            }
        }
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        Log::msg(Log::INFO, "Total processing time: ", elapsed, " seconds");
        return result;
    }

    // Multiple sets: run concurrently
    Log::progress("Processing ", optionsSet.size(), " sets with up to ", maxJobs, " concurrent jobs");

#ifdef _OPENMP
    int totalOmpThreads = omp_get_max_threads();
    int ompPerJob = std::max(1, totalOmpThreads / maxJobs);
    Log::debug("OpenMP threads per job: ", ompPerJob, " (total: ", totalOmpThreads, ")");
#endif

    std::atomic<int> globalResult(0);
    std::vector<std::thread> threads;
    std::mutex jobMutex;
    std::condition_variable jobCv;
    std::atomic<int> activeJobs(0);

    for (LoadOptions & options : optionsSet) {
        if (!options.withSingles && options.fileNames.size() == 1) {
            Log::progress(tr("Skipping single image %1").arg(options.fileNames.front()));
            continue;
        }

        // Wait if at max capacity
        {
            std::unique_lock<std::mutex> lock(jobMutex);
            jobCv.wait(lock, [&]{ return activeJobs.load() < maxJobs; });
            activeJobs++;
        }

        // Capture by value for the thread
        LoadOptions threadOptions = options;
        SaveOptions threadSaveOptions = saveOptions;

        threads.emplace_back([threadOptions, threadSaveOptions, &globalResult, &activeJobs, &jobCv, &tr
#ifdef _OPENMP
            , ompPerJob
#endif
        ]() mutable {
#ifdef _OPENMP
            omp_set_num_threads(ompPerJob);
#endif
            auto setStart = std::chrono::steady_clock::now();
            ImageIO io;
            CoutProgressIndicator progress;
            int numImages = threadOptions.fileNames.size();
            int loadResult = io.load(threadOptions, progress);
            if (loadResult < numImages * 2) {
                int format = loadResult & 1;
                int i = loadResult >> 1;
                if (format) {
                    Log::progress(tr("Error loading %1, it has a different format.").arg(threadOptions.fileNames[i]));
                } else {
                    Log::progress(tr("Error loading %1, file not found.").arg(threadOptions.fileNames[i]));
                }
                globalResult.store(1);
            } else {
                SaveOptions setOptions = threadSaveOptions;
                if (!setOptions.fileName.isEmpty()) {
                    setOptions.fileName = io.replaceArguments(setOptions.fileName, "");
                    int extPos = setOptions.fileName.lastIndexOf('.');
                    if (extPos > setOptions.fileName.length() || setOptions.fileName.mid(extPos) != ".dng") {
                        setOptions.fileName += ".dng";
                    }
                } else {
                    setOptions.fileName = io.buildOutputFileName();
                }
                setOptions.fileName = applyOutputDir(setOptions.fileName, setOptions.outputDir);
                Log::progress(tr("Writing result to %1").arg(setOptions.fileName));
                double ghostScore = io.save(setOptions, progress);
                double setElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - setStart).count();
                Log::msg(Log::INFO, setOptions.fileName, "  ", setElapsed, "s  ghost=", ghostScore);

                // Auto-reprocess with deghosting if ghost score exceeds threshold
                const double ghostThreshold = 5.0;
                const float autoDeghostSigma = 3.0f;
                if (ghostScore > ghostThreshold && setOptions.deghostSigma <= 0.0f) {
                    Log::progress("Ghost score ", ghostScore, " > ", ghostThreshold,
                                  ", reprocessing with deghosting (sigma=", autoDeghostSigma, ")");
                    setOptions.deghostSigma = autoDeghostSigma;
                    double dgScore = io.save(setOptions, progress);
                    double dgElapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - setStart).count();
                    Log::msg(Log::INFO, setOptions.fileName, "  ", dgElapsed, "s  ghost=", dgScore, " (deghosted)");
                }
            }

            activeJobs--;
            jobCv.notify_one();
        });
    }

    for (auto & t : threads) {
        t.join();
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
    Log::msg(Log::INFO, "Total processing time: ", elapsed, " seconds");
    return globalResult.load();
}


void Launcher::parseCommandLine() {
    auto tr = [&] (const char * text) { return QCoreApplication::translate("Help", text); };
    bool explicitBps = false;
    bool explicitC = false;
    for (int i = 1; i < argc; ++i) {
        if (string("-o") == argv[i]) {
            if (++i < argc) {
                saveOptions.fileName = argv[i];
            }
        } else if (string("-m") == argv[i]) {
            if (++i < argc) {
                saveOptions.maskFileName = argv[i];
                saveOptions.saveMask = true;
            }
        } else if (string("-v") == argv[i]) {
            Log::setMinimumPriority(1);
        } else if (string("-vv") == argv[i]) {
            Log::setMinimumPriority(0);
        } else if (string("--no-align") == argv[i]) {
            generalOptions.align = false;
        } else if (string("--align-features") == argv[i]) {
            generalOptions.alignFeatures = true;
        } else if (string("--no-crop") == argv[i]) {
            generalOptions.crop = false;
        } else if (string("--batch") == argv[i] || string("-B") == argv[i]) {
            generalOptions.batch = true;
        } else if (string("--single") == argv[i]) {
            generalOptions.withSingles = true;
        } else if (string("--help") == argv[i]) {
            help = true;
        } else if (string("-b") == argv[i]) {
            if (++i < argc) {
                try {
                    int value = stoi(argv[i]);
                    if (value == 32 || value == 24 || value == 16) {
                        saveOptions.bps = value;
                        explicitBps = true;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("-w") == argv[i]) {
            if (++i < argc) {
                try {
                    generalOptions.customWl = stoi(argv[i]);
                    generalOptions.useCustomWl = true;
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                    generalOptions.useCustomWl = false;
                }
            }
        } else if (string("-g") == argv[i]) {
            if (++i < argc) {
                try {
                    generalOptions.batchGap = stod(argv[i]);
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("-r") == argv[i]) {
            if (++i < argc) {
                try {
                    saveOptions.featherRadius = stoi(argv[i]);
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("-p") == argv[i]) {
            if (++i < argc) {
                string previewWidth(argv[i]);
                if (previewWidth == "full") {
                    saveOptions.previewSize = 2;
                } else if (previewWidth == "half") {
                    saveOptions.previewSize = 1;
                } else if (previewWidth == "none") {
                    saveOptions.previewSize = 0;
                } else {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("-c") == argv[i]) {
            if (++i < argc) {
                try {
                    int level = stoi(argv[i]);
                    saveOptions.compressionLevel = std::min(12, std::max(1, level));
                    explicitC = true;
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--deghost") == argv[i]) {
            if (++i < argc) {
                try {
                    saveOptions.deghostSigma = std::max(0.0f, stof(argv[i]));
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--deghost-mode") == argv[i]) {
            if (++i < argc) {
                string mode(argv[i]);
                if (mode == "legacy") {
                    saveOptions.deghostMode = DeghostMode::Legacy;
                } else if (mode == "robust") {
                    saveOptions.deghostMode = DeghostMode::Robust;
                } else {
                    cerr << tr("Invalid --deghost-mode, must be 'legacy' or 'robust'.") << endl;
                }
            }
        } else if (string("--deghost-iterations") == argv[i]) {
            if (++i < argc) {
                try {
                    int val = stoi(argv[i]);
                    saveOptions.deghostIterations = std::max(1, std::min(5, val));
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--clip-percentile") == argv[i]) {
            if (++i < argc) {
                try {
                    double pct = stod(argv[i]);
                    if (pct >= 90.0 && pct <= 100.0) {
                        saveOptions.clipPercentile = pct;
                    } else {
                        cerr << tr("--clip-percentile must be between 90 and 100.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("-L") == argv[i]) {
            if (++i < argc) {
                saveOptions.acrProfilePath = QString::fromLocal8Bit(argv[i]);
            }
        } else if (string("--auto-curves") == argv[i]) {
            saveOptions.autoCurves = true;
        } else if (string("--resize-long") == argv[i]) {
            if (++i < argc) {
                try {
                    int val = stoi(argv[i]);
                    if (val > 0) saveOptions.resizeLong = val;
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--ev-shift") == argv[i]) {
            if (++i < argc) {
                try {
                    saveOptions.evShift = stod(argv[i]);
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--response-mode") == argv[i]) {
            if (++i < argc) {
                string mode(argv[i]);
                if (mode == "linear") {
                    generalOptions.responseMode = ResponseMode::Linear;
                } else if (mode == "nonlinear") {
                    generalOptions.responseMode = ResponseMode::Nonlinear;
                } else {
                    cerr << tr("Invalid --response-mode, must be 'linear' or 'nonlinear'.") << endl;
                }
            }
        } else if (string("--hot-pixel") == argv[i]) {
            if (++i < argc) {
                try {
                    generalOptions.hotPixelSigma = std::max(0.0f, stof(argv[i]));
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--sub-pixel") == argv[i]) {
            saveOptions.subPixelAlign = true;
        } else if (string("--highlight-pull") == argv[i]) {
            if (++i < argc) {
                try {
                    float val = stof(argv[i]);
                    if (val >= 0.0f && val <= 1.0f) {
                        saveOptions.highlightPull = val;
                    } else {
                        cerr << tr("--highlight-pull must be between 0 and 1.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--highlight-rolloff") == argv[i]) {
            if (++i < argc) {
                try {
                    float val = stof(argv[i]);
                    if (val >= 0.5f && val <= 0.95f) {
                        saveOptions.highlightRolloff = val;
                    } else {
                        cerr << tr("--highlight-rolloff must be between 0.5 and 0.95.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--highlight-knee") == argv[i]) {
            if (++i < argc) {
                try {
                    float val = stof(argv[i]);
                    if (val >= 1.0f && val <= 10.0f) {
                        saveOptions.highlightKnee = val;
                    } else {
                        cerr << tr("--highlight-knee must be between 1 and 10.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--bilateral-range-sigma") == argv[i]) {
            if (++i < argc) {
                try {
                    float val = stof(argv[i]);
                    if (val >= 0.1f && val <= 2.0f) {
                        saveOptions.bilateralRangeSigma = val;
                    } else {
                        cerr << tr("--bilateral-range-sigma must be between 0.1 and 2.0.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--highlight-mask-blur") == argv[i]) {
            if (++i < argc) {
                try {
                    int val = stoi(argv[i]);
                    if (val >= 0 && val <= 30) {
                        saveOptions.highlightMaskBlur = val;
                    } else {
                        cerr << tr("--highlight-mask-blur must be between 0 and 30.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--highlight-scale-blur") == argv[i]) {
            if (++i < argc) {
                try {
                    int val = stoi(argv[i]);
                    if (val >= 0 && val <= 30) {
                        saveOptions.highlightScaleBlur = val;
                    } else {
                        cerr << tr("--highlight-scale-blur must be between 0 and 30.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("--highlight-boost-cap") == argv[i]) {
            if (++i < argc) {
                try {
                    float val = stof(argv[i]);
                    if (val >= 0.0f && val <= 16.0f) {
                        saveOptions.highlightBoostCap = val;
                    } else {
                        cerr << tr("--highlight-boost-cap must be between 0 and 16.") << endl;
                    }
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                }
            }
        } else if (string("-O") == argv[i] || string("--output-dir") == argv[i]) {
            if (++i < argc) {
                saveOptions.outputDir = QString::fromLocal8Bit(argv[i]);
            }
        } else if (string("-d") == argv[i]) {
            if (++i < argc) {
                scanDirectory(QString::fromLocal8Bit(argv[i]), generalOptions.fileNames);
            }
        } else if (string("-j") == argv[i]) {
            if (++i < argc) {
                try {
                    maxJobs = stoi(argv[i]);
                    if (maxJobs < 1) maxJobs = 1;
                } catch (std::invalid_argument & e) {
                    cerr << tr("Invalid %1 parameter, using default.").arg(argv[i - 1]) << endl;
                    maxJobs = 0;
                }
            }
        } else if (argv[i][0] != '-') {
            QString arg = QString::fromLocal8Bit(argv[i]);
            QFileInfo fi(arg);
            if (fi.isDir()) {
                scanDirectory(arg, generalOptions.fileNames);
            } else {
                generalOptions.fileNames.push_back(arg);
            }
        }
    }
    // -c implies -b 16 (JPEG XL) unless -b was explicitly given
    if (explicitC && !explicitBps) {
        saveOptions.bps = 16;
    }
    // Set default maxJobs if not specified
    if (maxJobs == 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        maxJobs = std::max(1u, hw / 2);
    }
    // Default to batch mode when files are provided
    if (!generalOptions.fileNames.empty()) {
        generalOptions.batch = true;
    }
    // Default output directory to "merged" subfolder alongside inputs
    if (saveOptions.outputDir.isEmpty() && saveOptions.fileName.isEmpty()) {
        saveOptions.outputDir = "merged";
    }
    // Auto-load default ACR profile if present alongside the binary
    if (saveOptions.acrProfilePath.isEmpty()) {
        QString candidate = QCoreApplication::applicationDirPath() + "/default_profile.xmp";
        if (QFileInfo::exists(candidate)) {
            saveOptions.acrProfilePath = candidate;
            Log::debug("Using default ACR profile: ", candidate);
        }
    }
}


void Launcher::showHelp() {
    auto tr = [&] (const char * text) { return QCoreApplication::translate("Help", text); };
    cout << tr("Usage") << ": HDRMerge [--help] [OPTIONS ...] [RAW_FILES ...]" << endl;
    cout << tr("Merges RAW_FILES into an HDR DNG raw image.") << endl;
#ifndef NO_GUI
    cout << tr("If neither -a nor -o, nor --batch options are given, the GUI will be presented.") << endl;
#endif
    cout << tr("If similar options are specified, only the last one prevails.") << endl;
    cout << endl;
    cout << tr("Options:") << endl;
    cout << "    " << "--help        " << tr("Shows this message.") << endl;
    cout << "    " << "-o OUT_FILE   " << tr("Sets OUT_FILE as the output file name.") << endl;
    cout << "    " << "              " << tr("The following parameters are accepted, most useful in batch mode:") << endl;
    cout << "    " << "              - %if[n]: " << tr("Replaced by the base file name of image n. Image file names") << endl;
    cout << "    " << "                " << tr("are first sorted in lexicographical order. Besides, n = -1 is the") << endl;
    cout << "    " << "                " << tr("last image, n = -2 is the previous to the last image, and so on.") << endl;
    cout << "    " << "              - %iF[n]: " << tr("Replaced by the base file name of image n without the extension.") << endl;
    cout << "    " << "              - %id[n]: " << tr("Replaced by the directory name of image n.") << endl;
    cout << "    " << "              - %in[n]: " << tr("Replaced by the numerical suffix of image n, if it exists.") << endl;
    cout << "    " << "                " << tr("For instance, in IMG_1234.CR2, the numerical suffix would be 1234.") << endl;
    cout << "    " << "              - %%: " << tr("Replaced by a single %.") << endl;
    cout << "    " << "-a            " << tr("Calculates the output file name as") << " %id[-1]/%iF[0]-%in[-1].dng." << endl;
    cout << "    " << "-B|--batch    " << tr("Batch mode: Input images are automatically grouped into bracketed sets,") << endl;
    cout << "    " << "              " << tr("by comparing the creation time. Implies -a if no output file name is given.") << endl;
    cout << "    " << "-g gap        " << tr("Batch gap, maximum difference in seconds between two images of the same set.") << endl;
    cout << "    " << "--single      " << tr("Include single images in batch mode (the default is to skip them.)") << endl;
    cout << "    " << "-b BPS        " << tr("Bits per sample: 16 (JPEG XL), 24 (default) or 32.") << endl;
    cout << "    " << "--no-align    " << tr("Do not auto-align source images.") << endl;
    cout << "    " << "--align-features " << tr("Use feature-based alignment (requires OpenCV). Falls back to MTB if unavailable.") << endl;
    cout << "    " << "--no-crop     " << tr("Do not crop the output image to the optimum size.") << endl;
    cout << "    " << "-m MASK_FILE  " << tr("Saves the mask to MASK_FILE as a PNG image.") << endl;
    cout << "    " << "              " << tr("Besides the parameters accepted by -o, it also accepts:") << endl;
    cout << "    " << "              - %of: " << tr("Replaced by the base file name of the output file.") << endl;
    cout << "    " << "              - %od: " << tr("Replaced by the directory name of the output file.") << endl;
    cout << "    " << "-r radius     " << tr("Mask blur radius, to soften transitions between images. Default is 3 pixels.") << endl;
    cout << "    " << "-p size       " << tr("Preview size. Can be full, half or none.") << endl;
    cout << "    " << "-v            " << tr("Verbose mode.") << endl;
    cout << "    " << "-vv           " << tr("Debug mode.") << endl;
    cout << "    " << "-w whitelevel " << tr("Use custom white level.") << endl;
    cout << "    " << "-c LEVEL      " << tr("Compression level 1-12. Implies -b 16 (JPEG XL) unless -b is also given.") << endl;
    cout << "    " << "--deghost S   " << tr("Sigma-clipping ghost detection. S is the sigma threshold (e.g. 3.0). 0=off (default).") << endl;
    cout << "    " << "--deghost-mode MODE " << tr("Deghost algorithm: 'legacy' (MAD) or 'robust' (reference-guided). Default: robust.") << endl;
    cout << "    " << "--deghost-iterations N " << tr("Refinement iterations for robust deghosting (1-5). Default: 1.") << endl;
    cout << "    " << "--clip-percentile P " << tr("Normalization percentile (90-100). Default 99.9. Use 100 for legacy behavior.") << endl;
    cout << "    " << "-j N          " << tr("Number of concurrent merge jobs in batch mode. Use -j 1 for sequential processing (each job gets all CPU cores). Default: half of CPU cores.") << endl;
    cout << "    " << "-L PROFILE    " << tr("Apply ACR/Lightroom .xmp preset to output DNGs. Overrides default_profile.xmp.") << endl;
    cout << "    " << "              " << tr("If default_profile.xmp exists next to the hdrmerge binary, it is used automatically.") << endl;
    cout << "    " << "--auto-curves " << tr("Generate per-image adaptive RGB tone curves via ONNX model.") << endl;
    cout << "    " << "--resize-long N " << tr("Resize output so longest edge is N pixels. CFA-aware Lanczos-3.") << endl;
    cout << "    " << "--ev-shift EV " << tr("Add EV stops to default rendering brightness. Does not change pixel data.") << endl;
    cout << "    " << "--response-mode MODE " << tr("Response curve mode: 'linear' (default) or 'nonlinear'. Linear is optimal for RAW.") << endl;
    cout << "    " << "--hot-pixel S " << tr("Hot/dead pixel correction. S is the sigma threshold (e.g. 3.0). 0=off (default).") << endl;
    cout << "    " << "--sub-pixel   " << tr("Apply sub-pixel alignment correction (experimental). Default off.") << endl;
    cout << "    " << "--highlight-pull S " << tr("Highlight pull strength [0, 1]. Compresses bright regions to recover window detail. 0=off (default).") << endl;
    cout << "    " << "--highlight-rolloff F " << tr("Rolloff start as fraction of saturation [0.5, 0.95]. Lower = earlier transition to shorter exposures. Default 0.9.") << endl;
    cout << "    " << "--highlight-knee K " << tr("Compression knee [1, 10]. Higher = gentler highlight compression. Default 2.0.") << endl;
    cout << "    " << "-O|--output-dir DIR " << tr("Write output files to DIR instead of alongside inputs.") << endl;
    cout << "    " << "-d DIR        " << tr("Scan directory for raw files.") << endl;
    cout << "    " << "RAW_FILES     " << tr("The input raw files or directories containing raw files.") << endl;
}


bool Launcher::checkGUI() {
    int numFiles = 0;
    bool useGUI = true;
    for (int i = 1; i < argc; ++i) {
        if (string("-o") == argv[i]) {
            if (++i < argc) {
                useGUI = false;
            }
        } else if (string("-a") == argv[i]) {
            useGUI = false;
        } else if (string("--batch") == argv[i]) {
            useGUI = false;
        } else if (string("-B") == argv[i]) {
            useGUI = false;
        } else if (string("--align-features") == argv[i]) {
            // flag only, no effect on GUI decision
        } else if (string("-c") == argv[i]) {
            ++i; // skip the value
        } else if (string("--deghost") == argv[i]) {
            ++i; // skip the value
        } else if (string("--deghost-mode") == argv[i]) {
            ++i; // skip the value
        } else if (string("--deghost-iterations") == argv[i]) {
            ++i; // skip the value
        } else if (string("--clip-percentile") == argv[i]) {
            ++i; // skip the value
        } else if (string("-j") == argv[i]) {
            ++i; // skip the value
        } else if (string("-L") == argv[i]) {
            ++i; // skip the value
        } else if (string("--auto-curves") == argv[i]) {
            // flag only, no effect on GUI decision
        } else if (string("--resize-long") == argv[i]) {
            ++i; // skip the value
        } else if (string("--ev-shift") == argv[i]) {
            ++i; // skip the value
        } else if (string("--response-mode") == argv[i]) {
            ++i; // skip the value
        } else if (string("--hot-pixel") == argv[i]) {
            ++i; // skip the value
        } else if (string("--sub-pixel") == argv[i]) {
            // flag only, no effect on GUI decision
        } else if (string("--highlight-pull") == argv[i]) {
            ++i; // skip the value
        } else if (string("--highlight-rolloff") == argv[i]) {
            ++i; // skip the value
        } else if (string("--highlight-knee") == argv[i]) {
            ++i; // skip the value
        } else if (string("-O") == argv[i] || string("--output-dir") == argv[i]) {
            ++i; // skip the value
        } else if (string("-d") == argv[i]) {
            if (++i < argc) {
                numFiles++;
            }
        } else if (string("--help") == argv[i]) {
            return false;
        } else if (argv[i][0] != '-') {
            numFiles++;
        }
    }
    return useGUI || numFiles == 0;
}


int Launcher::run() {
#ifndef NO_GUI
    bool useGUI = checkGUI();
#else
    bool useGUI = false;
#endif
    QApplication app(argc, argv, useGUI);

    // Settings
    QCoreApplication::setOrganizationName("J.Celaya");
    QCoreApplication::setApplicationName("HdrMerge");

    // Translation
    QTranslator qtTranslator;
    qtTranslator.load("qt_" + QLocale::system().name(),
                      QLibraryInfo::location(QLibraryInfo::TranslationsPath));
    app.installTranslator(&qtTranslator);

    QTranslator appTranslator;
    appTranslator.load("hdrmerge_" + QLocale::system().name(), ":/translators");
    app.installTranslator(&appTranslator);

    parseCommandLine();
    Log::debug("Using LibRaw ", libraw_version());

#ifdef NO_GUI
    if (generalOptions.fileNames.empty()) help = true;
#endif
    if (help) {
        showHelp();
        return 0;
    } else if (useGUI) {
        return startGUI();
    } else {
        return automaticMerge();
    }
}

} // namespace hdrmerge
