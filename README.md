# HDRMerge

HDRMerge combines two or more raw images into a single raw with an extended dynamic range. It can import any raw image supported by LibRaw, and outputs a DNG 1.4 image with floating point data. The output raw is built using Poisson-optimal weighted merging across exposures, so that shadows maintain as much detail as possible with minimal noise. This tool also offers automatic ghost detection and a GUI to manually refine the resulting image.

## Download & Installation

Find the latest builds on our [releases page](https://github.com/acmeguy/hdrmerge/releases).

Linux users can get HDRMerge from their package manager. If your package manager does not ship the latest version of HDRMerge, file a bug report using your distribution's bug tracker asking them to ship the latest version.

## Binaries are provided by some distributions

https://repology.org/project/hdrmerge/versions

## Compilation

If you would like to compile HDRMerge yourself, follow the instructions in the `INSTALL.md` file.

## Usage

Source images can be loaded from the Open option in the File menu, or passed as arguments in the command line. They must be made with the same camera. After loading them, HDRMerge will correct misalignments. With OpenCV available, it uses feature-based alignment (AKAZE/ORB) supporting translation, rotation, scale, and perspective correction. Without OpenCV, it falls back to median threshold bitmap (MTB) translation alignment. So, if your camera allows it, you can take the shots with bracketing in burst mode. A tripod is recommended for best results, but handheld shooting works well with feature-based alignment.

Once the input images are loaded, the interface presents you with a 100% preview of the result. The selected pixels from each input image are painted with a different color. You can then pan the result to inspect it.

When some objects were moving while you took the shots, there will appear "ghosts". In CLI mode, automatic ghost detection is available via `--deghost` (sigma-clipping or gradient mode). In the GUI, you can use the toolbar to add or remove pixels from each image but the last one, until all the pixels that belong to a moving object only come from one of the input images. Usually, you will want to only remove pixels, starting with the first layer and then going down. These operations can be undone and redone with the actions of the Edit menu.

Once the preview is satisfactory, the Save HDR option of the File menu generates the output DNG file. You can select the number of bits per sample (16, 24 or 32), the size of the embedded preview (full, half or no preview) and whether to save an image with the mask that was used to merge the input files. The default is 24 bits per sample, which preserves full 14-bit sensor data with headroom. 16 bits is sufficient for most workflows and produces smaller files (and enables JXL compression when available). 32 bits is available for maximum precision.

The program can also be run without GUI, in batch mode. This is accomplished either by providing an output file name with the "-o" switch, or by generating an automatic one with the "-a" switch. Entire directories can be scanned with `-d DIR`. Multiple merge jobs can run concurrently with `-j N`. Other switches control the output parameters, refer to the output of the "--help" switch.

HDRMerge merges raw files to produce an HDR image in DNG format. Once you have obtained your image from HDRMerge, you need to further process it in an application that supports HDR images in the DNG format, such as [RawTherapee](https://rawtherapee.com) or [darktable](https://darktable.org). 

## Licence

HDRMerge is released under the GNU General Public License v3.0.
See the file `LICENSE`.

## Contributing

Fork the project and send pull requests, or send patches by creating a new issue on our GitHub page:
https://github.com/acmeguy/hdrmerge/issues

## Reporting Bugs

Report bugs by creating a new issue on our GitHub page:
https://github.com/acmeguy/hdrmerge/issues

## Changelog:

- acmeguy fork (2026)
  - **Concurrency**: Concurrent batch processing (`-j N`, defaults to half of CPU cores) with thread pool and per-job OpenMP thread allocation. Thread-safe logging via mutex. OpenMP `reduction` clause replaces manual critical sections in ImageStack.
  - **Merge algorithm**: Replaced binary pixel selection with Poisson-optimal weighted merge for cleaner HDR compositing. Noise-model-aware shadow weighting with safe Poisson/variance blend. Multi-exposure noise estimation with per-channel noise profile (variance = S*signal + O). Double-precision CFA interpolation. Configurable normalization percentile (`--clip-percentile`, default 99.9). Per-channel saturation thresholds with Bayer-block rolloff consistency.
  - **Ghost detection**: Sigma-clipping ghost detection (`--deghost`) with two modes: legacy MAD-based (`--deghost-mode sigma`) and reference-guided (`--deghost-mode gradient`, default). Soft deghosting with Bayer-block rolloff, spatial ghost coherence, and configurable iteration count (`--deghost-iterations`).
  - **Alignment**: Feature-based alignment (`--align-features`) using OpenCV (AKAZE/ORB with progressive 4/6/8 DOF geometry estimation and MTB fallback). CFA-aware even-pixel rounding to preserve Bayer pattern alignment. Pairwise chain accumulation with darkest frame as reference. Sub-pixel residual diagnostic via parabolic SSD fitting.
  - **Response function**: Linear least-squares response model (default) with R-squared diagnostic. Falls back to nonlinear spline if R-squared < 0.995 and `--response-mode nonlinear` is set. Full unsaturated range used for spline fitting (previously only top 25%).
  - **DNG writer**: Streaming tile-based DNG writer — writes to temp file with header placeholder, patches header at end, reducing peak memory. DNG version bumped to 1.7 for JXL output. Writes BASELINEEXPOSURE, BASELINENOISE, NOISEPROFILE, and DEFAULTBLACKRENDER tags. Dual-illuminant color matrix passthrough (ColorMatrix2, ForwardMatrix1/2, CameraCalibration1/2). AsShotNeutral metadata. ForwardMatrix1 computed from inverse camXyz when absent.
  - **Metadata**: ACR/Lightroom XMP profile injection (`-L PROFILE`). Default HDR settings (ProcessVersion=11.0, HDREditMode=1). Adaptive per-channel RGB tone curves via ONNX model (`--auto-curves`). Lens metadata synthesis (Xmp.aux.Lens, LensInfo, serial numbers). Application order: source EXIF, then defaults, then ACR profile overrides, then adaptive curves.
  - **Compression**: libdeflate for DNG tile compression (preferred over zlib). zlib-ng with NEON-optimized DEFLATE. JXL compression for 16-bit output. Configurable compression level (`-c 1..12`, default 6).
  - **ARM NEON**: Vectorized fattenMask (uint8x16 max operations), boxBlurT (float32x4 8-column batches), and float-to-half conversion for Apple Silicon.
  - **Batch grouping**: Two-phase bracket detection — first groups images by time gap (`--batch-gap`, default 2s), then subdivides by EV pattern (repeated exposure values quantized to third-stops trigger new bracket set). Auto-scans directories for raw files (NEF, CR2, CR3, ARW, RAF, ORF, RW2, PEF, and more). Output directory support (`-O DIR`, defaults to `merged/` subfolder).
  - **CLI**: Directory scanning (`-d DIR`) with auto-detection of directory arguments. Wall-clock and per-set timing output. Hot pixel correction (`--hot-pixel-sigma`). CFA-aware Lanczos-3 resize (`--resize-long N`). EV shift for rendering brightness (`--ev-shift`). Sub-pixel alignment (`--sub-pixel`). Default bps changed from 16 to 24.
  - **Internals**: Replaced 256-entry popcount LUT with `__builtin_popcount()`. `-march=native` for Release builds. `NO_GUI` CMake option for CLI-only builds.
  - **Bug fixes**: Fixed black/pink dot artifacts from unbounded variance weight in shadow merge. Deghost allocation fix.
  - **Build**: Requires C++11, Qt5, LibRaw >= 0.21, zlib-ng, libdeflate. Optional: OpenCV (core, features2d, calib3d, imgproc), libjxl, ONNX Runtime, ALGLIB. macOS: JXL dylibs auto-bundled into app via rpath fixup script.
- v0.6 (not released yet)
  - Allow user to specify custom white level in case of artifacts with automatically computed white level from LibRaw.
  - Added support for raw files from Fufjifilm X-Trans sensors.
  - Speed optimization.
  - Assume aperture of f/8 if the aperture is invalid.
  - Migrated from Qt4 to Qt5.
  - Enable compilation in Windows.
  - Documentation updated.
  - Repository tree restructured.
- v0.5.0:
  - First Mac OS X build! Thanks to Philip Ries for his help.
  - Several bug fixes:
    - Fix dealing with images with non-ANSI file names.
    - Calculate response function with non-linear behavior.
    - Fix file locking issues by transfering Exif tags in memory.
    - Correctly calculate the response function of very dark images.
- v0.4.5:
  - Better compatibility with other programs, by producing a DNG file that maintains the original layout: frame and active area sizes, black and white levels, etc. *Note that, if you use RawTherapee, you need v4.1.23 or higher to open these files.
  - Batch mode in command line! Merge several sets of HDR images at once.
  - Creation of menu launchers and a Windows installer.
  - Support for CYGM and Fujifilm X-Trans sensors (experimental).
  - Several bug-fixes.
  - Improved accuracy and performance.
- v0.4.4:
  - Better support for more camera models.
  - Better rendering of the embedded preview image.
  - Change the edit brush radius with Alt+Mouse wheel.
  - Several bug fixes.
    - The original embedded preview is not included in the output anymore.
    - Fixed some glitches with the edit tools.
- v0.4.3:
  - Fix segmentation fault error painting the preview of some rotated images.
  - Fix DateTime tag in Windows hosts.
- v0.4.2:
  - Improved GUI:
    - A slider to control the brush radius.
    - A slider to control the preview exposure.
    - Movable toolbars.
    - Layer selector with color codes.
    - Improved brush visibility on different backgrounds.
    - Posibility of saving the output options.
  - First release with a Windows version, both 32- and 64-bit.
- v0.4.1:
  - Bugfixes release
- v0.4:
  - Great performance improvements with OpenMP.
  - Not depend anymore on DNG & XMP SDK! Windows and Mac version soon...
  - More robust MBT alignment.
  - More control on the logging output.
  - The user may disable alignment and/or cropping. This is most useful to obtain an image of the same size as the inputs. Some programs have this requirement to apply a flat-field image, for instance.
- v0.3: This is the first public version of HDRMerge
  - Supports most raw format supported by LibRaw (No foveon of Fuji formats for the moment).
  - Automatic alignment of small translations.
  - Automatic crop to the optimal size.
  - Automatic merge mask creation. The mask identifies the best source image for each pixel of the output.
  - Editable merge mask, to manually select pixels from specific source images.
  - Writes DNG files with 16, 24 and 32 bits per pixel.
  - Writes full, half or no preview to the output image.
  - Copies the EXIF data from the least exposed source image.

## Acknowledgments

I would like to thank all the people that have contributed ideas, critics and samples to improve HDRMerge. In particular, to the team of [RawTherapee](https://github.com/Beep6581/RawTherapee).

Also, HDRMerge implements or is based on the techniques described in the following works:
- Ward, G. (2003). Fast, robust image registration for compositing high dynamic range photographs from hand-held exposures. *Journal of graphics tools*, 8(2), 17-30.
- Guillermo Luijk, Zero Noise, <http://www.guillermoluijk.com/tutorial/zeronoise/index.html>
- Jens Mueller, dngconvert, <https://github.com/jmue/dngconvert>
- Jarosz, W. (2001). Fast image convolutions. In SIGGRAPH Workshop. Code from Ivan Kuckir, <http://blog.ivank.net/fastest-gaussian-blur.html>

There is also a community forum for discussions and connecting with other users (as well as other Free Software projects) at <https://discuss.pixls.us>, hosted by [PIXLS.US](https://pixls.us).

## Links

Upstream: https://github.com/jcelaya/hdrmerge
This fork: https://github.com/acmeguy/hdrmerge
Forum: https://discuss.pixls.us/c/software/hdrmerge
