# HDRMerge Optimization — Session Prompt

Copy everything below the line into a new Claude Code session to continue implementation.

---

## Project

HDRMerge — C++11 HDR exposure merging tool. Combines multiple raw camera files (NEF/CR2/ARW) into a single floating-point DNG. Runs on Apple Silicon (M3/M4).

## Key Files

- **Build**: `cd build && cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DCMAKE_BUILD_TYPE=Release && make -j$(sysctl -n hw.ncpu)`
- **Binary**: `build/hdrmerge.app/Contents/MacOS/hdrmerge`
- **Core processing**: `src/ImageStack.cpp` (alignment, compositing, fattenMask), `src/Image.cpp` (alignment, response curves)
- **DNG output**: `src/DngFloatWriter.cpp` (tile compression, float conversion), `src/TiffDirectory.cpp`
- **Blur**: `src/BoxBlur.cpp` (triple box blur for feathering)
- **Bitmap**: `src/Bitmap.cpp` (MTB alignment bitmaps, `__builtin_popcount`)
- **CLI/batch**: `src/Launcher.cpp`, `src/Launcher.hpp` (CLI parsing, concurrent jobs)
- **Options**: `src/LoadSaveOptions.hpp` (SaveOptions struct: bps, featherRadius, etc.)
- **Logging**: `src/Log.hpp` (thread-safe singleton with mutex)
- **Build system**: `CMakeLists.txt` (Qt5, LibRaw, zlib, OpenMP, ALGLIB)

## Dependencies

Qt5, LibRaw (libraw_r.23 via Homebrew), zlib (currently stock `/usr/lib/libz.1.dylib`), OpenMP, ALGLIB (compiled from source at `~/alglib/cpp`), libdeflate 1.25 (installed at `/opt/homebrew/Cellar/libdeflate/1.25/`)

## Test Setup

- **RAW files**: `/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/` (1198 NEF files, Nikon Z 9, ~50 MB each)
- **Original outputs**: `/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/org/` (DNG files produced by current codebase)
- **Test output dir**: `/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/stepNN/`

**Test set** (3 bracket groups, 13 files):

| Set | Files | Brackets | Reference DNG |
|-----|-------|----------|---------------|
| A | `_EBX4622.NEF` `_EBX4623.NEF` `_EBX4624.NEF` | 3 | `org/_EBX4622-4624.dng` |
| B | `_EBX4640.NEF` .. `_EBX4644.NEF` | 5 | `org/_EBX4640-4644.dng` |
| C | `_EBX4650.NEF` .. `_EBX4654.NEF` | 5 | `org/_EBX4650-4654.dng` |

**Test command** (example for Set A):
```bash
/usr/bin/time -l build/hdrmerge.app/Contents/MacOS/hdrmerge -v \
  -o "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/test/stepNN/_EBX4622-4624.dng" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4622.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4623.NEF" \
  "/Volumes/Oryggi/Eignamyndir/RAW/20.02.26/_EBX4624.NEF"
```

## The Plan

Read `docs/implementation-plan.md` — it contains the complete 17-step optimization plan organized in 7 phases. Each step specifies the exact research reference, code to change, implementation details, verification checklist, and rollback instructions.

Read `docs/research-modern-hdr-techniques.md` — the research document with all references that the plan is based on.

## How to Proceed

1. Read both documents above to understand the full plan
2. Check the metrics log in the plan to see which steps are already completed
3. Pick up from the next incomplete step
4. For each step:
   - Create git branch `opt/stepNN-description`
   - Implement the change as specified in the plan
   - Build and run all 3 test sets (A, B, C)
   - Record metrics (wall time, file size, peak RSS, byte-identical check)
   - Verify against the checklist in the plan
   - Commit with message referencing the research section
5. One step at a time. Test after each change. Do not combine steps.

## Important Constraints

- **Quality is paramount** — no `-ffast-math`, no lossy optimizations
- Steps 0-9 must produce **bit-identical** output to baseline (pure performance, no quality change)
- Steps 12-13 intentionally change output (quality improvement) — verify visually
- Always record metrics before moving to the next step
- If a step fails verification, fix it or roll back before proceeding
