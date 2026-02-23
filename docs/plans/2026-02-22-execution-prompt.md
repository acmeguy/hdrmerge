# Execution Prompt for HDR Merge Pipeline Plan

Copy everything below the line into a new Claude Code session.

---

Read `docs/plans/2026-02-22-merge-pipeline-sota-plan.md` — this is a reviewed and research-validated implementation plan for the HDR merge pipeline. All claims have been verified against SOTA literature and the current codebase.

Execute the plan in the recommended order:

**Step 0: Hot pixel reorder**
Move `correctHotPixels()` in `src/ImageIO.cpp` to run before `align()` (currently at line 199, needs to move before line 177). The detection algorithm itself does not change — only the call site. Build and run a batch merge on `/Volumes/Oryggi/Eignamyndir/RAW/test/` to verify no regression.

**Step 1: M4 — Linear-first response mode**
In `src/Image.cpp`, make the scalar linear fit the primary path in `computeResponseFunction()` (currently it's only the fallback at lines 159-175). Fit each image against the reference directly — eliminate the pairwise chain at `src/ImageStack.cpp:242-245`. Add R^2 linearity diagnostic. Add `--response-mode linear|nonlinear` CLI switch. Keep the spline path behind `nonlinear` mode only. Build, run batch merge, compare output DNG pixel values against the current baseline to quantify the improvement.

**Step 2: M3 — Calibrated noise model**
Replace `estimateNoiseProfile()` at `src/ImageStack.cpp:683-736`. Use MAD (not sample variance) for O from black margins. Implement S estimation from active pixels. Update compose weighting to use `w_k = t_k^2 / (S * z_k + O)`. Validate NoiseProfile output. Build and test.

**Step 3: M1 — Feature alignment** (requires OpenCV)
Implement the feature-based path in `align()` with AKAZE/ORB, progressive DOF escalation (4->6->8->MTB), CLAHE-normalized grayscale previews, and optional ECC refinement. Build with `-DHAVE_OPENCV=ON` and test.

**Step 4: M2 — Reference-guided deghosting**
Replace MAD-based ghost detection with reference-guided pipeline. Use M3's noise model for threshold awareness. Add Tukey/Huber robust weighting, refinement iterations, guided filter spatial regularization. Add `--deghost-mode` and `--deghost-iterations` CLI switches.

**Step 5: M5 — X-Trans interpolation** (blocked on Fuji test data)
Replace fixed cfaStep=6 with precomputed 6x6x4 lookup table. Only affects green channel (R/B are already optimal at distance 6). Add unit tests for all 144 direction-slots.

**Important context:**
- Build command: `cd build && cmake .. -DALGLIB_ROOT=/Users/stefanbaxter/alglib/cpp -DNO_GUI=ON && make -j$(sysctl -n hw.ncpu)`
- Binary: `build/hdrmerge.app/Contents/MacOS/hdrmerge`
- Test files: `/Volumes/Oryggi/Eignamyndir/RAW/test/` (23 NEF files, 5 bracket sets, Nikon Z 9)
- Batch mode is now the default when files are provided on CLI
- Default output goes to `merged/` subfolder alongside inputs
- Quality is paramount — no `-ffast-math`, no lossy optimizations
- On ARM64 (Apple Silicon), `__SSE2__` is undefined — scalar fallback for fattenMask
- DNG compression uses zlib `compress()` (RFC 1950) — Apple libcompression is incompatible

Start with Step 0 (hot pixel reorder) and proceed sequentially. After each step, build, run the test batch, and verify no regression before moving to the next step. Use the plan document for detailed technical notes, acceptance criteria, and specific line references.
