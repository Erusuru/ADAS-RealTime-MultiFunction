# Accuracy upgrade + benchmark guide

## What changed, in one paragraph

The original pipeline was dlib's HOG face detector + dlib's ResNet-34 face
encoder (via the `face_recognition` package), matched with a plain
1-nearest-neighbour Euclidean distance. The new default pipeline is
InsightFace's SCRFD face detector + its ArcFace ResNet-50 encoder
(`buffalo_l` model pack), matched with cosine similarity and a top-k
majority vote instead of trusting a single nearest sample. Both models run
through `onnxruntime`, so they use your RTX 5060 automatically once the
right onnxruntime package is installed. Nothing else about the app
changed: `main.py`, the GUI windows, encryption, and GDPR erasure all work
exactly as before -- `face_engine.py` is a thin dispatcher, so setting
`config.RECOGNITION_ENGINE` back to `"dlib"` reverts everything instantly
if you ever need to.

## Files

**New:**
- `image_quality.py` -- shared blur check, split out of the old `face_engine.py`
- `face_engine_dlib.py` -- the original pipeline, unchanged in behavior, kept for comparison
- `face_engine_insightface.py` -- the new default pipeline
- `face_engine.py` -- dispatcher; picks whichever engine `config.RECOGNITION_ENGINE` names
- `benchmark.py` -- old-vs-new accuracy AND FPS comparison (supersedes `test.py`)
- `requirements.txt` -- updated dependencies

**Changed:**
- `config.py` -- engine selection + InsightFace tuning added; old dlib tuning kept for the benchmark's baseline
- `security.py` -- each person's record is now tagged with which engine encoded them, so the two engines' embeddings (128-d Euclidean vs. 512-d cosine) can never get compared against each other by accident
- `enrollment_window.py` -- surfaces the "can't mix engines" error from `security.py` as a normal message box
- `people_window.py` -- shows which engine each person was enrolled under
- `recognition_window.py` -- warns on open if anyone in the database needs re-enrollment
- `test.py` -- replaced with a short pointer to `benchmark.py` (the old code would silently break under the new default engine -- see the file for why)

**Unchanged:** `main.py`, `main_window.py`, `logs_window.py`, `enrollment.py`, `rotate_key.py`.

## Setup

1. Install dependencies -- see `requirements.txt`. The one real decision is
   which onnxruntime package to install (only one at a time):
   - **`onnxruntime-gpu`** -- fastest, but needs a matching CUDA + cuDNN
     runtime installed system-wide. onnxruntime is mid-transition from
     CUDA-12 to CUDA-13 GPU packages, so check the current pairing at
     onnxruntime.ai before installing anything, and make sure your NVIDIA
     driver is recent enough for the RTX 5060's Blackwell/sm_120 GPU.
   - **`onnxruntime-directml`** -- simpler on Windows: no CUDA toolkit
     install, uses the RTX 5060 through DirectX 12, which your existing GPU
     driver already provides. A bit slower than a well-configured CUDA
     setup, but nothing to fight with.

   `config.INSIGHTFACE_PROVIDERS` lists both `CUDAExecutionProvider` and
   `DmlExecutionProvider` ahead of a `CPUExecutionProvider` fallback, so the
   code works with either choice (or neither, falling back to CPU) with no
   changes needed.

2. First run downloads the `buffalo_l` model pack automatically (cached
   under `~/.insightface/models/`); that needs network access once.

3. **Re-enroll everyone.** dlib's 128-d embeddings and ArcFace's 512-d
   embeddings live in different, incompatible spaces -- there's no way to
   convert one into the other after the fact. Anyone already in
   `database.enc` will show up in "Registered people" tagged
   `dlib (needs re-enroll)` and will be treated as Unknown by live
   recognition until you re-run their enrollment.

## Running the benchmark

```
python benchmark.py --dataset-path "C:\path\to\lfw-deepfunneled"
```

Optional flags: `--num-enrolled` (default 50), `--min-images` (default 5),
`--max-unknown` (default 200), `--fps-frames` (default 150), `--target-fpir`
(default `0.05,0.01`), `--skip-fps`. Run `python benchmark.py --help` for
the full list.

It prints three sections and also writes a timestamped Markdown copy to
`reports/`:

1. **Accuracy at each engine's default threshold** -- the same TPR / FRR /
   misclassification / TNR / FAR / overall-accuracy table your old
   `test.py` printed, now run for three configurations: the original dlib
   pipeline, InsightFace with plain 1-NN matching (isolates what the
   detector/encoder swap alone is worth), and InsightFace with the top-k
   vote (the live app's actual default). Comparing rows 2 and 3 tells you
   how much the voting scheme adds on top of the encoder swap; comparing
   rows 1 and 2 tells you how much the encoder swap alone is worth.

2. **Threshold sweep** -- dlib's distance and InsightFace's cosine
   similarity live on different scales, so a single fixed-threshold table
   isn't a fully fair comparison. This section instead reports, per
   configuration, what threshold you'd need to hold the impostor accept
   rate (FPIR) at ~5% and ~1%, and what genuine-reject rate (FNIR) you'd
   pay at that threshold -- an operating-point comparison instead of one
   arbitrary number, following the methodology InsightFace's own guide
   recommends for 1:N identification testing. With only a few hundred
   impostor probes from LFW this can't reliably estimate FPIR below a
   percent or so (a defensible estimate needs roughly
   10 / target_FPIR non-mate comparisons) -- read it as directional, and if
   you want a tighter estimate, raise `--max-unknown` against a bigger
   dataset.

3. **FPS** -- detection + encoding throughput on single-face frames resized
   to your live `config.FRAME_WIDTH x FRAME_HEIGHT` (640x480), for whichever
   engines are installed, plus which onnxruntime execution provider actually
   ended up running the model (CUDA vs. DirectML vs. CPU -- worth checking,
   since "requested CUDA" and "CUDA actually loaded" are two different
   things if a driver/toolkit version doesn't match). This is a ceiling
   estimate: LFW faces are large, centered, and sharp compared to a real
   hallway camera frame with a smaller, more distant, possibly moving face,
   so treat it as best-case and watch the live app's own "N face(s) in
   view" status label for real-world numbers.

I can't produce real numbers for your RTX 5060 / Ryzen AI 350 machine from
here -- this sandbox has no GPU, no webcam, and no network access to
install packages or download models. Run `benchmark.py` locally and share
the console output (or the generated Markdown report) and I can help you
read it, tune `config.COSINE_SIM_THRESHOLD`, or chase down anything that
looks off.

## What to expect, based on public numbers (not your data)

For context while you wait on your own run: dlib's bundled encoder scores
about 99.38% on the LFW verification benchmark, and InsightFace's
`buffalo_l` ArcFace encoder scores in the same 99.5-99.85% range on LFW --
but LFW is old and close to saturated for both, so it barely shows the gap
that actually matters for a hallway camera. That gap is much more visible
on harder, more realistic benchmarks: on IJB-C at a fixed 0.01% false-accept
rate, InsightFace's own published guidance puts R50/`w600k_r50`-class
models around 95-96% true-accept and R100-class around 96-97.5%, which
stresses angle, distance, and lighting variation the way a school entrance
actually does. SCRFD detection also recalls meaningfully more
angled/partial/small faces than dlib's HOG detector, which matters just as
much as the encoder for a walk-up camera -- a face the detector never finds
can't be recognized no matter how good the encoder is.

None of that is a promise about your camera, your lighting, or your
specific set of enrolled people -- it's why `benchmark.py` exists. Trust
its number over any public figure.

## Why not the NPU (Ryzen AI 350)?

The XDNA NPU is reachable from onnxruntime via the VitisAI execution
provider, but it requires the model to be quantized to INT8/BF16 and
precompiled ahead of time, and current public reports show it's still
rough around the edges (some operators silently falling back to CPU,
incomplete Linux support). The RTX 5060 already has spare headroom for
this workload without any of that, so it's the simpler and almost
certainly faster path for now. If you later want to offload detection to
the NPU to free the GPU up for something else, that's worth benchmarking
separately -- not a change to make blind.

## Other accuracy levers you haven't used yet

- `config.INSIGHTFACE_DET_SIZE` -- bump it (e.g. to 960x960) if people are
  recognized at close range but missed further from the camera; costs FPS.
- `config.FRAME_WIDTH` / `FRAME_HEIGHT` -- your camera and GPU both have
  headroom to go past 640x480, which directly helps far-away faces clear
  `MIN_FACE_WIDTH_PX`.
- `config.MATCH_TOPK` -- more votes damps out one bad enrollment sample
  further, at the cost of needing a few more well-separated identities in
  the gallery before it's meaningful.
- Liveness/anti-spoof detection isn't part of this change -- worth a
  separate look for an access-control system, since nothing above protects
  against a printed photo or a phone screen held up to the camera.
