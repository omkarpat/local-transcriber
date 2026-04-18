# VoxLocal — local-transcriber

A local-first iOS transcription app. Audio is captured on-device, segmented
by a voice-activity detector, and transcribed by a small ASR model — all
running locally on the phone. A small HTTP service then adds punctuation,
sentence casing, and inverse text normalization (e.g. "ten dollars" → "$10")
to finalized segments. That service is designed to run locally during
development but can be deployed to a server later.

The repo is split into two components:

- **`VoxLocal/`** — the SwiftUI iOS app. Runs Moonshine (ASR) + Silero (VAD)
  via ONNX Runtime with the CoreML execution provider. This is the part
  that ships to a user's device.
- **`server/`** — a FastAPI + ONNX Runtime service that wraps a multilingual
  punctuation model and NeMo's WFST-based inverse text normalizer. The app
  POSTs finalized transcript segments here and streams back punctuated,
  normalized text via Server-Sent Events.

Design docs live in `plan.md`, `phase1-plan.md`, `phase2-plan.md`, and the
two `punctuation-service-*-plan.md` files.

## Swift app setup (`VoxLocal/`)

Requirements:

- macOS with **Xcode 26** or newer (the project uses `PBXFileSystemSynchronizedRootGroup`, Xcode 15+).
- An iOS 17+ simulator or device. Primary target is iPhone 14 or newer.

Steps:

1. Open `VoxLocal/VoxLocal.xcodeproj` in Xcode.
2. Let Swift Package Manager resolve `onnxruntime-swift-package-manager`
   on first open (it pulls ORT 1.24.2 with the CoreML EP bundled).
3. Select the **VoxLocal** scheme and ⌘R.

The Moonshine ONNX files are **not** checked in — on first launch the app
downloads them into `Library/Application Support/Models/moonshine-tiny/`.
Silero VAD ships in the bundle.

By default the app points at `http://127.0.0.1:8000` with
`X-API-Key: dev-key-change-me`. If the Python server isn't running, the
on-device pipeline still works; only punctuation/ITN is skipped. To change
the endpoint or key, edit `VoxLocal/VoxLocal/Services/PunctuationClient.swift`.

## Python server setup (`server/`)

The server needs a one-time model export (downloads ~1 GB, produces an
INT8-quantized ONNX model) and depends on `pynini`, which compiles against
OpenFst C++ headers.

macOS:

```bash
brew install openfst
```

Debian/Ubuntu:

```bash
apt-get install libfst-dev
```

Then from `server/`:

```bash
cd server
python -m venv .venv && source .venv/bin/activate

# pynini 2.1.6.post1 on PyPI doesn't build against openfst >= 1.8.4;
# install 2.1.7 first, then nemo_text_processing with --no-deps.
CPPFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  pip install pynini==2.1.7
pip install -r requirements.txt -r requirements-export.txt --no-deps nemo_text_processing

python export_model.py   # one-time, 3–5 minutes
```

Run it:

```bash
PUNCTUATION_API_KEY=dev-key-change-me \
  uvicorn app:app --host 127.0.0.1 --port 8000
```

Smoke-test with `curl http://127.0.0.1:8000/healthz`. A Dockerfile is
provided too — see `server/README.md` for the Docker path, the full list
of environment knobs (`MODEL_DIR`, `NUM_THREADS`, `ITN_CACHE_DIR`, …), and
operational notes.
