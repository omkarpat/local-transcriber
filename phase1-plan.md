# Phase 1: Core Local Pipeline — Detailed Task Plan

**Goal**: Mic → VAD → ASR → text on screen, fully on-device.
**Timeline**: Weeks 1–4 (~80–100 dev hours)
**Exit Criteria**: User can record speech and see raw (unpunctuated) transcript appear in real-time, fully offline, on an iPhone.

---

## Progress

- [x] **Task 1** — Project Scaffolding & ONNX Runtime Setup (commit `d93fe2f`)
  - Folder structure: `App/`, `Audio/`, `VAD/`, `ASR/`, `Pipeline/`, `Views/`, `Utilities/`
  - SPM deps: `onnxruntime-swift` 1.24.2, `swift-collections` 1.4.1
  - `Utilities/OnnxRuntimeSetup.swift` — shared `ORTEnv` singleton + CoreML EP session-options factory
  - `Utilities/SmokeTest.swift` + bundled `VAD/silero_vad.onnx` (2.3MB)
  - Smoke test verified on iPhone 17 Pro simulator: session creates, CoreML EP partitions the graph (18/39 nodes accelerated), inputs `[input, state, sr]`, outputs `[output, stateN]`
  - `CLAUDE.md` at repo root captures Xcode 26 synchronized-group gotcha
- [x] **Task 2** — Audio Capture Manager (commit `50d9fbe`)
  - `Audio/AudioFormat.swift` — 16kHz / mono / Float32 constants + `AVAudioFormat` factory
  - `Audio/CircularAudioBuffer.swift` — lock-free SPSC ring buffer using Swift 6 `Synchronization.Atomic`; drop-oldest overflow with running `overflowSamples` counter; capacity rounded to next power of 2 for mask-based indexing
  - `Audio/Permissions.swift` — mic permission via `AVAudioApplication.requestRecordPermission()` (iOS 17+ API)
  - `Audio/AudioCaptureManager.swift` — `AVAudioSession` (`.record` / `.measurement` / `.duckOthers`), `AVAudioEngine` input tap at hardware rate, `AVAudioConverter` to 16kHz mono Float32, writes to ring buffer; exposes `currentRMS` and `framesCaptured` via atomics
  - `Views/AudioCaptureDebugView.swift` + `ContentView.swift` — NavigationStack entry with live level bar, Written/Drained/Overflow counters; debug view drains the ring buffer every 50ms so overflow stays at 0 end-to-end
  - `NSMicrophoneUsageDescription` added via `INFOPLIST_KEY_*` in both Debug and Release build configs (project uses `GENERATE_INFOPLIST_FILE = YES`, no file-based plist)
  - Verified on iPhone 17 Pro simulator: RMS responds to mic, Written grows at 16kHz, Drained tracks it, Overflow stays at 0
- [x] **Task 3** — Silero VAD Integration (commit `2919708`)
  - `VAD/VADConfiguration.swift` — thresholds and durations (speech threshold 0.5, min speech 250ms, end-of-speech silence 700ms, pre-speech padding 300ms, max utterance 30s). Uses Swift's `Duration` type with integer sample/frame conversions.
  - `VAD/SileroVADModel.swift` — ONNX wrapper for Silero VAD v5. Manages three persistent tensors: recurrent state `Float32[2,1,128]`, 64-sample context, and Int64 sample rate. Per-call input is `[64 context + 512 new] = 576 samples` at shape `[1, 576]`. CoreML EP on by default, CPU-only fallback available for diagnostics.
  - `Utilities/WAVWriter.swift` — writes Float32 mono samples to 16kHz 16-bit PCM WAV in the app's Documents folder (for offline Moonshine debugging in Task 4).
  - `VAD/VADProcessor.swift` — detached `Task` consumer loop that reads 512-sample frames from the `CircularAudioBuffer`, runs `SileroVADModel`, and implements an idle → pendingSpeech → speaking → pendingSilence state machine. Pre-roll buffer holds the last 300ms so word onsets aren't clipped. Events fan out via `AsyncStream<VADEvent>`.
  - `Views/AudioCaptureDebugView.swift` extended — speech probability bar + color-coded state pill (gray/orange/green), utterance count, last utterance duration and filename. Subscribes to the VAD event stream on a `Task`; each `utteranceEnded` event writes a WAV.
  - Verified on iPhone 17 Pro simulator: probability cycles 1.0 during speech / <0.02 in silence, state pill cycles through IDLE → LISTENING → SPEAKING → IDLE, utterances close cleanly after ~700ms of silence, WAVs land in Documents, ring-buffer overflow stays at 0.
- [x] **Task 4** — Moonshine ASR Integration (complete through subtasks 4.1–4.6)
  - Subtasks carved up as: **4.1** inspect + document (done), **4.1b** ModelAssets + first-run downloader (done), **4.2a** forward pass (done), **4.2b** tokenizer (done), **4.3** partial transcription during active speech (done), **4.4** VAD→ASR wiring (done), **4.5+4.6** benchmark + error handling (done). **Task 4 complete.**
  - [x] **4.3** — Partial transcription during active speech (commit TBD)
    - `ASR/TranscriptUpdate.swift` — enum gains a `.partial(TranscriptPartial)` case alongside existing `.finalized`/`.failed`. `TranscriptResult`/`TranscriptFailure` both gain `utteranceID: UUID`; a new `TranscriptPartial` struct carries the same shape minus `timings` (partials are best-effort, not instrumented). The enum's `utteranceID` computed property lets the UI group partials + final for one utterance onto a single row.
    - `VAD/VADConfiguration.swift` — new `partialTranscriptionInterval: Duration? = nil` (nil = off, matches pre-partial behavior). Measured in frames internally, so the effective interval rounds to the nearest multiple of `frameDuration` (~32 ms).
    - `VAD/VADProcessor.swift` — `VADEvent` now threads `utteranceID: UUID` through `.utteranceStarted`/`.utteranceEnded` and adds a new `.utteranceProgress(utteranceID, samples, duration)` case. The processor mints a UUID on the `.pendingSpeech → .speaking` transition, reuses it across all progress events + the final utteranceEnded, and emits `.utteranceProgress` every `partialFrameCount` frames while in `.speaking` or `.pendingSilence`. Frame-counting (not a `Timer` / `ContinuousClock`) is deterministic, adds no drift, and doesn't spawn a second async task racing the read loop.
    - `ASR/MoonshineTranscriber.swift` — new `enqueuePartial(utteranceID:samples:duration:)` runs Moonshine on the buffer-so-far and yields a `.partial` update. Gated by an actor-isolated `PartialGate` (`Set<UUID>` of in-flight utterances) — if a partial for the same utterance is still running when the next tick fires, the new tick is dropped (let-finish, not cancel-and-restart). Partials re-transcribe the full growing buffer each tick rather than attempting LocalAgreement-style prefix-stitching; compute is O(utterance length) per tick but bounded by VAD's 30 s cap, and avoids the class of word-boundary bugs that come with overlapping-window ASR on a batch model. Final `enqueue` now also takes `utteranceID` and threads it through `TranscriptResult`. Partial inference errors are logged and dropped (UI only surfaces finalized failures).
    - `Views/AudioCaptureDebugView.swift` — `transcripts` changes from `[TranscriptUpdate]` (append-only) to `[TranscriptRow]` keyed by `utteranceID` with upsert-in-place logic: new `.partial` inserts at top or replaces the existing partial row; `.finalized` replaces the row and locks it (late partials returning after final are ignored); `.failed` same treatment. Partial rows render italic + `…` suffix + secondary foreground to distinguish from finalized rows. `apply(_:)` gains a `.utteranceProgress` branch dispatching to `enqueuePartial`. New "Partial interval" `Picker` in the controls: `Off / 500 / 750 / 1000 / 1250 / 1500 / 1750 / 2000 / 2500 / 3000 ms` (default 1500 ms), disabled while running (change requires stop→start); nil maps to `partialTranscriptionInterval = nil` to restore pre-partial behavior exactly.
    - Verified on iPhone 17 simulator: long utterance shows a single top row whose italic text grows every ~1.5 s and flips to non-italic authoritative text on silence; short utterances skip partials entirely; Off recovers the pre-partial flow.
  - [x] **4.5 + 4.6** — Benchmarking and basic error handling (commit `b7ae70c`)
    - `ASR/MoonshineModel.swift` — `transcribe(samples:)` now returns a `TranscriptionResult { tokens, timings }` where `TranscriptionTimings` records per-phase wall time (encoder, first-decode-call, cache-decode-calls, total) plus a cache-call count. Callers that only needed tokens now read `result.tokens`; all pipeline sites updated.
    - `ASR/TranscriptUpdate.swift` — converted from a struct to an enum with `.finalized(TranscriptResult)` and `.failed(TranscriptFailure)`. `TranscriptResult` carries the decoded text + timings; `TranscriptFailure` carries the error message. Both have UUID IDs and satisfy `Identifiable` via a switch on the enum. Phase 1 ignores partial/streaming results — they're always `.finalized`.
    - `ASR/MoonshineTranscriber.swift` — now emits `.failed(TranscriptFailure)` when inference throws instead of silently dropping + logging. Success path still goes through `.finalized` with the full `TranscriptionTimings` carried through.
    - `Views/AudioCaptureDebugView.swift` — renders the transcript list via a switch on the update variant. Failures appear with a ⚠︎ prefix, red color, and the error description. Model-load failure is already surfaced via `errorMessage` + `isLoadingModels = false` from 4.4.
    - `Utilities/SmokeTest.swift` — added `runMoonshineBenchmark(iterations:)`: loads a fresh `MoonshineModel` for CoreML EP on, runs one warmup + N measured iterations, then repeats for CoreML off (CPU only). Prints min/median/mean/p95 for total, encoder, 1st-decode, cache-loop phases, with RTF against the audio duration. Discards warmup to exclude graph-compile costs from the stats.
    - `Views/ContentView.swift` — new "Benchmark Moonshine (CoreML vs CPU)" button dispatches the benchmark off the main actor.
    - New `phase1-benchmarks.md` captures the first round of simulator numbers. Noteworthy finding: **CPU-only beats CoreML EP on the simulator** (no real ANE to dispatch to), ~8× faster session load, ~3× faster inference. Expected to flip on physical device (Task 7). Both configs are already well under the Phase 1 RTF < 0.5 target.
    - **Deferred for 4.6** (noted in the file): inference timeout (ORT doesn't support mid-run cancellation from Swift), memory pressure `didReceiveMemoryWarning` handling (low priority — total RAM use is ~200 MB). Add when device testing shows them biting.
  - [x] **4.4** — MoonshineTranscriber consuming VAD events (commit `631215d`)
    - `ASR/TranscriptUpdate.swift` — Sendable value type with `id`, `text`, `utteranceDuration`, `inferenceDuration`, `tokenCount`; computed `realTimeFactor` for UI display.
    - `ASR/MoonshineTranscriber.swift` — `nonisolated final class @unchecked Sendable` holding a pre-loaded MoonshineModel + MoonshineTokenizer. Push-based `enqueue(samples:duration:)` spawns a `Task.detached` per utterance (parallel transcription; RTF << 1 means back-to-back utterances don't queue up under normal speech cadence). Emits completed results via `AsyncStream<TranscriptUpdate>`.
    - `Views/AudioCaptureDebugView.swift` — `AudioCaptureDebugModel` now loads model + tokenizer once on first Start (spinner shown via `isLoadingModels`). Loading is wrapped in `Task.detached` so the ~2s ORT session init doesn't block MainActor. On `utteranceEnded`, the debug model both dumps a WAV (preserved for offline iteration) AND calls `transcriber.enqueue(...)`. Subscribes to `transcriber.updates` and shows a scrolling transcript list (newest first, capped at 20).
    - `VAD/SileroVADModel.swift` — now explicitly `nonisolated`. Without this, the project's default `@MainActor` isolation forced a main-actor hop inside `VADProcessor`'s detached task, which made CoreAudio / ORT spam "This method should not be called on the main thread" warnings on every frame.
    - Verified on iPhone 17 Pro simulator: speak → pause (>700ms) → transcript shows up in the list with duration + RTF + token count. Model loads once per session and is reused across start/stop cycles. No main-thread warnings after the fix.
  - **Former Phase 1 limitation, now resolved in 4.3**: finalized transcripts still only emit after VAD closes an utterance (≥ 700 ms of silence per `VADConfiguration.endOfSpeechSilence`), but `.partial` updates now fire every `partialTranscriptionInterval` (default 1500 ms) while speech is ongoing so the UI can show a best-guess transcript growing live during long utterances. See 4.3 for design notes and the let-finish vs cancel-and-restart tradeoff.
  - [x] **4.2b** — swift-transformers tokenizer (commit `a940b93`)
    - Added `swift-transformers` 1.3.0 SPM dependency (product **Tokenizers** only; Hub and Transformers aren't needed). Added via Xcode's Package Dependencies GUI — no pbxproj hand-editing.
    - `ASR/MoonshineTokenizer.swift` — thin wrapper around `AutoTokenizer.from(modelFolder:)` that points at `ModelAssets.installDirectory(for:)`. Single method `decode(tokenIDs: [Int64], skipSpecialTokens: Bool = true) -> String`. `nonisolated final class @unchecked Sendable` so it can be loaded off the main actor.
    - `Utilities/SmokeTest.swift` — `runMoonshineInference()` is now `async`; loads both model + tokenizer (2.07s on sim, dominated by ORT session init), decodes the token IDs, prints the transcript string alongside the raw IDs and RTF.
    - `Views/ContentView.swift` — button wraps call in `Task.detached(priority: .userInitiated) { await SmokeTest.runMoonshineInference() }`.
    - Verified on iPhone 17 Pro simulator: same 14-token sequence from 4.2a decodes to `"Yes, it's it's working really nice now."` — matches the utterance; the "it's it's" subsequence really was a double-tap in the source audio, not a greedy-decode stutter.
  - [x] **4.2a** — MoonshineModel forward pass (commit `fdaf2a0`)
    - `ASR/MoonshineModel.swift` — `nonisolated final class @unchecked Sendable` holding three `ORTSession`s (encoder + no-cache decoder + with-past decoder). `transcribe(samples:) -> [Int64]` runs encoder once, seeds KV cache from `decoder_model.onnx`, then loops `decoder_with_past_model.onnx` with argmax greedy decode until EOS (token 2) or 512 tokens. Self-attn KV grows per step; cross-attn KV is captured from the first call and reused unchanged on every subsequent call.
    - `Utilities/WAVReader.swift` — reads 16-bit PCM WAV back to `[Float]` via `AVAudioFile` (no hand-rolled RIFF parser). Helper `mostRecentUtterance(in:)` picks the latest `utterance-*.wav` in Documents so the smoke test can transcribe the last thing the VAD captured without requiring a fresh recording.
    - `Utilities/SmokeTest.swift` — new `runMoonshineInference()` that grabs latest utterance, loads MoonshineModel, runs greedy decode, prints RTF + token IDs.
    - `Views/ContentView.swift` — added a "Run Moonshine on latest utterance" button. Wraps the SmokeTest call in `Task.detached(priority: .userInitiated)` because AVAudioFile and ORT run-loops complain with runtime warnings if invoked on the main thread.
    - Verified on iPhone 17 Pro simulator: 3.6s utterance → **14 tokens in 0.31s, RTF=0.09**, final token == 2 (EOS), deterministic across repeated runs on the same WAV. CoreML EP covers ~50% of nodes per session (encoder 299/569, decoder_no_cache 397/805, decoder_with_past 351/649); the remaining CPU fallback is expected for shape-related ops. Real accuracy sanity-check waits for the tokenizer in 4.2b.
  - [x] **4.1b** — ModelAssets + first-run downloader (commit `dace0b7`)
    - `ASR/ModelAssets.swift` — `ModelBundle`, `RemoteFile`, and the `moonshineTiny` constant pinning HF source URLs and exact byte sizes (captured 2026-04-16). Helpers: `installDirectory(for:)` (creates `Library/Application Support/Models/<id>/` with `isExcludedFromBackup = true` set on the parent so children inherit), `isInstalled(_:)` (size-matches every file), `remove(_:)` (for dev reset).
    - `ASR/ModelDownloader.swift` — `nonisolated final class` that vends an `AsyncThrowingStream<DownloadProgress, Error>`. Sequential `URLSession.download(from:)` per file, HTTP status check, atomic move via sibling `.tmp` → final name, size verification before commit. Skip-if-already-present logic lets partial installs resume without redownloading completed files. Logs start / per-file / completion / error events to stdout with elapsed time + MB/s.
    - `App/AppState.swift` — `@Observable` main-actor class. States: `.checking / .downloading(DownloadProgress) / .ready / .failed(String)`. `bootstrap()` is idempotent; `startDownload()` owns the current `Task`; `resetInstall()` is dev-only and wipes the install dir. `#if DEBUG`-gated `static func previewing(_:)` lets SwiftUI previews construct state variants without breaking the `private(set)` on `installState`.
    - `Views/ModelDownloadView.swift` — first-run UI with progress bar, current-file name + index/count, bytes-done/total via `ByteCountFormatter`, error + retry state. Two `#Preview`s.
    - `App/VoxLocalApp.swift` — root `WindowGroup` now wraps a `RootView` that switches between `ContentView` (when `.ready`) and `ModelDownloadView` (otherwise). `.task { appState.bootstrap() }` triggers the check on first view appearance.
    - `Views/ContentView.swift` — added a Developer section with a destructive "Reinstall Models" button wired to `appState.resetInstall()`, so we can exercise the first-run flow without reinstalling.
    - Deferred (noted in deviations): SHA256 integrity check, resume-on-failure mid-file, retry backoff, WiFi-only toggle, background `URLSession` config. Size-match is the only integrity check for now.
    - Build: `xcodebuild … iPhone 17 Pro simulator` completes clean. Runtime verification of the actual download flow is pending user-driven simulator run (hit ▶ in Xcode).
  - [x] **4.1** — Moonshine Tiny ONNX inspection (commit `fa8de68`)
    - Source repo: [`onnx-community/moonshine-tiny-ONNX`](https://huggingface.co/onnx-community/moonshine-tiny-ONNX) — HF-transformers-style layout with tokenizer JSONs at top level and `onnx/` subfolder containing the models. Preferred over `UsefulSensors/moonshine` because tokenizer files live alongside the models.
    - Variant chosen: **float** (`encoder_model.onnx` 30.9 MB + `decoder_model_merged.onnx` 78.2 MB + tokenizer/config JSONs ~3.9 MB = ~113 MB total). Quantized int8 variant exists at ~28 MB but deferred; correctness baseline first, optimization later.
    - Decoder pattern: `_merged` unifies the no-cache (first call) and with-past (cached call) decoders via a `use_cache_branch` bool input. One file, standard HF Optimum export format, what swift-transformers expects.
    - **Encoder I/O**: input `input_values [batch, num_samples]` float32 (raw PCM, 16 kHz — **no mel spectrogram**, which is a Moonshine design point vs. Whisper). Output `last_hidden_state [batch, T, 288]` where `T ≈ floor(floor(floor(num_samples/64 - 127/64)/3)/2) - 1` ≈ num_samples/384 → **~40 encoder frames per second of audio**.
    - **Decoder I/O (merged, 27 inputs / 25 outputs)**:
      - `input_ids [batch, dec_seq]` int64; `encoder_hidden_states [batch, enc_seq, 288]` float32; 24 past KV tensors (6 layers × {self-K, self-V, cross-K, cross-V}, each `[batch, 8, past_len, 36]`); `use_cache_branch [1]` bool.
      - Outputs: `logits [batch, dec_seq, 32768]` + 24 `present.*` KV tensors (shape grows by 1 step for self-attn; cross-attn KV is invariant after first call).
    - Config facts: `hidden_size = 288`, `vocab_size = 32768`, 6 enc + 6 dec layers, 8 KV heads (no GQA), head dim 36, `bos/decoder_start = 1`, `eos = 2`, `max_position_embeddings = 512`. `preprocessor_config.json` declares only `sampling_rate: 16000` — no mel, no normalization params to replicate.
    - **Inference loop** (greedy baseline, what 4.2a will implement):
      1. Run encoder once on the full utterance PCM.
      2. First decoder call: `input_ids = [1]`, `use_cache_branch = false`, past KVs as empty `[1,8,0,36]` tensors. Take `argmax(logits[:, -1, :])` → first token.
      3. Loop: feed the new token as `input_ids`, promote last step's `present.*` → `past_key_values.*`, flip `use_cache_branch = true`. Stop on token 2 (EOS) or length 512.
    - Files live in `scratch/moonshine-tiny/` (gitignored). Throwaway inspection script: `scratch/moonshine-tiny/inspect_io.py`.
- [ ] **Task 5** — Transcription Pipeline Orchestrator
- [ ] **Task 6** — Minimal Transcription UI
- [ ] **Task 7** — Physical Device Testing & Battery Profiling

### Notes & deviations from plan

- **Task 5 scope expansion (5.6–5.9)**: cloud punctuation was originally Phase 3 and optional in `plan.md`. Per a product decision, it's been promoted to **mandatory** and moved into Phase 1 Task 5. Local Moonshine stays the source of truth for partial + raw finalized output; a separate `Services/PunctuationClient` posts finalized text to the FastAPI+ONNX service in `server/` (`oliverguhr/fullstop-punctuation-multilingual-base`, see `punctuation-service-plan.md`) and the UI upserts the polished response in place. Timeouts / network failures degrade silently to the raw local transcript — the app still works offline, just without punctuation. Server runs locally (uvicorn or Docker) for integration testing; ECS deploy remains future work.
- Deployment target is **iOS 26.4** (Xcode 26 default), not iOS 17. Still satisfies "17+" from the architecture doc. Can lower later if broader device support becomes a goal.
- Swift build uses `SWIFT_APPROACHABLE_CONCURRENCY = YES` + `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`. Means types default to `@MainActor`; background-thread utilities (like `OnnxRuntimeSetup`) must opt out with `nonisolated`.
- Task 1.6 ("smoke test on physical device") was run on the **simulator** instead of a registered iPhone, because no device has been registered with the Personal Team yet. Simulator run is sufficient to validate the toolchain; physical-device ANE verification is deferred to Task 7.
- Task 1 used Silero VAD as the smoke-test model (rather than a synthetic one-op model). Rationale: the model is needed for Task 3 anyway, and loading a real recurrent-state model stresses the CoreML EP more realistically than a trivial graph.
- Task 2 reordered subtasks: 2.4 (permissions) was written before 2.3 (capture manager) so the manager could assume permission was already granted upstream. Plan-defined order was 2.1 → 2.2 → 2.3 → 2.4 → 2.5; actual order was 2.1 → 2.2 → 2.4 → 2.3 → 2.5.
- Task 2 used Swift 6 stdlib `Synchronization.Atomic<T>` rather than `swift-atomics` SPM or `TPCircularBuffer`. Rationale: iOS 18+ target makes it available, no third-party C dependency, ~80 lines of Swift we fully understand. `Atomic<Float>` doesn't exist, so `rms` is stored as `Atomic<UInt32>` via `Float.bitPattern`.
- Task 2 tap thread hazard: project default is `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, so classes are `@MainActor` by default. The `AVAudioEngine` tap closure runs on the real-time audio thread and must not cross to the main actor. Solution: a separate `nonisolated`, `@unchecked Sendable` `AudioTapState` class holds everything the tap touches (converter, ring buffer, RMS atomic, output buffer); the tap closure captures that reference, not the manager's `self`.
- Task 2 AVAudioSession mode is `.measurement` rather than `.default`. Rationale: `.measurement` disables iOS's default speech-processing chain (AGC, noise suppression) so the VAD / ASR see the raw mic signal.
- Task 2 format conversion: the hardware mic runs at 48kHz (or the route's native rate); requesting 16kHz on `AVAudioSession.preferredSampleRate` is a hint only and often overridden. The tap is installed at hardware rate, `AVAudioConverter` does the 48→16kHz / stereo→mono conversion per-buffer inside the tap. Pre-allocated output buffer, no allocations in the hot path.
- Task 2 physical-device testing (permission dialog, real mic routing, battery impact) deferred to Task 7 along with the rest of on-device verification.
- Task 3 non-obvious finding: Silero VAD v5's ONNX export does **not** accept a bare 512-sample frame — it expects `[64 context + 512 new] = 576` samples per call, where the 64-sample context is the last 64 samples of the previous frame. The official `silero-vad` Python package hides this inside its `OnnxWrapper` class. Feeding only 512 samples makes the model silently return ~0.001 for everything, including clean speech. Our `SileroVADModel` now maintains the 64-sample context buffer alongside the LSTM state. Verified against Python ORT ground truth on JFK's inaugural speech (68% of frames cross 0.5 threshold).
- Task 3 VAD warm-up: the LSTM state is zero on init; first ~6 frames (~200ms) of real speech may produce muted probability even after context fix is in place. Current design relies on the natural human-tap-to-speak delay (>500ms) to warm the model on ambient silence. Explicit init-time warm-up deferred unless Task 7 testing shows onset clipping.
- Task 3 added `Utilities/WAVWriter.swift` as a debug-only side-channel: each detected utterance is dumped to `Documents/utterance-<ms>.wav`. Purpose is to have a corpus of real captured utterances to feed offline into Moonshine during Task 4 development (so we can iterate on ASR without re-recording each time).
- Task 3 also added `OnnxRuntimeSetup.makeCPUSessionOptions()` and `SileroVADModel(useCoreML: Bool)` as diagnostic tooling (so we can force CPU-only inference when debugging CoreML EP issues). The normal path still uses CoreML EP.
- Task 4 scope split: plan-defined Task 4 was a single block. Broken into 4.1 (inspect), 4.1b (**new** — ModelAssets + first-run downloader), 4.2a (forward pass), 4.2b (tokenizer), 4.4 (VAD→ASR wiring), 4.5+4.6 (benchmark + error). Rationale: smaller commits, and inserting 4.1b early means Moonshine ONNX files never get committed to git (Silero VAD at 2.3 MB is fine to commit; Moonshine at ~110 MB would bloat the repo permanently, and removing it later requires destructive history rewrites — especially relevant if we later swap to Whisper).
- Task 4.1b is **not** in the original plan. Models load from `Library/Application Support/Models/moonshine-tiny/` (with `isExcludedFromBackupKey = true` so they don't inflate iCloud backups). First-run UI kicks off a `URLSession` download from the HF repo. Deferred for simplicity: SHA256 verification, resume-on-failure, retry UI, WiFi-only toggle — add when they start hurting.
- Task 4.1 deviation from plan 4.1 wording: original says "Moonshine v2 has separate encoder and decoder ONNX files." Reality is the HF Optimum export produces three decoder flavors (no-cache, with-past, merged). We use **`decoder_model_merged.onnx`** — one file that handles both first-call and cached-call via a `use_cache_branch` bool input. Standard pattern, matches what swift-transformers expects.
- Task 4.1 deviation: plan 4.2 says "Moonshine uses a BPE/SentencePiece tokenizer — either port the tokenizer to Swift or use a lightweight C++ binding." Decision: use [`swift-transformers`](https://github.com/huggingface/swift-transformers) SPM — it reads the `tokenizer.json` directly, one import, no C++. Will be wired in Task 4.2b.
- Task 4.2a deviation: switched from **`decoder_model_merged.onnx`** (single file, Task 4.1's original choice) to the **split pair** `decoder_model.onnx` + `decoder_with_past_model.onnx`. Reason: the merged decoder has a `use_cache_branch` bool tensor input, but the Objective-C ONNX Runtime binding in this SPM (verified in `ort_enums.mm`) does not expose a BOOL element type — its enum lists only float / int8 / uint8 / int32 / uint32 / int64 / uint64 / string. The two alternatives were (a) swap to split decoders (+66 MB on disk, pure Swift), or (b) write a small Objective-C++ shim exposing a bool-tensor constructor (~113 MB on disk, but Swift/C++ interop). Chose (a) because bytes are easier to reverse than language interop; the plan records this so we can revisit if the 66 MB ever hurts. `ModelAssets.moonshineTiny.files` updated accordingly; existing installs will redownload on next launch. (The orphan `decoder_model_merged.onnx` left behind isn't cleaned up automatically; "Reinstall Models" in ContentView wipes the dir.)
- Task 4.2a: `MoonshineModel` is explicitly marked `nonisolated` to opt out of the project's default-`@MainActor` policy. Inference is compute-heavy and must never block the UI. The button in `ContentView` wraps the call in `Task.detached(priority: .userInitiated)` — without that wrap, AVAudioFile + ORT runtime both emit "This method should not be called on the main thread" warnings.

---

## Task Breakdown

### 1. Project Scaffolding & ONNX Runtime Setup

| Field | Detail |
|---|---|
| **Estimate** | 1–2 days |
| **Dependencies** | None (starting point) |
| **Risk** | 🟡 Medium — ONNX Runtime Swift package can have version/compatibility issues with Xcode |

**Subtasks**:

- **1.1** Create Xcode project (iOS 17+, Swift 6 strict concurrency, SwiftUI lifecycle)
- **1.2** Add `onnxruntime-swift` SPM dependency; verify it builds for both simulator and physical device
- **1.3** Add `swift-collections` SPM dependency (for `Deque`-based audio buffering)
- **1.4** Set up project folder structure matching the architecture doc (`Audio/`, `VAD/`, `ASR/`, `Pipeline/`, `Views/`, `Utilities/`)
- **1.5** Create `OnnxRuntimeSetup.swift` — shared `OrtEnvironment` singleton with CoreML execution provider configuration
- **1.6** Verify a trivial ONNX model loads and runs inference on a physical device (smoke test for the full toolchain)

**Acceptance Criteria**: Project builds on simulator + device. ONNX Runtime initializes successfully with CoreML EP enabled. Folder structure in place.

---

### 2. Audio Capture Manager

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 1 (project scaffolding) |
| **Risk** | 🟢 Low — AVAudioEngine is well-documented; main risk is threading discipline |

**Subtasks**:

- **2.1** Create `AudioFormat.swift` — constants: 16kHz sample rate, mono, Float32 PCM
- **2.2** Implement `CircularAudioBuffer.swift` — lock-free ring buffer for passing audio from the real-time audio thread to processing threads. Consider using `TPCircularBuffer` or a custom implementation with `OSAtomicFifoEnqueue`/`Dequeue`
- **2.3** Implement `AudioCaptureManager.swift`:
  - Configure `AVAudioSession` (category: `.record`, mode: `.default`)
  - Set up `AVAudioEngine` input node
  - Install tap at 16kHz mono Float32
  - Write captured frames into the `CircularAudioBuffer`
  - Expose start/stop methods
- **2.4** Implement `Permissions.swift` — microphone permission request flow with proper SwiftUI integration (pre-prompt explanation → system dialog → handle denial gracefully)
- **2.5** Build a minimal debug view that shows audio is being captured (e.g., display RMS level or frame count updating in real-time)

**Acceptance Criteria**: App requests mic permission, captures audio at 16kHz/mono/Float32, writes to ring buffer without audio thread glitches. Debug view confirms audio is flowing.

---

### 3. Silero VAD Integration

| Field | Detail |
|---|---|
| **Estimate** | 3–4 days |
| **Dependencies** | Task 1 (ONNX Runtime), Task 2 (audio capture) |
| **Risk** | 🟡 Medium — Silero VAD's ONNX model has specific input/output tensor shapes and internal state (h/c tensors) that need careful handling |

**Subtasks**:

- **3.1** Download Silero VAD ONNX model (`silero_vad.onnx`, ~2MB); bundle in app target
- **3.2** Implement `SileroVADModel.swift`:
  - Load the ONNX model via `OrtSession`
  - Understand input tensor shape (batch × samples, typically 512 samples at 16kHz = 32ms frames)
  - Manage hidden state tensors (`h` and `c`) across frames — these must persist between calls
  - Return speech probability per frame
- **3.3** Implement `VADConfiguration.swift`:
  - Speech threshold (default: 0.5)
  - Minimum speech duration (default: 250ms — avoid firing on clicks/pops)
  - Silence duration to end utterance (default: 700ms)
  - Pre-speech padding (default: 300ms — capture the start of speech that triggered VAD)
- **3.4** Implement `VADProcessor.swift`:
  - Reads from `CircularAudioBuffer` on a dedicated thread (QoS: `.userInteractive`)
  - Processes 32ms frames through `SileroVADModel`
  - Implements state machine: `silence → speech_start → speaking → speech_end`
  - On `speech_start`: begin accumulating audio frames (including pre-speech padding)
  - On `speech_end`: package accumulated audio as a complete utterance, emit via callback/AsyncStream
  - Handle edge cases: very long utterances (>30s) — force-segment to prevent unbounded memory growth
- **3.5** Build a debug view overlay showing VAD state (silence/speech) and speech probability in real-time

**Acceptance Criteria**: VAD correctly detects speech onset and offset. Silence is filtered. Utterances are segmented and emitted as discrete audio chunks. No false triggers on background noise at reasonable thresholds.

---

### 4. Moonshine ASR Integration

| Field | Detail |
|---|---|
| **Estimate** | 5–7 days |
| **Dependencies** | Task 1 (ONNX Runtime), Task 3 (VAD provides segmented utterances) |
| **Risk** | 🔴 High — This is the most complex integration. Moonshine's ONNX export, encoder/decoder architecture, and streaming behavior need careful study. CoreML EP compatibility is not guaranteed for all ops. |

**Subtasks**:

- **4.1** Download Moonshine Tiny ONNX model files from HuggingFace (`UsefulSensors/moonshine`). Understand the model structure — Moonshine v2 has separate encoder and decoder ONNX files. Document input/output tensor names and shapes.
- **4.2** Implement `MoonshineModel.swift`:
  - Load encoder + decoder ONNX sessions
  - Implement the inference loop: audio → encoder → decoder (autoregressive token generation)
  - Handle tokenizer: Moonshine uses a BPE/SentencePiece tokenizer — either port the tokenizer to Swift or use a lightweight C++ binding
  - Map output token IDs → text
- **4.3** Implement streaming partial results:
  - Moonshine v2's streaming encoder processes audio in sliding windows
  - Emit partial text as tokens are generated (before the utterance is complete)
  - Design the `TranscriptSegment` data model: `id: UUID`, `partialText: String`, `finalText: String?`, `timestamp: Date`, `audioRange: ClosedRange<TimeInterval>`
- **4.4** Implement `MoonshineTranscriber.swift`:
  - Receives segmented audio from VAD
  - Runs ASR inference on a dedicated thread (QoS: `.userInteractive`)
  - Emits `TranscriptUpdate.partial` and `TranscriptUpdate.finalized` via `AsyncStream`
- **4.5** Benchmark on physical device:
  - Measure real-time factor (RTF) — target: < 0.5 (inference takes less than half the audio duration)
  - Measure time-to-first-token for a typical utterance
  - Test with CoreML EP enabled vs. CPU-only — compare latency and verify correctness
  - If CoreML EP fails on certain ops, identify which ops and decide on fallback strategy
- **4.6** Implement basic error handling:
  - Model loading failure (corrupted file, incompatible device)
  - Inference timeout (utterance too long, device too slow)
  - Memory pressure (handle `didReceiveMemoryWarning`)

**Acceptance Criteria**: Given a segmented audio utterance from VAD, Moonshine produces a text transcript. Partial results stream as tokens are generated. RTF < 0.5 on iPhone 14+. CoreML EP status is documented (works / partially works / fallback to CPU).

---

### 5. Transcription Pipeline Orchestrator

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Tasks 2, 3, 4 (all pipeline components) |
| **Risk** | 🟠 Medium-high — Concurrency coordination between audio, VAD, ASR, and (as of 5.6+) a network client. Swift 6 strict concurrency will surface issues; network dependency adds a new failure class. |

**Subtasks**:

- **5.1** Implement `TranscriptionPipeline` actor:
  - Owns `AudioCaptureManager`, `VADProcessor`, `MoonshineTranscriber`
  - `startSession()`: initializes audio capture → VAD → ASR chain
  - `stopSession()`: tears down cleanly, returns accumulated transcript
  - Exposes `AsyncStream<TranscriptUpdate>` for the UI to observe
- **5.2** Implement `PipelineConfiguration.swift`:
  - Model variant selection (tiny vs. base — base not used yet but design for it)
  - VAD sensitivity settings
  - Debug flags (log audio levels, VAD decisions, ASR timing)
- **5.3** Implement `TranscriptionSession.swift`:
  - Accumulates transcript segments in order
  - Tracks session metadata: start time, duration, segment count
  - Provides `fullTranscript` computed property (concatenated finalized text)
- **5.4** Wire up error propagation: errors from any pipeline stage surface as `TranscriptUpdate.error` on the stream
- **5.5** Test the full pipeline end-to-end on device: speak → see text appear
- **5.6** Add `.punctuated(TranscriptPunctuated)` case to `TranscriptUpdate`; thread `utteranceID` from finalized → punctuated so the UI can upsert in place (commit TBD).
- **5.7** Implement `Services/PunctuationClient.swift` + `PunctuationConfig.swift`. URLSession with 2 s request / 4 s resource timeout, `POST /punctuate`, `X-API-Key` header. Returns `Optional<TranscriptPunctuated>` — `nil` on any failure (silent fallback). Config loaded from `Info.plist` (via `.xcconfig` at build time) with a localhost fallback for dev (commit TBD).
- **5.8** Wire punctuation dispatch into the pipeline. `AppState` instantiates the shared `PunctuationClient` alongside the preloaded transcriber. `AudioCaptureDebugModel.attach(transcriber:punctuationClient:)` stores both; when the subscription loop sees a `.finalized(...)` it fires a `Task` that calls `punctuate(...)` and upserts `.punctuated(...)` on success. UI `upsert` gets a `.punctuated` branch that replaces the finalized row's text in place (commit TBD).
- **5.9** Real FastAPI + ONNX Runtime server in `server/`. `export_model.py` downloads `oliverguhr/fullstop-punctuation-multilingual-base`, exports to ONNX via `optimum`, and dynamic-INT8-quantizes — ~140 MB artifact in `server/model-onnx/` (gitignored). `app.py` loads tokenizer + ONNX session in a FastAPI lifespan hook, routes `POST /punctuate` with `X-API-Key` auth, implements sliding-window chunking (256-token windows, 32-token overlap, prefer-center tiebreak) for inputs > 254 content tokens, reassembles punctuated text from per-word labels, applies deterministic casing post-step. `/healthz` returns 503 until the lifespan hook finishes so ALB won't route traffic to a cold task. `Dockerfile` pins OMP/MKL thread counts, one uvicorn worker per container (commit TBD).

**Acceptance Criteria**: The pipeline orchestrates audio → VAD → ASR → cloud punctuation as a single cohesive unit. Starting/stopping a session is clean. Transcript updates stream to observers in real-time; punctuated replacements update rows in place without disturbing scroll position. Server timeouts / network failures degrade silently to the raw local transcript. No thread safety violations under Swift 6 strict concurrency.

---

### 6. Minimal Transcription UI

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 5 (pipeline wired up) |
| **Risk** | 🟢 Low — SwiftUI basics; main concern is smooth scrolling with rapid text updates |

**Subtasks**:

- **6.1** Implement `TranscriptionView.swift`:
  - Large scrolling text area showing transcript
  - Auto-scrolls to bottom as new text appears
  - Partial results shown in gray/italic; finalized in black
  - Record/Stop button (large, prominent)
  - Elapsed time display
- **6.2** Implement `AudioLevelIndicator.swift` — simple waveform or level meter showing mic input (confirms the app is listening)
- **6.3** Implement `SessionTimer.swift` — elapsed time counter (MM:SS format)
- **6.4** Wire `TranscriptionView` to `TranscriptionPipeline` via `@Observable` view model pattern
- **6.5** Handle UI edge cases:
  - Permission denied state (show explanation + settings link)
  - Model loading state (show spinner on first launch)
  - Error state (pipeline failure → user-friendly message)

**Acceptance Criteria**: User taps Record, speaks, and sees words appear on screen in near real-time. UI is responsive (no frame drops during rapid text updates). Start/stop works reliably.

---

### 7. Physical Device Testing & Battery Profiling

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 6 (full pipeline + UI working) |
| **Risk** | 🟡 Medium — Battery drain or thermal issues could require architectural changes |

**Subtasks**:

- **7.1** Test on primary device (iPhone 14+):
  - 5-minute session: verify transcript accuracy is reasonable
  - 15-minute session: check for memory leaks (Instruments → Leaks)
  - 30-minute session: measure battery drain (Instruments → Energy Log)
- **7.2** Profile memory usage (Instruments → Allocations):
  - Peak memory during active transcription
  - Memory after stopping session (verify models can be unloaded)
  - Target: < 400MB peak
- **7.3** Profile CPU/GPU usage:
  - Verify CoreML EP is being used (check Instruments → Core ML)
  - Identify hot spots in the pipeline
- **7.4** Test edge cases:
  - Background noise (coffee shop, street) — does VAD filter it?
  - Rapid speech — does ASR keep up?
  - Long silence → speech → silence cycles
  - Device rotation during recording
- **7.5** Document findings: latency measurements, battery drain rate, memory profile, any issues discovered. Create a `phase1-benchmarks.md` with results.

**Acceptance Criteria**: 30-minute session completes without crashes or memory issues. Battery drain is documented and < 20%/hour. Latency from speech → text on screen is < 500ms. Known issues are documented.

---

## Dependency Graph

```
Task 1 (Scaffolding)
  ├──► Task 2 (Audio Capture)
  │       └──► Task 3 (VAD) ──────┐
  └──► Task 3 (VAD)               │
        └──► Task 4 (Moonshine) ◄─┘
              └──► Task 5 (Pipeline)
                    └──► Task 6 (UI)
                          └──► Task 7 (Testing)
```

Tasks 2 and 3 can be partially parallelized — audio capture can be built and tested independently before wiring VAD to it. Task 4 (Moonshine) is the critical path and highest risk item.

---

## Summary

| Task | Estimate | Risk | Critical Path? |
|---|---|---|---|
| 1. Project Scaffolding | 1–2 days | 🟡 Medium | Yes |
| 2. Audio Capture | 2–3 days | 🟢 Low | Yes |
| 3. Silero VAD | 3–4 days | 🟡 Medium | Yes |
| 4. Moonshine ASR | 5–7 days | 🔴 High | **Yes — bottleneck** |
| 5. Pipeline Orchestrator | 2–3 days | 🟡 Medium | Yes |
| 6. Minimal UI | 2–3 days | 🟢 Low | Yes |
| 7. Device Testing | 2–3 days | 🟡 Medium | Yes |
| **Total** | **17–25 days** | | |

**Critical risk**: Task 4 (Moonshine integration). If ONNX Runtime + CoreML EP doesn't work well with Moonshine's ops, the fallback is CPU-only inference which may be too slow, or switching to Whisper Tiny as a backup ASR model. **Recommendation**: Start a spike on Task 4.1–4.2 in parallel with Task 2 to de-risk early.
