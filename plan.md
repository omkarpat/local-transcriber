# VoxLocal: Local-First iOS Transcription App

## Technical Plan & Architecture Document

---

## 1. Product Vision

A transcription app that runs entirely on-device, delivering real-time speech-to-text with punctuation, capitalization, and near-instant display — no internet connection required. An optional cloud component provides enhanced correction for word errors, proper nouns, and formatting when connectivity is available.

### Core Principles

- **Local-first**: The app is fully functional offline. Cloud is a quality enhancer, not a dependency.
- **Streaming UX**: Words appear on screen as the user speaks, not after a delay.
- **Privacy by default**: Audio never leaves the device unless the user explicitly opts in to cloud correction.
- **Battery-conscious**: Model selection and inference scheduling are optimized for sustained 1-hour+ sessions.

### Target Users

- Journalists recording interviews
- Students transcribing lectures
- Professionals capturing meeting notes
- Anyone who needs quick, private, offline transcription

---

## 2. Architecture Overview

The app uses a local-first pipeline: streaming ASR on-device for fast word recognition, with punctuation + refinement delivered by an optional cloud service via streaming HTTP. The cloud service is a quality enhancer — the app is fully functional with the raw ASR transcript when offline.

```
┌──────────────────────────────────────────────────────────────┐
│                        iOS App                                │
│                                                               │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Audio    │───►│ VAD          │───►│ Moonshine v2       │  │
│  │ Capture  │    │ (Silero)     │    │ Streaming ASR      │  │
│  │ (AVAudio │    │              │    │ (ONNX Runtime +    │  │
│  │  Engine) │    │ Filters      │    │  CoreML EP)        │  │
│  │          │    │ silence,     │    │                    │  │
│  │ 16kHz    │    │ segments     │    │ Outputs:           │  │
│  │ mono     │    │ utterances   │    │ raw lowercase      │  │
│  └──────────┘    └──────────────┘    │ text, no punct     │  │
│                                       └─────────┬──────────┘  │
│                                                 │              │
│  ┌──────────────┐                     ┌─────────▼──────────┐  │
│  │ Display      │◄────────partials────│                    │  │
│  │ Layer        │◄────────finalized───│ TranscriptionPipe- │  │
│  │              │                     │ line (actor)       │  │
│  │ - Live       │                     │                    │  │
│  │   partial    │         refinements ├────────────┐       │  │
│  │   (italic)   │◄────────────────────┤            │       │  │
│  │ - Finalized  │                     │  Fires     │       │  │
│  │   raw text   │                     │  HTTP+SSE  │       │  │
│  │ - Refined    │                     │  request   │       │  │
│  │   (merged    │                     │  per       │       │  │
│  │   paragraph) │                     │  finalized │       │  │
│  └──────────────┘                     │  utterance │       │  │
│                                       └────────────┼───────┘  │
│  ┌──────────────────────────────────────────────┐  │          │
│  │ Local Storage (SwiftData)                     │  │          │
│  │ - Session transcripts                         │  │          │
│  │ - Retry queue for offline refinement          │  │          │
│  │ - User preferences                            │  │          │
│  └──────────────────────────────────────────────┘  │          │
└─────────────────────────────────────────────────────┼──────────┘
                                                      │
                                          HTTP POST + SSE stream
                                          (X-API-Key auth)
                                                      │
┌─────────────────────────────────────────────────────▼──────────┐
│  Punctuation Refinement Service (stateless; Fargate first,     │
│  EC2/batched workers later if scale demands it)                │
│                                                                 │
│  FastAPI + ONNX Runtime (CPU-only, INT8 quantized)             │
│                                                                 │
│  Stage 1: Spoken commands  ("comma" → ",", "new paragraph")    │
│     │                                                           │
│  Stage 2: Punct + truecasing  (NeMo joint DistilBERT)          │
│     │                                                           │
│  Stage 3: ITN  (NeMo Text Processing — "twenty twenty six"     │
│                 → "2026", "five dollars" → "$5")               │
│                                                                 │
│  Each stage emits an SSE event. Client renders progressively.   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. On-Device ASR: Moonshine v2

### Why Moonshine v2

- **Streaming encoder**: Ergodic sliding-window self-attention gives bounded time-to-first-token regardless of utterance length. Users see words appear in real-time.
- **Variable-length input**: Unlike Whisper's fixed 30-second chunks, Moonshine processes exactly the audio you give it. Short utterances are very fast.
- **Cross-platform C++ core**: Same library on iOS, Android, macOS. ONNX Runtime with CoreML execution provider for Apple hardware acceleration.
- **Small footprint**: Moonshine Base is 62M params (~400MB on disk). Moonshine Tiny is 27M params (~190MB).
- **MIT license**: No commercial restrictions.

### Model Variants to Ship

| Model | Size on Disk | Use Case |
|---|---|---|
| `moonshine-tiny-en` | ~190MB | Default. Best battery life, good accuracy. |
| `moonshine-base-en` | ~400MB | User-selectable "High Accuracy" mode. |

Models are bundled in the app binary (or downloaded on first launch to reduce initial download size). Stored in the app's Documents directory, loaded into memory on session start.

### ONNX Runtime Configuration

```swift
// Pseudocode for iOS ONNX Runtime setup
let env = OrtEnvironment()
let sessionOptions = OrtSessionOptions()

// Enable CoreML execution provider for Apple Neural Engine / GPU
sessionOptions.appendCoreMLExecutionProvider(
    with: .init(
        useCPUAndGPU: false,      // prefer ANE
        enableOnSubgraphs: true,
        onlyEnableDeviceWithANE: false
    )
)

// Memory-map the model for fast cold start
sessionOptions.enableMemoryPattern = true

// Load the ONNX model
let session = try OrtSession(
    env: env,
    modelPath: modelPath,
    sessionOptions: sessionOptions
)
```

### Audio Capture Pipeline

```
AVAudioEngine (input node)
  → Install tap: 16kHz, mono, Float32
  → Ring buffer (thread-safe, lock-free)
  → VAD processing thread
    → Silero VAD: determines speech/silence boundaries
    → On speech start: begin accumulating audio frames
    → On speech end (isFinal): flush to ASR
  → ASR inference thread
    → Moonshine v2 streaming: process accumulated frames
    → Emit partial transcript (displayed immediately)
    → On utterance complete: emit final transcript
    → Forward to punctuation model
```

### Key Implementation Details

- **Audio format**: 16kHz sample rate, mono channel, Float32 PCM. This is Moonshine's native input format.
- **Buffer strategy**: Use a lock-free ring buffer (e.g., TPCircularBuffer) to pass audio from the audio thread to the processing thread without blocking.
- **Thread model**: Audio capture on the real-time audio thread (highest priority). VAD + ASR on a dedicated background thread (QoS: `.userInteractive`). Punctuation on a second background thread (QoS: `.userInitiated`).
- **Memory budget**: Target ~300MB total for both models + audio buffers. This is safe on any iPhone from iPhone 11 onward (4GB+ RAM).

---

## 4. Punctuation & Refinement: Cloud Service

### The Problem

Moonshine outputs raw lowercase text: `so the main thing we need to discuss today is the quarterly budget`. We need: `So, the main thing we need to discuss today is the quarterly budget.`

### Approach: Stateless Cloud Service (Not On-Device)

**Design decision (revised)**: Punctuation + formatting runs in a cloud service, not on-device. Earlier versions of this plan proposed a distilled BERT-tiny running locally; in practice a small stateless FastAPI service running on ECS CPU tasks delivered better accuracy, shorter time-to-ship, and zero on-device memory/battery cost — at the price of a ~100ms network RTT and a silent fallback when offline. Both tradeoffs were worth it.

The service is **optional**. When offline or on server failure the app shows the raw ASR transcript. No UI error — silent fallback preserves the local-first promise.

### Service Architecture: V1 (Shipped)

- **Runtime**: FastAPI + ONNX Runtime (CPU, INT8 quantized XLM-R base)
- **Endpoint**: `POST /punctuate` — single request/response with punctuated + regex-cased text
- **Deployment**: ECS Fargate initially (4 vCPU / 8 GB task) behind ALB, single-AZ for now. Keep the HTTP+SSE contract stable if/when the workers move to ECS on EC2 or a batched inference tier at higher sustained load.
- **Auth**: shared `X-API-Key` header; no per-user auth yet
- **Model**: `oliverguhr/fullstop-punctuation-multilingual-base`, INT8 ONNX

See `punctuation-service-plan.md` for the V1 service detail.

### Service Architecture: V2 (In Progress)

V2 keeps the same deployment shape but adds three pipeline stages and a streaming wire contract:

1. **Stage 1 — Spoken commands**: "comma" → ",", "new paragraph" → segment break
2. **Stage 2 — Punct + truecasing**: swap XLM-R for NeMo joint DistilBERT (English-only; kills the regex casing step)
3. **Stage 3 — ITN**: NeMo Text Processing for "twenty twenty six" → "2026", "five dollars" → "$5"

And a new endpoint:

- **`POST /punctuate/stream`** — returns `text/event-stream` with one `event: stage` per pipeline stage. Client renders progressively-improved text as each stage lands.

See `punctuation-service-v2-plan.md` for the V2 plan including the wire contract, two-ID model (utteranceID + segmentID), deterministic uuidv5 segment IDs for retry idempotency, and stateless-server rationale.

Architectural posture: keep SSE + FastAPI as the edge contract. If scale pressure shows up, change the compute substrate first (EC2 workers, batching), not the client-visible API.

### Client Integration (iOS)

```swift
// PunctuationClient.swift — existing V1 client (single-shot)
func punctuate(
    utteranceID: UUID,
    text: String,
    utteranceDuration: Duration
) async -> TranscriptRefinement?

// V2 addition — streaming consumer via URLSession.bytes(for:)
func punctuateStream(
    utteranceID: UUID,
    text: String
) -> AsyncStream<PunctuationStageEvent>
```

`TranscriptionPipeline` fires a request per finalized utterance and yields `.refined(TranscriptRefinement)` updates onto its `updates` stream. Each refinement carries both `utteranceID` (provenance) and `segmentID` (UI upsert key). Today they're equal; when the server starts splitting on "new paragraph" commands, `segmentID` diverges and the UI renders multiple paragraphs per utterance without any client changes.

### Retry Semantics

When a request fails (timeout, non-2xx, mid-stream drop), the utterance is enqueued on a per-pipeline retry queue with exponential backoff (5s → 60s cap). Deterministic server-side segment IDs (`uuidv5(utteranceID, index)`) mean a replayed utterance produces the same segment IDs as the first attempt, so retries update in place rather than creating orphan rows.

### Latency Budget

Audio → displayed raw text target: <500ms (on-device only).
Audio → displayed refined text target: <700ms p50 for 30–60 token utterances.

| Stage | Target Latency | Notes |
|---|---|---|
| Audio capture → VAD | ~100ms | Silero VAD processes 30ms frames |
| VAD → Moonshine ASR | ~150–200ms | Streaming partial results |
| **Raw text displayed** | **~250–300ms** | Local path complete here |
| POST → SSE first byte | ~20–40ms | ALB + LAN |
| Stage 1 commands | ~1ms | Rule-based |
| Stage 2 punct+case | ~60–90ms | NeMo DistilBERT INT8 |
| Stage 3 ITN | ~5–20ms | WFST traversal |
| **Refined text displayed** | **~400–500ms** | Within conversational feel |

Network tail events (500ms+) fall back silently to the raw transcript via the retry queue.

---

## 5. Future Cloud Capability: LLM Correction (Phase 4+)

V1 and V2 punctuation services handle *formatting* problems. Neither can fix ASR recognition errors — homophones like "405" → "four or five", out-of-vocabulary terms, proper-noun confusions. Those need either a better ASR (Moonshine retraining or contextual biasing) or an LLM rewrite pass with world knowledge.

**This layer is deferred to Phase 4+.** When it arrives, the candidate architecture is:

- **Deployment**: HTTP POST to a separate Bedrock-backed endpoint (or direct API call depending on latency requirements)
- **Invocation rule**: opt-in per session or triggered by a low-confidence signal; V2's deterministic pipeline runs first and returns immediately
- **Contract**: extend the V2 SSE stream with an optional `event: stage` carrying `stage: "llm_correction"`. The client already renders progressively, so adding one more late-arriving event is a no-op on the UI.
- **Budget**: 500ms–2s p99 is acceptable because this stage runs *after* the deterministic pipeline has already shown text. Network tail doesn't stall the UI.

Earlier versions of this plan specified WebSocket + API Gateway + Bedrock + Cognito as the main cloud path. That architecture is shelved — the stateless HTTP+SSE shape we're using for V1/V2 is simpler, faster to iterate on, and extends cleanly into Phase 4's LLM fallback. If scale becomes the problem, the first move is to change the worker substrate (ECS on EC2, batching), not to replace SSE. Cognito auth, per-user sessions, and WebSocket connection management become relevant only if we add features that actually need them (e.g., bidirectional streaming, live collaboration).

---

## 6. iOS App Structure

### Project Setup

- **Language**: Swift 6 (strict concurrency)
- **UI Framework**: SwiftUI
- **Minimum iOS**: 17.0 (for SwiftData, modern concurrency)
- **Architecture**: MVVM with Swift Concurrency (async/await, actors)

### Module Breakdown

```
VoxLocal/
├── App/
│   ├── VoxLocalApp.swift            # App entry point
│   └── AppState.swift               # Global state (active session, settings)
│
├── Audio/
│   ├── AudioCaptureManager.swift    # AVAudioEngine setup, tap installation
│   ├── CircularAudioBuffer.swift    # Lock-free ring buffer for audio frames
│   └── AudioFormat.swift            # Constants (16kHz, mono, Float32)
│
├── VAD/
│   ├── SileroVADModel.swift         # ONNX Runtime inference for Silero VAD
│   ├── VADProcessor.swift           # Speech start/end detection logic
│   └── VADConfiguration.swift       # Thresholds, min speech duration, etc.
│
├── ASR/
│   ├── MoonshineModel.swift         # ONNX Runtime inference for Moonshine
│   ├── MoonshineTranscriber.swift   # Streaming transcription orchestrator
│   ├── TranscriptSegment.swift      # Data model: text, timestamps, confidence
│   └── ModelManager.swift           # Model download, storage, version management
│
├── Services/
│   ├── PunctuationClient.swift      # HTTP + SSE client for the cloud
│   │                                # refinement service. Single-shot
│   │                                # `punctuate()` for V1; streaming
│   │                                # `punctuateStream()` for V2.
│   └── PunctuationConfig.swift      # Endpoint + API key config
│
├── Pipeline/
│   ├── TranscriptionPipeline.swift  # Orchestrates Audio→VAD→ASR→Punct→Cloud
│   ├── PipelineConfiguration.swift  # Model selection, cloud on/off, etc.
│   └── TranscriptionSession.swift   # Session lifecycle, transcript accumulation
│
├── Storage/
│   ├── TranscriptStore.swift        # SwiftData models + CRUD
│   ├── SessionRecord.swift          # Persisted session: date, duration, text
│   └── ExportManager.swift          # Export to .txt, .srt, .vtt, clipboard
│
├── Views/
│   ├── HomeView.swift               # Session list, start new session
│   ├── TranscriptionView.swift      # Live transcription display
│   ├── TranscriptDetailView.swift   # Review/edit past transcripts
│   ├── SettingsView.swift           # Model selection, cloud toggle, etc.
│   └── Components/
│       ├── LiveTextView.swift       # Streaming text with correction animations
│       ├── AudioLevelIndicator.swift# Waveform / level meter
│       └── SessionTimer.swift       # Elapsed time display
│
└── Utilities/
    ├── OnnxRuntimeSetup.swift       # Shared ONNX environment + CoreML config
    ├── Permissions.swift            # Microphone permission handling
    └── Haptics.swift                # Feedback for start/stop recording
```

### Key Swift Actors

```swift
// Thread-safe transcription pipeline orchestrator
actor TranscriptionPipeline {
    private let manager: AudioCaptureManager
    private let transcriber: MoonshineTranscriber       // ASR (long-lived)
    private let punctuationClient: PunctuationClient    // cloud refinement (long-lived)
    private var vad: VADProcessor?                      // per-session

    // Published streams for the UI
    let updates: AsyncStream<TranscriptUpdate>          // partial/final/refined
    let events: AsyncStream<PipelineEvent>              // started/stopped/speech

    func start(configuration: PipelineConfiguration) async throws { ... }
    func stop() { ... }
}

// Transcript update types the UI observes
enum TranscriptUpdate {
    case partial(TranscriptPartial)           // ASR streaming partial (italic)
    case finalized(TranscriptResult)          // ASR final, raw local text
    case refined(TranscriptRefinement)        // Cloud-polished; may fire multiple
                                              // times per utterance as each V2 stage
                                              // (commands → punct+case → ITN) lands
    case failed(TranscriptFailure)
}

// Each refinement carries two IDs. `utteranceID` is provenance (which ASR
// utterance this came from — used for retries). `segmentID` is the UI
// upsert key (which paragraph this text renders in). They're equal until
// the V2 server starts splitting utterances on "new paragraph" commands,
// at which point one utterance can produce multiple segments and the UI
// renders multiple paragraphs — without any client changes.
```

### Background Audio

The app must continue transcribing when backgrounded (user switches to another app during a lecture/meeting):

```swift
// Info.plist
UIBackgroundModes: ["audio"]

// AudioCaptureManager.swift
let audioSession = AVAudioSession.sharedInstance()
try audioSession.setCategory(.record, mode: .default, options: [])
try audioSession.setActive(true)
```

---

## 7. UX Design Principles

### Live Transcription View

- **Merged paragraph**: The transcript is a single flowing paragraph, not a list of utterance rows. Finalized utterances concatenate chronologically; the in-flight partial trails at the end in italic. This matches how dictated text reads and leaves room for future paragraph breaks driven by the V2 server.
- **Update phases** on each segment:
    1. **Partial** (secondary color, italic): ASR streaming output, may change
    2. **Finalized raw** (primary color): Local ASR result, no punctuation
    3. **Refined** (primary color, overwrites in place): Each cloud pipeline stage progressively improves the segment — commands → punct+case → ITN. With V2 SSE streaming, users see text improve in real time rather than jumping from raw to fully-polished in one update.
- **Segment identity**: Each refinement is keyed by `segmentID`. In V2 single-segment mode every utterance is its own segment and they all concatenate into one paragraph. When the V2 server starts emitting new segment IDs on spoken "new paragraph" commands (or cross-utterance silence), the UI renders paragraph breaks automatically with no client changes.
- **Auto-scroll**: Stick to bottom on new text; pause when the user scrolls up to re-read; resume when they return to the bottom. Show a "scroll to bottom" button when paused.
- **Pause/resume recording**: Single tap to pause, tap again to resume.

### Session Management

- Sessions are auto-saved continuously (every 30 seconds to SwiftData).
- Session list shows: date, duration, first line preview, word count.
- Swipe to delete, long-press to export.

### Export Formats

- **Plain text** (.txt): Clean transcript
- **Subtitles** (.srt / .vtt): With timestamps from Moonshine
- **Clipboard**: One-tap copy of full transcript
- **Share sheet**: System share for email, Messages, Notes, etc.

---

## 8. Model Management

### Initial Download Strategy

To keep the App Store binary under 200MB, models are downloaded on first launch:

```
App Launch
  → Check for models in Documents/models/
  → If missing:
      → Show "Setting up..." screen with progress bar
      → Download from CloudFront CDN (or bundled asset catalog)
      → Verify SHA256 checksum
      → Store in Documents/models/
  → Load models into ONNX Runtime
```

### Model Sizes to Download

| Model | Size | Required? |
|---|---|---|
| Silero VAD | ~2MB | Yes (bundled in app) |
| Moonshine Tiny | ~190MB | Yes (default) |
| Moonshine Base | ~400MB | Optional (user downloads for "HD" mode) |
| Punctuation model | ~60-120MB | Yes |
| **Total (default)** | **~270MB** | |

### Model Updates

- Models are versioned (e.g., `moonshine-tiny-en-v2.1.onnx`).
- App checks for model updates on launch (if online), downloads in background.
- Old models are cleaned up after successful migration.

---

## 9. Development Phases

### Phase 1: Core Local Pipeline + V1 Refinement (Weeks 1-4) — DONE except device testing

**Goal**: Mic → ASR → text on screen, fully on-device. Plus V1 cloud refinement as a silent-fallback HTTP service.

- [x] Set up Xcode project, ONNX Runtime Swift package
- [x] Implement `AudioCaptureManager` with AVAudioEngine
- [x] Integrate Silero VAD via ONNX Runtime
- [x] Integrate Moonshine Tiny via ONNX Runtime + CoreML EP
- [x] Build `TranscriptionPipeline` actor orchestrating Audio → VAD → ASR
- [x] Build minimal `TranscriptionView` showing streaming text
- [x] Deploy V1 `POST /punctuate` service (FastAPI + ONNX on CPU); integrate `PunctuationClient` with silent fallback and retry queue
- [ ] Test on physical device (iPhone 14+): verify real-time performance
- [ ] Measure battery drain over 30-minute session

**Exit criteria**: User can record speech and see raw transcript appear in real-time, fully offline. When online, the transcript upgrades to punctuated + cased text via cloud refinement, with no UI disruption on server failures.

### Phase 2: V2 Streaming Refinement + Persistence + UX (Weeks 5-7)

**Goal**: Upgrade cloud refinement to progressive SSE streaming; add session persistence, export, and polish.

See `phase2-plan.md` for the detailed task breakdown.

- [ ] Task 1: Adopt V2 streaming refinement
    - Commit 1 (**DONE**): rename `.punctuated` → `.refined`; introduce `TranscriptRefinement` with `utteranceID` + `segmentID`; merge user-facing view into a single flowing paragraph
    - Commit 2: server `POST /punctuate/stream` with SSE + deterministic uuidv5 segment IDs + explicit in-flight backpressure / overload handling (tracked in `punctuation-service-v2-plan.md`)
    - Commit 3: client `PunctuationClient.punctuateStream()` via `URLSession.bytes(for:)`, wire into pipeline
- [ ] Task 2: Progressive paragraph rendering (auto-scroll, stage transitions, paragraph breaks when server splits)
- [ ] Task 3: SwiftData persistence for sessions
- [ ] Task 4: `HomeView` with session list
- [ ] Task 5: Export (.txt, .srt, .vtt, clipboard, share sheet)
- [ ] Task 6: Background audio
- [ ] Task 7: Settings + Moonshine Base "HD" model download

**Exit criteria**: User sees text progressively improve across stages (commands → punct+case → ITN) with deterministic fallback to raw on server failures. Sessions are saved and exportable. App works in background.

### Phase 3: V2 Server Stages & Eval (Weeks 8-10)

**Goal**: Ship the three V2 pipeline stages behind feature flags, with labeled eval on real Moonshine output.

See `punctuation-service-v2-plan.md` for the service-side plan.

- [ ] Wire the SSE contract in production (start by emitting a single `stage: "full"` event wrapping V1 output — no behaviour change, just contract)
- [ ] Ship ITN as Stage 3 (NeMo Text Processing); emit dedicated `stage: "itn"` event
- [ ] Swap XLM-R → NeMo joint DistilBERT for punct+case; A/B on labeled eval set; gate behind flag
- [ ] Add Stage 1 spoken commands; build labeled eval set for precision/recall tuning
- [ ] Wire deterministic `silence_before_ms` from client VAD timeline to enable cross-utterance paragraph splits
- [ ] Observability: per-stage latency histograms, p50/p95/p99/p999 tracked separately, plus in-flight request count, overload rejects, disconnects, and fallback rate

**Exit criteria**: All three stages in production behind flags with documented quality wins on labeled eval. p99 < 250ms for 30–60 token inputs. No regression on V1 punctuation F1.

### Phase 4: LLM Correction Layer & Optimization (Weeks 11-14)

**Goal**: Add optional LLM correction for ASR errors (homophones, proper nouns), plus polish for App Store.

- [ ] Design LLM-correction endpoint as Phase 3 SSE extension (one more stage event, runs after deterministic pipeline)
- [ ] Deploy Bedrock-backed (Claude Haiku) or direct-API correction service
- [ ] Wire opt-in per-session flag; route low-confidence utterances only (telemetry-driven threshold)
- [ ] Profile and optimize memory usage (Instruments → Allocations)
- [ ] Profile and optimize battery drain (Instruments → Energy Log)
- [ ] Accessibility audit (VoiceOver, Dynamic Type)
- [ ] App Store assets, screenshots, privacy policy
- [ ] TestFlight beta

**Exit criteria**: LLM correction visibly fixes recognition errors on an opt-in basis without regressing latency for the common path. Polished, battery/memory-optimized app ready for App Store submission.

---

## 10. Testing Strategy

### Unit Tests

- Audio buffer read/write correctness
- VAD speech boundary detection (with known audio fixtures)
- Punctuation model label → text reconstruction
- WebSocket message encoding/decoding
- Offline queue ordering and deduplication

### Integration Tests

- Full pipeline: audio file → transcript (compare against reference)
- Cloud round-trip: send utterance → receive correction → verify merge
- Session persistence: create → background → restore → verify

### Performance Tests

- **Latency**: Measure audio-to-display latency per segment (target: <500ms)
- **Real-time factor**: ASR processing speed vs. audio duration (target: RTF < 0.5)
- **Memory**: Peak memory during 1-hour session (target: <500MB)
- **Battery**: Drain rate during continuous transcription (target: <15%/hour)
- **Thermal**: Device temperature after 30 minutes sustained use

### Device Matrix

| Device | RAM | Chip | Priority |
|---|---|---|---|
| iPhone 16 Pro | 8GB | A18 Pro | Primary |
| iPhone 14 | 6GB | A15 | Secondary |
| iPhone 12 | 4GB | A14 | Minimum spec |
| iPad Air M2 | 8GB | M2 | Bonus |

---

## 11. Dependencies

### Swift Packages

| Package | Purpose | License |
|---|---|---|
| `onnxruntime-swift` | ONNX Runtime inference with CoreML EP | MIT |
| `swift-collections` | Deque for audio buffering | Apache 2.0 |

### Model Artifacts

| Model | Source | License |
|---|---|---|
| Moonshine Tiny/Base | `UsefulSensors/moonshine` (HuggingFace) | MIT |
| Silero VAD | `snakers4/silero-vad` (GitHub) | MIT |
| Punctuation model | `oliverguhr/fullstop` or custom | MIT |

### AWS Services (Cloud Component)

| Service | Purpose | Phase |
|---|---|---|
| ECS Fargate (initial) / ECS on EC2 (scale-up) | Run the FastAPI refinement service (CPU-only ONNX inference) | 1 |
| Application Load Balancer | Terminate HTTPS, auth via `X-API-Key`, route to ECS tasks | 1 |
| ECR | Container image registry for the service | 1 |
| CloudWatch | Task + latency histograms, alarm on error rate | 1 |
| S3 + CloudFront | Model artifact hosting (ONNX weights, tokenizer, labels.json) | 1 |
| Amazon Bedrock | Optional LLM correction pass (Phase 4) | 4 |

Deferred until a concrete feature needs them: API Gateway (only if WebSockets become necessary), Cognito (only if per-user auth becomes necessary), DynamoDB (only if session-level server state becomes necessary). V1/V2 deliberately avoid all three.

---

## 12. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Moonshine ONNX + CoreML EP performance insufficient | Users see lag | Medium | Fall back to CPU EP; benchmark early in Phase 1; keep Whisper as backup |
| Cloud refinement service tail latency (network or task churn) | Text takes too long to polish | Medium | Silent fallback to raw local transcript; retry queue handles server blips; hard 2s client timeout keeps the UI responsive |
| V2 NeMo model ONNX export loses label mappings | Service returns garbled text | Medium | Ship `labels.json` alongside the model; verify round-trip in CI; gate the swap behind a feature flag |
| Spoken command false positives ("comma" → ",") | Text looks worse, not better | Medium | Precision-first tuning (>95% precision target even at 80% recall); labeled eval set on real Moonshine output before launch |
| Moonshine hallucinations on silence/noise | Phantom text appears | Medium | Silero VAD filters silence; implement token-rate heuristic to detect repetition |
| App Store review: large model download on first launch | Rejection | Low | Offer "download models" screen with clear size disclosure; comply with cellular download limits |
| Battery drain exceeds user tolerance | Negative reviews | Medium | Profile early; offer "battery saver" mode with Tiny model + lower sample rate |

---

## 13. Future Enhancements (Post-v1)

- **Speaker diarization**: Identify and label different speakers (Moonshine supports this in their roadmap)
- **Multilingual support**: Moonshine Flavors models for non-English languages
- **Apple Watch companion**: Tiny model for wrist-based dictation
- **Widgets**: Live Activity showing transcription status on lock screen
- **Shortcuts integration**: Start transcription via Siri or Shortcuts automation
- **Audio playback sync**: Tap a word to hear the audio at that timestamp
- **Search across sessions**: Full-text search over all saved transcripts
- **iCloud sync**: Sync transcripts across devices
- **macOS app**: Shared codebase via Moonshine's cross-platform C++ core
- **Custom vocabulary**: User-defined word list to improve recognition of names, jargon
- **Fine-tuned ASR**: Domain-specific Moonshine fine-tuning for medical, legal, etc.
