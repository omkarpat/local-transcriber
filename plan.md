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

The app uses a two-stage on-device pipeline: a streaming ASR model for fast word recognition, followed by a lightweight punctuation restoration model for formatting. An optional cloud layer provides deeper correction.

```
┌─────────────────────────────────────────────────────────────┐
│                       iOS App                                │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Audio    │───►│ VAD          │───►│ Moonshine v2      │  │
│  │ Capture  │    │ (Silero)     │    │ Streaming ASR     │  │
│  │ (AVAudio │    │              │    │ (ONNX Runtime +   │  │
│  │  Engine) │    │ Filters      │    │  CoreML EP)       │  │
│  │          │    │ silence,     │    │                   │  │
│  │ 16kHz    │    │ segments     │    │ Outputs:          │  │
│  │ mono     │    │ utterances   │    │ raw lowercase     │  │
│  └──────────┘    └──────────────┘    │ text, no punct    │  │
│                                       └────────┬──────────┘  │
│                                                │              │
│                                       ┌────────▼──────────┐  │
│                                       │ Punctuation Model │  │
│  ┌──────────────┐                     │ (On-Device)       │  │
│  │ Display      │◄────────────────────│                   │  │
│  │ Layer        │                     │ Token classifier  │  │
│  │              │                     │ ~15-30M params    │  │
│  │ - Live       │                     │ Adds . , ? ! :    │  │
│  │   partial    │                     │ and Capitalization │  │
│  │   results    │                     └────────┬──────────┘  │
│  │ - Finalized  │                              │              │
│  │   punctuated │                     ┌────────▼──────────┐  │
│  │   text       │                     │ Cloud Correction  │  │
│  │ - Cloud-     │◄────────────────────│ (Optional)        │  │
│  │   corrected  │                     │ WebSocket → AWS   │  │
│  │   text       │                     │ Bedrock           │  │
│  └──────────────┘                     └───────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Local Storage (SwiftData / Core Data)                 │   │
│  │ - Session transcripts    - User preferences           │   │
│  │ - Offline correction     - Export history              │   │
│  │   queue                                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
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

## 4. On-Device Punctuation Restoration

### The Problem

Moonshine outputs raw lowercase text: `so the main thing we need to discuss today is the quarterly budget`. We need: `So the main thing we need to discuss today is the quarterly budget.`

### Approach: Lightweight Token Classifier

A punctuation restoration model is a token-level classifier: for each word, predict what punctuation follows it (if any) and whether the next word should be capitalized.

**Label set**: `O` (nothing), `PERIOD`, `COMMA`, `QUESTION`, `EXCLAMATION`, `COLON`, `CAPITALIZE` (can be combined: `PERIOD+CAPITALIZE`).

### Model Options (Ranked by Mobile Feasibility)

#### Option A: Distilled BERT-Tiny Punctuation (Recommended for v1)

- **Base model**: A pruned/distilled XLM-RoBERTa or BERT-tiny (~15-30M params)
- **Size on disk**: ~60-120MB (INT8 quantized ONNX)
- **Inference time**: <50ms per utterance on iPhone (CoreML)
- **Training data**: Generate pairs from clean text (punctuated) → strip punctuation → train classifier
- **Existing starting point**: `oliverguhr/fullstop-punctuation-multilingual-base` (278M params) can be distilled down

#### Option B: Ali CT-Transformer Punctuation

- Used by the ManySpeech library specifically to complement Moonshine
- Available as ONNX
- Designed for CTC/transducer model output

#### Option C: Custom Fine-Tuned Tiny Model (Recommended for v2)

- Fine-tune a ~15M param model on Moonshine-specific output
- Training pipeline: Clean text → TTS → Moonshine → noisy transcript → correction pairs
- This captures Moonshine's specific error distribution (what words it tends to get wrong, where it splits utterances)
- Your QLoRA/fine-tuning experience maps directly to this task

### Punctuation Pipeline Integration

```swift
class PunctuationRestorer {
    private let session: OrtSession  // ONNX Runtime session
    private let tokenizer: BPETokenizer

    func restore(_ rawText: String) -> String {
        // 1. Tokenize input
        let tokens = tokenizer.encode(rawText)

        // 2. Run inference → per-token label predictions
        let labels = runInference(tokens)

        // 3. Post-process: insert punctuation, apply capitalization
        return applyLabels(rawText, labels)
    }
}
```

### Latency Budget

The entire pipeline from audio → displayed punctuated text should be under 500ms:

| Stage | Target Latency | Notes |
|---|---|---|
| Audio capture → VAD | ~100ms | Silero VAD processes 30ms frames |
| VAD → Moonshine ASR | ~150-200ms | Streaming partial results |
| ASR → Punctuation model | ~30-50ms | Token classifier on completed utterance |
| UI update | ~16ms | Main thread, next frame |
| **Total** | **~300-370ms** | Well within conversational feel |

---

## 5. Optional Cloud Correction Layer

### When Cloud Adds Value

- Correcting ASR word errors (homophones, rare words, proper nouns)
- Improving punctuation on ambiguous sentences
- Inverse text normalization (e.g., "five dollars" → "$5")
- Domain-specific terminology

### Architecture: API Gateway WebSocket + Lambda + Bedrock

```
iOS App
  ↕ WSS (persistent connection)
API Gateway WebSocket
  → $connect: Auth Lambda (Cognito JWT validation)
  → transcribe: Orchestrator Lambda
      → Bedrock (Claude Haiku / Amazon Nova Micro)
      → Verify correction via edit-distance
      → POST back via @connections/{connectionId}
  → heartbeat: no-op (keeps connection alive)
  → $disconnect: Cleanup Lambda
```

### Protocol

```json
// Client → Server
{
    "action": "transcribe",
    "sessionId": "sess_abc123",
    "sequenceNum": 42,
    "text": "So the main thing we need to discuss today is the quarterly budget.",
    "isFinal": true,
    "timestampMs": 1713275400000
}

// Server → Client
{
    "type": "correction",
    "sessionId": "sess_abc123",
    "sequenceNum": 42,
    "correctedText": "So, the main thing we need to discuss today is the quarterly budget.",
    "changes": ["added_comma:4"],
    "latencyMs": 230
}
```

### Connection Lifecycle

- **Session start**: App opens WebSocket if online. Falls back to local-only if offline.
- **Heartbeat**: Every 5 minutes to prevent API Gateway's 10-minute idle timeout.
- **Hard limit**: API Gateway enforces a 2-hour max connection duration. For sessions >2 hours, implement auto-reconnect with `sessionId` + `lastSequenceNum` to resume without loss.
- **Offline queue**: If the connection drops, completed utterances queue locally (SwiftData) and flush when reconnected.
- **Graceful degradation**: Cloud corrections are applied as "upgrades" to already-displayed local text. The UI animates the transition subtly (e.g., a brief highlight on changed words).

### AWS Cost Estimate (MVP Scale)

| Component | Unit Cost | 1,000 sessions/month (1hr each) |
|---|---|---|
| API GW connection minutes | $0.25/M minutes | $0.015 |
| API GW messages | $1.00/M messages | ~$0.72 |
| Lambda invocations | $0.20/M + compute | ~$2.00 |
| Bedrock (Haiku) tokens | ~$0.25/M input, $1.25/M output | ~$5-10 |
| DynamoDB | On-demand pricing | ~$1.00 |
| **Total** | | **~$10-15/month** |

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
├── Punctuation/
│   ├── PunctuationModel.swift       # ONNX Runtime inference for punct model
│   ├── PunctuationRestorer.swift    # Token classification + post-processing
│   └── TextFormatter.swift          # Capitalization rules, edge cases
│
├── Cloud/
│   ├── WebSocketManager.swift       # URLSessionWebSocketTask wrapper
│   ├── CloudCorrectionService.swift # Send utterance, receive correction
│   ├── ConnectionState.swift        # Connected/disconnected/reconnecting
│   ├── OfflineQueue.swift           # Queue corrections for when back online
│   └── CorrectionProtocol.swift     # JSON message encoding/decoding
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
    private let audioCapture: AudioCaptureManager
    private let vad: VADProcessor
    private let asr: MoonshineTranscriber
    private let punctuation: PunctuationRestorer
    private let cloud: CloudCorrectionService?

    // Published stream of transcript updates for the UI
    let transcriptStream: AsyncStream<TranscriptUpdate>

    func startSession(config: PipelineConfiguration) async throws { ... }
    func stopSession() async -> SessionRecord { ... }
}

// Transcript update types the UI observes
enum TranscriptUpdate {
    case partial(text: String, segmentId: UUID)           // ASR streaming partial
    case finalized(text: String, segmentId: UUID)         // ASR final + punctuated
    case corrected(text: String, segmentId: UUID)         // Cloud-corrected
    case error(TranscriptionError)
}
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

- **Streaming text**: Words appear at the bottom of a scrolling text view as they're spoken. The view auto-scrolls to follow.
- **Three-phase rendering**:
    1. **Partial** (gray, italic): ASR streaming output, may change
    2. **Finalized** (black, normal): Punctuated local result
    3. **Corrected** (black, normal, brief highlight): Cloud-enhanced, subtle animation on changed words
- **Confidence indicator**: Optional subtle underline on low-confidence words (tap to see alternatives).
- **Pause/resume**: Single tap to pause, tap again to resume. Audio continues recording during pause for seamless playback.

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

### Phase 1: Core Local Pipeline (Weeks 1-4)

**Goal**: Mic → ASR → text on screen, fully on-device.

- [ ] Set up Xcode project, ONNX Runtime Swift package
- [ ] Implement `AudioCaptureManager` with AVAudioEngine
- [ ] Integrate Silero VAD via ONNX Runtime
- [ ] Integrate Moonshine Tiny via ONNX Runtime + CoreML EP
- [ ] Build `TranscriptionPipeline` actor orchestrating Audio → VAD → ASR
- [ ] Build minimal `TranscriptionView` showing streaming text
- [ ] Test on physical device (iPhone 14+): verify real-time performance
- [ ] Measure battery drain over 30-minute session

**Exit criteria**: User can record speech and see raw (unpunctuated) transcript appear in real-time, fully offline, on an iPhone.

### Phase 2: Punctuation & Polish (Weeks 5-7)

**Goal**: Add punctuation restoration, session persistence, basic UX.

- [ ] Integrate or train punctuation model (start with Ali CT-Transformer or distilled BERT)
- [ ] Run punctuation model as second pass on finalized utterances
- [ ] Implement three-phase text rendering (partial → finalized → punctuated)
- [ ] Build SwiftData persistence layer for sessions
- [ ] Build `HomeView` with session list
- [ ] Implement export (clipboard, .txt, share sheet)
- [ ] Add background audio support
- [ ] Add Moonshine Base as downloadable "HD" model option

**Exit criteria**: User gets punctuated, capitalized transcripts. Sessions are saved and exportable. App works in background.

### Phase 3: Cloud Correction (Weeks 8-10)

**Goal**: Optional cloud enhancement via AWS.

- [ ] Deploy AWS infrastructure (CDK/Terraform):
    - API Gateway WebSocket
    - Lambda functions ($connect, transcribe, $disconnect)
    - Bedrock model access (Claude Haiku / Nova Micro)
    - DynamoDB session table
    - Cognito user pool
- [ ] Implement `WebSocketManager` with auto-reconnect + heartbeat
- [ ] Implement `CloudCorrectionService` with offline queue
- [ ] Build correction verification (edit-distance check in Lambda)
- [ ] Design and implement correction prompt for Bedrock
- [ ] Add cloud toggle in Settings
- [ ] Implement correction animation in `LiveTextView`
- [ ] Test: 1-hour session with intermittent connectivity

**Exit criteria**: Cloud corrections visibly improve transcript quality. App degrades gracefully to local-only when offline.

### Phase 4: Fine-Tuning & Optimization (Weeks 11-14)

**Goal**: Custom punctuation model, performance optimization, App Store readiness.

- [ ] Build synthetic training data pipeline:
    - Clean text corpus → strip punctuation → Moonshine output pairs
    - Use TTS to generate audio, run through Moonshine, collect error patterns
- [ ] Fine-tune/distill a tiny punctuation model (~15M params) for Moonshine output
- [ ] Quantize to INT8, convert to ONNX, benchmark on device
- [ ] Profile and optimize memory usage (Instruments → Allocations)
- [ ] Profile and optimize battery drain (Instruments → Energy Log)
- [ ] Optimize model cold-start time (memory-mapped ONNX, lazy loading)
- [ ] Accessibility audit (VoiceOver, Dynamic Type)
- [ ] App Store assets, screenshots, privacy policy
- [ ] TestFlight beta

**Exit criteria**: Polished app with custom punctuation model, optimized for battery/memory, ready for App Store submission.

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

| Service | Purpose |
|---|---|
| API Gateway (WebSocket) | Persistent client connection |
| Lambda (Python 3.12) | Orchestration, auth, correction logic |
| Amazon Bedrock | LLM inference for text correction |
| DynamoDB | Session state, connection mapping |
| Cognito | Mobile authentication |
| CloudWatch | Monitoring and alerting |
| S3 | Model hosting CDN origin |
| CloudFront | Model download CDN |

---

## 12. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Moonshine ONNX + CoreML EP performance insufficient | Users see lag | Medium | Fall back to CPU EP; benchmark early in Phase 1; keep Whisper as backup |
| Punctuation model too large for device | Exceeds memory budget | Low | Quantize aggressively (INT4); distill smaller; use simpler rule-based fallback |
| API Gateway 2-hour WebSocket limit | Long sessions disconnect | Medium | Auto-reconnect with sessionId + sequenceNum; tested in Phase 3 |
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
