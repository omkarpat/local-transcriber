# Phase 2: Punctuation & Polish — Detailed Task Plan

**Goal**: Add punctuation restoration, session persistence, export, and a polished UX.
**Timeline**: Weeks 5–7 (~60–80 dev hours)
**Dependencies**: Phase 1 complete (working local ASR pipeline)
**Exit Criteria**: User gets punctuated, capitalized transcripts. Sessions are saved and exportable. App works in background.

---

## Task Breakdown

### 1. Punctuation Model Integration

| Field | Detail |
|---|---|
| **Estimate** | 4–5 days |
| **Dependencies** | Phase 1 complete; ONNX Runtime already set up |
| **Risk** | 🔴 High — Model selection, ONNX export compatibility, and inference quality are all unknowns until tested on device |

**Subtasks**:

- **1.1** Evaluate candidate models (1 day — research + quick benchmarks on laptop):
  - **Option A**: `oliverguhr/fullstop-punctuation-multilingual-base` — download, test on sample Moonshine output, measure size (278M params full; need to quantize/distill)
  - **Option B**: Ali CT-Transformer punctuation (from ManySpeech) — check if ONNX export exists, test compatibility
  - **Option C**: Smaller alternatives on HuggingFace (`felflare/bert-restore-punctuation`, etc.)
  - **Decision criteria**: Size on disk (target: < 120MB quantized), inference speed (target: < 50ms/utterance on device), quality on Moonshine-style output (lowercase, no punct, occasional word errors)
- **1.2** Export/convert the chosen model to ONNX format:
  - If PyTorch: use `torch.onnx.export` with dynamic axes for variable-length input
  - Apply INT8 quantization (`onnxruntime.quantization`)
  - Verify the quantized model produces acceptable output vs. the full-precision version
- **1.3** Port or bundle the model's tokenizer for Swift:
  - If BPE-based: use a lightweight Swift BPE tokenizer (or compile a C++ tokenizer and bridge)
  - If SentencePiece: use the SentencePiece Swift/C++ library
  - Test that tokenization matches the Python reference exactly (token ID mismatches → garbage output)
- **1.4** Implement `PunctuationModel.swift`:
  - Load ONNX model via `OrtSession` (can reuse the shared CoreML EP environment)
  - Input: tokenized text → output: per-token label predictions
  - Labels: `O`, `PERIOD`, `COMMA`, `QUESTION`, `EXCLAMATION`, `COLON`, each optionally combined with `CAPITALIZE`
- **1.5** Implement `PunctuationRestorer.swift`:
  - Tokenize raw ASR text
  - Run inference
  - Post-process: insert punctuation characters, apply capitalization
  - Handle edge cases: sentence-initial capitalization, acronyms, numbers
- **1.6** Implement `TextFormatter.swift` — additional rule-based formatting:
  - Always capitalize "I"
  - Capitalize after sentence-ending punctuation
  - Handle common contractions ("i'm" → "I'm")
  - Trim extra whitespace
- **1.7** Benchmark on device:
  - Measure inference time per utterance (varying lengths: 5, 15, 50, 100 words)
  - Measure model loading time
  - Verify total pipeline latency stays under 500ms target

**Acceptance Criteria**: Punctuation model runs on device, adds periods/commas/question marks and capitalization to Moonshine output with reasonable accuracy. Inference < 50ms per utterance. Model size < 120MB on disk.

---

### 2. Three-Phase Text Rendering

| Field | Detail |
|---|---|
| **Estimate** | 3–4 days |
| **Dependencies** | Task 1 (punctuation model working), Phase 1 Task 6 (basic UI exists) |
| **Risk** | 🟡 Medium — Animating text transitions smoothly in SwiftUI while maintaining scroll position is tricky |

**Subtasks**:

- **2.1** Redesign `TranscriptUpdate` enum to support all three phases:
  ```
  .partial(text, segmentId)    → gray, italic — ASR streaming
  .finalized(text, segmentId)  → black, normal — punctuated result
  .corrected(text, segmentId)  → highlight animation — cloud (Phase 3, but design for it now)
  ```
- **2.2** Implement `LiveTextView.swift` — the core text display component:
  - Renders an array of `TranscriptSegment` objects
  - Each segment transitions through visual phases: partial → finalized
  - Use `AttributedString` for per-segment styling
  - Smooth transition animation when partial text is replaced by finalized text (crossfade or type-replace)
- **2.3** Handle segment replacement gracefully:
  - Partial text may differ from finalized text (ASR corrections + punctuation added)
  - Avoid jarring jumps — animate the transition
  - Maintain scroll position when earlier segments update
- **2.4** Implement auto-scroll behavior:
  - Auto-scroll to bottom when new text appears
  - Pause auto-scroll if user scrolls up (they're reviewing earlier text)
  - Resume auto-scroll when user scrolls back to bottom
  - Show a "scroll to bottom" button when auto-scroll is paused
- **2.5** Test with rapid speech: ensure UI stays responsive (60fps) even with frequent updates

**Acceptance Criteria**: Live text view shows partial results in gray/italic, transitions smoothly to finalized punctuated text in black. Auto-scroll works intuitively. No frame drops during rapid transcription.

---

### 3. SwiftData Persistence Layer

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Phase 1 Task 5 (TranscriptionSession model exists) |
| **Risk** | 🟢 Low — SwiftData is straightforward for this use case |

**Subtasks**:

- **3.1** Define SwiftData models in `SessionRecord.swift`:
  ```swift
  @Model class SessionRecord {
      var id: UUID
      var createdAt: Date
      var duration: TimeInterval
      var transcript: String          // full finalized text
      var segments: [SegmentRecord]   // individual segments with timestamps
      var wordCount: Int
      var modelUsed: String           // "moonshine-tiny" / "moonshine-base"
  }
  
  @Model class SegmentRecord {
      var id: UUID
      var text: String
      var startTime: TimeInterval     // offset from session start
      var endTime: TimeInterval
  }
  ```
- **3.2** Implement `TranscriptStore.swift`:
  - CRUD operations for `SessionRecord`
  - Auto-save during active session (every 30 seconds)
  - Query: all sessions sorted by date, search by text content
  - Delete: single session, bulk delete
- **3.3** Wire auto-save into `TranscriptionPipeline`:
  - Every 30 seconds during an active session, persist the current transcript state
  - On `stopSession()`, do a final save with complete transcript and duration
- **3.4** Handle migration: even though this is v1, set up a SwiftData schema version so future updates can migrate

**Acceptance Criteria**: Sessions are automatically saved during recording. Closing and reopening the app preserves all past sessions. Sessions include segment-level timestamps.

---

### 4. Home View & Session Management

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 3 (persistence layer) |
| **Risk** | 🟢 Low — Standard SwiftUI list/navigation patterns |

**Subtasks**:

- **4.1** Implement `HomeView.swift`:
  - List of past sessions, most recent first
  - Each row shows: date (relative, e.g., "Today", "Yesterday", "Apr 12"), duration (MM:SS), first-line preview (truncated), word count
  - Prominent "New Session" button (FAB-style or toolbar)
  - Empty state: friendly illustration + "Tap to start your first transcription"
- **4.2** Implement `TranscriptDetailView.swift`:
  - Full transcript display with segment timestamps
  - Copy-to-clipboard button
  - Export button (share sheet)
  - Edit mode: allow the user to manually correct words in the transcript
  - Delete session (with confirmation)
- **4.3** Implement list interactions:
  - Swipe to delete (with undo)
  - Long-press context menu: Export, Delete, Duplicate
- **4.4** Implement navigation: `HomeView` → `TranscriptionView` (new session) or `TranscriptDetailView` (past session)

**Acceptance Criteria**: Users can browse past sessions, view full transcripts, and manage sessions (delete, export). Navigation feels native and responsive.

---

### 5. Export Functionality

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 3 (session data with timestamps), Task 4 (detail view) |
| **Risk** | 🟢 Low — File formatting is straightforward |

**Subtasks**:

- **5.1** Implement `ExportManager.swift` with export formats:
  - **Plain text (.txt)**: Clean transcript, one paragraph per segment, no timestamps
  - **SRT subtitles (.srt)**: Standard SRT format with sequence number, timestamp range (HH:MM:SS,mmm), and segment text
  - **WebVTT (.vtt)**: WEBVTT header + timestamp → text cues
  - **Clipboard**: One-tap copy of full transcript text
- **5.2** Implement file export via `UIActivityViewController` (share sheet):
  - Write export file to a temp directory
  - Present share sheet with the file
  - Clean up temp files after sharing
- **5.3** Add export triggers:
  - Share button in `TranscriptDetailView` toolbar
  - Export option in session list context menu
  - Format picker (sheet or action sheet) when user taps export
- **5.4** Test export: verify .srt and .vtt files are valid (import into VLC or a subtitle viewer)

**Acceptance Criteria**: Users can export transcripts as .txt, .srt, or .vtt files, or copy to clipboard. Share sheet integrates with all standard iOS share targets. Subtitle files are correctly formatted.

---

### 6. Background Audio Support

| Field | Detail |
|---|---|
| **Estimate** | 1–2 days |
| **Dependencies** | Phase 1 Tasks 2, 5 (audio capture + pipeline) |
| **Risk** | 🟡 Medium — Background audio on iOS has specific requirements and can be interrupted by system events |

**Subtasks**:

- **6.1** Configure `Info.plist`:
  - Add `UIBackgroundModes: ["audio"]`
  - Add microphone usage description string
- **6.2** Configure `AVAudioSession` properly:
  - Category: `.record`
  - Handle interruptions (phone calls, Siri, other apps requesting audio):
    - Register for `AVAudioSession.interruptionNotification`
    - On interruption began: pause pipeline, show notification to user
    - On interruption ended: resume pipeline automatically (if `shouldResume` flag is set)
  - Handle route changes (headphones plugged/unplugged, Bluetooth connect/disconnect)
- **6.3** Test background scenarios:
  - Switch to another app during recording → verify transcription continues
  - Receive phone call during recording → verify graceful pause/resume
  - Lock screen during recording → verify transcription continues
  - Leave app in background for 30+ minutes → verify no termination
- **6.4** Add a subtle persistent notification or status bar indicator so the user knows recording is active in the background

**Acceptance Criteria**: Transcription continues when app is backgrounded. Audio interruptions are handled gracefully. No background termination during active sessions.

---

### 7. Settings & Model Management

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 1 (punctuation model), Phase 1 Tasks (ASR model) |
| **Risk** | 🟡 Medium — Model download/management adds complexity (network errors, storage space, partial downloads) |

**Subtasks**:

- **7.1** Implement `SettingsView.swift`:
  - **Model selection**: Toggle between Moonshine Tiny (default) and Moonshine Base ("HD" mode)
  - **Download management**: Show model sizes, download progress, delete downloaded models
  - **VAD sensitivity**: Slider or presets (quiet room / normal / noisy environment)
  - **Auto-save interval**: Configurable (15s / 30s / 60s)
  - **About**: App version, model versions, licenses, privacy policy link
- **7.2** Implement `ModelManager.swift`:
  - Check for model files in `Documents/models/`
  - Download models from CDN (use `URLSession` download task with progress tracking)
  - Verify SHA256 checksum after download
  - Resume interrupted downloads
  - Report download progress via `AsyncStream` for UI
  - Delete models (with confirmation — warn about re-download)
- **7.3** Implement first-launch model download flow:
  - On first launch, show a setup screen: "VoxLocal needs to download speech models (~270MB)"
  - Progress bar for each model (Moonshine Tiny + punctuation model)
  - Allow use of app only after required models are downloaded
  - Moonshine Base is optional — downloadable from Settings later
- **7.4** Store user preferences with `@AppStorage` / `UserDefaults`:
  - Selected model variant
  - VAD sensitivity
  - Auto-save interval
  - Whether cloud correction is enabled (design for Phase 3, default off)

**Acceptance Criteria**: Users can select between model variants. Moonshine Base downloads on demand with progress indication. Settings persist across app launches. First-launch setup works smoothly.

---

### 8. Integration Testing & UX Polish

| Field | Detail |
|---|---|
| **Estimate** | 3–4 days |
| **Dependencies** | All above tasks |
| **Risk** | 🟡 Medium — Integration issues between components; UX edge cases |

**Subtasks**:

- **8.1** End-to-end integration test:
  - Fresh install → first-launch model download → record session → see punctuated text → stop → view in session list → export as .txt and .srt
  - Verify the complete flow works without crashes
- **8.2** Test punctuation quality on real-world scenarios:
  - Interview-style Q&A (two speakers alternating)
  - Lecture/monologue (long continuous speech)
  - Casual conversation (incomplete sentences, filler words)
  - Technical content (numbers, abbreviations, jargon)
  - Document punctuation accuracy qualitatively — note systematic errors for Phase 4
- **8.3** UX polish pass:
  - Add `Haptics.swift` — haptic feedback on record start/stop
  - Smooth all animations (text transitions, screen transitions)
  - Verify Dynamic Type support (text scales with system font size setting)
  - Dark mode support (ensure all views look good in both modes)
  - App icon and launch screen (placeholder is fine, but it should exist)
- **8.4** Performance regression check:
  - Re-run Phase 1 benchmarks with punctuation model added
  - Verify total pipeline latency is still < 500ms
  - Verify memory usage is still within budget (< 500MB peak with both models)
  - Verify battery drain hasn't significantly increased
- **8.5** Bug bash: spend a focused session using the app as a real user would. Note and fix issues.

**Acceptance Criteria**: Complete user flow works end-to-end. Punctuation quality is documented. Performance targets are met with the punctuation model added. App feels polished and responsive.

---

## Dependency Graph

```
Phase 1 Complete
  ├──► Task 1 (Punctuation Model) ──► Task 2 (Text Rendering) ──┐
  ├──► Task 3 (Persistence) ──► Task 4 (Home View) ──────────────┤
  │                        └──► Task 5 (Export) ──────────────────┤
  ├──► Task 6 (Background Audio)                                  │
  └──► Task 7 (Settings & Models)                                 │
                                                                   │
                                    Task 8 (Integration & Polish) ◄┘
```

Tasks 1, 3, 6, and 7 can all begin in parallel at the start of Phase 2 since they have no interdependencies. Task 2 (text rendering) depends on Task 1. Tasks 4 and 5 depend on Task 3. Task 8 requires all others to be substantially complete.

---

## Summary

| Task | Estimate | Risk | Parallelizable? |
|---|---|---|---|
| 1. Punctuation Model | 4–5 days | 🔴 High | Yes — start immediately |
| 2. Three-Phase Text Rendering | 3–4 days | 🟡 Medium | After Task 1 |
| 3. SwiftData Persistence | 2–3 days | 🟢 Low | Yes — start immediately |
| 4. Home View & Sessions | 2–3 days | 🟢 Low | After Task 3 |
| 5. Export Functionality | 2–3 days | 🟢 Low | After Task 3 |
| 6. Background Audio | 1–2 days | 🟡 Medium | Yes — start immediately |
| 7. Settings & Model Mgmt | 2–3 days | 🟡 Medium | Yes — start immediately |
| 8. Integration & Polish | 3–4 days | 🟡 Medium | After all others |
| **Total** | **19–27 days** | | |

**Critical risk**: Task 1 (punctuation model). Model quality, size, and on-device performance are unknowns. **Recommendation**: Spend the first day of Phase 2 purely on model evaluation (Task 1.1) — test 2-3 candidate models on sample Moonshine output before committing to one. Have a fallback plan: a simple rule-based punctuator (regex + heuristics) that covers 80% of cases while a better model is being prepared for Phase 4.

**Parallelism opportunity**: With two developers, one could work Tasks 1→2 (punctuation pipeline) while the other works Tasks 3→4→5 (persistence + UI). Tasks 6 and 7 are smaller and can be picked up by whoever finishes first.
