# Phase 2: Refinement, Persistence & Polish — Detailed Task Plan

**Goal**: Adopt the V2 streaming refinement service, add session persistence, export, and polish the UX.
**Timeline**: Weeks 5–7 (~60–80 dev hours)
**Dependencies**: Phase 1 complete — local ASR pipeline working, V1 cloud punctuation service (`/punctuate`) integrated and shipping refined text on `TranscriptUpdate.refined`.
**Exit Criteria**: User gets progressively-refined transcripts streamed from the V2 service, with sessions saved and exportable, and the app working in the background.

> **Scope correction from original plan**: The earlier version of this doc assumed an on-device punctuation model would be Phase 2 Task 1. That work turned out differently — Phase 1 shipped a stateless cloud service (FastAPI + ONNX + XLM-R quantized) that already returns punctuated + cased text. So Phase 2's punctuation-related work is **not** to integrate a model, it's to **upgrade the existing cloud path to V2 streaming** (ITN, better truecasing, spoken commands) using SSE so the UI can progressively improve the text as each stage completes. See `punctuation-service-v2-plan.md` for the server-side plan.

---

## Task Breakdown

### 1. Adopt V2 Streaming Refinement

| Field | Detail |
|---|---|
| **Estimate** | 3–4 days (client); 4–6 days (server — tracked in `punctuation-service-v2-plan.md`) |
| **Dependencies** | Phase 1 cloud punctuation integrated; `TranscriptRefinement` struct + `.refined` case already in place |
| **Risk** | 🟡 Medium — mostly integration risk; the protocol is well-defined, but SSE stream lifecycle + retry idempotency have edge cases worth exercising |

**Subtasks** (client-side; the server track is independent):

- **1.1** Ship Commit 1 (**DONE** — this session): rename `.punctuated` → `.refined`, introduce `TranscriptRefinement` carrying both `utteranceID` (provenance) and `segmentID` (UI upsert key, equal to `utteranceID` until the server starts splitting), merge the user-facing view into a single flowing paragraph. Retry path still single-shot against `/punctuate`; zero behaviour change.
- **1.2** Ship Commit 2 (server): new `/punctuate/stream` endpoint emitting SSE events per stage (`commands` → `punct_case` → `itn`). Stateless. `segment_id` is deterministic `uuidv5(utteranceID, index)` so retries reproduce the same IDs and update in place. Add explicit in-flight backpressure, overload responses, and stream-failure metrics as part of this commit. See v2 plan for the event shape.
- **1.3** Ship Commit 3 (client): `PunctuationClient.punctuateStream(...)` returning `AsyncStream<PunctuationStageEvent>` via `URLSession.bytes(for:)`; teach `TranscriptionPipeline.dispatchFirstTryPunctuation` to consume the stream and yield one `.refined` update per stage. Each stage's text overwrites the prior stage for that `segmentID` — the UI sees the paragraph progressively improve.
- **1.4** Retry hardening: `PendingPunctuation` grows a `segmentIDAtDispatch` field so retries replay with the same incoming segment id. Deterministic uuidv5 on the server makes this largely moot today (single-segment), but paying the cost now keeps the contract honest when splitting lands.
- **1.5** Telemetry: per-stage latency logging (stage wall time, stream lifetime, partial vs. full completion) plus server-side counters for in-flight requests, overload rejects, stream failures, disconnects, and fallback rate. Keep payloads PII-free; guard client-side debug output behind the existing debug view toggle.
- **1.6** Smoke test end-to-end on simulator + physical device: verify mid-stream failures fall back cleanly to raw text, verify retries during server-down scenarios still produce the same rendered output, and verify burst traffic is shed explicitly instead of timing out silently.

**Acceptance Criteria**: User sees text progressively improve through commands → punct+case → ITN stages in real time. Stream failures degrade silently to raw local transcript. Retry queue survives server blips without producing duplicate or orphan segments in the UI. Under burst load, the server sheds excess work explicitly rather than letting p99 grow without bound.

---

### 2. Progressive Paragraph Rendering

| Field | Detail |
|---|---|
| **Estimate** | 2–3 days |
| **Dependencies** | Task 1.3 (streaming client wired); merged-paragraph view already in place |
| **Risk** | 🟡 Medium — scroll-position preservation during frequent text updates is the hard part |

**Subtasks**:

- **2.1** `TranscriptUpdate` cases (current shape — no more redesign needed):
  ```
  .partial(TranscriptPartial)       → secondary color, italic — ASR streaming
  .finalized(TranscriptResult)      → raw local text, primary color
  .refined(TranscriptRefinement)    → server-polished text, primary color, overwrites .finalized in-place
  .failed(TranscriptFailure)        → small inline ⚠︎ marker
  ```
  With V2 streaming, `.refined` may fire multiple times per utterance (once per stage). Each one overwrites prior text keyed on `segmentID`. No new case needed.
- **2.2** Auto-scroll behaviour for the merged paragraph:
  - Stick to bottom when new text lands (final stage or partial growing)
  - Pause auto-scroll if the user scrolls up to re-read earlier text
  - Resume when they return to the bottom; show a "scroll to bottom" button when paused
- **2.3** Subtle transition animation when a stage lands (e.g., briefly fade the delta, or just let SwiftUI's implicit animation handle the text change). Avoid type-by-type reveals — users perceive those as laggy.
- **2.4 (DONE)** Server-side paragraph splitting on `new paragraph` plus client-side silence-based reflow shipped. Renderer uses `Text(AttributedString)` with `\n\n` between same-utterance siblings and across utterances flagged `startsNewParagraph` (long silence, stop+restart, session start); single space otherwise. Dictation toggle in the header gates both the server-side split and the spoken-command interpretation.
- **2.5** Test with rapid dictation: paragraph stays at 60fps even when stage events arrive every ~30–50ms during cloud-refinement bursts.

**Acceptance Criteria**: Merged paragraph renders smoothly as stages land. Auto-scroll feels intuitive. No frame drops during rapid stage updates. Splitting produces clean paragraph breaks without disrupting scroll position.

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
  ├──► Task 1 (V2 Streaming Refinement) ──► Task 2 (Progressive Rendering) ──┐
  ├──► Task 3 (Persistence) ──► Task 4 (Home View) ───────────────────────────┤
  │                        └──► Task 5 (Export) ───────────────────────────────┤
  ├──► Task 6 (Background Audio)                                               │
  └──► Task 7 (Settings & Model Mgmt)                                          │
                                                                                │
                                             Task 8 (Integration & Polish) ◄────┘
```

Tasks 1, 3, 6, and 7 can all begin in parallel at the start of Phase 2 since they have no interdependencies. Task 2 (rendering) depends on Task 1.3 (streaming client wired). Tasks 4 and 5 depend on Task 3. Task 8 requires all others to be substantially complete.

Note that Task 1 has a **server-side track** (V2 service — ITN, NeMo joint model, spoken commands) and a **client-side track** (SSE consumer, retry hardening). The tracks are independent; client can ship against the existing V1 `/punctuate` contract while server work is in flight, and gain streaming benefit incrementally as each server stage lands.

---

## Summary

| Task | Estimate | Risk | Parallelizable? |
|---|---|---|---|
| 1. V2 Streaming Refinement | 3–4 days client + 4–6 days server | 🟡 Medium | Yes — start immediately |
| 2. Progressive Paragraph Rendering | 2–3 days | 🟡 Medium | After Task 1.3 |
| 3. SwiftData Persistence | 2–3 days | 🟢 Low | Yes — start immediately |
| 4. Home View & Sessions | 2–3 days | 🟢 Low | After Task 3 |
| 5. Export Functionality | 2–3 days | 🟢 Low | After Task 3 |
| 6. Background Audio | 1–2 days | 🟡 Medium | Yes — start immediately |
| 7. Settings & Model Mgmt | 2–3 days | 🟡 Medium | Yes — start immediately |
| 8. Integration & Polish | 3–4 days | 🟡 Medium | After all others |
| **Total** | **~18–26 days** (client) + **4–6 days** (server) | | |

**Critical risk**: the V2 server upgrade (Task 1 server track). Swapping the punctuation model to NeMo joint DistilBERT, adding WFST-based ITN, and wiring spoken commands all have unknowns — cold-start times, label-space compatibility after ONNX export, and ITN over-normalization in particular. **Recommendation**: ship ITN first behind a feature flag (smallest change, biggest user-perceived win), keep V1's punctuation model as fallback until the NeMo swap is validated on an eval set. The client ships against V1's `/punctuate` throughout; the SSE client lands in parallel and activates the moment `/punctuate/stream` is live.

**Parallelism opportunity**: one track per person — client streaming + rendering (Tasks 1-client → 2) alongside persistence + UI (Tasks 3 → 4 → 5). Background audio and settings can be filled in whenever. The server track runs independently.
