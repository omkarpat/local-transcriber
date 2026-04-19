import Foundation
import os

/// Orchestrates the full local pipeline: mic → VAD → ASR → optional cloud
/// punctuation. Owns the per-session audio + VAD stack; reuses a single
/// long-lived `MoonshineTranscriber` + `PunctuationClient` across
/// start/stop cycles so the ~2 s ORT graph setup only happens once at
/// app launch.
///
/// Exposes two async streams:
///   - `updates`  — `TranscriptUpdate` per-utterance content.
///   - `events`   — `PipelineEvent` lifecycle + speech-state signals.
///
/// Marked `actor` for safe shared mutable lifecycle state (isRunning,
/// current VAD + forwarding tasks). Heavy compute still runs in detached
/// tasks inside `MoonshineTranscriber`; the actor only coordinates.
actor TranscriptionPipeline {
    let updates: AsyncStream<TranscriptUpdate>
    let events: AsyncStream<PipelineEvent>
    /// Optional debug-only stream. Nil unless the actor was constructed
    /// with `debug: true`. Carries per-frame probability + VAD state so
    /// the debug view can render its level meter and state pill without
    /// re-running VAD.
    let debugEvents: AsyncStream<DebugEvent>?

    private let manager: AudioCaptureManager
    private let transcriber: MoonshineTranscriber
    private let punctuationClient: PunctuationClient
    private let updatesContinuation: AsyncStream<TranscriptUpdate>.Continuation
    private let eventsContinuation: AsyncStream<PipelineEvent>.Continuation
    private let debugContinuation: AsyncStream<DebugEvent>.Continuation?
    private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "Pipeline")

    private var vad: VADProcessor?
    private var vadForwardTask: Task<Void, Never>?
    private var transcriptForwardTask: Task<Void, Never>?
    private var configuration: PipelineConfiguration = .default
    private var running = false

    /// Punctuation requests that failed on first-try (server down, timeout,
    /// 5xx). Retried on a backoff timer until each one lands or the queue
    /// overflows. In-memory only — lost across app restarts, which is
    /// acceptable for Phase 1 (and matches the silent-fallback contract).
    private var pendingPunctuation: [PendingPunctuation] = []
    private var retryTask: Task<Void, Never>?
    /// Monotonic counter identifying which drain loop currently owns
    /// `retryTask`. A cancelled drain's end-of-loop cleanup only clears
    /// `retryTask` if its epoch still matches — otherwise it would
    /// clobber a newer drain's registration and leak orphaned tasks.
    private var retryEpoch: Int = 0

    /// Smallest wait between retry attempts. Picked to feel responsive
    /// ("network back? within 5 s everything catches up") without
    /// dog-piling a recovering server.
    private let minRetryDelay: Duration = .seconds(5)
    /// Cap on exponential backoff when the server keeps refusing us.
    private let maxRetryDelay: Duration = .seconds(60)
    /// Safety cap on the queue so a long outage doesn't balloon memory.
    /// At ~100-char transcripts this tops out around ~20 KB — cheap.
    private let maxPendingPunctuation = 100
    /// How many consecutive "no progress" retry rounds we tolerate
    /// before giving up and dropping the queue. With the current delay
    /// schedule this is roughly 5 + 10 + 20 + 40 + 60 ≈ 2 min 15 s of
    /// retrying after the server first stopped responding — long enough
    /// to ride out a server crash-and-restart, short enough that a
    /// truly dead server doesn't leak retries forever. Any partial
    /// success (drained > 0) resets the counter.
    private let maxConsecutiveNoProgress = 5

    /// Wall-clock instant of the most recent `utteranceEnded` event for
    /// the current session. Reset to `nil` on `start()` and on `stop()`
    /// so silence detection treats a stop+restart as "no prior context"
    /// and forces a paragraph break for the first utterance after.
    private var lastUtteranceEndTime: ContinuousClock.Instant?

    /// One-shot flag that forces the next utterance to begin a new
    /// paragraph regardless of measured silence. Set on `stop()` so the
    /// user's "stop recording" gesture is treated as a paragraph break
    /// even when they immediately resume. Consumed (and reset) by the
    /// next `utteranceStarted` event.
    private var nextUtteranceStartsParagraph: Bool = false

    /// Utterance IDs whose first segment row should render as a new
    /// visual paragraph. Populated at `utteranceStarted` time based on
    /// silence and stop-flag state; consumed by the update-forwarding
    /// task as it enriches partial/finalized events. Bounded growth —
    /// entries are removed once the corresponding `.finalized` is
    /// forwarded (or on `start()` reset).
    private var paragraphBreakUtteranceIDs: Set<UUID> = []

    private struct PendingPunctuation: Sendable {
        let utteranceID: UUID
        /// The `segmentID` that was current when this utterance was first
        /// dispatched. Retry replays with *this* value, not whatever the
        /// session's current running segment has become in the meantime —
        /// otherwise the server would derive different segment IDs on
        /// retry and create orphan rows in the UI. For V2 single-segment
        /// mode this equals `utteranceID`; it'll diverge once cross-
        /// utterance segment state lands in Phase 3.
        let segmentIDAtDispatch: UUID
        let text: String
        let utteranceDuration: Duration
        /// Captured at first-try dispatch so retries replay with the
        /// same Stage 1 mode the user had selected at the time the
        /// utterance was spoken. Toggling dictation mid-session must
        /// not retroactively reinterpret already-recorded utterances.
        let dictationModeAtDispatch: Bool
    }

    init(
        transcriber: MoonshineTranscriber,
        punctuationClient: PunctuationClient,
        manager: AudioCaptureManager = AudioCaptureManager(),
        debug: Bool = false
    ) {
        self.transcriber = transcriber
        self.punctuationClient = punctuationClient
        self.manager = manager

        var updatesCont: AsyncStream<TranscriptUpdate>.Continuation!
        self.updates = AsyncStream(bufferingPolicy: .bufferingNewest(64)) { c in
            updatesCont = c
        }
        self.updatesContinuation = updatesCont

        var eventsCont: AsyncStream<PipelineEvent>.Continuation!
        self.events = AsyncStream(bufferingPolicy: .bufferingNewest(32)) { c in
            eventsCont = c
        }
        self.eventsContinuation = eventsCont

        if debug {
            var debugCont: AsyncStream<DebugEvent>.Continuation!
            self.debugEvents = AsyncStream(bufferingPolicy: .bufferingNewest(256)) { c in
                debugCont = c
            }
            self.debugContinuation = debugCont
        } else {
            self.debugEvents = nil
            self.debugContinuation = nil
        }
    }

    var isRunning: Bool { running }

    /// Poll-friendly readout for UI level meters. Actor hop is cheap at
    /// ~20 Hz polling. Returns 0 when the pipeline is idle.
    var currentRMS: Float { manager.currentRMS }

    /// Total 16 kHz samples captured since the current session started.
    var framesCaptured: Int { manager.framesCaptured }

    /// Drop counter from the ring buffer. Growth here means the consumer
    /// (VAD) is falling behind producer (mic); only hits in pathological
    /// conditions in practice.
    var overflowSamples: Int { manager.ringBuffer.overflowSamples }

    /// How many finalized utterances are currently waiting on a retry
    /// punctuation call. Zero in the happy path; grows during server
    /// outages and drains back to zero once the server is reachable.
    var pendingPunctuationCount: Int { pendingPunctuation.count }

    /// Live read of the current dictation toggle. Reflects whatever the
    /// most recent `setDictationMode(_:)` call set; safe to read between
    /// recordings to surface the current state in the UI.
    var dictationMode: Bool { configuration.dictationMode }

    /// Update the dictation toggle mid-session. Takes effect for the
    /// next utterance that finishes after the call lands; in-flight
    /// dispatches and queued retries replay with the value captured at
    /// their original dispatch time, so toggling the UI doesn't
    /// retroactively reinterpret already-spoken utterances.
    func setDictationMode(_ enabled: Bool) {
        configuration.dictationMode = enabled
        log.info("dictation mode \(enabled ? "enabled" : "disabled", privacy: .public)")
    }

    /// Start a fresh session. Idempotent — already-running is a no-op.
    /// Throws on audio/permission failure; VAD errors surface later on
    /// `events` as `.failed(String)`.
    func start(configuration: PipelineConfiguration = .default) async throws {
        guard !running else { return }
        self.configuration = configuration
        do {
            try await manager.start()
        } catch {
            log.error("audio start failed: \(String(describing: error), privacy: .public)")
            throw error
        }

        // Reset paragraph-break state for the fresh session. Carrying
        // over from the previous session would either force a redundant
        // break (stop-flag still set) or compute silence against a stale
        // end time. Either way, fresh state matches the user's mental
        // model of "I just hit record."
        lastUtteranceEndTime = nil
        nextUtteranceStartsParagraph = false
        paragraphBreakUtteranceIDs.removeAll()

        let vad = VADProcessor(ringBuffer: manager.ringBuffer, configuration: configuration.vad)
        self.vad = vad
        startVADForwarding(vad: vad)
        startTranscriptForwarding()
        vad.start()

        running = true
        eventsContinuation.yield(.started)
        log.info("pipeline started")
    }

    /// Stop the current session. Safe to call repeatedly. `stop()` does
    /// not cancel in-flight ASR tasks — late `.finalized` or `.failed`
    /// updates for utterances that were already accumulating will still
    /// arrive on `updates` after we return, which is usually what the
    /// user wants (don't discard a word they just finished saying).
    func stop() {
        guard running else { return }
        vad?.stop()
        vad = nil
        vadForwardTask?.cancel()
        vadForwardTask = nil
        manager.stop()
        // Deliberately DO NOT cancel `retryTask` or clear
        // `pendingPunctuation` here. If the server was flaky during the
        // session, those utterances should still get refined when the
        // server recovers — the drain loop has its own termination
        // condition (see `drainLoop`) so it can't run forever.

        // The user's stop gesture is itself a paragraph-break signal: if
        // they hit record again, that next utterance should start a
        // fresh paragraph regardless of how brief the gap was. Reset
        // `lastUtteranceEndTime` too so silence comparisons after the
        // restart compute against the actual recording window, not the
        // stale gap that includes the stop+restart latency.
        nextUtteranceStartsParagraph = true
        lastUtteranceEndTime = nil

        running = false
        eventsContinuation.yield(.stopped)
        log.info("pipeline stopped")
    }

    // MARK: - Private

    private func startVADForwarding(vad: VADProcessor) {
        vadForwardTask?.cancel()
        let events = vad.events
        vadForwardTask = Task { [weak self] in
            for await event in events {
                guard let self else { return }
                await self.handle(vadEvent: event)
            }
        }
    }

    private func handle(vadEvent: VADEvent) {
        switch vadEvent {
        case .frame(let probability, let isSpeech):
            debugContinuation?.yield(.frame(probability: probability, isSpeech: isSpeech))
        case .utteranceStarted(let id):
            // Decide right now whether this utterance should begin a new
            // visual paragraph. We mark it iff the stop-recording flag
            // is set OR the silence since the prior utterance ended
            // crossed the configured threshold. The first utterance of
            // a session has no prior end time and no stop flag, so it
            // does NOT get marked — it's the first content; there's
            // nothing to break from.
            let now = ContinuousClock.now
            let breakFromSilence: Bool
            if let lastEnd = lastUtteranceEndTime {
                breakFromSilence = (now - lastEnd) >= configuration.paragraphSilenceThreshold
            } else {
                breakFromSilence = false
            }
            if nextUtteranceStartsParagraph || breakFromSilence {
                paragraphBreakUtteranceIDs.insert(id)
            }
            nextUtteranceStartsParagraph = false
            debugContinuation?.yield(.utteranceStarted(utteranceID: id))
            eventsContinuation.yield(.speechStateChanged(isActive: true))
        case .utteranceProgress(let id, let samples, let duration):
            transcriber.enqueuePartial(utteranceID: id, samples: samples, duration: duration)
        case .utteranceEnded(let id, let samples, let duration):
            // Record the end time so the *next* utteranceStarted can
            // measure the silence gap. Stamped synchronously here so it
            // reflects when the VAD event arrived, not when downstream
            // ASR work completes.
            lastUtteranceEndTime = ContinuousClock.now
            debugContinuation?.yield(.utteranceEnded(utteranceID: id, duration: duration))
            eventsContinuation.yield(.speechStateChanged(isActive: false))
            if configuration.debugDumpUtterances {
                dumpWAV(samples: samples)
            }
            transcriber.enqueue(utteranceID: id, samples: samples, duration: duration)
        case .error(let message):
            log.error("VAD error: \(message, privacy: .public)")
            eventsContinuation.yield(.failed(message))
            // VAD has already torn down its loop; bring the rest of the
            // stack to idle so the UI can reset cleanly. Retries for
            // already-finalized utterances keep running — same rationale
            // as `stop()`.
            vad?.stop()
            vad = nil
            vadForwardTask?.cancel()
            vadForwardTask = nil
            manager.stop()
            running = false
            eventsContinuation.yield(.stopped)
        }
    }

    /// Bridge the transcriber's stream onto our `updates` stream, and
    /// fire cloud-punctuation requests in parallel for finalized results.
    /// Survives across start/stop because the transcriber is long-lived;
    /// we only (re)install it on first start.
    ///
    /// Each partial/finalized is enriched with `startsNewParagraph` based
    /// on the per-utterance decision we made at `utteranceStarted` time.
    /// Once the `.finalized` event has been forwarded the entry is
    /// removed from `paragraphBreakUtteranceIDs` so the set doesn't
    /// grow without bound across long sessions. Refined and failed
    /// updates pass through untouched — the row already carries the
    /// flag from its initial insert.
    private func startTranscriptForwarding() {
        guard transcriptForwardTask == nil else { return }
        let updatesContinuation = self.updatesContinuation
        let transcriberUpdates = transcriber.updates
        transcriptForwardTask = Task { [weak self] in
            for await update in transcriberUpdates {
                guard let self else {
                    updatesContinuation.yield(update)
                    continue
                }
                let enriched = await self.enrichingParagraphFlag(update)
                updatesContinuation.yield(enriched)
                guard case .finalized(let result) = enriched else { continue }
                guard await self.configuration.enableCloudPunctuation else { continue }
                await self.dispatchFirstTryPunctuation(for: result)
            }
        }
    }

    /// Look up the paragraph-break decision for this update's utterance
    /// and stamp it onto partial/finalized payloads. Finalized
    /// consumes the entry — partials may keep arriving for the same
    /// utterance, but once finalized has been forwarded the per-utterance
    /// decision has done its job and the entry can be released.
    private func enrichingParagraphFlag(_ update: TranscriptUpdate) -> TranscriptUpdate {
        let breaks = paragraphBreakUtteranceIDs.contains(update.utteranceID)
        switch update {
        case .partial(var p):
            p.startsNewParagraph = breaks
            return .partial(p)
        case .finalized(var r):
            r.startsNewParagraph = breaks
            paragraphBreakUtteranceIDs.remove(update.utteranceID)
            return .finalized(r)
        case .refined, .failed:
            return update
        }
    }

    /// First-try refinement for a freshly-finalized utterance. Detached
    /// so multiple utterances can be in flight in parallel — the
    /// happy-path RTT on localhost is ~80 ms for the full V2 pipeline,
    /// but real networks vary. On failure (stream ends without a `final`
    /// event), enqueue for the retry loop to pick up.
    private func dispatchFirstTryPunctuation(for result: TranscriptResult) {
        // V2 single-segment: the segment for every utterance is the
        // utteranceID itself. When cross-utterance segment state lands
        // (Phase 3 — "new paragraph" detection + silence-based breaks),
        // capture `self.currentSegmentID` here instead. The `AtDispatch`
        // naming reminds readers this is frozen at first-try time and
        // must be reused verbatim on retry.
        let dispatchSegmentID = result.utteranceID
        // Capture the dictation toggle as it stood when the user finished
        // speaking. Toggling mid-session must not retroactively flip
        // already-recorded utterances between command-interpreted and
        // literal — the retry queue replays this same value.
        let dispatchDictationMode = configuration.dictationMode

        Task.detached { [weak self] in
            guard let self else { return }
            let succeeded = await self.consumeRefinementStream(
                utteranceID: result.utteranceID,
                segmentID: dispatchSegmentID,
                text: result.text,
                utteranceDuration: result.utteranceDuration,
                dictationMode: dispatchDictationMode
            )
            if !succeeded {
                await self.enqueueForRetry(
                    utteranceID: result.utteranceID,
                    segmentIDAtDispatch: dispatchSegmentID,
                    text: result.text,
                    utteranceDuration: result.utteranceDuration,
                    dictationMode: dispatchDictationMode
                )
            }
        }
    }

    /// Consume a refinement stream for one utterance. Yields each stage
    /// event onto `updates` — intermediate stages via `emitRefinement`,
    /// the terminal `final` event via `handleRefinementSuccess` (which
    /// also kicks the retry drain since a successful completion means
    /// the server is reachable). Returns `true` if the stream ended with
    /// a `final` event, `false` otherwise (mid-stream drop, non-2xx,
    /// stream exhausted without final).
    private func consumeRefinementStream(
        utteranceID: UUID,
        segmentID: UUID,
        text: String,
        utteranceDuration: Duration,
        dictationMode: Bool
    ) async -> Bool {
        let start = ContinuousClock.now
        var sawFinal = false
        for await event in punctuationClient.punctuateStream(
            utteranceID: utteranceID,
            segmentID: segmentID,
            text: text,
            dictationMode: dictationMode
        ) {
            let elapsed = ContinuousClock.now - start
            let refinement = TranscriptRefinement(
                id: UUID(),
                utteranceID: event.utteranceID,
                segmentID: event.segmentID,
                text: event.text,
                sourceText: text,
                utteranceDuration: utteranceDuration,
                roundTripDuration: elapsed
            )
            if event.isFinal {
                sawFinal = true
                handleRefinementSuccess(refinement)
            } else {
                emitRefinement(refinement)
            }
        }
        return sawFinal
    }

    /// Yield a refinement onto `updates` without kicking the retry drain.
    /// Intermediate stage events go through here — they're progress, not
    /// completion signals.
    private func emitRefinement(_ r: TranscriptRefinement) {
        updatesContinuation.yield(.refined(r))
    }

    /// A stream just completed with `final: true`. Publish the refinement
    /// and — if there's a backlog and no drain is currently running —
    /// kick a fresh one at zero delay since we just confirmed the server
    /// is reachable.
    ///
    /// We deliberately do NOT cancel an in-progress drain here. Cancel
    /// propagates into the drain's current `consumeRefinementStream`,
    /// aborting a retry request that the server is already processing;
    /// under continuous speech every success fires this, starving the
    /// queue because pending items never finish a stream. A running
    /// drain will pick up its next iteration on its own schedule (worst
    /// case: the current `maxRetryDelay` sleep finishes first).
    private func handleRefinementSuccess(_ r: TranscriptRefinement) {
        emitRefinement(r)
        guard retryTask == nil, !pendingPunctuation.isEmpty else { return }
        startDrainLoop(initialDelay: .zero)
    }

    /// Stash a failed request. Idempotent per utteranceID (guards against
    /// a double-enqueue if a retry attempt fails while the first-try
    /// callback is still in flight). Starts the retry loop if not already
    /// running.
    private func enqueueForRetry(
        utteranceID: UUID,
        segmentIDAtDispatch: UUID,
        text: String,
        utteranceDuration: Duration,
        dictationMode: Bool
    ) {
        if pendingPunctuation.contains(where: { $0.utteranceID == utteranceID }) { return }
        pendingPunctuation.append(PendingPunctuation(
            utteranceID: utteranceID,
            segmentIDAtDispatch: segmentIDAtDispatch,
            text: text,
            utteranceDuration: utteranceDuration,
            dictationModeAtDispatch: dictationMode
        ))
        if pendingPunctuation.count > maxPendingPunctuation {
            let drop = pendingPunctuation.count - maxPendingPunctuation
            pendingPunctuation.removeFirst(drop)
            log.warning("punctuation backlog at cap — dropped \(drop, privacy: .public) oldest item(s)")
        }
        ensureRetryLoop()
    }

    private func ensureRetryLoop() {
        guard retryTask == nil else { return }
        startDrainLoop(initialDelay: minRetryDelay)
    }

    /// Mint a fresh `retryEpoch` and spawn a drain task tagged with it.
    /// Only the drain whose epoch is still current may nil out
    /// `retryTask` at the end of its loop.
    private func startDrainLoop(initialDelay: Duration) {
        retryEpoch += 1
        let epoch = retryEpoch
        retryTask = Task { [weak self] in
            await self?.drainLoop(initialDelay: initialDelay, epoch: epoch)
        }
    }

    /// Keeps retrying until the queue drains, the actor goes away, or
    /// we hit the no-progress streak ceiling (see
    /// `maxConsecutiveNoProgress`). `initialDelay` lets the
    /// happy-path-recovered case (server just answered) skip straight to
    /// a drain without waiting 5 s. `epoch` identifies this specific
    /// drain — only the current-epoch drain is allowed to clear
    /// `retryTask` on exit, so a cancelled drain's late cleanup can't
    /// clobber a newer drain's registration.
    private func drainLoop(initialDelay: Duration, epoch: Int) async {
        var delay = initialDelay
        var noProgressStreak = 0
        while !Task.isCancelled && !pendingPunctuation.isEmpty {
            if delay > .zero {
                do { try await Task.sleep(for: delay) }
                catch { break }  // cancelled mid-sleep
            }
            let drained = await drainAsManyAsPossible()
            if pendingPunctuation.isEmpty { break }
            if drained > 0 {
                // Made partial progress — server is flaky but responding.
                // Reset the streak and keep the pressure on at min delay.
                delay = minRetryDelay
                noProgressStreak = 0
            } else {
                noProgressStreak += 1
                if noProgressStreak >= maxConsecutiveNoProgress {
                    log.info("punctuation retry giving up after \(noProgressStreak, privacy: .public) failed rounds; dropping \(self.pendingPunctuation.count, privacy: .public) pending")
                    pendingPunctuation.removeAll()
                    break
                }
                delay = (delay == .zero) ? minRetryDelay : min(delay * 2, maxRetryDelay)
                log.info("punctuation retry backing off to \(String(describing: delay), privacy: .public); \(self.pendingPunctuation.count, privacy: .public) pending (attempt \(noProgressStreak, privacy: .public)/\(self.maxConsecutiveNoProgress, privacy: .public))")
            }
        }
        if retryEpoch == epoch {
            retryTask = nil
        }
    }

    /// Pop items off the front of the queue and refine them one at a
    /// time until we hit a failure. Returns the count we got through —
    /// zero means the server is still down. The refinement stream yields
    /// intermediate stage events through `emitRefinement` while each
    /// utterance is in flight; we only pop an item after its stream
    /// completes successfully (final event received).
    private func drainAsManyAsPossible() async -> Int {
        var drained = 0
        while let next = pendingPunctuation.first {
            if Task.isCancelled { break }
            let succeeded = await consumeRefinementStream(
                utteranceID: next.utteranceID,
                segmentID: next.segmentIDAtDispatch,
                text: next.text,
                utteranceDuration: next.utteranceDuration,
                dictationMode: next.dictationModeAtDispatch
            )
            guard succeeded else { break }
            // Guard against the queue being mutated during the stream's
            // awaits. We only pop if the head is still the item we just
            // processed.
            if pendingPunctuation.first?.utteranceID == next.utteranceID {
                pendingPunctuation.removeFirst()
            }
            drained += 1
        }
        return drained
    }
}

/// Fine-grained signals emitted only when the pipeline is constructed
/// with `debug: true`. Feeds the `AudioCaptureDebugView` level meter,
/// state pill, and utterance counters without duplicating VAD wiring.
enum DebugEvent: Sendable, Equatable {
    case frame(probability: Float, isSpeech: Bool)
    case utteranceStarted(utteranceID: UUID)
    case utteranceEnded(utteranceID: UUID, duration: Duration)
}

extension TranscriptionPipeline {
    private func dumpWAV(samples: [Float]) {
        let stamp = Int(Date().timeIntervalSince1970 * 1000)
        let url = WAVWriter.documentsDirectory().appendingPathComponent("utterance-\(stamp).wav")
        do {
            try WAVWriter.write(samples: samples, to: url)
            log.info("Wrote utterance WAV: \(url.path, privacy: .public)")
        } catch {
            log.error("WAV write failed: \(String(describing: error), privacy: .public)")
        }
    }
}
