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

    /// Smallest wait between retry attempts. Picked to feel responsive
    /// ("network back? within 5 s everything catches up") without
    /// dog-piling a recovering server.
    private let minRetryDelay: Duration = .seconds(5)
    /// Cap on exponential backoff when the server keeps refusing us.
    private let maxRetryDelay: Duration = .seconds(60)
    /// Safety cap on the queue so a long outage doesn't balloon memory.
    /// At ~100-char transcripts this tops out around ~20 KB — cheap.
    private let maxPendingPunctuation = 100

    private struct PendingPunctuation: Sendable {
        let utteranceID: UUID
        let text: String
        let utteranceDuration: Duration
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
            debugContinuation?.yield(.utteranceStarted(utteranceID: id))
            eventsContinuation.yield(.speechStateChanged(isActive: true))
        case .utteranceProgress(let id, let samples, let duration):
            transcriber.enqueuePartial(utteranceID: id, samples: samples, duration: duration)
        case .utteranceEnded(let id, let samples, let duration):
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
            // stack to idle so the UI can reset cleanly.
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
    private func startTranscriptForwarding() {
        guard transcriptForwardTask == nil else { return }
        let updatesContinuation = self.updatesContinuation
        let transcriberUpdates = transcriber.updates
        transcriptForwardTask = Task { [weak self] in
            for await update in transcriberUpdates {
                updatesContinuation.yield(update)
                guard case .finalized(let result) = update else { continue }
                guard let self, await self.configuration.enableCloudPunctuation else { continue }
                await self.dispatchFirstTryPunctuation(for: result)
            }
        }
    }

    /// First-try punctuation for a freshly-finalized utterance. Detached
    /// so multiple utterances can be in flight in parallel — the
    /// happy-path RTT on localhost is ~80 ms, but real networks vary.
    /// On failure, enqueue for the retry loop to pick up.
    private func dispatchFirstTryPunctuation(for result: TranscriptResult) {
        let client = punctuationClient
        Task.detached { [weak self] in
            let punctuated = await client.punctuate(
                utteranceID: result.utteranceID,
                text: result.text,
                utteranceDuration: result.utteranceDuration
            )
            guard let self else { return }
            if let punctuated {
                await self.handlePunctuationSuccess(punctuated)
            } else {
                await self.enqueueForRetry(
                    utteranceID: result.utteranceID,
                    text: result.text,
                    utteranceDuration: result.utteranceDuration
                )
            }
        }
    }

    /// A punctuate call just landed. Publish the result and — if we have
    /// a backlog — kick an immediate drain, since we just confirmed the
    /// server is reachable.
    private func handlePunctuationSuccess(_ p: TranscriptPunctuated) {
        updatesContinuation.yield(.punctuated(p))
        guard !pendingPunctuation.isEmpty else { return }
        retryTask?.cancel()
        retryTask = Task { [weak self] in
            await self?.drainLoop(initialDelay: .zero)
        }
    }

    /// Stash a failed request. Idempotent per utteranceID (guards against
    /// a double-enqueue if a retry attempt fails while the first-try
    /// callback is still in flight). Starts the retry loop if not already
    /// running.
    private func enqueueForRetry(utteranceID: UUID, text: String, utteranceDuration: Duration) {
        if pendingPunctuation.contains(where: { $0.utteranceID == utteranceID }) { return }
        pendingPunctuation.append(PendingPunctuation(
            utteranceID: utteranceID,
            text: text,
            utteranceDuration: utteranceDuration
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
        retryTask = Task { [weak self] in
            await self?.drainLoop(initialDelay: self?.minRetryDelay ?? .seconds(5))
        }
    }

    /// Keeps retrying until the queue drains or the owning actor goes
    /// away. `initialDelay` lets the happy-path-recovered case (server
    /// just answered) skip straight to a drain without waiting 5 s.
    private func drainLoop(initialDelay: Duration) async {
        var delay = initialDelay
        while !Task.isCancelled && !pendingPunctuation.isEmpty {
            if delay > .zero {
                do { try await Task.sleep(for: delay) }
                catch { break }  // cancelled mid-sleep
            }
            let drained = await drainAsManyAsPossible()
            if pendingPunctuation.isEmpty { break }
            if drained > 0 {
                // Made partial progress — server is flaky but responding.
                // Keep the pressure on at min delay.
                delay = minRetryDelay
            } else {
                // Zero progress — back off.
                delay = (delay == .zero) ? minRetryDelay : min(delay * 2, maxRetryDelay)
                log.info("punctuation retry backing off to \(String(describing: delay), privacy: .public); \(self.pendingPunctuation.count, privacy: .public) pending")
            }
        }
        retryTask = nil
    }

    /// Pop items off the front of the queue and punctuate them one at a
    /// time until we hit a failure. Returns the count we got through —
    /// zero means the server is still down.
    private func drainAsManyAsPossible() async -> Int {
        var drained = 0
        while let next = pendingPunctuation.first {
            if Task.isCancelled { break }
            let punctuated = await punctuationClient.punctuate(
                utteranceID: next.utteranceID,
                text: next.text,
                utteranceDuration: next.utteranceDuration
            )
            guard let punctuated else { break }
            // Guard against the queue being mutated during the await.
            // We only pop if the head is still the item we just sent.
            if pendingPunctuation.first?.utteranceID == next.utteranceID {
                pendingPunctuation.removeFirst()
            }
            updatesContinuation.yield(.punctuated(punctuated))
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
