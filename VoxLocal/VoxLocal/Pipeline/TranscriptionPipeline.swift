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
        let punctuate = self.punctuationClient
        let transcriberUpdates = transcriber.updates
        transcriptForwardTask = Task { [weak self] in
            for await update in transcriberUpdates {
                updatesContinuation.yield(update)
                guard case .finalized(let result) = update else { continue }
                guard let self, await self.configuration.enableCloudPunctuation else { continue }
                // Detached so the punctuation hop doesn't serialize with
                // the transcriber's stream — multiple utterances can be in
                // flight simultaneously.
                Task.detached {
                    if let punctuated = await punctuate.punctuate(
                        utteranceID: result.utteranceID,
                        text: result.text,
                        utteranceDuration: result.utteranceDuration
                    ) {
                        updatesContinuation.yield(.punctuated(punctuated))
                    }
                }
            }
        }
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
