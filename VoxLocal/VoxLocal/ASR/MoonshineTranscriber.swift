import Foundation
import os

/// Push-based transcriber. Holds a pre-loaded `MoonshineModel` +
/// `MoonshineTokenizer`; callers push VAD-segmented audio via `enqueue`
/// for the finalized result, or via `enqueuePartial` for in-progress
/// best-guess snapshots while the user is still speaking. Completed
/// `TranscriptUpdate`s surface on the `updates` async stream.
///
/// Each `enqueue` / `enqueuePartial` call spawns a detached task, so two
/// utterances transcribe in parallel rather than queuing. RTF is well
/// below 1 on Apple Silicon so this is fine; we'll serialize only if
/// that stops holding.
///
/// Partials are gated by `PartialGate`: at most one partial per
/// utterance may be in flight. Extra ticks while a partial is running
/// are dropped (let-finish), so slow inference never builds a backlog.
///
/// `nonisolated` because transcription is compute-heavy and never touches UI.
nonisolated final class MoonshineTranscriber: @unchecked Sendable {
    let updates: AsyncStream<TranscriptUpdate>

    private let model: MoonshineModel
    private let tokenizer: MoonshineTokenizer
    private let continuation: AsyncStream<TranscriptUpdate>.Continuation
    private let partialGate = PartialGate()
    private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "Transcriber")

    init(model: MoonshineModel, tokenizer: MoonshineTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        var cont: AsyncStream<TranscriptUpdate>.Continuation!
        self.updates = AsyncStream(bufferingPolicy: .bufferingNewest(32)) { c in
            cont = c
        }
        self.continuation = cont
    }

    /// Kick off transcription of a finalized utterance. Returns
    /// immediately; the `.finalized` (or `.failed`) update shows up on
    /// `updates` when inference settles.
    func enqueue(utteranceID: UUID, samples: [Float], duration: Duration) {
        let model = self.model
        let tokenizer = self.tokenizer
        let continuation = self.continuation
        let log = self.log
        Task.detached(priority: .userInitiated) {
            let start = Date()
            do {
                let result = try model.transcribe(samples: samples)
                let text = tokenizer.decode(tokenIDs: result.tokens)
                let inferenceDuration = Duration.seconds(Date().timeIntervalSince(start))
                continuation.yield(.finalized(TranscriptResult(
                    id: UUID(),
                    utteranceID: utteranceID,
                    text: text,
                    utteranceDuration: duration,
                    inferenceDuration: inferenceDuration,
                    tokenCount: result.tokens.count,
                    timings: result.timings
                )))
            } catch {
                log.error("transcription failed: \(String(describing: error))")
                continuation.yield(.failed(TranscriptFailure(
                    id: UUID(),
                    utteranceID: utteranceID,
                    utteranceDuration: duration,
                    message: String(describing: error)
                )))
            }
        }
    }

    /// Kick off a best-guess transcription of an in-progress utterance.
    /// Drops the call if a partial for the same `utteranceID` is still
    /// running — the next tick will have more audio anyway. Inference
    /// errors during a partial are logged and dropped; only finalized
    /// failures surface to the UI.
    func enqueuePartial(utteranceID: UUID, samples: [Float], duration: Duration) {
        let model = self.model
        let tokenizer = self.tokenizer
        let continuation = self.continuation
        let gate = self.partialGate
        let log = self.log
        Task.detached(priority: .userInitiated) {
            guard await gate.tryAcquire(utteranceID) else {
                log.debug("dropping partial tick for \(utteranceID, privacy: .public) — prior still in flight")
                return
            }
            let start = Date()
            do {
                let result = try model.transcribe(samples: samples)
                let text = tokenizer.decode(tokenIDs: result.tokens)
                let inferenceDuration = Duration.seconds(Date().timeIntervalSince(start))
                continuation.yield(.partial(TranscriptPartial(
                    id: UUID(),
                    utteranceID: utteranceID,
                    text: text,
                    utteranceDuration: duration,
                    inferenceDuration: inferenceDuration
                )))
            } catch {
                log.error("partial transcription failed: \(String(describing: error))")
            }
            await gate.release(utteranceID)
        }
    }

    func finish() {
        continuation.finish()
    }
}

/// Tracks which utterances currently have a partial inference in flight
/// so we don't stack multiple Moonshine calls on the same utterance.
/// `tryAcquire` is the gate; `release` clears it on completion.
private actor PartialGate {
    private var inFlight: Set<UUID> = []

    func tryAcquire(_ id: UUID) -> Bool {
        if inFlight.contains(id) { return false }
        inFlight.insert(id)
        return true
    }

    func release(_ id: UUID) {
        inFlight.remove(id)
    }
}
