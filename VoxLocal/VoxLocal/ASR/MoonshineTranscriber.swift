import Foundation
import os

/// Push-based transcriber. Holds a pre-loaded `MoonshineModel` +
/// `MoonshineTokenizer`; callers call `enqueue(samples:duration:)` with
/// VAD-segmented audio and receive completed `TranscriptUpdate`s on the
/// `updates` async stream.
///
/// Each `enqueue` call spawns a detached task, so two utterances arriving
/// back-to-back transcribe in parallel rather than queuing. This is fine
/// on a single device since RTF is well below 1; we'll add serialization
/// only if it stops being fine.
///
/// `nonisolated` because transcription is compute-heavy and never touches UI.
nonisolated final class MoonshineTranscriber: @unchecked Sendable {
    let updates: AsyncStream<TranscriptUpdate>

    private let model: MoonshineModel
    private let tokenizer: MoonshineTokenizer
    private let continuation: AsyncStream<TranscriptUpdate>.Continuation
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

    /// Kick off transcription of a single utterance. Returns immediately;
    /// the completed `TranscriptUpdate` (either `.finalized` or `.failed`)
    /// shows up on `updates` when inference settles.
    func enqueue(samples: [Float], duration: Duration) {
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
                    utteranceDuration: duration,
                    message: String(describing: error)
                )))
            }
        }
    }

    func finish() {
        continuation.finish()
    }
}
