import Foundation

/// One completed ASR result. Emitted by `MoonshineTranscriber` on its
/// `updates` stream after an utterance has been fully decoded.
/// Partial/streaming results are deferred to a later task; everything
/// here is `finalized` for Phase 1.
nonisolated struct TranscriptUpdate: Sendable, Identifiable, Equatable {
    let id: UUID
    let text: String
    let utteranceDuration: Duration
    let inferenceDuration: Duration
    let tokenCount: Int

    /// Real-time factor — how long inference took relative to the audio
    /// duration. < 1.0 means faster than realtime; Phase 1 target is < 0.5.
    var realTimeFactor: Double {
        let utterance = Self.seconds(utteranceDuration)
        let inference = Self.seconds(inferenceDuration)
        guard utterance > 0 else { return .infinity }
        return inference / utterance
    }

    private static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
}
