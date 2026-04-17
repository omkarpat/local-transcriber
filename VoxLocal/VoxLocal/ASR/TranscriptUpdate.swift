import Foundation

/// One ASR result emitted by `MoonshineTranscriber` on its `updates`
/// stream. `.partial` is a best-guess snapshot while the user is still
/// speaking; `.finalized` is the authoritative transcript once VAD closes
/// the utterance; `.failed` surfaces inference errors so the UI can show
/// them instead of silently dropping.
///
/// Partials and the final result for one utterance share a `utteranceID`
/// so the UI can upsert them into a single row.
nonisolated enum TranscriptUpdate: Sendable, Identifiable, Equatable {
    case partial(TranscriptPartial)
    case finalized(TranscriptResult)
    case failed(TranscriptFailure)

    /// Per-inference ID (unique to each emitted update).
    var id: UUID {
        switch self {
        case .partial(let p): return p.id
        case .finalized(let r): return r.id
        case .failed(let f): return f.id
        }
    }

    /// Groups partials + final for the same utterance onto one UI row.
    var utteranceID: UUID {
        switch self {
        case .partial(let p): return p.utteranceID
        case .finalized(let r): return r.utteranceID
        case .failed(let f): return f.utteranceID
        }
    }

    var utteranceDuration: Duration {
        switch self {
        case .partial(let p): return p.utteranceDuration
        case .finalized(let r): return r.utteranceDuration
        case .failed(let f): return f.utteranceDuration
        }
    }
}

nonisolated struct TranscriptResult: Sendable, Identifiable, Equatable {
    let id: UUID
    let utteranceID: UUID
    let text: String
    let utteranceDuration: Duration
    let inferenceDuration: Duration    // wall clock: model + tokenizer
    let tokenCount: Int
    let timings: TranscriptionTimings  // stage-by-stage breakdown from MoonshineModel

    static func == (lhs: TranscriptResult, rhs: TranscriptResult) -> Bool {
        lhs.id == rhs.id && lhs.text == rhs.text && lhs.tokenCount == rhs.tokenCount
    }

    /// Real-time factor — inference wall time / audio duration. < 1 is
    /// faster than realtime; Phase 1 target is < 0.5.
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

/// Best-guess transcript for an utterance that hasn't ended yet. Produced
/// when `VADConfiguration.partialTranscriptionInterval` is set; replaced
/// in the UI by the `TranscriptResult` when VAD closes the utterance.
nonisolated struct TranscriptPartial: Sendable, Identifiable, Equatable {
    let id: UUID
    let utteranceID: UUID
    let text: String
    let utteranceDuration: Duration    // audio length at the tick that produced this partial
    let inferenceDuration: Duration
}

nonisolated struct TranscriptFailure: Sendable, Identifiable, Equatable {
    let id: UUID
    let utteranceID: UUID
    let utteranceDuration: Duration
    let message: String
}
