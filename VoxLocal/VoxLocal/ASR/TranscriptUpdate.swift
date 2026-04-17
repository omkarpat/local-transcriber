import Foundation

/// One ASR result emitted by `MoonshineTranscriber` on its `updates`
/// stream. Either the utterance transcribed cleanly or inference failed
/// — both surface to the UI so users see when something broke instead of
/// silently dropping.
///
/// Partial / streaming results are deferred past Phase 1 (see plan.md
/// Task 4.3).
nonisolated enum TranscriptUpdate: Sendable, Identifiable, Equatable {
    case finalized(TranscriptResult)
    case failed(TranscriptFailure)

    var id: UUID {
        switch self {
        case .finalized(let r): return r.id
        case .failed(let f): return f.id
        }
    }

    var utteranceDuration: Duration {
        switch self {
        case .finalized(let r): return r.utteranceDuration
        case .failed(let f): return f.utteranceDuration
        }
    }
}

nonisolated struct TranscriptResult: Sendable, Identifiable, Equatable {
    let id: UUID
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

nonisolated struct TranscriptFailure: Sendable, Identifiable, Equatable {
    let id: UUID
    let utteranceDuration: Duration
    let message: String
}
