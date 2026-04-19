import Foundation

/// One result emitted downstream for the UI to render. `.partial` is a
/// best-guess snapshot while the user is still speaking; `.finalized` is
/// the authoritative on-device transcript once VAD closes the utterance;
/// `.refined` is the server-polished replacement for a finalized row
/// (today a single terminal update, tomorrow possibly many stage-by-stage
/// updates as the V2 streaming pipeline lands); `.failed` surfaces
/// inference errors so the UI can show them instead of silently dropping.
///
/// All cases share `utteranceID` so the UI can upsert them into a single
/// row across partial → final → refined.
nonisolated enum TranscriptUpdate: Sendable, Identifiable, Equatable {
    case partial(TranscriptPartial)
    case finalized(TranscriptResult)
    case refined(TranscriptRefinement)
    case failed(TranscriptFailure)

    /// Per-inference ID (unique to each emitted update).
    var id: UUID {
        switch self {
        case .partial(let p): return p.id
        case .finalized(let r): return r.id
        case .refined(let r): return r.id
        case .failed(let f): return f.id
        }
    }

    /// Groups partials + final + refinements for the same utterance onto
    /// one UI row.
    var utteranceID: UUID {
        switch self {
        case .partial(let p): return p.utteranceID
        case .finalized(let r): return r.utteranceID
        case .refined(let r): return r.utteranceID
        case .failed(let f): return f.utteranceID
        }
    }

    var utteranceDuration: Duration {
        switch self {
        case .partial(let p): return p.utteranceDuration
        case .finalized(let r): return r.utteranceDuration
        case .refined(let r): return r.utteranceDuration
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
    /// Set by `TranscriptionPipeline` when this utterance should begin a
    /// new visual paragraph (long silence before its start, an immediately
    /// preceding stop+restart, or session beginning after a clear). The
    /// transcriber leaves it at the default; pipeline mutates the value
    /// before forwarding to consumers. Pure rendering hint — no effect on
    /// upsert keys, refinement, or retry behavior.
    var startsNewParagraph: Bool = false

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
    /// See `TranscriptResult.startsNewParagraph` — same semantics; tracked
    /// from the partial so the UI can render the running partial in its
    /// final paragraph slot before `.finalized` lands.
    var startsNewParagraph: Bool = false
}

nonisolated struct TranscriptFailure: Sendable, Identifiable, Equatable {
    let id: UUID
    let utteranceID: UUID
    let utteranceDuration: Duration
    let message: String
}

/// Server-polished replacement for a finalized transcript. Emitted when
/// the punctuation service returns a 2xx within the client timeout; the
/// UI replaces the finalized row's text with `text` in place (keyed by
/// `segmentID`). On server failure nothing is emitted — the raw finalized
/// row stays on screen (silent fallback).
///
/// Two IDs: `utteranceID` is provenance (which Moonshine utterance this
/// text originated from — used for retries + debugging). `segmentID` is
/// the UI's upsert key — the unit of rendering. Today they're equal
/// because one utterance produces exactly one refined segment. When the
/// server-side V2 pipeline starts splitting an utterance on spoken "new
/// paragraph" commands, `segmentID` will diverge and the UI will render
/// multiple paragraphs per utterance without any additional client work.
nonisolated struct TranscriptRefinement: Sendable, Identifiable, Equatable {
    let id: UUID
    let utteranceID: UUID
    let segmentID: UUID                // == utteranceID until server splits
    let text: String                   // punctuated + cased (+ ITN later)
    let sourceText: String             // raw local text we sent
    let utteranceDuration: Duration    // carried over from the finalized result
    let roundTripDuration: Duration    // network wall time, for telemetry
}
