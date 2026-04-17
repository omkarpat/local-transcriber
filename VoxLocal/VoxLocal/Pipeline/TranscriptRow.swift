import Foundation

/// SwiftUI-shaped projection of `TranscriptUpdate`. One row per utterance,
/// keyed on `utteranceID` so `ForEach` can upsert in place as
/// `.partial` → `.finalized` → `.punctuated` arrive.
struct TranscriptRow: Identifiable, Equatable {
    enum Status: Equatable {
        case partial
        case final(tokenCount: Int, realTimeFactor: Double, isPunctuated: Bool)
        case failed(message: String)
    }

    let utteranceID: UUID
    var id: UUID { utteranceID }
    var status: Status
    var text: String
    var utteranceDuration: Duration
    var inferenceDuration: Duration
}

extension Array where Element == TranscriptRow {
    /// Upsert rules (shared between transcription view + debug view):
    /// - `.partial`: create a row at index 0, or replace an existing
    ///   `.partial` in place. Late partials arriving after `.final`/`.failed`
    ///   are ignored — final wins.
    /// - `.finalized`: replace in place, or insert at top. Clears any
    ///   `isPunctuated` flag until a later `.punctuated` arrives.
    /// - `.punctuated`: replace the row's text, only if the row is `.final`.
    ///   Marks the row as punctuated so the UI can optionally show a badge.
    /// - `.failed`: replace in place, or insert at top.
    /// Caller is responsible for the cap (`removeLast(count - maxRows)`).
    mutating func upsert(_ update: TranscriptUpdate) {
        let utteranceID = update.utteranceID
        let existingIndex = firstIndex(where: { $0.utteranceID == utteranceID })

        switch update {
        case .partial(let p):
            if let idx = existingIndex {
                if case .partial = self[idx].status {
                    self[idx].text = p.text
                    self[idx].utteranceDuration = p.utteranceDuration
                    self[idx].inferenceDuration = p.inferenceDuration
                }
            } else {
                insert(TranscriptRow(
                    utteranceID: utteranceID,
                    status: .partial,
                    text: p.text,
                    utteranceDuration: p.utteranceDuration,
                    inferenceDuration: p.inferenceDuration
                ), at: 0)
            }

        case .finalized(let r):
            let row = TranscriptRow(
                utteranceID: utteranceID,
                status: .final(tokenCount: r.tokenCount, realTimeFactor: r.realTimeFactor, isPunctuated: false),
                text: r.text,
                utteranceDuration: r.utteranceDuration,
                inferenceDuration: r.inferenceDuration
            )
            if let idx = existingIndex {
                self[idx] = row
            } else {
                insert(row, at: 0)
            }

        case .punctuated(let p):
            guard let idx = existingIndex else { return }
            if case .final(let tokenCount, let rtf, _) = self[idx].status {
                self[idx].text = p.text
                self[idx].status = .final(tokenCount: tokenCount, realTimeFactor: rtf, isPunctuated: true)
            }

        case .failed(let f):
            let row = TranscriptRow(
                utteranceID: utteranceID,
                status: .failed(message: f.message),
                text: "",
                utteranceDuration: f.utteranceDuration,
                inferenceDuration: .zero
            )
            if let idx = existingIndex {
                self[idx] = row
            } else {
                insert(row, at: 0)
            }
        }
    }
}
