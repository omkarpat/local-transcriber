import Foundation

/// SwiftUI-shaped projection of `TranscriptUpdate`. One row per rendered
/// segment, keyed on `segmentID` (`ForEach` upserts by `id`). For a
/// plain single-paragraph utterance `segmentID == utteranceID`; when the
/// server's V2 pipeline splits an utterance on a `new paragraph`
/// command, extra sibling rows share the parent's `utteranceID` but get
/// their own server-minted `segmentID`.
struct TranscriptRow: Identifiable, Equatable {
    enum Status: Equatable {
        case partial
        case final(tokenCount: Int, realTimeFactor: Double, isRefined: Bool)
        case failed(message: String)
    }

    let utteranceID: UUID
    let segmentID: UUID
    var id: UUID { segmentID }
    var status: Status
    var text: String
    var utteranceDuration: Duration
    var inferenceDuration: Duration
}

extension Array where Element == TranscriptRow {
    /// Upsert rules (shared between transcription view + debug view):
    /// - `.partial`: create a row at index 0 with `segmentID = utteranceID`,
    ///   or replace an existing `.partial` in place. Late partials
    ///   arriving after `.final`/`.failed` are ignored — final wins.
    /// - `.finalized`: replace the single-segment row in place (match
    ///   by `utteranceID`), or insert at top. Clears any `isRefined`
    ///   flag until a later `.refined` arrives.
    /// - `.refined`: match by `segmentID` first — if the server kept
    ///   the incoming id (paragraph 0 of a split, or the single-segment
    ///   case), overwrite in place. If no match, the server minted a
    ///   new id for a split paragraph: find the first existing row
    ///   sharing the `utteranceID` and insert the new segment right
    ///   ahead of it in the newest-first array so display (reversed)
    ///   places subsequent paragraphs after earlier ones.
    /// - `.failed`: replace in place (match by `utteranceID`), or
    ///   insert at top.
    /// Caller is responsible for the cap (`removeLast(count - maxRows)`).
    mutating func upsert(_ update: TranscriptUpdate) {
        let utteranceID = update.utteranceID

        switch update {
        case .partial(let p):
            if let idx = firstIndex(where: { $0.utteranceID == utteranceID }) {
                if case .partial = self[idx].status {
                    self[idx].text = p.text
                    self[idx].utteranceDuration = p.utteranceDuration
                    self[idx].inferenceDuration = p.inferenceDuration
                }
            } else {
                insert(TranscriptRow(
                    utteranceID: utteranceID,
                    segmentID: utteranceID,
                    status: .partial,
                    text: p.text,
                    utteranceDuration: p.utteranceDuration,
                    inferenceDuration: p.inferenceDuration
                ), at: 0)
            }

        case .finalized(let r):
            let row = TranscriptRow(
                utteranceID: utteranceID,
                segmentID: utteranceID,
                status: .final(tokenCount: r.tokenCount, realTimeFactor: r.realTimeFactor, isRefined: false),
                text: r.text,
                utteranceDuration: r.utteranceDuration,
                inferenceDuration: r.inferenceDuration
            )
            if let idx = firstIndex(where: { $0.utteranceID == utteranceID }) {
                self[idx] = row
            } else {
                insert(row, at: 0)
            }

        case .refined(let r):
            if let idx = firstIndex(where: { $0.segmentID == r.segmentID }) {
                guard case .final(let tokenCount, let rtf, _) = self[idx].status else { return }
                self[idx].text = r.text
                self[idx].status = .final(tokenCount: tokenCount, realTimeFactor: rtf, isRefined: true)
                return
            }
            // No match by segmentID — this refinement carries a
            // server-minted id for an extra paragraph from a split
            // utterance. Slot it in ahead of the first existing row
            // that shares the `utteranceID` so it renders as a sibling
            // paragraph. If the parent row was already evicted by the
            // row cap, drop silently — there's nowhere coherent to put
            // an orphan.
            guard let parentIdx = firstIndex(where: { $0.utteranceID == r.utteranceID }) else { return }
            let parent = self[parentIdx]
            // `tokenCount`/`rtf` are raw-ASR metrics that apply to the
            // whole utterance. We zero them on sibling rows so the
            // debug view doesn't duplicate the parent's numbers against
            // a paragraph that wasn't independently transcribed.
            insert(TranscriptRow(
                utteranceID: r.utteranceID,
                segmentID: r.segmentID,
                status: .final(tokenCount: 0, realTimeFactor: 0, isRefined: true),
                text: r.text,
                utteranceDuration: parent.utteranceDuration,
                inferenceDuration: .zero
            ), at: parentIdx)

        case .failed(let f):
            let row = TranscriptRow(
                utteranceID: utteranceID,
                segmentID: utteranceID,
                status: .failed(message: f.message),
                text: "",
                utteranceDuration: f.utteranceDuration,
                inferenceDuration: .zero
            )
            if let idx = firstIndex(where: { $0.utteranceID == utteranceID }) {
                self[idx] = row
            } else {
                insert(row, at: 0)
            }
        }
    }
}
