import Foundation

/// Ordered accumulator of finalized segments for one recording session.
/// Value type — each `apply(...)` returns a new session so SwiftUI views
/// can diff cleanly. Partials and failures aren't stored; we only keep
/// what a user would reasonably want to copy/export.
struct TranscriptionSession: Sendable, Equatable {
    let startedAt: Date
    private(set) var segments: [Segment] = []

    /// One committed utterance. `text` is the punctuated version if the
    /// cloud service answered in time; otherwise the raw local transcript.
    struct Segment: Sendable, Equatable, Identifiable {
        let id: UUID                        // == utteranceID
        var text: String
        let utteranceDuration: Duration
        let capturedAt: Date
        var isPunctuated: Bool
    }

    init(startedAt: Date = Date()) {
        self.startedAt = startedAt
    }

    var fullTranscript: String {
        segments.map(\.text).joined(separator: " ")
    }

    var totalDuration: Duration {
        segments.reduce(.zero) { $0 + $1.utteranceDuration }
    }

    /// Append a finalized segment (first time we hear about it) or replace
    /// its text when a `.punctuated` update arrives. `.partial` and
    /// `.failed` are ignored here — session is the "clean record" view.
    mutating func apply(_ update: TranscriptUpdate) {
        switch update {
        case .partial, .failed:
            return
        case .finalized(let r):
            if let idx = segments.firstIndex(where: { $0.id == r.utteranceID }) {
                segments[idx].text = r.text
                segments[idx].isPunctuated = false
            } else {
                segments.append(Segment(
                    id: r.utteranceID,
                    text: r.text,
                    utteranceDuration: r.utteranceDuration,
                    capturedAt: Date(),
                    isPunctuated: false
                ))
            }
        case .punctuated(let p):
            guard let idx = segments.firstIndex(where: { $0.id == p.utteranceID }) else { return }
            segments[idx].text = p.text
            segments[idx].isPunctuated = true
        }
    }
}
