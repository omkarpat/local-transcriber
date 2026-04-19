import Foundation

/// Knobs the UI can set before calling `TranscriptionPipeline.start(...)`.
/// Kept as a `Sendable` value type so it crosses actor boundaries for free
/// and so SwiftUI `@Bindable` views can mutate copies locally.
nonisolated struct PipelineConfiguration: Sendable, Equatable {
    /// Forwarded verbatim to the VAD. Covers thresholds, pre/post padding,
    /// partial-transcription cadence, and the max-utterance hard cap.
    var vad: VADConfiguration = .default

    /// When true, finalized local transcripts get POSTed to the cloud
    /// punctuation service and the polished reply is upserted into the
    /// transcript row. When false, the raw local text is final — useful
    /// for fully-offline mode or when testing without the server.
    var enableCloudPunctuation: Bool = true

    /// Developer-only: write every finalized utterance to
    /// `Documents/utterance-<ms>.wav`. Feeds the Moonshine smoke tests
    /// (which replay the latest utterance offline) and is otherwise
    /// disabled. Off by default.
    var debugDumpUtterances: Bool = false

    /// Silence between an utterance ending and the next utterance starting
    /// at or above this value forces the next utterance to begin a new
    /// paragraph. Below it, the next utterance flows into the running
    /// paragraph with a single space separator. The first utterance of a
    /// session and the first utterance after a stop+restart also start
    /// a new paragraph regardless of this threshold. Set to `.zero` to
    /// disable silence-based breaks (only `new paragraph` and stop will
    /// then break).
    var paragraphSilenceThreshold: Duration = .seconds(2)

    /// When true, the server interprets spoken commands ("comma" → ",",
    /// "new paragraph" → segment break, etc.) and the client respects
    /// server-emitted multi-segment splits. When false, spoken commands
    /// pass through as literal text and within-utterance splits don't
    /// happen. Off by default — a conservative posture so casual speech
    /// like "can you add a new paragraph" is captured verbatim. The
    /// transcription header exposes a toggle for users who want
    /// dictation behavior.
    var dictationMode: Bool = false

    static let `default` = PipelineConfiguration()
}
