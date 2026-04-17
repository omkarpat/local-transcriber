import Foundation

/// Knobs the UI can set before calling `TranscriptionPipeline.start(...)`.
/// Kept as a `Sendable` value type so it crosses actor boundaries for free
/// and so SwiftUI `@Bindable` views can mutate copies locally.
struct PipelineConfiguration: Sendable, Equatable {
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

    static let `default` = PipelineConfiguration()
}
