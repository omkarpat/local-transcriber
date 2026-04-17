import Foundation

/// Lifecycle + non-utterance signals from `TranscriptionPipeline`. Kept
/// on a separate `AsyncStream` from `TranscriptUpdate` so utterance
/// content and pipeline state stay cleanly separated — a UI can subscribe
/// to just the stream it needs.
enum PipelineEvent: Sendable, Equatable {
    /// The pipeline transitioned from idle to running.
    case started

    /// The pipeline transitioned from running to idle (by explicit stop,
    /// cancellation, or an error — see `.failed` for the error case).
    case stopped

    /// A non-recoverable failure in the audio/VAD stack. The pipeline is
    /// already transitioning to idle when this fires; callers should
    /// surface the message and reset UI state.
    case failed(String)

    /// The user just started/stopped speaking per the VAD state machine.
    /// Useful for a "listening" pulse, mic outline color change, etc.
    /// `isActive == true` means we're inside an utterance (possibly still
    /// below the min-speech threshold); `false` means we've returned to
    /// idle.
    case speechStateChanged(isActive: Bool)
}
