import Foundation

struct VADConfiguration: Sendable {
    /// Silero VAD output ≥ this value is treated as "speech present" in a frame.
    var speechThreshold: Float = 0.5

    /// Contiguous speech shorter than this is discarded (filters clicks/pops).
    var minSpeechDuration: Duration = .milliseconds(250)

    /// Silence longer than this after speech closes the utterance.
    var endOfSpeechSilence: Duration = .milliseconds(700)

    /// Audio kept before the triggering frame so we don't clip word onsets.
    var preSpeechPadding: Duration = .milliseconds(300)

    /// Hard cap on utterance length to prevent unbounded buffering.
    var maxUtteranceDuration: Duration = .seconds(30)

    static let `default` = VADConfiguration()
}

extension VADConfiguration {
    /// Frame size Silero VAD expects at 16kHz: 512 samples ≈ 32ms.
    static let frameSamples = 512

    var frameDuration: Duration { .milliseconds(32) }

    func frames(in duration: Duration) -> Int {
        let seconds = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        return Int((seconds * AudioFormat.sampleRate / Double(Self.frameSamples)).rounded(.up))
    }

    func samples(in duration: Duration) -> Int {
        let seconds = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        return Int((seconds * AudioFormat.sampleRate).rounded(.up))
    }
}
