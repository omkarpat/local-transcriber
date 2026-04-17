import AVFoundation

nonisolated enum AudioFormat {
    static let sampleRate: Double = 16_000
    static let channelCount: AVAudioChannelCount = 1
    static let commonFormat: AVAudioCommonFormat = .pcmFormatFloat32
    static let isInterleaved = false

    static func makeAVAudioFormat() -> AVAudioFormat {
        guard let format = AVAudioFormat(
            commonFormat: commonFormat,
            sampleRate: sampleRate,
            channels: channelCount,
            interleaved: isInterleaved
        ) else {
            fatalError("AudioFormat: failed to construct AVAudioFormat for 16kHz mono Float32")
        }
        return format
    }
}
