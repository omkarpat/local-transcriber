import AVFoundation
import Synchronization
import os

enum AudioCaptureError: Error {
    case permissionDenied
    case converterUnavailable(from: AVAudioFormat, to: AVAudioFormat)
    case engineStartFailed(underlying: Error)
    case sessionConfigurationFailed(underlying: Error)
}

/// Shared state between the main actor and the real-time audio tap thread.
/// The tap closure must never touch the main actor, so this isolates everything
/// the tap needs behind a `nonisolated`, `Sendable` reference.
final class AudioTapState: @unchecked Sendable {
    let ringBuffer: CircularAudioBuffer
    let converter: AVAudioConverter
    let targetFormat: AVAudioFormat
    // 16kHz → ~200ms of headroom per hop; scaled up in init to cover any realistic input buffer.
    private let outputCapacity: AVAudioFrameCount

    // Packed: bit pattern of a Float RMS estimate (0...1).
    private let rmsBits = Atomic<UInt32>(0)
    private let totalFrames = Atomic<Int>(0)

    init(ringBuffer: CircularAudioBuffer,
         converter: AVAudioConverter,
         targetFormat: AVAudioFormat,
         outputCapacity: AVAudioFrameCount) {
        self.ringBuffer = ringBuffer
        self.converter = converter
        self.targetFormat = targetFormat
        self.outputCapacity = outputCapacity
    }

    var currentRMS: Float {
        Float(bitPattern: rmsBits.load(ordering: .relaxed))
    }

    var framesCaptured: Int {
        totalFrames.load(ordering: .relaxed)
    }

    /// Called on the real-time audio thread.
    func handleInput(_ input: AVAudioPCMBuffer) {
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
            return
        }

        var error: NSError?
        var consumed = false
        let status = converter.convert(to: output, error: &error) { _, statusOut in
            if consumed {
                statusOut.pointee = .noDataNow
                return nil
            }
            consumed = true
            statusOut.pointee = .haveData
            return input
        }

        switch status {
        case .haveData, .inputRanDry:
            break
        case .error, .endOfStream:
            return
        @unknown default:
            return
        }

        let frames = Int(output.frameLength)
        guard frames > 0, let samples = output.floatChannelData?.pointee else { return }

        var sumSquares: Float = 0
        for i in 0..<frames {
            let s = samples[i]
            sumSquares += s * s
        }
        let rms = (frames > 0) ? (sumSquares / Float(frames)).squareRoot() : 0
        rmsBits.store(rms.bitPattern, ordering: .relaxed)

        ringBuffer.write(samples, count: frames)
        totalFrames.wrappingAdd(frames, ordering: .relaxed)
    }
}

final class AudioCaptureManager {
    let ringBuffer: CircularAudioBuffer

    private let engine = AVAudioEngine()
    private let session = AVAudioSession.sharedInstance()
    private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "AudioCapture")

    private var tapState: AudioTapState?
    private(set) var isRunning = false

    /// Ring buffer holds ~10s of 16kHz mono Float32 by default (160 000 samples → 2^18 rounded).
    init(ringBufferCapacity: Int = 160_000) {
        self.ringBuffer = CircularAudioBuffer(capacity: ringBufferCapacity)
    }

    /// Current RMS of the most recent converted buffer; 0 when not running.
    var currentRMS: Float { tapState?.currentRMS ?? 0 }

    /// Total 16kHz frames captured since `start()`; 0 when not running.
    var framesCaptured: Int { tapState?.framesCaptured ?? 0 }

    func start() async throws {
        guard !isRunning else { return }

        let permission = await MicrophonePermission.request()
        guard permission == .granted else { throw AudioCaptureError.permissionDenied }

        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setPreferredSampleRate(AudioFormat.sampleRate)
            try session.setActive(true, options: [])
        } catch {
            throw AudioCaptureError.sessionConfigurationFailed(underlying: error)
        }

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        let targetFormat = AudioFormat.makeAVAudioFormat()

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw AudioCaptureError.converterUnavailable(from: inputFormat, to: targetFormat)
        }

        // Sized for the worst case: a 100ms input tap @ 48kHz stereo = 4800 frames.
        // After conversion to 16kHz that's 1600 frames; headroom for jitter.
        let outputCapacity: AVAudioFrameCount = 4096
        let state = AudioTapState(
            ringBuffer: ringBuffer,
            converter: converter,
            targetFormat: targetFormat,
            outputCapacity: outputCapacity
        )
        self.tapState = state

        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { buffer, _ in
            state.handleInput(buffer)
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            tapState = nil
            throw AudioCaptureError.engineStartFailed(underlying: error)
        }

        isRunning = true
        log.info("AudioCaptureManager started; inputFormat=\(inputFormat), target=\(targetFormat)")
    }

    func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? session.setActive(false, options: [.notifyOthersOnDeactivation])
        tapState = nil
        isRunning = false
        log.info("AudioCaptureManager stopped")
    }
}
