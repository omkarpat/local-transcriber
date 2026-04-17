import Foundation
import os

enum VADEvent: Sendable {
    case frame(probability: Float, isSpeech: Bool)
    case utteranceStarted
    case utteranceEnded(samples: [Float], duration: Duration)
    case error(String)
}

final class VADProcessor: @unchecked Sendable {
    let events: AsyncStream<VADEvent>

    private let ringBuffer: CircularAudioBuffer
    private let configuration: VADConfiguration
    private let continuation: AsyncStream<VADEvent>.Continuation
    private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "VAD")

    private var task: Task<Void, Never>?

    init(ringBuffer: CircularAudioBuffer, configuration: VADConfiguration = .default) {
        self.ringBuffer = ringBuffer
        self.configuration = configuration
        var cont: AsyncStream<VADEvent>.Continuation!
        self.events = AsyncStream(bufferingPolicy: .bufferingNewest(256)) { c in
            cont = c
        }
        self.continuation = cont
    }

    func start() {
        guard task == nil else { return }
        let ringBuffer = self.ringBuffer
        let configuration = self.configuration
        let continuation = self.continuation
        task = Task.detached(priority: .userInitiated) { [log] in
            do {
                let model = try SileroVADModel()
                try await Self.loop(
                    model: model,
                    ringBuffer: ringBuffer,
                    configuration: configuration,
                    continuation: continuation
                )
            } catch {
                log.error("VADProcessor exited with error: \(error)")
                continuation.yield(.error(String(describing: error)))
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    deinit {
        task?.cancel()
        continuation.finish()
    }

    // MARK: - Loop

    private enum Mode {
        case idle
        case pendingSpeech(frames: Int)
        case speaking
        case pendingSilence(silenceFrames: Int)
    }

    private static func loop(
        model: SileroVADModel,
        ringBuffer: CircularAudioBuffer,
        configuration: VADConfiguration,
        continuation: AsyncStream<VADEvent>.Continuation
    ) async throws {
        let frameSamples = VADConfiguration.frameSamples
        let frameBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: frameSamples)
        defer { frameBuffer.deallocate() }

        let minSpeechFrames = configuration.frames(in: configuration.minSpeechDuration)
        let endSilenceFrames = configuration.frames(in: configuration.endOfSpeechSilence)
        let preRollSamples = configuration.samples(in: configuration.preSpeechPadding)
        let maxUtteranceSamples = configuration.samples(in: configuration.maxUtteranceDuration)

        var mode: Mode = .idle
        var preRoll = PreRoll(capacity: preRollSamples)
        var utterance: [Float] = []
        utterance.reserveCapacity(maxUtteranceSamples)

        while !Task.isCancelled {
            if ringBuffer.availableToRead < frameSamples {
                try? await Task.sleep(for: .milliseconds(10))
                continue
            }
            let n = ringBuffer.read(into: frameBuffer.baseAddress!, maxCount: frameSamples)
            guard n == frameSamples else { continue }

            let probability = try model.process(samples: frameBuffer.baseAddress!)
            let isSpeech = probability >= configuration.speechThreshold
            continuation.yield(.frame(probability: probability, isSpeech: isSpeech))

            switch mode {
            case .idle:
                preRoll.append(frameBuffer)
                if isSpeech {
                    mode = .pendingSpeech(frames: 1)
                    utterance.removeAll(keepingCapacity: true)
                    preRoll.copyInto(&utterance)
                    utterance.append(contentsOf: UnsafeBufferPointer(start: frameBuffer.baseAddress, count: frameSamples))
                }

            case .pendingSpeech(let frames):
                utterance.append(contentsOf: UnsafeBufferPointer(start: frameBuffer.baseAddress, count: frameSamples))
                if isSpeech {
                    let next = frames + 1
                    if next >= minSpeechFrames {
                        mode = .speaking
                        continuation.yield(.utteranceStarted)
                    } else {
                        mode = .pendingSpeech(frames: next)
                    }
                } else {
                    // didn't confirm — discard and reset
                    mode = .idle
                    utterance.removeAll(keepingCapacity: true)
                    model.reset()
                    preRoll.append(frameBuffer)
                }

            case .speaking:
                utterance.append(contentsOf: UnsafeBufferPointer(start: frameBuffer.baseAddress, count: frameSamples))
                if !isSpeech {
                    mode = .pendingSilence(silenceFrames: 1)
                }
                if utterance.count >= maxUtteranceSamples {
                    emit(utterance: &utterance, continuation: continuation)
                    mode = .idle
                    model.reset()
                    preRoll.reset()
                }

            case .pendingSilence(let silenceFrames):
                utterance.append(contentsOf: UnsafeBufferPointer(start: frameBuffer.baseAddress, count: frameSamples))
                if isSpeech {
                    mode = .speaking
                } else {
                    let next = silenceFrames + 1
                    if next >= endSilenceFrames {
                        emit(utterance: &utterance, continuation: continuation)
                        mode = .idle
                        model.reset()
                        preRoll.reset()
                        preRoll.append(frameBuffer)
                    } else {
                        mode = .pendingSilence(silenceFrames: next)
                    }
                }
            }
        }
    }

    private static func emit(utterance: inout [Float], continuation: AsyncStream<VADEvent>.Continuation) {
        let samples = utterance
        let duration = Duration.seconds(Double(samples.count) / AudioFormat.sampleRate)
        continuation.yield(.utteranceEnded(samples: samples, duration: duration))
        utterance.removeAll(keepingCapacity: true)
    }
}

/// Tiny ring buffer holding the last N samples of pre-roll audio.
/// Used by VADProcessor only; not thread-safe.
private struct PreRoll {
    let capacity: Int
    private var storage: [Float]
    private var writeIndex = 0
    private var filled = 0

    init(capacity: Int) {
        self.capacity = capacity
        self.storage = Array(repeating: 0, count: capacity)
    }

    mutating func reset() {
        writeIndex = 0
        filled = 0
    }

    mutating func append(_ frame: UnsafeMutableBufferPointer<Float>) {
        guard capacity > 0 else { return }
        for i in 0..<frame.count {
            storage[writeIndex] = frame[i]
            writeIndex = (writeIndex + 1) % capacity
            if filled < capacity { filled += 1 }
        }
    }

    func copyInto(_ dst: inout [Float]) {
        guard filled > 0 else { return }
        if filled < capacity {
            dst.append(contentsOf: storage[0..<filled])
        } else {
            dst.append(contentsOf: storage[writeIndex..<capacity])
            dst.append(contentsOf: storage[0..<writeIndex])
        }
    }
}
