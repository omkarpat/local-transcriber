import Foundation
import OnnxRuntimeBindings

enum SileroVADError: Error {
    case modelNotFound
    case unexpectedOutputShape
    case unexpectedStateShape
}

/// Silero VAD v5 ONNX wrapper.
///
/// Contract:
/// - Input  `input`  Float32[1, 512]
/// - Input  `state`  Float32[2, 1, 128]  (recurrent; persist across calls)
/// - Input  `sr`     Int64 scalar (16000)
/// - Output `output` Float32[1, 1]       (speech probability in [0, 1])
/// - Output `stateN` Float32[2, 1, 128]  (feed back as `state` next call)
///
/// Not thread-safe. Intended for use on a single consumer task.
final class SileroVADModel: @unchecked Sendable {
    static let frameSamples = 512
    private static let contextSamples = 64
    private static let totalInputSamples = frameSamples + contextSamples
    private static let stateCount = 2 * 1 * 128
    private static let stateByteCount = stateCount * MemoryLayout<Float>.size
    private static let contextByteCount = contextSamples * MemoryLayout<Float>.size

    private let session: ORTSession
    private let inputShape: [NSNumber] = [1, NSNumber(value: totalInputSamples)]
    private let stateShape: [NSNumber] = [2, 1, 128]

    private let stateData: NSMutableData
    private let contextData: NSMutableData
    private let sampleRateData: NSMutableData

    init(useCoreML: Bool = true) throws {
        guard let modelURL = Bundle.main.url(forResource: "silero_vad", withExtension: "onnx") else {
            throw SileroVADError.modelNotFound
        }
        let options = try (useCoreML
            ? OnnxRuntimeSetup.makeSessionOptions()
            : OnnxRuntimeSetup.makeCPUSessionOptions())
        self.session = try ORTSession(
            env: OnnxRuntimeSetup.shared,
            modelPath: modelURL.path,
            sessionOptions: options
        )

        self.stateData = NSMutableData(length: Self.stateByteCount)!
        self.contextData = NSMutableData(length: Self.contextByteCount)!
        var sr: Int64 = Int64(AudioFormat.sampleRate)
        self.sampleRateData = NSMutableData(bytes: &sr, length: MemoryLayout<Int64>.size)
    }

    /// Zero the recurrent state and context buffer. Call between unrelated sessions.
    func reset() {
        memset(stateData.mutableBytes, 0, Self.stateByteCount)
        memset(contextData.mutableBytes, 0, Self.contextByteCount)
    }

    /// Run one frame through the model. `samples` must point to exactly 512 Float32 values.
    /// Silero v5 expects [64 context + 512 new samples] = 576 total inputs.
    func process(samples: UnsafePointer<Float>) throws -> Float {
        let inputBytes = Self.totalInputSamples * MemoryLayout<Float>.size
        let inputData = NSMutableData(length: inputBytes)!
        let inputPtr = inputData.mutableBytes.assumingMemoryBound(to: Float.self)
        memcpy(inputPtr, contextData.bytes, Self.contextByteCount)
        memcpy(inputPtr.advanced(by: Self.contextSamples), samples, Self.frameSamples * MemoryLayout<Float>.size)

        let inputTensor = try ORTValue(
            tensorData: inputData,
            elementType: .float,
            shape: inputShape
        )
        let stateTensor = try ORTValue(
            tensorData: stateData,
            elementType: .float,
            shape: stateShape
        )
        let srTensor = try ORTValue(
            tensorData: sampleRateData,
            elementType: .int64,
            shape: [1]
        )

        let outputs = try session.run(
            withInputs: ["input": inputTensor, "state": stateTensor, "sr": srTensor],
            outputNames: ["output", "stateN"],
            runOptions: nil
        )

        guard let outputTensor = outputs["output"], let stateNTensor = outputs["stateN"] else {
            throw SileroVADError.unexpectedOutputShape
        }

        let outputData = try outputTensor.tensorData() as Data
        guard outputData.count >= MemoryLayout<Float>.size else {
            throw SileroVADError.unexpectedOutputShape
        }
        let probability: Float = outputData.withUnsafeBytes { $0.load(as: Float.self) }

        let stateNData = try stateNTensor.tensorData() as Data
        guard stateNData.count == Self.stateByteCount else {
            throw SileroVADError.unexpectedStateShape
        }
        stateNData.withUnsafeBytes { raw in
            memcpy(stateData.mutableBytes, raw.baseAddress!, Self.stateByteCount)
        }

        // Save last 64 samples of this frame as the next call's context.
        let contextSrc = samples.advanced(by: Self.frameSamples - Self.contextSamples)
        memcpy(contextData.mutableBytes, contextSrc, Self.contextByteCount)

        return probability
    }
}
