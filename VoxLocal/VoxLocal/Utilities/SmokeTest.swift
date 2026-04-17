
import Foundation
import OnnxRuntimeBindings

nonisolated enum SmokeTest {
    static func runSileroVADLoadTest() {
        do {
            guard let modelURL = Bundle.main.url(forResource: "silero_vad", withExtension: "onnx") else {
                print("[SmokeTest] silero_vad.onnx not found in app bundle")
                return
            }
            print("[SmokeTest] Model URL: \(modelURL.path)")

            let options = try OnnxRuntimeSetup.makeSessionOptions()
            let session = try ORTSession(
                env: OnnxRuntimeSetup.shared,
                modelPath: modelURL.path,
                sessionOptions: options
            )

            let inputNames = try session.inputNames()
            let outputNames = try session.outputNames()

            print("[SmokeTest] Session created successfully")
            print("[SmokeTest] Inputs:  \(inputNames)")
            print("[SmokeTest] Outputs: \(outputNames)")
        } catch {
            print("[SmokeTest] Failed: \(error)")
        }
    }

    /// Feed the VAD model a few known synthetic signals and print the probability it returns.
    /// If probability varies meaningfully between silence and tone, the model itself is healthy
    /// and any VAD misbehavior is in our audio pipeline. If it stays ~0 on all, the model/export
    /// is suspect.
    static func runSileroVADInferenceSanity() {
        do {
            let model = try SileroVADModel(useCoreML: false)

            func feed(label: String, samples: [Float]) throws {
                precondition(samples.count == 512)
                let prob = try samples.withUnsafeBufferPointer { buf in
                    try model.process(samples: buf.baseAddress!)
                }
                let padded = label.padding(toLength: 14, withPad: " ", startingAt: 0)
                let probStr = String(format: "%.4f", prob)
                print("[SanityVAD] \(padded) prob=\(probStr)")
            }

            // 1) pure silence
            model.reset()
            for _ in 0..<4 {
                try feed(label: "silence", samples: Array(repeating: 0, count: 512))
            }

            // 2) 440 Hz sine @ 0.3 amplitude (speech-level loudness)
            model.reset()
            let sr: Float = 16_000
            let sine = (0..<512).map { i in
                0.3 * sin(2 * .pi * 440 * Float(i) / sr)
            }
            for _ in 0..<4 {
                try feed(label: "sine440@0.3", samples: sine)
            }

            // 3) white noise @ 0.3 amplitude (VAD typically sees this as "speechy")
            model.reset()
            var rng = SystemRandomNumberGenerator()
            let noise = (0..<512).map { _ -> Float in
                let u = Float(UInt32.random(in: 0...UInt32.max, using: &rng)) / Float(UInt32.max)
                return (u - 0.5) * 0.6
            }
            for _ in 0..<4 {
                try feed(label: "noise@0.3", samples: noise)
            }

            // 4) scaled-up version in case the model expects int16-scaled floats
            model.reset()
            let loudSine = sine.map { $0 * 10_000 }
            for _ in 0..<4 {
                try feed(label: "sine*10k", samples: loudSine)
            }
        } catch {
            print("[SanityVAD] Failed: \(error)")
        }
    }
}
