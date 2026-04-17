
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

    /// Run Moonshine Tiny against the most recently captured utterance in
    /// Documents/. Loads tokenizer + model, runs greedy decode, prints the
    /// decoded transcript plus timing info. Expects the user to have
    /// recorded at least one utterance via the Audio Capture debug screen.
    static func runMoonshineInference() async {
        do {
            guard let wavURL = WAVReader.mostRecentUtterance() else {
                print("[MoonshineSmoke] No utterance-*.wav found in Documents. Record one via Audio Capture first.")
                return
            }
            print("[MoonshineSmoke] WAV: \(wavURL.lastPathComponent)")

            let samples = try WAVReader.readFloat32Samples(from: wavURL)
            let durationMS = Int(Double(samples.count) * 1000.0 / AudioFormat.sampleRate)
            print("[MoonshineSmoke] samples=\(samples.count)  duration=\(durationMS)ms")

            let loadStart = Date()
            let model = try MoonshineModel()
            let tokenizer = try await MoonshineTokenizer.load()
            print(String(format: "[MoonshineSmoke] loaded model + tokenizer  %.2fs", Date().timeIntervalSince(loadStart)))

            let inferStart = Date()
            let result = try model.transcribe(samples: samples)
            let inferTime = Date().timeIntervalSince(inferStart)
            let rtf = inferTime / (Double(durationMS) / 1000.0)

            print("[MoonshineSmoke] generated \(result.tokens.count) tokens in \(String(format: "%.2fs", inferTime)) (RTF=\(String(format: "%.2f", rtf)))")
            print("[MoonshineSmoke] token IDs: \(result.tokens)")
            let text = tokenizer.decode(tokenIDs: result.tokens)
            print("[MoonshineSmoke] transcript: \"\(text)\"")
        } catch {
            print("[MoonshineSmoke] Failed: \(error)")
        }
    }

    /// Benchmark Moonshine Tiny against the most recent utterance. Runs
    /// `iterations` transcribe passes with CoreML EP on, then with CoreML
    /// EP off (CPU only), prints per-phase min / median / p95 stats to the
    /// console. One warmup pass is discarded per configuration.
    static func runMoonshineBenchmark(iterations: Int = 5) async {
        guard let wavURL = WAVReader.mostRecentUtterance() else {
            print("[MoonshineBench] No utterance-*.wav found in Documents. Record one via Audio Capture first.")
            return
        }
        do {
            let samples = try WAVReader.readFloat32Samples(from: wavURL)
            let durationSeconds = Double(samples.count) / AudioFormat.sampleRate
            print("[MoonshineBench] WAV: \(wavURL.lastPathComponent)  duration=\(String(format: "%.2fs", durationSeconds))  iters=\(iterations)")

            try await bench(label: "CoreML EP", samples: samples, durationSeconds: durationSeconds, iterations: iterations, useCoreML: true)
            try await bench(label: "CPU only ", samples: samples, durationSeconds: durationSeconds, iterations: iterations, useCoreML: false)
        } catch {
            print("[MoonshineBench] Failed: \(error)")
        }
    }

    private static func bench(label: String, samples: [Float], durationSeconds: Double, iterations: Int, useCoreML: Bool) async throws {
        let loadStart = Date()
        let model = try MoonshineModel(useCoreML: useCoreML)
        let loadTime = Date().timeIntervalSince(loadStart)
        // Warmup — first pass always pays graph-compile costs that skew stats.
        _ = try model.transcribe(samples: samples)

        var totals: [Double] = []
        var encoders: [Double] = []
        var firstDecodes: [Double] = []
        var cacheDecodes: [Double] = []
        var tokenCounts: [Int] = []
        for _ in 0..<iterations {
            let r = try model.transcribe(samples: samples)
            totals.append(seconds(r.timings.total))
            encoders.append(seconds(r.timings.encoder))
            firstDecodes.append(seconds(r.timings.firstDecodeCall))
            cacheDecodes.append(seconds(r.timings.cacheDecodeCalls))
            tokenCounts.append(r.tokens.count)
        }
        let tokCount = tokenCounts.first ?? 0
        print(String(format: "[MoonshineBench] %@  load=%.2fs  tokens=%d  (N=%d)", label, loadTime, tokCount, iterations))
        printStats(label: "  total      ", values: totals, audioSec: durationSeconds)
        printStats(label: "  encoder    ", values: encoders, audioSec: durationSeconds)
        printStats(label: "  1st-decode ", values: firstDecodes, audioSec: nil)
        printStats(label: "  cache-loop ", values: cacheDecodes, audioSec: nil)
    }

    private static func printStats(label: String, values: [Double], audioSec: Double?) {
        let sorted = values.sorted()
        let median = sorted[sorted.count / 2]
        let p95 = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
        let minV = sorted.first ?? 0
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        var line = String(format: "%@ min=%.3fs  median=%.3fs  mean=%.3fs  p95=%.3fs", label, minV, median, mean, p95)
        if let audioSec, audioSec > 0 {
            line += String(format: "  (RTF median=%.2f)", median / audioSec)
        }
        print(line)
    }

    private static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
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
