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
}
