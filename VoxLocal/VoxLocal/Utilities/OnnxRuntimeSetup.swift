import Foundation
import OnnxRuntimeBindings

nonisolated enum OnnxRuntimeSetup {
    static let shared: ORTEnv = {
        do {
            return try ORTEnv(loggingLevel: .warning)
        } catch {
            fatalError("ORTEnv init failed: \(error)")
        }
    }()

    static func makeSessionOptions() throws -> ORTSessionOptions {
        _ = shared
        let options = try ORTSessionOptions()
        let coreML = ORTCoreMLExecutionProviderOptions()
        coreML.enableOnSubgraphs = true
        try options.appendCoreMLExecutionProvider(with: coreML)
        return options
    }
}
