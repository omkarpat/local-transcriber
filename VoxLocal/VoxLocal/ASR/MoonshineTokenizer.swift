import Foundation
import Tokenizers

enum MoonshineTokenizerError: Error, CustomStringConvertible {
    case loadFailed(String)

    var description: String {
        switch self {
        case .loadFailed(let msg): return "Moonshine tokenizer load failed: \(msg)"
        }
    }
}

/// Wraps a `Tokenizers.Tokenizer` loaded from the files inside a model
/// bundle directory (must contain `tokenizer.json` + `tokenizer_config.json`).
/// `swift-transformers`' `AutoTokenizer.from(modelFolder:)` reads those
/// JSONs and hands back a concrete tokenizer keyed on the config's
/// tokenizer class — for Moonshine that's a BPE tokenizer.
///
/// `nonisolated` because tokenizer loading is async/throws and we don't
/// want to pin it to the main actor.
nonisolated final class MoonshineTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    static func load(bundle: ModelBundle = ModelAssets.moonshineTiny) async throws -> MoonshineTokenizer {
        let folder: URL
        do {
            folder = try ModelAssets.installDirectory(for: bundle)
        } catch {
            throw MoonshineTokenizerError.loadFailed("could not resolve install directory: \(error)")
        }
        do {
            let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
            return MoonshineTokenizer(tokenizer: tokenizer)
        } catch {
            throw MoonshineTokenizerError.loadFailed(error.localizedDescription)
        }
    }

    private init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Decode model-generated token IDs into a human-readable string.
    /// Skips special tokens (BOS/EOS/pad) by default so the output is
    /// just the transcript the user cares about.
    func decode(tokenIDs: [Int64], skipSpecialTokens: Bool = true) -> String {
        let ints = tokenIDs.map { Int($0) }
        return tokenizer.decode(tokens: ints, skipSpecialTokens: skipSpecialTokens)
    }
}
