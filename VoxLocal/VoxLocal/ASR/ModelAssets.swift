import Foundation

/// A single remote file that makes up part of a model bundle.
/// `nonisolated` because the project default is `@MainActor` and this is
/// plain Sendable data that must be readable from the downloader's
/// detached task without hopping actors.
nonisolated struct RemoteFile: Sendable {
    let relativePath: String
    let sourceURL: URL
    let expectedBytes: Int64
}

/// A named collection of remote files that together form one ASR model.
/// Decoupling the Swift-side model identity from file layout lets us swap
/// Moonshine → Whisper later without touching inference code — a new
/// `ModelBundle` is the only thing required.
nonisolated struct ModelBundle: Sendable, Identifiable {
    let id: String
    let displayName: String
    let files: [RemoteFile]

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.expectedBytes } }
}

nonisolated enum ModelAssets {
    /// Moonshine Tiny (float variant) sourced from onnx-community/moonshine-tiny-ONNX
    /// on Hugging Face. Byte sizes captured 2026-04-16; bump if the HF repo
    /// updates. See `phase1-plan.md` Task 4.1 for the full I/O spec.
    ///
    /// We intentionally use the **split** decoder pair (decoder_model +
    /// decoder_with_past_model) rather than `decoder_model_merged.onnx`.
    /// The merged variant requires a bool `use_cache_branch` input, but the
    /// Objective-C ONNX Runtime binding used by this package does not
    /// expose a BOOL tensor element type. Using the split pair is ~66 MB
    /// larger on disk but keeps the inference path pure Swift. See
    /// phase1-plan.md Task 4.2a notes.
    static let moonshineTiny: ModelBundle = {
        let base = "https://huggingface.co/onnx-community/moonshine-tiny-ONNX/resolve/main"
        return ModelBundle(
            id: "moonshine-tiny",
            displayName: "Moonshine Tiny",
            files: [
                RemoteFile(relativePath: "encoder_model.onnx",
                           sourceURL: URL(string: "\(base)/onnx/encoder_model.onnx")!,
                           expectedBytes: 30_882_331),
                RemoteFile(relativePath: "decoder_model.onnx",
                           sourceURL: URL(string: "\(base)/onnx/decoder_model.onnx")!,
                           expectedBytes: 77_906_761),
                RemoteFile(relativePath: "decoder_with_past_model.onnx",
                           sourceURL: URL(string: "\(base)/onnx/decoder_with_past_model.onnx")!,
                           expectedBytes: 73_867_894),
                RemoteFile(relativePath: "tokenizer.json",
                           sourceURL: URL(string: "\(base)/tokenizer.json")!,
                           expectedBytes: 3_761_754),
                RemoteFile(relativePath: "tokenizer_config.json",
                           sourceURL: URL(string: "\(base)/tokenizer_config.json")!,
                           expectedBytes: 135_735),
                RemoteFile(relativePath: "config.json",
                           sourceURL: URL(string: "\(base)/config.json")!,
                           expectedBytes: 921),
                RemoteFile(relativePath: "generation_config.json",
                           sourceURL: URL(string: "\(base)/generation_config.json")!,
                           expectedBytes: 147),
                RemoteFile(relativePath: "preprocessor_config.json",
                           sourceURL: URL(string: "\(base)/preprocessor_config.json")!,
                           expectedBytes: 128),
                RemoteFile(relativePath: "special_tokens_map.json",
                           sourceURL: URL(string: "\(base)/special_tokens_map.json")!,
                           expectedBytes: 3),
            ]
        )
    }()

    /// Returns (creating if needed) `Library/Application Support/Models/<bundle.id>/`.
    /// Marked `nonisolated` so it is callable from any actor; pure filesystem work.
    nonisolated static func installDirectory(for bundle: ModelBundle) throws -> URL {
        let fm = FileManager.default
        let appSupport = try fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        var modelsDir = appSupport.appendingPathComponent("Models", isDirectory: true)
        try fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        // Tell iCloud to skip backups for this entire subtree. Set once on the
        // parent; children inherit. Must go on a var so we can mutate the
        // URLResourceValues bag.
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? modelsDir.setResourceValues(values)

        let bundleDir = modelsDir.appendingPathComponent(bundle.id, isDirectory: true)
        try fm.createDirectory(at: bundleDir, withIntermediateDirectories: true)
        return bundleDir
    }

    /// Resolves the on-disk URL for a single file within a bundle.
    nonisolated static func fileURL(in bundle: ModelBundle, named relativePath: String) throws -> URL {
        try installDirectory(for: bundle).appendingPathComponent(relativePath)
    }

    /// True iff every file in the bundle exists on disk at the expected size.
    /// Exact-size match is the poor-man's integrity check until we add SHA256.
    nonisolated static func isInstalled(_ bundle: ModelBundle) -> Bool {
        guard let dir = try? installDirectory(for: bundle) else { return false }
        let fm = FileManager.default
        for file in bundle.files {
            let url = dir.appendingPathComponent(file.relativePath)
            guard fm.fileExists(atPath: url.path),
                  let attrs = try? fm.attributesOfItem(atPath: url.path),
                  let size = attrs[.size] as? Int64,
                  size == file.expectedBytes
            else { return false }
        }
        return true
    }

    /// Blow away the bundle's install directory. Useful for dev (force
    /// redownload on next launch) and eventually for "uninstall model".
    nonisolated static func remove(_ bundle: ModelBundle) throws {
        let dir = try installDirectory(for: bundle)
        try FileManager.default.removeItem(at: dir)
    }
}
