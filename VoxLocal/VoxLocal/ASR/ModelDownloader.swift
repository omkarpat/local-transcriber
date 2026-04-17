import Foundation

/// Snapshot of where we are inside a multi-file model download.
/// Emitted once per file boundary (start + end) plus an overall terminal
/// `.allCompleted`. No byte-level progress inside a single file for now —
/// URLSession.download(for:) is a coarse one-shot API. Good enough for the
/// ~108 MB Moonshine bundle; can be swapped for delegate-based streaming
/// progress later if the 78 MB decoder feels too sticky in the UI.
nonisolated struct DownloadProgress: Sendable, Equatable {
    let bundleID: String
    let fileIndex: Int          // 0-based index of currentFileName within the bundle
    let fileCount: Int
    let currentFileName: String
    let bytesCompleted: Int64   // cumulative across all files finished so far
    let bytesTotal: Int64
    let phase: Phase

    enum Phase: Sendable, Equatable {
        case fileStarted
        case fileCompleted
        case allCompleted
    }

    var fractionCompleted: Double {
        guard bytesTotal > 0 else { return 0 }
        return Double(bytesCompleted) / Double(bytesTotal)
    }
}

nonisolated enum ModelDownloaderError: Error, LocalizedError {
    case httpStatus(file: String, status: Int)
    case sizeMismatch(file: String, expected: Int64, actual: Int64)

    var errorDescription: String? {
        switch self {
        case .httpStatus(let file, let status):
            return "Download of \(file) failed with HTTP \(status)."
        case .sizeMismatch(let file, let expected, let actual):
            return "\(file): expected \(expected) bytes, got \(actual)."
        }
    }
}

/// Downloads a `ModelBundle` into its `ModelAssets.installDirectory`, one
/// file at a time, atomically (tempfile → rename). Emits progress via an
/// `AsyncThrowingStream` so callers on the main actor can drive UI updates.
/// `nonisolated` so URLSession work never pins the main thread.
nonisolated final class ModelDownloader: Sendable {
    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func download(_ bundle: ModelBundle) -> AsyncThrowingStream<DownloadProgress, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached { [session] in
                do {
                    try await Self.perform(
                        bundle: bundle,
                        session: session,
                        yield: { continuation.yield($0) }
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func perform(
        bundle: ModelBundle,
        session: URLSession,
        yield: @Sendable (DownloadProgress) -> Void
    ) async throws {
        let installDir = try ModelAssets.installDirectory(for: bundle)
        var bytesCompleted: Int64 = 0
        let bytesTotal = bundle.totalBytes
        let bundleStart = Date()
        print("[ModelDownloader] start bundle=\(bundle.id) files=\(bundle.files.count) total=\(formatBytes(bytesTotal))")
        print("[ModelDownloader] installDir=\(installDir.path)")

        for (index, file) in bundle.files.enumerated() {
            try Task.checkCancellation()

            yield(DownloadProgress(
                bundleID: bundle.id,
                fileIndex: index,
                fileCount: bundle.files.count,
                currentFileName: file.relativePath,
                bytesCompleted: bytesCompleted,
                bytesTotal: bytesTotal,
                phase: .fileStarted
            ))

            let destination = installDir.appendingPathComponent(file.relativePath)

            // Skip-if-already-present so partial installs (e.g. interrupted run,
            // then retry) don't redownload the JSONs that finished last time.
            let fm = FileManager.default
            if fm.fileExists(atPath: destination.path),
               let attrs = try? fm.attributesOfItem(atPath: destination.path),
               let size = attrs[.size] as? Int64,
               size == file.expectedBytes {
                print("[ModelDownloader] skip \(file.relativePath) (already present, \(formatBytes(size)))")
                bytesCompleted += size
                yield(DownloadProgress(
                    bundleID: bundle.id,
                    fileIndex: index,
                    fileCount: bundle.files.count,
                    currentFileName: file.relativePath,
                    bytesCompleted: bytesCompleted,
                    bytesTotal: bytesTotal,
                    phase: .fileCompleted
                ))
                continue
            }

            print("[ModelDownloader] fetch [\(index + 1)/\(bundle.files.count)] \(file.relativePath) (\(formatBytes(file.expectedBytes)))")
            let fileStart = Date()
            let (tempURL, response) = try await session.download(from: file.sourceURL)

            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                try? fm.removeItem(at: tempURL)
                print("[ModelDownloader] http \(http.statusCode) for \(file.relativePath)")
                throw ModelDownloaderError.httpStatus(file: file.relativePath, status: http.statusCode)
            }

            // Move to a sibling `.tmp` first, verify size, then rename into place.
            // If we crash between these steps the final file never appears, so
            // `ModelAssets.isInstalled(…)` correctly reports "not installed" on
            // next launch and we redownload from scratch.
            let stagingURL = destination.appendingPathExtension("tmp")
            try? fm.removeItem(at: stagingURL)
            try fm.moveItem(at: tempURL, to: stagingURL)

            let attrs = try? fm.attributesOfItem(atPath: stagingURL.path)
            let actualSize = (attrs?[.size] as? Int64) ?? 0
            guard actualSize == file.expectedBytes else {
                try? fm.removeItem(at: stagingURL)
                print("[ModelDownloader] size mismatch \(file.relativePath) expected=\(file.expectedBytes) actual=\(actualSize)")
                throw ModelDownloaderError.sizeMismatch(
                    file: file.relativePath,
                    expected: file.expectedBytes,
                    actual: actualSize
                )
            }

            try? fm.removeItem(at: destination)
            try fm.moveItem(at: stagingURL, to: destination)
            bytesCompleted += actualSize

            let elapsed = Date().timeIntervalSince(fileStart)
            let mbps = elapsed > 0 ? Double(actualSize) / elapsed / 1_000_000 : 0
            print(String(format: "[ModelDownloader] done  %@  %.1fs  %.1f MB/s", file.relativePath, elapsed, mbps))

            yield(DownloadProgress(
                bundleID: bundle.id,
                fileIndex: index,
                fileCount: bundle.files.count,
                currentFileName: file.relativePath,
                bytesCompleted: bytesCompleted,
                bytesTotal: bytesTotal,
                phase: .fileCompleted
            ))
        }

        let totalElapsed = Date().timeIntervalSince(bundleStart)
        print(String(format: "[ModelDownloader] bundle complete  %.1fs  %@ total", totalElapsed, formatBytes(bytesCompleted)))

        yield(DownloadProgress(
            bundleID: bundle.id,
            fileIndex: bundle.files.count - 1,
            fileCount: bundle.files.count,
            currentFileName: bundle.files.last?.relativePath ?? "",
            bytesCompleted: bytesCompleted,
            bytesTotal: bytesTotal,
            phase: .allCompleted
        ))
    }

    private static func formatBytes(_ n: Int64) -> String {
        let f = ByteCountFormatter()
        f.countStyle = .file
        return f.string(fromByteCount: n)
    }
}
