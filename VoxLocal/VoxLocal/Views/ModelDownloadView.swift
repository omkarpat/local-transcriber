import SwiftUI

/// First-run screen. Shown whenever `AppState.installState` is anything
/// other than `.ready`. Responsible for presenting progress / error /
/// retry affordances; not for driving the download itself.
struct ModelDownloadView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "waveform.badge.microphone")
                .font(.system(size: 56))
                .foregroundStyle(.tint)
                .symbolRenderingMode(.hierarchical)

            Text("Setting up VoxLocal")
                .font(.title2)
                .fontWeight(.semibold)

            content
                .frame(maxWidth: 420)
        }
        .padding(32)
    }

    @ViewBuilder
    private var content: some View {
        switch appState.installState {
        case .checking:
            VStack(spacing: 12) {
                ProgressView()
                Text("Preparing models…")
                    .foregroundStyle(.secondary)
            }

        case .downloading(let progress):
            VStack(alignment: .leading, spacing: 12) {
                ProgressView(value: progress.fractionCompleted)
                HStack {
                    Text(progress.currentFileName)
                        .font(.callout.monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Text("\(progress.fileIndex + 1) / \(progress.fileCount)")
                        .foregroundStyle(.secondary)
                        .font(.callout)
                }
                Text(Self.formatBytes(progress.bytesCompleted, of: progress.bytesTotal))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

        case .ready:
            // Transient — RootView swaps us out on .ready. Render something
            // neutral in case there's a frame of overlap.
            ProgressView()

        case .failed(let message):
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .font(.title)
                Text(message)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Button("Retry") { appState.startDownload() }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private static func formatBytes(_ done: Int64, of total: Int64) -> String {
        let f = ByteCountFormatter()
        f.countStyle = .file
        return "\(f.string(fromByteCount: done)) of \(f.string(fromByteCount: total))"
    }
}

#Preview("Downloading") {
    ModelDownloadView()
        .environment(AppState.previewing(.downloading(DownloadProgress(
            bundleID: "moonshine-tiny",
            fileIndex: 1,
            fileCount: 8,
            currentFileName: "decoder_model_merged.onnx",
            bytesCompleted: 30_882_331,
            bytesTotal: 113_008_569,
            phase: .fileStarted
        ))))
}

#Preview("Failed") {
    ModelDownloadView()
        .environment(AppState.previewing(.failed("The Internet connection appears to be offline.")))
}
