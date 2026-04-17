import Foundation
import Observation

/// Top-level app state driving the first-run bootstrap gate.
/// Main-actor-isolated so SwiftUI views can read it directly; download
/// work itself runs on background tasks inside `ModelDownloader`.
@Observable
final class AppState {
    enum InstallState: Equatable {
        case checking
        case downloading(DownloadProgress)
        case ready
        case failed(String)
    }

    private(set) var installState: InstallState = .checking
    let bundle: ModelBundle
    private let downloader: ModelDownloader
    private var currentTask: Task<Void, Never>?

    init(bundle: ModelBundle = ModelAssets.moonshineTiny,
         downloader: ModelDownloader = ModelDownloader()) {
        self.bundle = bundle
        self.downloader = downloader
    }

    /// Decide whether to skip the downloader or kick it off. Idempotent —
    /// calling twice while a download is in flight is a no-op.
    func bootstrap() {
        guard currentTask == nil else { return }
        if ModelAssets.isInstalled(bundle) {
            print("[AppState] bundle \(bundle.id) already installed; skipping download")
            installState = .ready
            return
        }
        print("[AppState] bundle \(bundle.id) missing; starting download")
        startDownload()
    }

    /// Manually trigger a fresh download. Used by the retry button on the
    /// failure screen and by the dev-only "reinstall models" debug action.
    func startDownload() {
        currentTask?.cancel()
        installState = .checking
        let bundle = self.bundle
        let downloader = self.downloader
        currentTask = Task { [weak self] in
            do {
                for try await progress in downloader.download(bundle) {
                    self?.installState = .downloading(progress)
                    if progress.phase == .allCompleted {
                        print("[AppState] install ready")
                        self?.installState = .ready
                    }
                }
            } catch is CancellationError {
                print("[AppState] download cancelled")
            } catch {
                print("[AppState] download failed: \(error.localizedDescription)")
                self?.installState = .failed(error.localizedDescription)
            }
            self?.currentTask = nil
        }
    }

    /// Dev-only: delete the installed bundle and redownload on next bootstrap.
    /// Exposed from `ContentView` debug actions so we can exercise the first-run
    /// UI without reinstalling the app.
    func resetInstall() {
        currentTask?.cancel()
        currentTask = nil
        try? ModelAssets.remove(bundle)
        installState = .checking
        startDownload()
    }

    #if DEBUG
    /// Construct an `AppState` wedged into a specific `InstallState` for SwiftUI
    /// previews only. Keeps `installState`'s setter private in the real app.
    static func previewing(_ state: InstallState) -> AppState {
        let s = AppState()
        s.installState = state
        return s
    }
    #endif
}
