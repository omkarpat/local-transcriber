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

    /// Warmup status of the ASR pipeline (Moonshine model + tokenizer).
    /// Kicked off automatically once `installState` becomes `.ready` so
    /// the ~2 s CoreML graph setup happens at launch instead of on the
    /// user's first tap of Start.
    enum TranscriberStatus: Equatable {
        case idle        // install not ready yet
        case loading
        case ready       // `transcriber` is non-nil
        case failed(String)
    }

    private(set) var installState: InstallState = .checking
    private(set) var transcriberStatus: TranscriberStatus = .idle
    private(set) var transcriber: MoonshineTranscriber?
    /// Long-lived pipeline wrapping the shared transcriber + punctuation
    /// client. Built once `transcriber` becomes non-nil; persists across
    /// start/stop cycles so its ASR stream stays continuous.
    private(set) var pipeline: TranscriptionPipeline?
    /// Shared across the app so any view dispatching a finalized
    /// transcript can post it for cloud punctuation. Instantiated in
    /// `init` — config is two strings, no heavy setup.
    let punctuationClient: PunctuationClient
    let bundle: ModelBundle
    private let downloader: ModelDownloader
    private var currentTask: Task<Void, Never>?

    init(bundle: ModelBundle = ModelAssets.moonshineTiny,
         downloader: ModelDownloader = ModelDownloader(),
         punctuationClient: PunctuationClient = PunctuationClient()) {
        self.bundle = bundle
        self.downloader = downloader
        self.punctuationClient = punctuationClient
    }

    /// Decide whether to skip the downloader or kick it off. Idempotent —
    /// calling twice while a download is in flight is a no-op.
    func bootstrap() {
        guard currentTask == nil else { return }
        if ModelAssets.isInstalled(bundle) {
            print("[AppState] bundle \(bundle.id) already installed; skipping download")
            installState = .ready
            loadTranscriber()
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
                        self?.loadTranscriber()
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
        transcriber = nil
        pipeline = nil
        transcriberStatus = .idle
        startDownload()
    }

    /// Load Moonshine + tokenizer off the main actor, exactly once per
    /// app lifetime. Idempotent — subsequent calls while loading or
    /// already loaded are no-ops. Downloader → transcriber is the only
    /// legitimate path; we reset to `.idle` on bundle removal via
    /// `resetInstall()`.
    private func loadTranscriber() {
        guard case .idle = transcriberStatus else { return }
        transcriberStatus = .loading
        Task { [weak self] in
            do {
                // Detached so ~2 s of file I/O + CoreML graph setup
                // doesn't block the main actor.
                let t = try await Task.detached(priority: .userInitiated) { () -> MoonshineTranscriber in
                    let model = try MoonshineModel()
                    let tokenizer = try await MoonshineTokenizer.load()
                    return MoonshineTranscriber(model: model, tokenizer: tokenizer)
                }.value
                guard let self else { return }
                self.transcriber = t
                // Construct the pipeline with debug: true so the debug
                // view has its fine-grained event stream available.
                // Non-debug views simply don't subscribe to it.
                self.pipeline = TranscriptionPipeline(
                    transcriber: t,
                    punctuationClient: self.punctuationClient,
                    debug: true
                )
                self.transcriberStatus = .ready
                print("[AppState] transcriber ready")
            } catch {
                self?.transcriberStatus = .failed(error.localizedDescription)
                print("[AppState] transcriber load failed: \(error.localizedDescription)")
            }
        }
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
