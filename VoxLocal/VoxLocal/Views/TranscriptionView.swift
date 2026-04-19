import SwiftUI

/// User-facing transcription screen. Big scrolling transcript, prominent
/// Record/Stop button, elapsed time, compact mic level + speaking indicator.
/// All orchestration goes through `TranscriptionPipeline`; this view just
/// projects its streams into `@Observable` state for SwiftUI.
@Observable
@MainActor
final class TranscriptionModel {
    /// Actor reference held weakly so the model doesn't outlive AppState.
    /// In practice both live for the app's lifetime, but `weak` is the
    /// idiomatic form for "reference I don't own".
    @ObservationIgnored private weak var pipeline: TranscriptionPipeline?
    @ObservationIgnored private var updatesTask: Task<Void, Never>?
    @ObservationIgnored private var eventsTask: Task<Void, Never>?
    @ObservationIgnored private var metricsTask: Task<Void, Never>?
    @ObservationIgnored private var timerTask: Task<Void, Never>?

    private(set) var rows: [TranscriptRow] = []
    private(set) var isRunning = false
    private(set) var isSpeaking = false
    private(set) var rms: Float = 0
    private(set) var elapsed: Duration = .zero
    private(set) var errorMessage: String?

    /// Partial-transcription cadence. 1.5 s is the same default the debug
    /// view used — a good balance between responsiveness and not pounding
    /// Moonshine on every VAD frame. `nil` turns partials off (user-facing
    /// escape hatch if someone wants to isolate finalized-only behavior).
    var partialIntervalMs: Int? = 1500

    /// Max rows retained in the scroll view. Older utterances drop off the
    /// bottom; the user can still read them via the debug view's larger
    /// history if they care. 40 comfortably covers a typical session.
    static let maxRows = 40

    private var sessionStart: Date?

    func attach(to pipeline: TranscriptionPipeline) {
        guard self.pipeline !== pipeline else { return }
        self.pipeline = pipeline
        subscribeToUpdates(pipeline)
        subscribeToEvents(pipeline)
    }

    func detach() {
        updatesTask?.cancel(); updatesTask = nil
        eventsTask?.cancel(); eventsTask = nil
        metricsTask?.cancel(); metricsTask = nil
        timerTask?.cancel(); timerTask = nil
    }

    func toggle() async {
        if isRunning { await stop() } else { await start() }
    }

    func start() async {
        guard let pipeline else {
            errorMessage = "Models are still loading."
            return
        }
        errorMessage = nil
        var config = PipelineConfiguration.default
        config.vad.partialTranscriptionInterval = partialIntervalMs.map { .milliseconds($0) }
        do {
            try await pipeline.start(configuration: config)
            sessionStart = Date()
            startMetricsLoop()
            startTimerLoop()
        } catch {
            errorMessage = friendlyMessage(for: error)
        }
    }

    func stop() async {
        guard let pipeline else { return }
        await pipeline.stop()
        metricsTask?.cancel(); metricsTask = nil
        timerTask?.cancel(); timerTask = nil
    }

    func clearTranscript() {
        rows.removeAll()
        elapsed = .zero
        sessionStart = nil
    }

    private func subscribeToUpdates(_ pipeline: TranscriptionPipeline) {
        updatesTask?.cancel()
        let stream = pipeline.updates
        updatesTask = Task { [weak self] in
            for await update in stream {
                guard let self else { return }
                self.rows.upsert(update)
                if self.rows.count > Self.maxRows {
                    self.rows.removeLast(self.rows.count - Self.maxRows)
                }
            }
        }
    }

    private func subscribeToEvents(_ pipeline: TranscriptionPipeline) {
        eventsTask?.cancel()
        let stream = pipeline.events
        eventsTask = Task { [weak self] in
            for await event in stream {
                guard let self else { return }
                switch event {
                case .started:
                    self.isRunning = true
                    self.errorMessage = nil
                case .stopped:
                    self.isRunning = false
                    self.isSpeaking = false
                    self.flattenPendingPartials()
                case .failed(let message):
                    self.errorMessage = message
                    self.isRunning = false
                    self.isSpeaking = false
                case .speechStateChanged(let active):
                    self.isSpeaking = active
                }
            }
        }
    }

    /// Any `.partial` rows still visible when the pipeline stops belong to
    /// an utterance VAD never closed (the user stopped mid-speech). Without
    /// this, those rows stay italic/grey forever because no `.finalized`
    /// is coming. Promote them to `.final` using the ASR's most recent
    /// partial text — good enough for display, and if a late `.finalized`
    /// or `.refined` eventually lands for the same utteranceID the existing
    /// upsert logic will overwrite the row.
    ///
    /// Rebuilding the array (rather than mutating in place) so `@Observable`
    /// reliably picks up the change and re-renders.
    private func flattenPendingPartials() {
        rows = rows.map { row in
            guard case .partial = row.status else { return row }
            var promoted = row
            promoted.status = .final(tokenCount: 0, realTimeFactor: 0, isRefined: false)
            return promoted
        }
    }

    private func startMetricsLoop() {
        metricsTask?.cancel()
        guard let pipeline else { return }
        metricsTask = Task { [weak self] in
            while !Task.isCancelled {
                let rms = await pipeline.currentRMS
                self?.rms = rms
                try? await Task.sleep(for: .milliseconds(66))  // ~15 Hz, enough for a level meter
            }
        }
    }

    private func startTimerLoop() {
        timerTask?.cancel()
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                if let self, let start = self.sessionStart {
                    self.elapsed = .seconds(Date().timeIntervalSince(start))
                }
                try? await Task.sleep(for: .milliseconds(200))  // 5 Hz, MM:SS only ticks once/sec
            }
        }
    }

    /// Turn an `AudioCaptureError` or similar into a human-readable line.
    private func friendlyMessage(for error: Error) -> String {
        if let captureError = error as? AudioCaptureError {
            switch captureError {
            case .permissionDenied:
                return "Microphone access was denied. Enable it in Settings → VoxLocal to record."
            case .converterUnavailable:
                return "This device's microphone format can't be converted to 16 kHz mono."
            case .engineStartFailed(let underlying):
                return "Audio engine failed to start: \(underlying.localizedDescription)"
            case .sessionConfigurationFailed(let underlying):
                return "Audio session setup failed: \(underlying.localizedDescription)"
            }
        }
        return error.localizedDescription
    }
}

struct TranscriptionView: View {
    @Environment(AppState.self) private var appState
    @State private var model = TranscriptionModel()

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal)
                .padding(.top, 8)

            Divider().padding(.vertical, 8)

            transcriptArea
                .frame(maxHeight: .infinity)

            Divider()

            controls
                .padding()
        }
        .navigationTitle("Transcribe")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: appState.transcriberStatus, initial: true) { _, status in
            if status == .ready, let pipeline = appState.pipeline {
                model.attach(to: pipeline)
            }
        }
        .onDisappear { model.detach() }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    model.clearTranscript()
                } label: {
                    Image(systemName: "trash")
                }
                .disabled(model.rows.isEmpty || model.isRunning)
            }
        }
    }

    // MARK: - Subviews

    private var header: some View {
        HStack(spacing: 16) {
            Text(formatElapsed(model.elapsed))
                .font(.system(.title3, design: .monospaced))
                .foregroundStyle(model.isRunning ? .primary : .secondary)

            MicLevelMeter(rms: model.rms, isActive: model.isRunning)
                .frame(height: 10)

            SpeakingIndicator(isSpeaking: model.isSpeaking, isRunning: model.isRunning)
        }
    }

    @ViewBuilder
    private var transcriptArea: some View {
        if model.rows.isEmpty {
            emptyState
        } else {
            ScrollView {
                Text(paragraph)
                    .font(.body)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(.horizontal)
                    .padding(.vertical, 8)
            }
        }
    }

    /// Renders all rows as a flowing transcript, one paragraph per row.
    /// Every row — whether a split-sibling within an utterance or the
    /// first segment of a new utterance — gets a `\n\n` break. The
    /// in-flight `.partial`, if any, trails as its own paragraph in
    /// italic/secondary so the user can see what's being transcribed
    /// right now. Failures get a small red marker.
    ///
    /// `model.rows` is newest-first (upsert inserts at index 0), so we
    /// reverse for display.
    private var paragraph: AttributedString {
        var result = AttributedString("")
        for row in model.rows.reversed() {
            let content: AttributedString?
            switch row.status {
            case .partial:
                var partial = AttributedString(row.text.isEmpty ? "…" : row.text + " …")
                partial.foregroundColor = .secondary
                partial.inlinePresentationIntent = .emphasized
                content = partial
            case .final:
                content = row.text.isEmpty ? nil : AttributedString(row.text)
            case .failed:
                var marker = AttributedString("⚠︎")
                marker.foregroundColor = .red
                content = marker
            }
            guard let content else { continue }
            if !result.characters.isEmpty {
                result.append(AttributedString("\n\n"))
            }
            result.append(content)
        }
        return result
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform")
                .font(.system(size: 48, weight: .light))
                .foregroundStyle(.tertiary)
            Text(model.isRunning ? "Listening…" : "Tap record to start")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var controls: some View {
        VStack(spacing: 10) {
            if let message = model.errorMessage {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }

            if appState.transcriberStatus == .loading {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Loading ASR models…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if case .failed(let message) = appState.transcriberStatus {
                Text("Model load failed: \(message)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            Button {
                Task { await model.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: model.isRunning ? "stop.circle.fill" : "record.circle.fill")
                        .font(.system(size: 22))
                    Text(model.isRunning ? "Stop" : "Record")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(model.isRunning ? .red : .accentColor)
            .controlSize(.large)
            .disabled(appState.transcriberStatus != .ready || appState.pipeline == nil)
        }
    }

    private func formatElapsed(_ duration: Duration) -> String {
        let totalSeconds = Int(Double(duration.components.seconds))
        let m = totalSeconds / 60
        let s = totalSeconds % 60
        return String(format: "%02d:%02d", m, s)
    }
}

// MARK: - Level meter

private struct MicLevelMeter: View {
    let rms: Float
    let isActive: Bool

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(.quaternary)
                if isActive {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.accentColor.gradient)
                        .frame(width: proxy.size.width * CGFloat(clamped(rms * 4)))
                        .animation(.easeOut(duration: 0.08), value: rms)
                }
            }
        }
    }

    private func clamped(_ value: Float) -> Float {
        min(max(value, 0), 1)
    }
}

private struct SpeakingIndicator: View {
    let isSpeaking: Bool
    let isRunning: Bool

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 10, height: 10)
            .animation(.easeInOut(duration: 0.2), value: isSpeaking)
    }

    private var color: Color {
        if !isRunning { return .gray.opacity(0.4) }
        return isSpeaking ? .green : .yellow.opacity(0.7)
    }
}

#Preview {
    NavigationStack {
        TranscriptionView()
    }
    .environment(AppState.previewing(.ready))
}
