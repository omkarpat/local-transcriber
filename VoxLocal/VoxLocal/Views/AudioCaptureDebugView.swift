import SwiftUI
import os

enum DebugVADState: String {
    case idle = "idle"
    case listening = "listening"
    case speaking = "speaking"
}

/// Developer-facing inspector for the pipeline. Subscribes to the same
/// `TranscriptionPipeline` instance as the user-facing `TranscriptionView`
/// but also taps into the optional `debugEvents` stream for per-frame VAD
/// probability + in-flight counters. No wiring of its own — all audio/VAD/
/// ASR lifecycle goes through the shared actor.
@Observable
@MainActor
final class AudioCaptureDebugModel {
    @ObservationIgnored private weak var pipeline: TranscriptionPipeline?
    @ObservationIgnored private var updatesTask: Task<Void, Never>?
    @ObservationIgnored private var eventsTask: Task<Void, Never>?
    @ObservationIgnored private var debugTask: Task<Void, Never>?
    @ObservationIgnored private var metricsTask: Task<Void, Never>?
    @ObservationIgnored private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "DebugView")

    var isRunning = false
    var rms: Float = 0
    var probability: Float = 0
    var vadState: DebugVADState = .idle
    var framesCaptured: Int = 0
    var overflowSamples: Int = 0
    var utteranceCount: Int = 0
    var lastUtteranceDuration: Duration = .zero
    var errorMessage: String?
    var transcripts: [TranscriptRow] = []   // most recent first; partials upsert in place

    /// Partial-transcription tick interval in ms. `nil` means partials are
    /// off (behavior matches pre-partial builds). Applied on `.start()`;
    /// changing while running requires stop→start.
    var partialIntervalMs: Int? = 1500

    /// When false, the pipeline won't POST finalized text to the cloud
    /// punctuation server on this session. Useful for isolating local-only
    /// behavior while iterating on VAD / ASR.
    var cloudPunctuationEnabled: Bool = true

    /// When true, each finalized utterance is also written to
    /// `Documents/utterance-<ms>.wav` so the "Run Moonshine on latest
    /// utterance" smoke test has something to replay.
    var dumpUtteranceWAV: Bool = false

    func attach(to pipeline: TranscriptionPipeline) {
        guard self.pipeline !== pipeline else { return }
        self.pipeline = pipeline
        subscribeToUpdates(pipeline)
        subscribeToEvents(pipeline)
        subscribeToDebugEvents(pipeline)
    }

    func detach() {
        updatesTask?.cancel(); updatesTask = nil
        eventsTask?.cancel(); eventsTask = nil
        debugTask?.cancel(); debugTask = nil
        metricsTask?.cancel(); metricsTask = nil
    }

    func toggle() async {
        if isRunning { await stop() } else { await start() }
    }

    private func start() async {
        guard let pipeline else {
            errorMessage = "Models are still loading — try again in a moment."
            return
        }
        var config = PipelineConfiguration.default
        config.vad.partialTranscriptionInterval = partialIntervalMs.map { .milliseconds($0) }
        config.enableCloudPunctuation = cloudPunctuationEnabled
        config.debugDumpUtterances = dumpUtteranceWAV
        do {
            try await pipeline.start(configuration: config)
            startMetricsLoop()
            errorMessage = nil
        } catch {
            errorMessage = String(describing: error)
        }
    }

    private func stop() async {
        guard let pipeline else { return }
        await pipeline.stop()
        metricsTask?.cancel(); metricsTask = nil
    }

    private func subscribeToUpdates(_ pipeline: TranscriptionPipeline) {
        updatesTask?.cancel()
        let stream = pipeline.updates
        updatesTask = Task { [weak self] in
            for await update in stream {
                guard let self else { return }
                self.transcripts.upsert(update)
                if self.transcripts.count > 20 {
                    self.transcripts.removeLast(self.transcripts.count - 20)
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
                    self.vadState = .idle
                    self.errorMessage = nil
                case .stopped:
                    self.isRunning = false
                    self.vadState = .idle
                    self.probability = 0
                case .failed(let message):
                    self.errorMessage = message
                    self.isRunning = false
                case .speechStateChanged:
                    break  // finer-grained info comes from debugEvents below
                }
            }
        }
    }

    private func subscribeToDebugEvents(_ pipeline: TranscriptionPipeline) {
        guard let stream = pipeline.debugEvents else {
            log.warning("pipeline has no debug stream; build AppState with debug: true")
            return
        }
        debugTask?.cancel()
        debugTask = Task { [weak self] in
            for await event in stream {
                guard let self else { return }
                switch event {
                case .frame(let probability, let isSpeech):
                    self.probability = probability
                    if self.vadState == .idle && isSpeech { self.vadState = .listening }
                case .utteranceStarted:
                    self.vadState = .speaking
                case .utteranceEnded(_, let duration):
                    self.vadState = .idle
                    self.utteranceCount += 1
                    self.lastUtteranceDuration = duration
                }
            }
        }
    }

    private func startMetricsLoop() {
        metricsTask?.cancel()
        guard let pipeline else { return }
        metricsTask = Task { [weak self] in
            while !Task.isCancelled {
                let (rms, frames, overflow) = await (
                    pipeline.currentRMS,
                    pipeline.framesCaptured,
                    pipeline.overflowSamples
                )
                self?.rms = rms
                self?.framesCaptured = frames
                self?.overflowSamples = overflow
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
    }
}

struct AudioCaptureDebugView: View {
    @Environment(AppState.self) private var appState
    @State private var model = AudioCaptureDebugModel()

    private var isLoadingModels: Bool {
        appState.transcriberStatus == .loading
    }

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 20) {
            Text("Audio + VAD")
                .font(.title2).bold()

            VStack(alignment: .leading, spacing: 6) {
                Text("Mic RMS").font(.caption).foregroundStyle(.secondary)
                LevelBar(level: model.rms, color: .blue, scale: 4)
                    .frame(height: 16)
            }
            .padding(.horizontal)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Speech probability").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    StatePill(state: model.vadState)
                }
                LevelBar(level: model.probability, color: .green, scale: 1)
                    .frame(height: 16)
            }
            .padding(.horizontal)

            VStack(alignment: .leading, spacing: 6) {
                metricRow("Utterances", value: "\(model.utteranceCount)")
                metricRow("Last duration", value: format(duration: model.lastUtteranceDuration))
                metricRow("Captured (s)", value: String(format: "%.1f", Double(model.framesCaptured) / AudioFormat.sampleRate))
                metricRow("Overflow", value: "\(model.overflowSamples)")
            }
            .font(.system(.footnote, design: .monospaced))
            .padding(.horizontal)

            HStack {
                Text("Partial interval").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Picker("Partial interval", selection: $model.partialIntervalMs) {
                    Text("Off").tag(Int?.none)
                    ForEach([500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000], id: \.self) { ms in
                        Text("\(ms) ms").tag(Int?.some(ms))
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .disabled(model.isRunning)
            }
            .padding(.horizontal)

            VStack {
                Toggle("Cloud punctuation", isOn: $model.cloudPunctuationEnabled)
                    .font(.caption)
                    .disabled(model.isRunning)
                Toggle("Dump utterance WAV (for smoke tests)", isOn: $model.dumpUtteranceWAV)
                    .font(.caption)
                    .disabled(model.isRunning)
            }
            .padding(.horizontal)

            HStack(spacing: 12) {
                Button(model.isRunning ? "Stop" : "Start") {
                    Task { await model.toggle() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isLoadingModels || appState.pipeline == nil)

                if isLoadingModels {
                    ProgressView()
                    Text("Loading ASR models…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if case .failed(let message) = appState.transcriberStatus {
                    Text("Model load failed: \(message)")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            if let message = model.errorMessage {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            transcriptList
        }
        .padding()
        .onChange(of: appState.transcriberStatus, initial: true) { _, status in
            if status == .ready, let pipeline = appState.pipeline {
                model.attach(to: pipeline)
            }
        }
        .onDisappear { model.detach() }
    }

    @ViewBuilder
    private var transcriptList: some View {
        if model.transcripts.isEmpty {
            Text("Transcripts will appear here as utterances finish.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                Text("Transcripts (newest first)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(model.transcripts) { row in
                            transcriptRow(row)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .frame(maxHeight: 180)
            }
            .padding(.horizontal)
        }
    }

    @ViewBuilder
    private func transcriptRow(_ row: TranscriptRow) -> some View {
        switch row.status {
        case .partial:
            VStack(alignment: .leading, spacing: 2) {
                Text(row.text.isEmpty ? "…" : row.text + " …")
                    .font(.body.italic())
                    .foregroundStyle(.secondary)
                Text("\(format(duration: row.utteranceDuration))  ·  partial")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        case .final(let tokenCount, let rtf, let punctuated):
            VStack(alignment: .leading, spacing: 2) {
                Text(row.text.isEmpty ? "(empty)" : row.text)
                    .font(.body)
                Text("\(format(duration: row.utteranceDuration))  ·  \(String(format: "RTF %.2f", rtf))  ·  \(tokenCount) tok  ·  \(punctuated ? "punctuated" : "raw")")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 2) {
                Text("⚠︎ transcription failed")
                    .font(.body)
                    .foregroundStyle(.red)
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
        }
    }

    private func metricRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value)
        }
    }

    private func format(duration: Duration) -> String {
        let seconds = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.2fs", seconds)
    }
}

private struct StatePill: View {
    let state: DebugVADState

    var body: some View {
        Text(state.rawValue.uppercased())
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(background)
            .foregroundStyle(.white)
            .clipShape(Capsule())
    }

    private var background: Color {
        switch state {
        case .idle: return .gray
        case .listening: return .orange
        case .speaking: return .green
        }
    }
}

private struct LevelBar: View {
    let level: Float
    let color: Color
    let scale: Float

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                RoundedRectangle(cornerRadius: 4)
                    .fill(color)
                    .frame(width: proxy.size.width * CGFloat(min(max(level * scale, 0), 1)))
            }
        }
    }
}

#Preview {
    AudioCaptureDebugView()
        .environment(AppState.previewing(.ready))
}
