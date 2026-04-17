import SwiftUI
import os

enum DebugVADState: String {
    case idle = "idle"
    case listening = "listening"
    case speaking = "speaking"
}

/// UI row for one utterance. `id == utteranceID` so SwiftUI's `ForEach`
/// updates the existing row in place when a partial is replaced by the
/// final result.
struct TranscriptRow: Identifiable, Equatable {
    enum Status: Equatable {
        case partial
        case final(tokenCount: Int, realTimeFactor: Double)
        case failed(message: String)
    }

    let utteranceID: UUID
    var id: UUID { utteranceID }
    var status: Status
    var text: String
    var utteranceDuration: Duration
    var inferenceDuration: Duration
}

@Observable
final class AudioCaptureDebugModel {
    let manager = AudioCaptureManager()
    @ObservationIgnored private var vad: VADProcessor?
    @ObservationIgnored private var eventTask: Task<Void, Never>?
    @ObservationIgnored private var transcriptTask: Task<Void, Never>?
    @ObservationIgnored private var transcriber: MoonshineTranscriber?
    @ObservationIgnored private var punctuationClient: PunctuationClient?
    @ObservationIgnored private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "DebugView")

    var isRunning = false
    var rms: Float = 0
    var probability: Float = 0
    var vadState: DebugVADState = .idle
    var framesCaptured: Int = 0
    var overflowSamples: Int = 0
    var utteranceCount: Int = 0
    var lastUtteranceDuration: Duration = .zero
    var lastUtterancePath: String?
    var errorMessage: String?
    var transcripts: [TranscriptRow] = []   // most recent first; partials upsert in place

    /// Partial-transcription tick interval in ms. `nil` means partials
    /// are off (behavior matches pre-partial builds). Applied on
    /// `.start()`; changing while running requires stop→start.
    var partialIntervalMs: Int? = 1500

    func toggle() async {
        if isRunning {
            stop()
            return
        }
        await start()
    }

    private func start() async {
        guard transcriber != nil else {
            errorMessage = "Models are still loading — try again in a moment."
            return
        }
        do {
            try await manager.start()
            var config = VADConfiguration.default
            config.partialTranscriptionInterval = partialIntervalMs.map { .milliseconds($0) }
            let vad = VADProcessor(ringBuffer: manager.ringBuffer, configuration: config)
            self.vad = vad
            subscribe(to: vad.events)
            vad.start()
            isRunning = true
            errorMessage = nil
        } catch {
            errorMessage = String(describing: error)
            isRunning = false
        }
    }

    private func stop() {
        vad?.stop()
        vad = nil
        eventTask?.cancel()
        eventTask = nil
        manager.stop()
        vadState = .idle
        probability = 0
        isRunning = false
        // `transcriber` is owned by AppState and lives for the app
        // lifetime; `transcripts` stays across start/stop cycles so the
        // user can keep reading history.
    }

    private func subscribe(to events: AsyncStream<VADEvent>) {
        eventTask?.cancel()
        eventTask = Task { [weak self] in
            for await event in events {
                guard let self else { return }
                await MainActor.run {
                    self.apply(event)
                }
            }
        }
    }

    /// Bind the preloaded `MoonshineTranscriber` + `PunctuationClient`
    /// from `AppState` to this model's event pipeline. Idempotent —
    /// re-calling with the same transcriber instance is a no-op. The
    /// view calls this via `.onChange` as soon as `AppState.transcriber`
    /// becomes non-nil.
    func attach(transcriber: MoonshineTranscriber, punctuationClient: PunctuationClient) {
        self.punctuationClient = punctuationClient
        guard self.transcriber !== transcriber else { return }
        self.transcriber = transcriber
        subscribeToTranscripts(transcriber)
    }

    private func subscribeToTranscripts(_ t: MoonshineTranscriber) {
        transcriptTask?.cancel()
        transcriptTask = Task { [weak self] in
            for await update in t.updates {
                guard let self else { return }
                await MainActor.run {
                    self.upsert(update)
                }
                if case .finalized(let result) = update {
                    self.dispatchPunctuation(for: result)
                }
            }
        }
    }

    /// Fire the cloud punctuation request in parallel with the raw
    /// finalized render. Result is either a polished replacement
    /// (upserted by `utteranceID`) or nil (silent fallback — raw row
    /// stays on screen unchanged). `Task { … }` (not detached) so the
    /// closure inherits MainActor; the `await` on `punctuate` hops off
    /// actor for the network request and comes back to MainActor for
    /// the upsert.
    private func dispatchPunctuation(for result: TranscriptResult) {
        guard let client = punctuationClient else { return }
        Task { [weak self] in
            let maybe = await client.punctuate(
                utteranceID: result.utteranceID,
                text: result.text,
                utteranceDuration: result.utteranceDuration
            )
            guard let punctuated = maybe else { return }
            self?.upsert(.punctuated(punctuated))
        }
    }

    /// Inserts or updates the row for `update.utteranceID`. Rules:
    /// - `.partial`: create a new row at top, or replace an existing
    ///   `.partial` in place. If the row is already `.final`/`.failed`
    ///   (late partial returning after final), ignore — final wins.
    /// - `.finalized`: replace the row in place, or insert at top if
    ///   no row exists yet (short utterances may skip partials).
    /// - `.punctuated`: replace the row's text in place, only if the
    ///   row is currently `.final`. Never inserts; if the finalized
    ///   upsert hasn't run yet (shouldn't happen — we upsert final
    ///   synchronously before dispatching punctuation) or the row is
    ///   already `.failed`, silently drop.
    /// - `.failed`: same as finalized.
    /// Capped at 20 rows.
    @MainActor
    private func upsert(_ update: TranscriptUpdate) {
        let utteranceID = update.utteranceID
        let existingIndex = transcripts.firstIndex(where: { $0.utteranceID == utteranceID })
        switch update {
        case .partial(let p):
            if let idx = existingIndex {
                if case .partial = transcripts[idx].status {
                    transcripts[idx].text = p.text
                    transcripts[idx].utteranceDuration = p.utteranceDuration
                    transcripts[idx].inferenceDuration = p.inferenceDuration
                }
                // final / failed rows ignore late partials
            } else {
                transcripts.insert(TranscriptRow(
                    utteranceID: utteranceID,
                    status: .partial,
                    text: p.text,
                    utteranceDuration: p.utteranceDuration,
                    inferenceDuration: p.inferenceDuration
                ), at: 0)
            }
        case .finalized(let r):
            let row = TranscriptRow(
                utteranceID: utteranceID,
                status: .final(tokenCount: r.tokenCount, realTimeFactor: r.realTimeFactor),
                text: r.text,
                utteranceDuration: r.utteranceDuration,
                inferenceDuration: r.inferenceDuration
            )
            if let idx = existingIndex {
                transcripts[idx] = row
            } else {
                transcripts.insert(row, at: 0)
            }
        case .punctuated(let p):
            guard let idx = existingIndex else { return }
            if case .final = transcripts[idx].status {
                transcripts[idx].text = p.text
            }
        case .failed(let f):
            let row = TranscriptRow(
                utteranceID: utteranceID,
                status: .failed(message: f.message),
                text: "",
                utteranceDuration: f.utteranceDuration,
                inferenceDuration: .zero
            )
            if let idx = existingIndex {
                transcripts[idx] = row
            } else {
                transcripts.insert(row, at: 0)
            }
        }
        if transcripts.count > 20 {
            transcripts.removeLast(transcripts.count - 20)
        }
    }

    @MainActor
    private func apply(_ event: VADEvent) {
        switch event {
        case .frame(let prob, let isSpeech):
            probability = prob
            if vadState == .idle && isSpeech { vadState = .listening }
        case .utteranceStarted(_):
            vadState = .speaking
        case .utteranceProgress(let id, let samples, let duration):
            transcriber?.enqueuePartial(utteranceID: id, samples: samples, duration: duration)
        case .utteranceEnded(let id, let samples, let duration):
            vadState = .idle
            utteranceCount += 1
            lastUtteranceDuration = duration
            dumpUtterance(samples: samples)
            transcriber?.enqueue(utteranceID: id, samples: samples, duration: duration)
        case .error(let message):
            errorMessage = message
        }
    }

    private func dumpUtterance(samples: [Float]) {
        let stamp = Int(Date().timeIntervalSince1970 * 1000)
        let url = WAVWriter.documentsDirectory().appendingPathComponent("utterance-\(stamp).wav")
        do {
            try WAVWriter.write(samples: samples, to: url)
            lastUtterancePath = url.lastPathComponent
            log.info("Wrote utterance: \(url.path)")
        } catch {
            log.error("WAV write failed: \(error)")
        }
    }

    func refreshCaptureMetrics() {
        rms = manager.currentRMS
        framesCaptured = manager.framesCaptured
        overflowSamples = manager.ringBuffer.overflowSamples
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
                metricRow("Last file", value: model.lastUtterancePath ?? "—")
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

            HStack(spacing: 12) {
                Button(model.isRunning ? "Stop" : "Start") {
                    Task { await model.toggle() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isLoadingModels || appState.transcriber == nil)

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
        .task {
            while !Task.isCancelled {
                model.refreshCaptureMetrics()
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
        .onChange(of: appState.transcriberStatus, initial: true) { _, status in
            if status == .ready, let t = appState.transcriber {
                model.attach(transcriber: t, punctuationClient: appState.punctuationClient)
            }
        }
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
        case .final(let tokenCount, let rtf):
            VStack(alignment: .leading, spacing: 2) {
                Text(row.text.isEmpty ? "(empty)" : row.text)
                    .font(.body)
                Text("\(format(duration: row.utteranceDuration))  ·  \(String(format: "RTF %.2f", rtf))  ·  \(tokenCount) tok")
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
