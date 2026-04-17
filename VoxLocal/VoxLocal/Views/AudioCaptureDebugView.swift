import SwiftUI
import os

enum DebugVADState: String {
    case idle = "idle"
    case listening = "listening"
    case speaking = "speaking"
}

@Observable
final class AudioCaptureDebugModel {
    let manager = AudioCaptureManager()
    @ObservationIgnored private var vad: VADProcessor?
    @ObservationIgnored private var eventTask: Task<Void, Never>?
    @ObservationIgnored private var transcriptTask: Task<Void, Never>?
    @ObservationIgnored private var transcriber: MoonshineTranscriber?
    @ObservationIgnored private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "DebugView")

    var isRunning = false
    var isLoadingModels = false
    var rms: Float = 0
    var probability: Float = 0
    var vadState: DebugVADState = .idle
    var framesCaptured: Int = 0
    var overflowSamples: Int = 0
    var utteranceCount: Int = 0
    var lastUtteranceDuration: Duration = .zero
    var lastUtterancePath: String?
    var errorMessage: String?
    var transcripts: [TranscriptUpdate] = []   // most recent first

    func toggle() async {
        if isRunning {
            stop()
            return
        }
        await start()
    }

    private func start() async {
        do {
            try await ensureTranscriber()
            try await manager.start()
            let vad = VADProcessor(ringBuffer: manager.ringBuffer)
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
        // Keep `transcriber`, `transcriptTask`, and `transcripts` across
        // start/stop cycles — no point reloading the 2s model every time,
        // and the user likely wants to keep reading the history.
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

    private func ensureTranscriber() async throws {
        if transcriber != nil { return }
        isLoadingModels = true
        defer { isLoadingModels = false }
        // Session loads do ~2s of file I/O + CoreML graph setup. Never on
        // the main actor — CoreAudio and the ORT runtime both fire "This
        // method should not be called on the main thread" warnings
        // otherwise. Task.detached hops to a cooperative background thread.
        let t = try await Task.detached(priority: .userInitiated) { () -> MoonshineTranscriber in
            let model = try MoonshineModel()
            let tokenizer = try await MoonshineTokenizer.load()
            return MoonshineTranscriber(model: model, tokenizer: tokenizer)
        }.value
        transcriber = t
        subscribeToTranscripts(t)
    }

    private func subscribeToTranscripts(_ t: MoonshineTranscriber) {
        transcriptTask?.cancel()
        transcriptTask = Task { [weak self] in
            for await update in t.updates {
                guard let self else { return }
                await MainActor.run {
                    // Newest first, cap history so the view doesn't grow unbounded.
                    self.transcripts.insert(update, at: 0)
                    if self.transcripts.count > 20 {
                        self.transcripts.removeLast(self.transcripts.count - 20)
                    }
                }
            }
        }
    }

    @MainActor
    private func apply(_ event: VADEvent) {
        switch event {
        case .frame(let prob, let isSpeech):
            probability = prob
            if vadState == .idle && isSpeech { vadState = .listening }
        case .utteranceStarted:
            vadState = .speaking
        case .utteranceEnded(let samples, let duration):
            vadState = .idle
            utteranceCount += 1
            lastUtteranceDuration = duration
            dumpUtterance(samples: samples)
            transcriber?.enqueue(samples: samples, duration: duration)
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
    @State private var model = AudioCaptureDebugModel()

    var body: some View {
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

            HStack(spacing: 12) {
                Button(model.isRunning ? "Stop" : "Start") {
                    Task { await model.toggle() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.isLoadingModels)

                if model.isLoadingModels {
                    ProgressView()
                    Text("Loading ASR models…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                        ForEach(model.transcripts) { update in
                            transcriptRow(update)
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
    private func transcriptRow(_ update: TranscriptUpdate) -> some View {
        switch update {
        case .finalized(let r):
            VStack(alignment: .leading, spacing: 2) {
                Text(r.text.isEmpty ? "(empty)" : r.text)
                    .font(.body)
                Text("\(format(duration: r.utteranceDuration))  ·  \(String(format: "RTF %.2f", r.realTimeFactor))  ·  \(r.tokenCount) tok")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        case .failed(let f):
            VStack(alignment: .leading, spacing: 2) {
                Text("⚠︎ transcription failed")
                    .font(.body)
                    .foregroundStyle(.red)
                Text(f.message)
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
}
