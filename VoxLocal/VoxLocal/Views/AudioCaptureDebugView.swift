import SwiftUI

@Observable
final class AudioCaptureDebugModel {
    let manager = AudioCaptureManager()

    var isRunning = false
    var rms: Float = 0
    var framesCaptured: Int = 0
    var framesDrained: Int = 0
    var overflowSamples: Int = 0
    var errorMessage: String?

    @ObservationIgnored private let drainBufferCapacity = 4096
    @ObservationIgnored private let drainBuffer: UnsafeMutableBufferPointer<Float>

    init() {
        drainBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 4096)
    }

    deinit {
        drainBuffer.deallocate()
    }

    func toggle() async {
        if isRunning {
            manager.stop()
            isRunning = false
            return
        }
        do {
            try await manager.start()
            isRunning = true
            errorMessage = nil
        } catch {
            errorMessage = String(describing: error)
            isRunning = false
        }
    }

    func refreshMetrics() {
        rms = manager.currentRMS
        framesCaptured = manager.framesCaptured
        overflowSamples = manager.ringBuffer.overflowSamples

        // Debug drain: pretend to be the VAD consumer so overflow stays at 0.
        // Replaced by VADProcessor in Task 3.
        while true {
            let n = manager.ringBuffer.read(into: drainBuffer.baseAddress!, maxCount: drainBufferCapacity)
            if n == 0 { break }
            framesDrained += n
        }
    }
}

struct AudioCaptureDebugView: View {
    @State private var model = AudioCaptureDebugModel()

    var body: some View {
        VStack(spacing: 24) {
            Text("Audio Capture")
                .font(.title2).bold()

            LevelBar(level: model.rms)
                .frame(height: 20)
                .padding(.horizontal)

            VStack(alignment: .leading, spacing: 6) {
                metricRow("RMS", value: String(format: "%.4f", model.rms))
                metricRow("Written (16kHz)", value: "\(model.framesCaptured)")
                metricRow("Drained", value: "\(model.framesDrained)")
                metricRow("Seconds", value: String(format: "%.2f", Double(model.framesCaptured) / AudioFormat.sampleRate))
                metricRow("Overflow", value: "\(model.overflowSamples)")
            }
            .font(.system(.body, design: .monospaced))
            .padding(.horizontal)

            Button(model.isRunning ? "Stop" : "Start") {
                Task { await model.toggle() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            if let message = model.errorMessage {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
        .padding()
        .task {
            while !Task.isCancelled {
                model.refreshMetrics()
                try? await Task.sleep(for: .milliseconds(50))
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
}

private struct LevelBar: View {
    let level: Float

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(.quaternary)
                RoundedRectangle(cornerRadius: 4)
                    .fill(.tint)
                    .frame(width: proxy.size.width * CGFloat(min(max(level * 4, 0), 1)))
            }
        }
    }
}

#Preview {
    AudioCaptureDebugView()
}
