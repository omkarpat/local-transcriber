//
//  ContentView.swift
//  VoxLocal
//
//  Created by Omkar Patil on 4/16/26.
//

import SwiftUI

struct ContentView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        NavigationStack {
            TranscriptionView()
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        NavigationLink {
                            DeveloperView()
                        } label: {
                            Image(systemName: "hammer")
                        }
                    }
                }
        }
    }
}

/// Secondary menu for developer-only utilities: the VAD/ASR debug
/// inspector plus the smoke-test + benchmark buttons. Not intended for
/// end users; reachable via the hammer icon on the main screen.
struct DeveloperView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        List {
            Section("Inspector") {
                NavigationLink("Audio + VAD Debug View") {
                    AudioCaptureDebugView()
                }
            }

            Section("Smoke Tests") {
                Button("Run ONNX Smoke Test") {
                    SmokeTest.runSileroVADLoadTest()
                }
                Button("Run VAD Sanity Inference") {
                    SmokeTest.runSileroVADInferenceSanity()
                }
                Button("Run Moonshine on latest utterance") {
                    // Dispatch off the main actor — AVAudioFile and ORT session
                    // runs are both expensive I/O that complain loudly when
                    // invoked on the UI thread.
                    Task.detached(priority: .userInitiated) {
                        await SmokeTest.runMoonshineInference()
                    }
                }
                Button("Benchmark Moonshine (CoreML vs CPU)") {
                    Task.detached(priority: .userInitiated) {
                        await SmokeTest.runMoonshineBenchmark()
                    }
                }
            }

            Section("Install") {
                Button(role: .destructive) {
                    appState.resetInstall()
                } label: {
                    Text("Reinstall Models")
                }
            }
        }
        .navigationTitle("Developer")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    ContentView()
        .environment(AppState.previewing(.ready))
}
