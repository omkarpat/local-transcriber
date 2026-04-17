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
            List {
                NavigationLink("Audio Capture") {
                    AudioCaptureDebugView()
                }
                Button("Run ONNX Smoke Test") {
                    SmokeTest.runSileroVADLoadTest()
                }
                Button("Run VAD Sanity Inference") {
                    SmokeTest.runSileroVADInferenceSanity()
                }

                Section("Developer") {
                    Button(role: .destructive) {
                        appState.resetInstall()
                    } label: {
                        Text("Reinstall Models")
                    }
                }
            }
            .navigationTitle("VoxLocal")
        }
    }
}

#Preview {
    ContentView()
}
