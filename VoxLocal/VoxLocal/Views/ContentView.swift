//
//  ContentView.swift
//  VoxLocal
//
//  Created by Omkar Patil on 4/16/26.
//

import SwiftUI

struct ContentView: View {
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
            }
            .navigationTitle("VoxLocal")
        }
    }
}

#Preview {
    ContentView()
}
