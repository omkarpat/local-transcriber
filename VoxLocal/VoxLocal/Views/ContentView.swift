//
//  ContentView.swift
//  VoxLocal
//
//  Created by Omkar Patil on 4/16/26.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")

            Button("Run ONNX Smoke Test") {
                SmokeTest.runSileroVADLoadTest()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
