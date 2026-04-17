//
//  VoxLocalApp.swift
//  VoxLocal
//
//  Created by Omkar Patil on 4/16/26.
//

import SwiftUI

@main
struct VoxLocalApp: App {
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(appState)
                .task { appState.bootstrap() }
        }
    }
}

/// Gates the main app UI behind the first-run model install. Everything
/// downstream can assume models are present once `ContentView` renders.
private struct RootView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        switch appState.installState {
        case .ready:
            ContentView()
        default:
            ModelDownloadView()
        }
    }
}
