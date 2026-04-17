import AVFoundation

enum MicrophonePermission {
    case undetermined
    case denied
    case granted

    static var current: MicrophonePermission {
        switch AVAudioApplication.shared.recordPermission {
        case .undetermined: return .undetermined
        case .denied: return .denied
        case .granted: return .granted
        @unknown default: return .denied
        }
    }

    static func request() async -> MicrophonePermission {
        if case let state = current, state != .undetermined {
            return state
        }
        let granted = await AVAudioApplication.requestRecordPermission()
        return granted ? .granted : .denied
    }
}
