import AVFoundation
import Foundation

enum WAVReaderError: Error, CustomStringConvertible {
    case fileNotFound(URL)
    case readFailed(String)

    var description: String {
        switch self {
        case .fileNotFound(let url): return "WAV file not found: \(url.path)"
        case .readFailed(let msg): return "WAV read failed: \(msg)"
        }
    }
}

/// Reads a mono 16-bit PCM WAV file (produced by `WAVWriter`) back into
/// Float32 samples suitable for Moonshine's `input_values` tensor.
/// Uses `AVAudioFile` so the 16-bit → Float32 conversion is handled by
/// CoreAudio; we don't parse the RIFF header by hand.
nonisolated enum WAVReader {
    static func readFloat32Samples(from url: URL) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WAVReaderError.fileNotFound(url)
        }
        do {
            let file = try AVAudioFile(forReading: url,
                                       commonFormat: .pcmFormatFloat32,
                                       interleaved: false)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                                frameCapacity: AVAudioFrameCount(file.length)) else {
                throw WAVReaderError.readFailed("could not allocate PCM buffer")
            }
            try file.read(into: buffer)
            guard let channelData = buffer.floatChannelData else {
                throw WAVReaderError.readFailed("buffer has no float channel data")
            }
            let count = Int(buffer.frameLength)
            return Array(UnsafeBufferPointer(start: channelData[0], count: count))
        } catch let error as WAVReaderError {
            throw error
        } catch {
            throw WAVReaderError.readFailed(error.localizedDescription)
        }
    }

    /// Returns the URL of the most recently modified `utterance-*.wav` file
    /// in the Documents directory, or nil if none exists.
    /// Used by the smoke test to grab the last thing the VAD captured so we
    /// can transcribe it without another recording pass.
    static func mostRecentUtterance(in directory: URL = WAVWriter.documentsDirectory()) -> URL? {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        let utterances = entries.filter { url in
            url.lastPathComponent.hasPrefix("utterance-") &&
            url.pathExtension.lowercased() == "wav"
        }
        return utterances.max { a, b in
            let aDate = (try? a.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            let bDate = (try? b.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            return aDate < bDate
        }
    }
}
