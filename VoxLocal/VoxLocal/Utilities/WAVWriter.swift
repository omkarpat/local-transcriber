import Foundation

nonisolated enum WAVWriter {
    /// Write Float32 mono samples (range ≈ [-1, 1]) to a 16kHz / 16-bit PCM WAV file.
    /// Returns the URL that was written.
    @discardableResult
    static func write(samples: [Float],
                      sampleRate: Int = Int(AudioFormat.sampleRate),
                      to url: URL) throws -> URL {
        let bitsPerSample: UInt16 = 16
        let numChannels: UInt16 = 1
        let byteRate: UInt32 = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign: UInt16 = numChannels * bitsPerSample / 8
        let dataSize: UInt32 = UInt32(samples.count) * UInt32(blockAlign)
        let riffSize: UInt32 = 36 + dataSize

        var data = Data(capacity: Int(riffSize) + 8)
        data.append("RIFF".data(using: .ascii)!)
        data.append(le(riffSize))
        data.append("WAVE".data(using: .ascii)!)

        data.append("fmt ".data(using: .ascii)!)
        data.append(le(UInt32(16)))               // fmt chunk size
        data.append(le(UInt16(1)))                // PCM
        data.append(le(numChannels))
        data.append(le(UInt32(sampleRate)))
        data.append(le(byteRate))
        data.append(le(blockAlign))
        data.append(le(bitsPerSample))

        data.append("data".data(using: .ascii)!)
        data.append(le(dataSize))

        data.reserveCapacity(data.count + samples.count * 2)
        for sample in samples {
            let clipped = max(-1.0, min(1.0, sample))
            let scaled = Int16(clipped * Float(Int16.max))
            data.append(le(UInt16(bitPattern: scaled)))
        }

        try data.write(to: url, options: .atomic)
        return url
    }

    static func documentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private static func le(_ v: UInt16) -> Data {
        withUnsafeBytes(of: v.littleEndian) { Data($0) }
    }

    private static func le(_ v: UInt32) -> Data {
        withUnsafeBytes(of: v.littleEndian) { Data($0) }
    }
}
