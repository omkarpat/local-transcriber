import Foundation
import Synchronization

/// Lock-free single-producer / single-consumer ring buffer for Float32 samples.
///
/// Producer: the AVAudioEngine tap (real-time audio thread) calls `write`.
/// Consumer: the VAD processing thread calls `read`.
///
/// If the consumer falls more than `capacity` samples behind, the producer
/// overwrites the oldest samples and increments `overflowSamples`. This
/// prioritizes keeping recent audio over retaining stale audio.
nonisolated final class CircularAudioBuffer: @unchecked Sendable {
    let capacity: Int
    private let mask: Int
    private let storage: UnsafeMutableBufferPointer<Float>

    private let writeIndex = Atomic<Int>(0)
    private let readIndex = Atomic<Int>(0)
    private let overflowCount = Atomic<Int>(0)

    init(capacity requested: Int) {
        precondition(requested > 0, "capacity must be positive")
        var cap = 1
        while cap < requested { cap <<= 1 }
        self.capacity = cap
        self.mask = cap - 1
        let raw = UnsafeMutablePointer<Float>.allocate(capacity: cap)
        raw.initialize(repeating: 0, count: cap)
        self.storage = UnsafeMutableBufferPointer(start: raw, count: cap)
    }

    deinit {
        storage.baseAddress?.deinitialize(count: capacity)
        storage.deallocate()
    }

    var availableToRead: Int {
        let w = writeIndex.load(ordering: .acquiring)
        let r = readIndex.load(ordering: .relaxed)
        return Swift.min(w - r, capacity)
    }

    var overflowSamples: Int {
        overflowCount.load(ordering: .relaxed)
    }

    /// Producer-side. Called from the real-time audio thread. Never blocks.
    func write(_ samples: UnsafePointer<Float>, count: Int) {
        guard count > 0 else { return }
        let w = writeIndex.load(ordering: .relaxed)
        let base = storage.baseAddress!

        let startSlot = w & mask
        let firstChunk = Swift.min(count, capacity - startSlot)
        base.advanced(by: startSlot).update(from: samples, count: firstChunk)
        if firstChunk < count {
            base.update(from: samples.advanced(by: firstChunk), count: count - firstChunk)
        }

        writeIndex.store(w + count, ordering: .releasing)

        let r = readIndex.load(ordering: .relaxed)
        let used = (w + count) - r
        if used > capacity {
            overflowCount.wrappingAdd(used - capacity, ordering: .relaxed)
        }
    }

    /// Consumer-side. Returns the number of samples actually copied (≤ maxCount).
    func read(into dst: UnsafeMutablePointer<Float>, maxCount: Int) -> Int {
        guard maxCount > 0 else { return 0 }
        let base = storage.baseAddress!

        while true {
            let w = writeIndex.load(ordering: .acquiring)
            let r = readIndex.load(ordering: .relaxed)
            let behind = Swift.max(0, (w - capacity) - r)
            let effectiveR = r + behind
            let available = w - effectiveR
            let n = Swift.min(maxCount, available)
            if n == 0 { return 0 }

            let startSlot = effectiveR & mask
            let firstChunk = Swift.min(n, capacity - startSlot)
            dst.update(from: base.advanced(by: startSlot), count: firstChunk)
            if firstChunk < n {
                dst.advanced(by: firstChunk).update(from: base, count: n - firstChunk)
            }

            let wAfter = writeIndex.load(ordering: .acquiring)
            if wAfter - effectiveR > capacity {
                continue
            }

            readIndex.store(effectiveR + n, ordering: .releasing)
            return n
        }
    }
}
