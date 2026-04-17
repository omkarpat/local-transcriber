import Foundation
import OnnxRuntimeBindings

enum MoonshineError: Error, CustomStringConvertible {
    case modelFileMissing(String)
    case unexpectedEncoderShape([Int])
    case unexpectedLogitsShape([Int])
    case missingOutput(String)

    var description: String {
        switch self {
        case .modelFileMissing(let name): return "Moonshine model file missing: \(name)"
        case .unexpectedEncoderShape(let s): return "Encoder produced unexpected shape \(s)"
        case .unexpectedLogitsShape(let s): return "Decoder produced unexpected logits shape \(s)"
        case .missingOutput(let n): return "Decoder output missing tensor '\(n)'"
        }
    }
}

/// Moonshine Tiny ASR wrapper.
///
/// Runs an encoder + a split decoder pair (no-cache for the first token,
/// with-past for subsequent tokens). Greedy decode only; no tokenizer yet —
/// `transcribe(samples:)` returns the raw Int64 token IDs for Task 4.2a.
/// See phase1-plan.md Task 4.1 for the full tensor I/O spec.
///
/// Not thread-safe. Intended to be driven from a single consumer task.
/// `nonisolated` so it can run off the main actor (inference is
/// compute-heavy — never block the UI).
nonisolated final class MoonshineModel: @unchecked Sendable {
    // From config.json (2026-04-16 snapshot); see phase1-plan.md Task 4.1.
    static let decoderStartTokenID: Int64 = 1
    static let eosTokenID: Int64 = 2
    static let maxGeneratedTokens: Int = 512
    static let vocabSize: Int = 32_768
    static let encoderHiddenSize: Int = 288
    static let numDecoderLayers: Int = 6
    static let kvHeads: Int = 8
    static let headDim: Int = 36

    private let encoder: ORTSession
    private let decoderNoCache: ORTSession
    private let decoderWithPast: ORTSession

    init(bundle: ModelBundle = ModelAssets.moonshineTiny, useCoreML: Bool = true) throws {
        func loadSession(_ filename: String) throws -> ORTSession {
            let url = try ModelAssets.fileURL(in: bundle, named: filename)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MoonshineError.modelFileMissing(filename)
            }
            let opts = try (useCoreML
                ? OnnxRuntimeSetup.makeSessionOptions()
                : OnnxRuntimeSetup.makeCPUSessionOptions())
            return try ORTSession(env: OnnxRuntimeSetup.shared, modelPath: url.path, sessionOptions: opts)
        }
        self.encoder = try loadSession("encoder_model.onnx")
        self.decoderNoCache = try loadSession("decoder_model.onnx")
        self.decoderWithPast = try loadSession("decoder_with_past_model.onnx")
    }

    /// Greedy-decode a single utterance and return the generated token IDs
    /// (not including the decoder-start token, up to and including EOS if
    /// reached, capped at `maxGeneratedTokens`).
    func transcribe(samples: [Float]) throws -> [Int64] {
        let encoderOutput = try runEncoder(samples: samples)
        return try decodeGreedy(encoderHiddenStates: encoderOutput)
    }

    // MARK: - Encoder

    private struct EncoderOutput {
        let tensor: ORTValue            // keeps the backing bytes alive
        let data: NSMutableData         // same bytes, reusable across decoder calls
        let encoderSeqLen: Int
    }

    private func runEncoder(samples: [Float]) throws -> EncoderOutput {
        let byteCount = samples.count * MemoryLayout<Float>.size
        let inputData = NSMutableData(length: byteCount)!
        samples.withUnsafeBufferPointer { buf in
            _ = memcpy(inputData.mutableBytes, buf.baseAddress!, byteCount)
        }
        let inputShape: [NSNumber] = [1, NSNumber(value: samples.count)]
        let inputTensor = try ORTValue(tensorData: inputData, elementType: .float, shape: inputShape)

        let outputs = try encoder.run(
            withInputs: ["input_values": inputTensor],
            outputNames: ["last_hidden_state"],
            runOptions: nil
        )
        guard let hidden = outputs["last_hidden_state"] else {
            throw MoonshineError.missingOutput("last_hidden_state")
        }
        let info = try hidden.tensorTypeAndShapeInfo()
        let shape = info.shape.map { $0.intValue }
        guard shape.count == 3, shape[0] == 1, shape[2] == Self.encoderHiddenSize else {
            throw MoonshineError.unexpectedEncoderShape(shape)
        }
        let hsData = try hidden.tensorData() as Data
        let mutable = NSMutableData(data: hsData)
        return EncoderOutput(tensor: hidden, data: mutable, encoderSeqLen: shape[1])
    }

    // MARK: - Decoder

    /// Holds the evolving decoder KV cache across steps.
    /// - Self-attn K/V grow by 1 along the sequence dim each step.
    /// - Cross-attn K/V are fixed after the first call (depends only on encoder output).
    private struct KVCache {
        var selfKeys:  [NSMutableData]   // 6 layers, shape [1, 8, N, 36]
        var selfVals:  [NSMutableData]
        var crossKeys: [NSMutableData]   // 6 layers, shape [1, 8, T_enc, 36]
        var crossVals: [NSMutableData]
        var selfSeqLen: Int               // N (grows per step)
        let crossSeqLen: Int              // T_enc (fixed)
    }

    private func decodeGreedy(encoderHiddenStates encOut: EncoderOutput) throws -> [Int64] {
        var tokens: [Int64] = []

        // --- First call: decoder_model.onnx, takes input_ids + encoder_hidden_states,
        //     emits logits and the full 24-tensor KV cache (self + cross).
        let encoderShape: [NSNumber] = [
            1,
            NSNumber(value: encOut.encoderSeqLen),
            NSNumber(value: Self.encoderHiddenSize),
        ]
        let encoderTensor = try ORTValue(tensorData: encOut.data,
                                         elementType: .float,
                                         shape: encoderShape)

        let firstInputIDs: [Int64] = [Self.decoderStartTokenID]
        let firstInputData = makeInt64Data(firstInputIDs)
        let firstInputTensor = try ORTValue(tensorData: firstInputData,
                                            elementType: .int64,
                                            shape: [1, NSNumber(value: firstInputIDs.count)])

        var firstOutputNames: Set<String> = ["logits"]
        for layer in 0..<Self.numDecoderLayers {
            firstOutputNames.insert("present.\(layer).decoder.key")
            firstOutputNames.insert("present.\(layer).decoder.value")
            firstOutputNames.insert("present.\(layer).encoder.key")
            firstOutputNames.insert("present.\(layer).encoder.value")
        }
        let firstOutputs = try decoderNoCache.run(
            withInputs: ["input_ids": firstInputTensor, "encoder_hidden_states": encoderTensor],
            outputNames: firstOutputNames,
            runOptions: nil
        )

        let firstToken = try argmaxLastPosition(logitsValue: try required(firstOutputs, "logits"))
        tokens.append(firstToken)
        if firstToken == Self.eosTokenID { return tokens }

        // Seed the KV cache from the first call's present.* outputs.
        var cache = try buildKVCache(fromFirstCallOutputs: firstOutputs)

        // --- Loop: decoder_with_past_model.onnx, takes one new token + full past KVs.
        var currentToken = firstToken
        while tokens.count < Self.maxGeneratedTokens {
            let inputIDData = makeInt64Data([currentToken])
            let inputIDTensor = try ORTValue(tensorData: inputIDData,
                                             elementType: .int64,
                                             shape: [1, 1])

            var inputs: [String: ORTValue] = ["input_ids": inputIDTensor]
            let selfShape: [NSNumber] = [1, NSNumber(value: Self.kvHeads),
                                         NSNumber(value: cache.selfSeqLen),
                                         NSNumber(value: Self.headDim)]
            let crossShape: [NSNumber] = [1, NSNumber(value: Self.kvHeads),
                                          NSNumber(value: cache.crossSeqLen),
                                          NSNumber(value: Self.headDim)]
            for layer in 0..<Self.numDecoderLayers {
                inputs["past_key_values.\(layer).decoder.key"] = try ORTValue(
                    tensorData: cache.selfKeys[layer], elementType: .float, shape: selfShape)
                inputs["past_key_values.\(layer).decoder.value"] = try ORTValue(
                    tensorData: cache.selfVals[layer], elementType: .float, shape: selfShape)
                inputs["past_key_values.\(layer).encoder.key"] = try ORTValue(
                    tensorData: cache.crossKeys[layer], elementType: .float, shape: crossShape)
                inputs["past_key_values.\(layer).encoder.value"] = try ORTValue(
                    tensorData: cache.crossVals[layer], elementType: .float, shape: crossShape)
            }

            // decoder_with_past emits logits + 12 present.decoder.* (self-attn only;
            // cross-attn KVs are invariant so not re-emitted).
            var outputNames: Set<String> = ["logits"]
            for layer in 0..<Self.numDecoderLayers {
                outputNames.insert("present.\(layer).decoder.key")
                outputNames.insert("present.\(layer).decoder.value")
            }
            let outputs = try decoderWithPast.run(withInputs: inputs,
                                                  outputNames: outputNames,
                                                  runOptions: nil)

            let nextToken = try argmaxLastPosition(logitsValue: try required(outputs, "logits"))
            tokens.append(nextToken)
            if nextToken == Self.eosTokenID { break }

            // Update self-attn cache (grows by 1 along seq dim); cross-attn untouched.
            for layer in 0..<Self.numDecoderLayers {
                let kT = try required(outputs, "present.\(layer).decoder.key")
                let vT = try required(outputs, "present.\(layer).decoder.value")
                cache.selfKeys[layer] = NSMutableData(data: try kT.tensorData() as Data)
                cache.selfVals[layer] = NSMutableData(data: try vT.tensorData() as Data)
            }
            cache.selfSeqLen += 1
            currentToken = nextToken
        }

        return tokens
    }

    // MARK: - Helpers

    private func makeInt64Data(_ values: [Int64]) -> NSMutableData {
        let byteCount = values.count * MemoryLayout<Int64>.size
        let data = NSMutableData(length: byteCount)!
        values.withUnsafeBufferPointer { buf in
            _ = memcpy(data.mutableBytes, buf.baseAddress!, byteCount)
        }
        return data
    }

    private func required(_ map: [String: ORTValue], _ name: String) throws -> ORTValue {
        guard let v = map[name] else { throw MoonshineError.missingOutput(name) }
        return v
    }

    /// Argmax across the vocab dimension of the final sequence position of `logits`.
    /// Expected shape: [1, seq_len, vocab_size].
    private func argmaxLastPosition(logitsValue: ORTValue) throws -> Int64 {
        let info = try logitsValue.tensorTypeAndShapeInfo()
        let shape = info.shape.map { $0.intValue }
        guard shape.count == 3, shape[0] == 1, shape[2] == Self.vocabSize else {
            throw MoonshineError.unexpectedLogitsShape(shape)
        }
        let data = try logitsValue.tensorData() as Data
        let seqLen = shape[1]
        let lastPositionStart = (seqLen - 1) * Self.vocabSize
        return data.withUnsafeBytes { raw -> Int64 in
            let floats = raw.bindMemory(to: Float.self)
            var bestIdx: Int = 0
            var bestVal: Float = -.infinity
            for i in 0..<Self.vocabSize {
                let v = floats[lastPositionStart + i]
                if v > bestVal { bestVal = v; bestIdx = i }
            }
            return Int64(bestIdx)
        }
    }

    private func buildKVCache(fromFirstCallOutputs outputs: [String: ORTValue]) throws -> KVCache {
        var selfKeys:  [NSMutableData] = []
        var selfVals:  [NSMutableData] = []
        var crossKeys: [NSMutableData] = []
        var crossVals: [NSMutableData] = []
        var selfSeqLen: Int = -1
        var crossSeqLen: Int = -1
        for layer in 0..<Self.numDecoderLayers {
            let sK = try required(outputs, "present.\(layer).decoder.key")
            let sV = try required(outputs, "present.\(layer).decoder.value")
            let cK = try required(outputs, "present.\(layer).encoder.key")
            let cV = try required(outputs, "present.\(layer).encoder.value")
            selfKeys.append(NSMutableData(data: try sK.tensorData() as Data))
            selfVals.append(NSMutableData(data: try sV.tensorData() as Data))
            crossKeys.append(NSMutableData(data: try cK.tensorData() as Data))
            crossVals.append(NSMutableData(data: try cV.tensorData() as Data))
            let sShape = (try sK.tensorTypeAndShapeInfo()).shape.map { $0.intValue }
            let cShape = (try cK.tensorTypeAndShapeInfo()).shape.map { $0.intValue }
            if selfSeqLen < 0 { selfSeqLen = sShape[2] }
            if crossSeqLen < 0 { crossSeqLen = cShape[2] }
        }
        return KVCache(
            selfKeys: selfKeys, selfVals: selfVals,
            crossKeys: crossKeys, crossVals: crossVals,
            selfSeqLen: selfSeqLen, crossSeqLen: crossSeqLen
        )
    }
}
