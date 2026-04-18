import Foundation
import os

/// HTTP client for the cloud refinement service (see
/// `punctuation-service-v2-plan.md`). Talks to `/punctuate/stream` via
/// Server-Sent Events and yields one `PunctuationStageEvent` per pipeline
/// stage as events arrive. The stream ends cleanly after the server
/// emits `final: true`; on any failure (timeout, non-2xx, malformed
/// frame, empty input) the stream just ends with no events or an
/// incomplete sequence. Caller uses `isFinal` to distinguish a completed
/// stream from a failed one — no errors propagate to the UI.
///
/// Marked `nonisolated` because the pipeline dispatches into it from a
/// `Task.detached` (first-try) and from the actor's drain loop. No UI
/// work happens here.
nonisolated final class PunctuationClient: @unchecked Sendable {
    private let config: PunctuationConfig
    private let session: URLSession
    private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "Punctuation")

    init(config: PunctuationConfig = .default) {
        self.config = config
        let sessionConfig = URLSessionConfiguration.ephemeral
        // Per-request budget: 4 s for the entire stream. Individual stage
        // events are expected within ~120 ms on a healthy server; 4 s is
        // generous tail tolerance before we give up and fall back to raw
        // text. Resource budget slightly larger to absorb slow first-byte.
        sessionConfig.timeoutIntervalForRequest = 4.0
        sessionConfig.timeoutIntervalForResource = 8.0
        sessionConfig.waitsForConnectivity = false
        self.session = URLSession(configuration: sessionConfig)
    }

    /// POST `text` to `/punctuate/stream` and yield a `PunctuationStageEvent`
    /// for every `event: stage` frame the server emits. The returned
    /// `AsyncStream` finishes when the server emits an event with
    /// `final: true`, when the connection drops, or when the consumer
    /// stops iterating (which cancels the underlying HTTP request).
    func punctuateStream(
        utteranceID: UUID,
        segmentID: UUID,
        text: String,
        silenceBeforeMs: Int? = nil
    ) -> AsyncStream<PunctuationStageEvent> {
        AsyncStream { continuation in
            let task = Task { [config, session, log] in
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else {
                    continuation.finish()
                    return
                }

                var request = URLRequest(
                    url: config.baseURL.appendingPathComponent("punctuate/stream")
                )
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                request.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key")
                do {
                    request.httpBody = try JSONEncoder().encode(StreamRequest(
                        text: text,
                        utteranceId: utteranceID,
                        segmentId: segmentID,
                        silenceBeforeMs: silenceBeforeMs
                    ))
                } catch {
                    log.error("stream encode failed: \(String(describing: error), privacy: .public)")
                    continuation.finish()
                    return
                }

                do {
                    let (bytes, response) = try await session.bytes(for: request)
                    guard
                        let http = response as? HTTPURLResponse,
                        (200..<300).contains(http.statusCode)
                    else {
                        if let http = response as? HTTPURLResponse {
                            log.error("stream HTTP \(http.statusCode, privacy: .public)")
                        } else {
                            log.error("stream non-HTTP response")
                        }
                        continuation.finish()
                        return
                    }

                    // SSE parser: fields are newline-delimited; a blank
                    // line dispatches the buffered event. Anything other
                    // than `event: stage` gets silently discarded — the
                    // client's contract is silent fallback on the UI side,
                    // so unknown events and errors just end the stream
                    // without emitting.
                    var currentEvent = "message"
                    var dataBuffer = ""

                    for try await line in bytes.lines {
                        if Task.isCancelled { break }

                        if line.isEmpty {
                            if !dataBuffer.isEmpty {
                                if currentEvent == "stage",
                                   let event = Self.decodeStage(dataBuffer) {
                                    continuation.yield(event)
                                    if event.isFinal {
                                        continuation.finish()
                                        return
                                    }
                                } else if currentEvent == "error" {
                                    log.error("server stream error: \(dataBuffer, privacy: .public)")
                                }
                            }
                            currentEvent = "message"
                            dataBuffer = ""
                            continue
                        }

                        // SSE comments start with ':' — skip entirely.
                        if line.hasPrefix(":") { continue }

                        // Field = "name: value" with an optional single
                        // leading space after the colon (per SSE spec).
                        if let colon = line.firstIndex(of: ":") {
                            let field = String(line[..<colon])
                            var value = String(line[line.index(after: colon)...])
                            if value.hasPrefix(" ") { value.removeFirst() }
                            switch field {
                            case "event": currentEvent = value
                            case "data":  dataBuffer += value
                            default:      break   // id:, retry:, etc.
                            }
                        }
                    }
                } catch {
                    log.error("stream read failed: \(String(describing: error), privacy: .public)")
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Parse one `data:` payload into a stage event. Returns nil on
    /// malformed JSON or missing required fields — the stream proceeds
    /// without emitting that event (silent fallback contract).
    private static func decodeStage(_ dataJSON: String) -> PunctuationStageEvent? {
        struct Wire: Decodable {
            let utteranceID: UUID
            let segmentID: UUID
            let stage: String
            let text: String
            let final: Bool

            enum CodingKeys: String, CodingKey {
                case utteranceID = "utterance_id"
                case segmentID = "segment_id"
                case stage, text, final
            }
        }
        guard let data = dataJSON.data(using: .utf8),
              let w = try? JSONDecoder().decode(Wire.self, from: data)
        else { return nil }
        return PunctuationStageEvent(
            utteranceID: w.utteranceID,
            segmentID: w.segmentID,
            stage: w.stage,
            text: w.text,
            isFinal: w.final
        )
    }
}

/// One event from the streaming refinement pipeline. Stage names are
/// opaque strings from the server (`"commands"`, `"punct_case"`, `"itn"`
/// today; more may come). Clients should not switch on them to drive
/// behavior — just render the text and watch `isFinal`.
nonisolated struct PunctuationStageEvent: Sendable, Equatable {
    let utteranceID: UUID
    let segmentID: UUID
    let stage: String
    let text: String
    let isFinal: Bool
}

private nonisolated struct StreamRequest: Encodable {
    let text: String
    let utteranceId: UUID
    let segmentId: UUID
    let silenceBeforeMs: Int?

    enum CodingKeys: String, CodingKey {
        case text
        case utteranceId = "utterance_id"
        case segmentId = "segment_id"
        case silenceBeforeMs = "silence_before_ms"
    }
}
