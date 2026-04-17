import Foundation
import os

/// HTTP client for the cloud punctuation service (see
/// `punctuation-service-plan.md`). Fire-and-forget semantics: returns an
/// optional `TranscriptPunctuated` on success, or `nil` on any failure
/// (timeout, network error, non-2xx, malformed body, empty input). The
/// caller is expected to fall back silently to the raw local transcript
/// when `nil` — no errors propagate to the UI.
///
/// Marked `nonisolated` because the pipeline dispatches into it from a
/// `Task.detached`; no UI work happens here.
nonisolated final class PunctuationClient: @unchecked Sendable {
    private let config: PunctuationConfig
    private let session: URLSession
    private let log = Logger(subsystem: "com.omkarpatil.VoxLocal", category: "Punctuation")

    init(config: PunctuationConfig = .default) {
        self.config = config
        let sessionConfig = URLSessionConfiguration.ephemeral
        // 2 s request budget — past this the caller considers the server
        // unavailable and keeps the raw local transcript. Resource budget
        // is slightly larger to absorb redirects / slow first-byte.
        sessionConfig.timeoutIntervalForRequest = 2.0
        sessionConfig.timeoutIntervalForResource = 4.0
        sessionConfig.waitsForConnectivity = false
        self.session = URLSession(configuration: sessionConfig)
    }

    /// POST `text` to `/punctuate` and return the polished version if the
    /// service answers cleanly inside the timeout. Returns `nil` on any
    /// failure (caller uses this as the fallback signal).
    func punctuate(
        utteranceID: UUID,
        text: String,
        utteranceDuration: Duration
    ) async -> TranscriptPunctuated? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        var request = URLRequest(url: config.baseURL.appendingPathComponent("punctuate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(config.apiKey, forHTTPHeaderField: "X-API-Key")
        do {
            request.httpBody = try JSONEncoder().encode(PunctuateRequest(text: text))
        } catch {
            log.error("request encode failed: \(String(describing: error), privacy: .public)")
            return nil
        }

        let start = Date()
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                log.error("non-HTTP response")
                return nil
            }
            guard (200..<300).contains(http.statusCode) else {
                log.error("HTTP \(http.statusCode, privacy: .public)")
                return nil
            }
            let decoded = try JSONDecoder().decode(PunctuateResponse.self, from: data)
            let elapsed = Duration.seconds(Date().timeIntervalSince(start))
            return TranscriptPunctuated(
                id: UUID(),
                utteranceID: utteranceID,
                text: decoded.text,
                sourceText: text,
                utteranceDuration: utteranceDuration,
                roundTripDuration: elapsed
            )
        } catch {
            log.error("\(String(describing: error), privacy: .public)")
            return nil
        }
    }
}

private nonisolated struct PunctuateRequest: Encodable {
    let text: String
}

private nonisolated struct PunctuateResponse: Decodable {
    let text: String
}
