import Foundation

/// Endpoint + credential config for the punctuation service.
///
/// In production the values are injected at build time via an
/// `.xcconfig` that feeds `Info.plist` keys `PunctuationBaseURL` and
/// `PunctuationAPIKey`. For local development (no xcconfig set up yet)
/// we fall back to a localhost FastAPI stub and a throwaway dev key.
/// The fallback URL is deliberately localhost so an accidental
/// uncommitted config reaching prod still can't leak text anywhere.
nonisolated struct PunctuationConfig: Sendable {
    let baseURL: URL
    let apiKey: String

    static let `default` = PunctuationConfig.loadFromBundle()

    private static func loadFromBundle() -> PunctuationConfig {
        // `127.0.0.1` (not `localhost`) so we bypass IPv6 resolution on
        // the iOS simulator — uvicorn binds IPv4 only by default and
        // `localhost` can resolve to `::1` first, producing an
        // immediate Connection Refused.
        let urlString = (Bundle.main.object(forInfoDictionaryKey: "PunctuationBaseURL") as? String)
            ?? "http://127.0.0.1:8000"
        let apiKey = (Bundle.main.object(forInfoDictionaryKey: "PunctuationAPIKey") as? String)
            ?? "dev-key-change-me"
        let url = URL(string: urlString) ?? URL(string: "http://127.0.0.1:8000")!
        return PunctuationConfig(baseURL: url, apiKey: apiKey)
    }
}
