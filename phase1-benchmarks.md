# Phase 1 Benchmarks

Measurements are collected by `SmokeTest.runMoonshineBenchmark()` (main-screen button "Benchmark Moonshine (CoreML vs CPU)"). Each run uses the most-recent `utterance-*.wav` the VAD wrote to `Documents/`, discards one warm-up pass per configuration, and reports min / median / mean / p95 over 5 iterations.

Per-phase breakdown:
- **encoder** — one pass of `encoder_model.onnx`.
- **1st-decode** — one pass of `decoder_model.onnx` (no-cache) to produce the first token and seed the KV cache.
- **cache-loop** — aggregate time across all `decoder_with_past_model.onnx` steps until EOS or max-length.
- **total** — wall time of the whole `transcribe(samples:)` call.

## Simulator — iPhone 17 Pro, iOS 26.4, Mac Silicon host

Captured 2026-04-17. Utterance 1.39s / 3 tokens.

| Phase          | CoreML EP (median / p95)   | CPU only (median / p95)    |
|----------------|----------------------------|----------------------------|
| Session load   | 1.68 s                     | 0.21 s                     |
| total          | 0.051 s / 0.054 s  **RTF 0.04** | 0.018 s / 0.020 s  **RTF 0.01** |
| encoder        | 0.012 s / 0.013 s          | 0.004 s / 0.007 s          |
| 1st-decode     | 0.007 s / 0.007 s          | 0.003 s / 0.003 s          |
| cache-loop     | 0.025 s / 0.025 s          | 0.003 s / 0.003 s          |

**Observations**:
- CPU-only beats CoreML EP on the simulator, by both load time (~8×) and inference (~3×). This is expected — the simulator's CoreML has no real Neural Engine to dispatch to, so CoreML EP just pays partitioning overhead without the ANE acceleration payoff.
- Both configurations sit far below the Phase 1 target of RTF < 0.5. Moonshine Tiny is genuinely small.
- CoreML EP partitions per session: encoder 299/569 nodes, decoder (no-cache) 397/805, decoder (with past) 351/649 — roughly half of nodes get routed to CoreML; the rest fall back to CPU for shape-related ops. Not directly tuneable without custom op fusion.
- The test utterance was short (3 tokens). Longer transcripts would amplify the cache-loop share of total time and make per-token decode cost more visible.

## Device — iPhone 14+ with real ANE

Pending. Captured in Task 7 (physical device testing & battery profiling). Expectation: CoreML EP flips to faster than CPU once the Neural Engine is actually available; first-run cold-load will also benefit from cached CoreML compile artifacts.

## Time-to-first-token

For our batch-per-utterance architecture, TTFT = `encoder + 1st-decode`. On the simulator that's ~19 ms (CoreML) / ~7 ms (CPU). Irrelevant in practice until streaming partials (plan.md Task 4.3) are wired in — right now the user only sees output after VAD closes the utterance (≥ 700 ms of silence), so pipeline latency is dominated by the VAD hangover, not model inference.

## Known simulator-only noise (ignore on device)

- `objc[...]: Class AKAlertImageURLProvider is implemented in both ... AuthKit.framework / AuthKitUI.framework` — known iOS 26.4 simulator glitch, unrelated to our code.
- `CoreMLExecutionProvider::GetCapability` / `VerifyEachNodeIsAssignedToAnEp` `[W:onnxruntime:]` lines — informational ORT logging about partial CoreML coverage; not errors.
