# Punctuation Service V2 — Scope & Implementation Plan

Builds on V1. Goal: make dictated output look like *written* text instead of *transcribed* text, and let the client render text that progressively improves as each stage completes. No new model training. CPU-only. Same ECS deployment shape.

## V2 Scope

Four features, all independently shippable:

1. **Inverse Text Normalization (ITN)** — spoken numbers/entities → written form
2. **Truecasing upgrade** — replace regex casing with NeMo joint Punctuation+Capitalization model
3. **Spoken punctuation commands** — "comma", "period", "new line" → actual marks
4. **Server-Sent Events (SSE) streaming contract** — one request emits a series of stage events so the client can render progressively-improved text instead of waiting for the full pipeline to complete

Out of scope (deferred):
- Custom dictionaries, command actions, style presets, disfluency removal, grammar correction, smart formatting, session-level state on the server

**Important framing:** V2 fixes *formatting* problems (missing punctuation, wrong casing, "twenty twenty six" instead of "2026"). It does *not* fix ASR recognition errors (homophones like "405" → "four or five", out-of-vocabulary terms, world-knowledge confusions). Those need to be solved at the ASR layer or with an LLM — see "What V2 Cannot Fix" below.

**Stateless posture (decided this session):** The server stays a pure function of its request. No session state, no conversation history, no rolling context. Paragraph-break decisions that need cross-utterance signals (e.g., long silence) are handled by having the **client supply the signal in the request** (`silence_before_ms`), not by the server maintaining memory. See "Paragraph Breaks & Segment IDs" below.

## Pipeline Architecture (V2)

```
Raw ASR text from Moonshine (client)
   │
   ▼
[Stage 1: Spoken Punctuation Commands]   ← rule-based, ~1ms
   │
   ▼
[Stage 2: Punctuation + Truecasing]      ← NeMo joint model, ~50-100ms
   │
   ▼
[Stage 3: ITN]                           ← WFST-based, ~5-20ms
   │
   ▼
Response
```

**Order matters:** spoken commands first (they're literal substitutions), then the model (so it sees natural English words, not digits or symbols), then ITN (operates on already-punctuated text). Reordering breaks things — see "Gotchas".

**Total p50 latency budget:** ~120ms. Same shape as V1, slightly heavier model.

---

## Wire Contract — SSE Streaming

### Why streaming

A single request synchronously running all three stages is ~100ms p50. That's fine, but extending to V3 (LLM fallback, 500ms–2s) would be painful to retrofit onto a single-response contract. Ship V2 as SSE from day one — the per-stage events cost almost nothing on localhost, and V3 adds one more event type instead of a breaking contract change.

SSE is the right edge protocol for this problem because the interaction is request-scoped, one-way, and stateless. Do not switch to WebSockets unless the product adds true bidirectional or sessionful behavior. Also note: SSE does **not** change the main scaling bottleneck here — model execution remains the thing that saturates first.

### Endpoint

Additive — V1 `/punctuate` stays for clients that haven't adopted streaming. V2 introduces `/punctuate/stream`.

```
POST /punctuate/stream
Headers: X-API-Key, Content-Type: application/json, Accept: text/event-stream
Body: {
  "text": "...",
  "utterance_id": "abc-123",
  "segment_id": "aaa-111",         // client's current running segment id
  "silence_before_ms": 2400        // optional; client-supplied signal for paragraph-break heuristics
}
```

Response is `text/event-stream`. Each event is `event: stage` with a JSON `data:` payload:

```
event: stage
data: {
  "utterance_id": "abc-123",
  "segment_id":   "aaa-111",
  "stage": "commands" | "punct_case" | "itn",
  "text": "...",
  "final": false
}
```

The last event of the stream has `"final": true`. On pipeline failure the server emits an `error` event and closes the stream — the client falls back silently to the raw local transcript, same posture as V1.

### Server implementation pointers

- FastAPI `StreamingResponse` with media_type `text/event-stream`. The generator must `await asyncio.to_thread(...)` for the blocking ONNX calls so bytes flush between stages; otherwise the loop stalls and the whole stream arrives at once, defeating the point.
- Response headers must include `Cache-Control: no-cache` and `X-Accel-Buffering: no` to defeat proxy/ALB buffering.
- Add a process-wide asyncio semaphore (~4-8 inflight) before rollout. On saturation, reject quickly with `429`/`503` rather than queueing unbounded work inside the FastAPI process.
- Check for disconnect before starting each stage. Once work has been dispatched into `asyncio.to_thread(...)`, the underlying ONNX inference is not cancellable in practice, so keep stages small and instrument abandoned work.
- If future stages can leave >30s gaps between bytes, emit SSE heartbeat comments (or raise ALB idle timeout) so the load balancer does not kill an otherwise-healthy stream.
- Track application-level stream failures explicitly. An SSE `error` event still rides on HTTP 200, so ALB success-rate dashboards alone are insufficient.

### Client implementation pointers (iOS)

- `URLSession.bytes(for:)` returns `AsyncBytes` which exposes a `.lines` async sequence. No third-party SSE library needed.
- Parse: blank line dispatches the buffered event; `event:` sets the type; `data:` (stripping one optional leading space) accumulates the JSON.
- Wrap the per-request Task in `AsyncStream.Continuation.onTermination` so cancellation propagates to `URLSession` and the server sees the disconnect.
- On failure (mid-stream network drop, error event, non-2xx), end the stream cleanly. Silent fallback — the raw transcript stays on screen.

---

## Paragraph Breaks & Segment IDs

### The two-ID model

Every refinement carries **two** identifiers:

- **`utterance_id`** — provenance. Which Moonshine utterance this text originated from. Stable for the life of the utterance; used for retries, debugging, "undo this utterance" features.
- **`segment_id`** — rendering identity. The unit the UI upserts against. The unit of a paragraph on screen.

For V2's shipping scope, a single utterance produces a single segment — `segment_id == utterance_id` by construction (or deterministically derived; see below). The UI renders one flowing paragraph per segment, growing as each utterance lands.

Paragraph splitting lives behind a server flag and ships when Stage 1 command detection is reliable: when the server detects a "new paragraph" within an utterance's text, it emits two events for that utterance with different `segment_id`s. The client keeps upserting by `segment_id` and naturally renders two paragraphs. **No client work needed to adopt splitting** beyond what already exists — that's the point of the two-ID contract.

### Deterministic segment IDs for retry idempotency

When a stream fails and the client's retry queue replays the utterance, the server sees the same `(utterance_id, incoming_segment_id, text)` and must produce the same outgoing segment IDs. Otherwise the first attempt's orphan segments persist in the client's rendering state.

Rule: any new segment id the server mints is `uuidv5(namespace=utterance_id, name=str(split_index))`. Reproducible across retries with zero server state.

### Client-supplied paragraph-break signals

The server never decides paragraph breaks on "long silence between utterances" because it doesn't see them. The client has the VAD timeline and supplies a `silence_before_ms` field. Stage 1 uses it as a secondary heuristic:

- Silence > threshold (e.g., 2s) → insert a paragraph break at the start of this utterance (mints a new segment id).
- Explicit "new paragraph" command → insert a paragraph break at that point within the utterance.

Both are local decisions — the request carries everything the server needs. The server stays a pure function of `(text, utterance_id, segment_id, silence_before_ms)`.

### Why the server never carries state

Considered and rejected: a stateful server that tracks session history and decides paragraph breaks globally. Costs: ALB stickiness (or shared Redis), task-churn fragility (state lost when ECS rotates), bigger privacy surface (transcripts retained beyond request), and harder V3 integration (LLM correction still benefits from context-as-parameter over session-memory). None of the benefits are load-bearing — client-supplied signals + deterministic IDs cover every V2 need.

### Client state model (for reference)

```swift
struct TranscriptRefinement {
    let id: UUID                 // this event's own id
    let utteranceID: UUID        // provenance
    let segmentID: UUID          // == utteranceID in V2 single-segment
    let text: String             // text for this segment at this stage
    let sourceText: String       // raw ASR, retained for debugging
    let utteranceDuration: Duration
    let roundTripDuration: Duration
}
```

UI upserts by `segmentID`: one paragraph per unique id. Today every utterance → one segment → one paragraph that grows with the session. When splitting ships, one utterance can emit multiple segment ids and the paragraph count grows naturally.

---

## Stage 1: Spoken Punctuation Commands

### What it does

Users dictating naturally say things like:
- "Send the report comma then call me period" → `"Send the report, then call me."`
- "Hi Sarah comma new line new paragraph thanks for the update period" → `"Hi Sarah,\n\nThanks for the update."`

This is muscle memory for dictation users. If V2 doesn't handle it, users feel the product is "dumber" than Dragon, Apple Dictation, etc.

### Approach

Rule-based preprocessor that runs on raw ASR output before the punctuation model. The hard problem isn't the substitution — it's *deciding when the word is a command vs the literal word*.

**Decision strategy: positional + lexical heuristics.** Word-timestamp-aware would be ideal but adds dependency on Moonshine emitting timing.

**Treat as command when:**
- Word appears alone in a short utterance (≤2 tokens)
- Word appears at the start or end of an utterance
- Word follows another command word ("comma new paragraph")
- Word is in a fixed set of "highly likely command" terms ("new line", "new paragraph" — almost never literal)

**Treat as literal when:**
- Word is mid-sentence with adjacent content words ("use a comma here", "the period of history")
- Surrounded by determiners/articles ("a comma", "the period")

Accept that this will misfire occasionally. Users can correct; misfires are visible and forgivable.

### Command map

```python
PUNCT_COMMANDS = {
    "comma": ",",
    "period": ".",
    "full stop": ".",
    "question mark": "?",
    "exclamation point": "!",
    "exclamation mark": "!",
    "colon": ":",
    "semicolon": ";",
    "dash": "—",
    "hyphen": "-",
    "open quote": "\"",
    "close quote": "\"",
    "open parenthesis": "(",
    "close parenthesis": ")",
    "ellipsis": "…",
}

LAYOUT_COMMANDS = {
    "new line": "\n",
    "new paragraph": "\n\n",
}

# Words that are almost never literal (high command probability)
HIGH_CONFIDENCE_COMMANDS = {"new line", "new paragraph", "exclamation point"}
```

### Implementation sketch

```python
import re

# Multi-word commands first to avoid partial matches
ALL_COMMANDS = sorted(
    {**PUNCT_COMMANDS, **LAYOUT_COMMANDS}.items(),
    key=lambda kv: -len(kv[0].split())
)

def apply_spoken_commands(text: str) -> str:
    tokens = text.lower().split()
    out = []
    i = 0
    while i < len(tokens):
        matched = False
        for phrase, replacement in ALL_COMMANDS:
            phrase_tokens = phrase.split()
            n = len(phrase_tokens)
            if tokens[i:i+n] == phrase_tokens and is_command_context(tokens, i, n, phrase):
                out.append(replacement)
                i += n
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def is_command_context(tokens, i, n, phrase):
    if phrase in HIGH_CONFIDENCE_COMMANDS:
        return True
    # Standalone short utterance
    if len(tokens) <= 2:
        return True
    # Preceded by article/determiner → literal
    if i > 0 and tokens[i-1] in {"a", "an", "the"}:
        return False
    # Default: treat as command
    return True
```

### Latency

Pure string ops, sub-millisecond. Negligible.

### Eval

Build a small labeled set (~200 utterances) split between command-uses and literal-uses of each ambiguous word ("comma", "period", "dash"). Track precision and recall separately. Aim for >95% precision on commands (false positives are jarring) even if recall is 80%.

---

## Stage 2: Truecasing Upgrade (NeMo Joint Model)

### What it does

Replaces V1's regex sentence-start casing with a model that handles:
- Proper nouns: "iphone" → "iPhone", "pytorch" → "PyTorch"
- Acronyms: "nasa" → "NASA", "api" → "API"
- Names: "omkar" → "Omkar", "san francisco" → "San Francisco"
- Mixed case: "macbook" → "MacBook", "github" → "GitHub"

Joint model also predicts punctuation in the same forward pass — replaces the V1 punctuation model entirely.

### Model

**Primary:** `nvidia/punctuation_en_distilbert`
- DistilBERT base (~66M params, ~260MB)
- Joint head: punctuation + capitalization
- English-only (acceptable for V2; revisit if multilingual needed)
- Permissive license

**Fallback if multilingual is required:** stick with `oliverguhr/fullstop-punctuation-multilingual-base` for punctuation + a separate truecasing model. Worse latency (two inferences), worse quality, but multilingual.

### Replaces, not augments

V1's `oliverguhr/fullstop-punctuation-multilingual-base` model goes away entirely. So does the regex `apply_casing` post-step. Cleaner pipeline.

### Implementation

```bash
# Download from NGC or HF
# (NeMo models often distributed as .nemo files; convert to ONNX)
pip install "nemo_toolkit[nlp]"
```

```python
# Convert NeMo → ONNX (one-time, build step)
from nemo.collections.nlp.models import PunctuationCapitalizationModel
model = PunctuationCapitalizationModel.restore_from("punct_cap.nemo")
model.export("punct_cap.onnx")
```

Then INT8 quantize the same way as V1:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("punct_cap.onnx", "punct_cap.int8.onnx", weight_type=QuantType.QInt8)
```

### Inference sketch

```python
def punctuate_and_case(text: str) -> str:
    encoded = tokenizer.encode(text)
    punct_logits, cap_logits = session.run(
        ["punct_logits", "cap_logits"],
        {"input_ids": encoded.ids, "attention_mask": encoded.attention_mask}
    )
    punct_labels = punct_logits.argmax(-1)
    cap_labels = cap_logits.argmax(-1)
    return reconstruct_text(encoded.tokens, punct_labels, cap_labels)
```

NeMo's two heads have separate label spaces:
- Punctuation: `O`, `.`, `,`, `?`
- Capitalization: `O` (lowercase), `U` (upper first letter), `A` (all caps)

Apply capitalization based on the predicted label per token, then insert punctuation after.

### Latency

DistilBERT base, INT8, on `c7g.xlarge`: ~30-60ms p50, ~80-100ms p99 for 30-60 token utterances. Faster than V1's XLM-R base because DistilBERT has half the layers.

### Eval

Build a held-out set with technical vocabulary (proper nouns, acronyms, mixed-case brands). Compare against V1 baseline. Track:
- F1 on punctuation marks (period, comma, question)
- Accuracy on first-letter capitalization
- Accuracy on proper-noun capitalization (this is where you'll see the win)

---

## Stage 3: Inverse Text Normalization (ITN)

### What it does

Converts spoken-form numbers and entities to written form:

| Spoken | Written |
|--------|---------|
| "twenty twenty six" | "2026" |
| "five dollars" | "$5" |
| "three p m" | "3 PM" |
| "one hundred percent" | "100%" |
| "july fourth" | "July 4" |
| "five point two" | "5.2" |
| "at gmail dot com" | "@gmail.com" |

This is the single highest-impact V2 feature for perceived quality.

### Approach

**Use NVIDIA NeMo Text Processing.** WFST-based, no ML inference, deterministic, sub-millisecond, MIT-licensed. Handles English well out of the box, supports German/Spanish/French if needed.

```bash
pip install nemo_text_processing
```

```python
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

itn = InverseNormalizer(lang='en', cache_dir='/app/itn_cache')
result = itn.inverse_normalize("twenty twenty six dollars", verbose=False)
# → "$2026"
```

**First call is slow** (compiles the WFST graphs, ~10-30s). Initialize at server startup, not per-request.

### Coverage

NeMo ITN handles out of the box:
- Cardinal numbers, ordinals
- Decimals, fractions
- Currency (USD, EUR, GBP)
- Dates (multiple formats)
- Times
- Telephone numbers
- Measurements (units)
- Math operators
- Electronic addresses (email, URLs)

What it doesn't handle and you may want to add as post-processing:
- App-specific shorthand ("at-mention sarah" → "@sarah")
- Markdown shortcuts ("hash tag wfh" → "#wfh")
- Emoji names ("smiley face" → "🙂") — probably skip, low signal

### Order: ITN runs *after* the punctuation model

Why: the punctuation model was trained on natural English text with words spelled out. Feed it "twenty twenty six" not "2026" — it handles word tokens better. Run ITN on the model's output as a post-step.

```python
def pipeline(text: str) -> str:
    text = apply_spoken_commands(text)            # Stage 1
    text = punctuate_and_case(text)               # Stage 2
    text = itn.inverse_normalize(text, verbose=False)  # Stage 3
    return text
```

### Latency

WFST traversal: ~5-20ms for typical utterances. Scales with sentence length but stays well under budget.

### Gotchas

- **Initialize once at startup.** Add to FastAPI's `lifespan` handler. Loading WFST graphs on first request adds ~20s.
- **Cache directory must be writable.** Set `cache_dir` to a path inside the container that survives restarts (or accept the rebuild cost on container start).
- **ITN can over-normalize.** "I want to be there at five" → "I want to be there at 5" (correct) vs "I'll be there in five" → "I'll be there in 5" (debatable). Accept this; it's overwhelmingly the user's expectation.
- **Capitalization of normalized output.** ITN may emit "PM" or "AM" — check that the truecasing pass doesn't downcase these. Run ITN *after* truecasing and ITN's output wins.

### Eval

Create a labeled set of 100-200 dictated utterances with known ITN expectations. Track exact-match accuracy on ITN-relevant spans. NeMo ITN should hit >95% on standard cases out of the box.

---

## Updated Container

```dockerfile
FROM python:3.12-slim

ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for NeMo text processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] onnxruntime tokenizers pydantic \
    nemo_text_processing

COPY punct_cap.int8.onnx tokenizer.json labels.json ./
COPY itn_cache/ ./itn_cache/
COPY app.py commands.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Container size jumps to ~600-800MB (NeMo text processing pulls Pynini and dependencies). Acceptable.

## Lifespan Setup

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy initialization, blocks readiness until done
    app.state.session = load_onnx_session("punct_cap.int8.onnx")
    app.state.tokenizer = load_tokenizer("tokenizer.json")
    app.state.itn = InverseNormalizer(lang="en", cache_dir="./itn_cache")
    # Warm-up call so first real request isn't slow
    _ = app.state.itn.inverse_normalize("one two three", verbose=False)
    yield
    # No shutdown work needed

app = FastAPI(lifespan=lifespan)
```

Health check (`/healthz`) should fail until lifespan startup completes — this prevents ALB from routing traffic to a cold task.

## Performance Expectations (V2)

### Per-request latency

Per `c7g.xlarge`, INT8 DistilBERT, 30-60 token utterances, with `intra_op_num_threads=2` (see Concurrency below):

| Stage | p50 | p99 |
|-------|-----|-----|
| Stage 1 (commands) | <1ms | <2ms |
| Stage 2 (punct+case) | 60-90ms | 120-150ms |
| Stage 3 (ITN) | 5-20ms | ~40ms |
| Network/parse/serialize | 5-10ms | 20-30ms |
| **Total** | **~100ms** | **~250ms** |

**Honest framing on the latency target.** The 250ms p99 is a realistic expectation under representative load on 30-60 token inputs. Inputs >256 tokens go through chunked inference and have explicit best-effort latency (no SLO). The p99 number is sensitive to:

- Cold caches after quiet periods (first request: +50-100ms)
- Python GC pauses
- Noisy neighbor on shared ECS hosts
- Long-input chunking (multi-chunk requests scale ~linearly with chunk count)

Don't treat sub-150ms as a guarantee. Measure in production and set SLOs based on observed distribution, not back-of-envelope per-stage math.

### Concurrency model — important

A single FastAPI task with one uvicorn worker runs **one inference at a time per ORT session**. ORT uses `intra_op_num_threads` for parallelism *within* a single inference, not for serving concurrent requests. This is a hard constraint, not a tunable.

Throughput per task depends entirely on inference wall-clock time and concurrency strategy:

| Config | Per-req inference time | Concurrent inferences | QPS/task |
|--------|------------------------|----------------------|----------|
| `intra=4`, no batching | ~50ms | 1 | ~14 |
| **`intra=2`, no batching** | **~80ms** | **2** | **~20-25** |
| `intra=1`, 4 uvicorn workers | ~150ms | 4 | ~25-30 (RAM-bound) |
| `intra=4`, dynamic batch=8 | ~100ms (at full batch) | 8 effective | ~80 |

**Recommended for V2: `intra_op_num_threads=2` on `c7g.xlarge`.** Trade ~30ms additional per-request inference latency for roughly 2x throughput per task. Multiple uvicorn workers are worse on transformer inference (model copies in RAM, process contention). Dynamic batching is the highest-throughput option but adds complexity (batch timeout tuning, padding overhead) — defer to V2.5 or V3 if throughput becomes a cost problem.

**Concurrency cap.** Add an asyncio semaphore in the request handler limiting in-flight requests per task to ~4-8. Without this, burst load queues inside the FastAPI process and p99 latency explodes silently.

```python
import asyncio
INFLIGHT = asyncio.Semaphore(4)

@app.post("/punctuate")
async def punctuate(req: Req):
    async with INFLIGHT:
        return run_pipeline(req.text)
```

### Throughput planning

Plan for **~25 QPS per `c7g.xlarge` task** as the realistic number with the recommended config. To handle sustained traffic:

| Sustained QPS | Tasks needed (with headroom) |
|---------------|------------------------------|
| 100 | 6 |
| 500 | 25 |
| 1000 | 50 |
| 5000 | 250 |

Spread across 2-3 AZs minimum. Set HPA target on `RequestCountPerTarget` at 60% of measured per-task peak (so ~15 RPS per target), not on CPU utilization — CPU lags request queueing under burst load.

Deployment guidance: keep Fargate for early production and moderate traffic. If sustained load grows into the several-hundred-RPS range, keep the SSE contract at the edge and move the workers first — ECS on EC2 and/or a batched inference tier are better scaling levers than replacing SSE.

### Cost

The first real cost wall is usually idle baseline (always-on tasks + ALB), not marginal request cost. Recompute current AWS pricing at rollout time; as a rule of thumb, once the service needs dozens of hot tasks, move to EC2 and/or batching before considering any API-protocol redesign.

## Gotchas

- **Pipeline ordering is fixed.** Commands → model → ITN. Reordering breaks things:
  - ITN before model: model sees "2026" instead of "twenty twenty six", less accurate
  - Commands after model: model adds period before "comma" word that should become ","
- **NeMo ITN cold start.** ~10-30s on first call to compile WFSTs. Pre-warm in lifespan.
- **DistilBERT max length is 512 tokens.** Same chunking strategy as V1 with 256-token windows + 32-token overlap. Long-input p99 is *not* covered by the 250ms budget — it scales with chunk count.
- **Joint model has two output heads.** Don't forget to apply both during reconstruction; easy to ship a service that adds punctuation but ignores capitalization predictions.
- **Spoken command precision matters more than recall.** A missed command is invisible; a false-positive command (turning literal "comma" into ",") is jarring. Tune toward conservative replacement.
- **NeMo `.nemo` → ONNX export sometimes loses metadata.** Verify label mappings round-trip correctly before deploying. Ship a `labels.json` alongside the model and load it explicitly.
- **Test on real Moonshine output, not synthetic.** Moonshine drops some fillers and produces lowercase output without punctuation — make sure your eval set reflects that distribution.
- **One inference at a time per ORT session.** Don't assume `intra_op_num_threads=4` means "4 parallel requests" — it means one request gets 4 threads of intra-op parallelism. See Concurrency Model above.
- **Burst failure mode is queueing, not socket count.** Without admission control, p99 blows up because requests pile up behind shared inference, then clients timeout and retry.
- **Hard request timeout.** Set FastAPI/uvicorn request timeout to ~400ms. Pathological inputs (e.g. adversarial long inputs that slip past the size check) can otherwise tail-blow p99.
- **Pre-warm new tasks before adding to ALB.** First 10-50 requests on a fresh task have cold ORT memory and worse latency. Either route synthetic warm-up traffic or accept temporary latency degradation after scale-out.
- **Client timeout can still burn server CPU.** Once a stage has entered `asyncio.to_thread(...)`, the work may finish even if the client has already given up. Track disconnects and abandoned work explicitly.
- **SSE errors are not HTTP errors.** Emit stream-failure and fallback counters so incidents are visible in dashboards even when the transport returns 200.
- **Track p50/p95/p99/p999 separately.** Averages hide the tail. Use CloudWatch histograms or Prometheus.

## V3 Hook: LLM Fallback Pattern

V2 has no fallback by design — every request goes through the deterministic pipeline. This is a latency *advantage*: no slow path, no provider tail latency, no branch overhead.

When fallback is added in V3 (likely for ASR error correction), the pattern should be:

1. Always run V2 pipeline first
2. Compute confidence signal (model entropy, ITN warning flags, low-confidence spans from Moonshine if exposed)
3. High confidence (95%+) → return immediately, single-digit ms overhead
4. Low confidence → hedge: fire to LLM in parallel with a hard 400ms timeout
5. Tag fallback requests in metrics; monitor rate and tune thresholds

Fallback latency budget is *separate* from V2's. LLM fallback paths will be 500ms-2s+ p99. Keep fallback rate under 5% so it doesn't dominate aggregate p99.

## Build Order

1. **SSE streaming contract** (wire change, no new models). Add `/punctuate/stream` alongside `/punctuate`; initially it just emits a single `stage: "full"` event with the V1 output. Ship the client's streaming consumer against this; everything that follows is additive. Half a day of work.
2. **ITN** (highest ROI, least risk). Add NeMo Text Processing to the pipeline as a post-step; emit a dedicated `stage: "itn"` event after the existing punct+case result. One day.
3. **Truecasing upgrade**. Export NeMo joint model to ONNX, INT8 quantize, A/B test against V1 model on a labeled eval set. Replace V1's punct+case stage (same event name, better output). Two-three days.
4. **Spoken punctuation commands**. Add Stage 1 preprocessor emitting `stage: "commands"`; build labeled eval set; tune precision/recall. One-two days plus eval iteration. Enables deterministic segment-id splitting at "new paragraph" commands.

Ship each independently behind a feature flag. Each stage is reversible if it regresses quality. The streaming contract is stable from step 1; later steps only add events, never rename or remove them.

## What V2 Cannot Fix: ASR Error Correction

This pipeline formats text. It does not correct ASR mistakes. Important distinction worth internalizing before users start filing bugs.

### The dividing line

- **Formatting problems** (in scope): the words are right, the presentation is wrong. Missing punctuation, wrong casing, "twenty twenty six" instead of "2026". Solvable with rules + small models on CPU.
- **Recognition errors** (out of scope): the words themselves are wrong. Homophones, near-homophones, out-of-vocabulary terms, world-knowledge confusions. The acoustic information needed to fix these is gone by the time text reaches this service.

### Concrete example

User says: *"Can you help me get off the 405 pretty soon?"*
Moonshine produces: *"can you help me get off the four or five pretty soon"*

V2 pipeline output: *"Can you help me get off the 4 or 5 pretty soon?"*

ITN sees the literal word "or" between two numbers and faithfully converts it to a numeric range. It has no signal that "405" was the intended utterance — that signal lives in the audio and in world knowledge ("405" is a freeway in LA), neither of which this service has access to.

### Why no text-only post-processor can fix this

The error happened upstream in the ASR. By the time text reaches V2, the acoustic features that could disambiguate "four-oh-five" from "four or five" are discarded. Text-only correction would require either world knowledge (LLM) or a brittle freeway-context dictionary that would generate false positives elsewhere.

### Where these fixes actually belong

In rough order of effectiveness:

1. **Better ASR** — bigger Moonshine model, domain-adapted retraining, or beam search with a stronger language model. Lives in the client.
2. **Contextual biasing at ASR time** — give Moonshine a hotword/vocabulary list of likely terms (freeways, technical terms, contact names) so it biases toward them during decoding. Moonshine offers paid retraining and has lighter-weight biasing on their roadmap. Lives in the client.
3. **LLM rewrite with world knowledge** — feed punctuated output to an LLM with "fix likely ASR errors". Works but breaks the CPU/no-LLM constraint. V3 conversation.
4. **Confusion-pair dictionary, narrow context** — e.g. "four or five" → "405" only when surrounded by freeway-context words. Cheap and deterministic but brittle and high false-positive risk. Don't build until you have telemetry showing it's worth it.

### Recommended posture for V2

Document these as known limitations. Don't build correction logic until you have data on which errors actually occur in production and at what frequency. Premature correction logic creates false positives that are more visible than the original errors.

If freeway-class errors become a top user complaint after launch, the right move is to push the fix to the ASR layer (Moonshine biasing) rather than adding a Stage 4 correction pass to V2. The post-processing pipeline is the wrong place to solve recognition problems.

## Known Follow-ups

Logged here during implementation; each is in-scope for V2 but not
blocking for the current work item.

- **Duplicate same-utterance dispatches on the client.** When a first-try
  `consumeRefinementStream` fails mid-flight and the same `utteranceID`
  then gets picked up by the retry drain, actor reentrancy can briefly
  run two concurrent streams for the same utterance. Logs show 2–3 `sse[UUID]`
  sequences for a single utterance. Not user-visible (UI upserts by
  `segmentID`) but doubles server CPU for that utterance and can tip a
  request past the 4 s client timeout under contention. Fix: track
  in-flight `utteranceID`s in `TranscriptionPipeline` and short-circuit
  duplicate `consumeRefinementStream` calls in both
  `dispatchFirstTryPunctuation` and `drainAsManyAsPossible`.

- **Server-side admission control.** The plan already calls for an
  asyncio semaphore (~4–8 in-flight) with 429/503 on saturation. Not
  wired up yet — add before any rollout beyond local dev. Required to
  bound tail latency under burst load and to reject duplicate dispatches
  cleanly if the client-side fix above misses a case.

- **Drain-loop wake-on-recovery.** `drainLoop` currently sleeps the full
  backoff (up to 60 s) even when another utterance has just confirmed
  the server is reachable. `handleRefinementSuccess` intentionally does
  not cancel the drain (cancelling the in-flight retry stream causes
  starvation). A wake signal — e.g. a `CheckedContinuation` the sleep
  races against — would let a success collapse the backoff to ~0 without
  touching the active stream. Bounded impact today (worst case 60 s),
  so only do this if the latency shows up in practice.

- **Multi-segment first-try drops.** Observed during paragraph-split
  testing: every multi-segment utterance's first-try stream timed out
  on the client (`NSURLErrorDomain -1001`) after Stage 1 events landed
  but before the server reached Stage 2. Retry then replayed and
  completed in ~500 ms with identical segment ids (uuidv5 determinism
  works as designed, so no UI orphan rows). Single-segment utterances
  are unaffected. Server logs show no error, so the suspicion is
  client-side: either the 4 s `timeoutIntervalForRequest` interacts
  badly with `URLSession.AsyncBytes.lines` when several `commands`
  events arrive back-to-back before any `punct_case` event, or the
  duplicate-dispatch issue above is doubling server CPU into the
  timeout. Investigate after the duplicate-dispatch fix lands; if the
  drops persist, raise the per-request timeout to ~8 s and bump
  `timeoutIntervalForResource` accordingly.

- **Comma-between-spelled-numbers guard before ITN.** NeMo's ITN
  greedily matches adjacent number-words against the time-of-day
  grammar (`"nine ten"` → `"09:10"`). Stage 2's punct_case model
  reliably inserts the comma between consecutive list items on short
  inputs but occasionally drops one on long repetitive runs (per-token
  argmax confidence drifts on repeating context). Result: `"...eight,
  nine ten."` from Stage 2 becomes `"...eight, 09:10."` after Stage 3
  even though the user clearly counted to ten. Fix is a tiny regex
  pre-pass between Stage 2 and Stage 3 that inserts a comma between
  any `\b(one|two|...|ten)\s+(one|two|...|ten)\b` pair before the
  WFST sees the text. Deterministic, microseconds, narrow. Worth
  keeping even after a punctuation model upgrade since
  token-classification models will always have some non-zero miss
  rate on long lists.

## Out of Scope (Explicit)

These are *not* in V2 — punt to V3 conversation:

- **ASR error correction** (homophones, near-homophones, recognition mistakes — see section above)
- Custom user dictionaries
- Command actions (delete, edit, format)
- Style presets per app context
- Disfluency removal (no good off-the-shelf option, deferred)
- Grammar correction (needs LLM, breaks CPU constraint)
- Multilingual support (NeMo joint model is EN-only)
- Smart formatting (lists, code blocks, headers)
- Spell correction (high false-positive risk on technical vocab)
- LLM escape hatch for hard cases

## Success Criteria

V2 is shippable when, on a labeled eval set of 200-500 real Moonshine outputs:

- ITN exact-match accuracy >95% on numeric/date/currency spans
- Truecasing accuracy on proper nouns >90% (vs V1's ~10% from regex-only)
- Spoken command precision >95%, recall >80%
- No regression on V1's punctuation F1

And on production-representative load on `c7g.xlarge` with `intra_op_num_threads=2`:

- p50 latency <100ms for 30-60 token inputs
- p99 latency <250ms for 30-60 token inputs
- p99 latency <500ms for inputs up to 256 tokens
- Inputs >256 tokens: best-effort, no SLO
- Sustained ~25 QPS per task without queue buildup
- No regression on V1's measured throughput at the same task count
