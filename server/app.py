"""
Punctuation service — FastAPI + ONNX Runtime wrapper around
`oliverguhr/fullstop-punctuation-multilingual-base` (XLM-RoBERTa base,
token classification). Takes an unpunctuated transcript and returns
punctuated + sentence-cased text.

Runtime contract — V1 (see `punctuation-service-plan.md`):

    POST /punctuate
    Headers: X-API-Key: <PUNCTUATION_API_KEY>
    Body: {"text": "…"}
    Response 200: {"text": "…"}

Runtime contract — V2 streaming (see `punctuation-service-v2-plan.md`):

    POST /punctuate/stream
    Headers: X-API-Key, Accept: text/event-stream
    Body: {
        "text": "…",
        "utterance_id": "…",
        "segment_id": "…",
        "silence_before_ms": 2400   (optional, reserved)
    }
    Response: text/event-stream with one or more `event: stage` frames.
    Each frame carries {utterance_id, segment_id, stage, text, final}.
    Stages emitted today, in order: `commands` (ASR punct strip +
    spoken-command substitution), `punct_case` (XLM-R + regex casing),
    `itn` (NeMo WFST normalization).

Model is loaded once at startup in the FastAPI lifespan hook so the first
request doesn't eat the 1-2 s graph init cost. `/healthz` returns 503
until the model is ready so ALB won't route traffic to a cold task.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
    InverseNormalizer,
)
from num2words import num2words
from pydantic import BaseModel
from tokenizers import Tokenizer

from commands import apply_spoken_commands, strip_asr_punctuation

# ---------- Config (env vars) ------------------------------------------------

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model-onnx"))
MODEL_PATH = MODEL_DIR / os.environ.get("MODEL_FILENAME", "model.int8.onnx")
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CONFIG_PATH = MODEL_DIR / "config.json"

API_KEY = os.environ.get("PUNCTUATION_API_KEY", "dev-key-change-me")
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 2))

# ITN (Stage 3 of the V2 pipeline) uses NeMo Text Processing's WFST-based
# InverseNormalizer. First construction compiles the FST graphs (~10-30 s
# in the worst case, ~1-2 s with cache); we do this once at startup.
ITN_CACHE_DIR = Path(os.environ.get("ITN_CACHE_DIR", Path(__file__).parent / "itn_cache"))
ITN_LANG = os.environ.get("ITN_LANG", "en")

# Reject inputs over this many chars at the edge — keeps pathological
# requests from blocking a worker. ~1500 chars ≈ 300 words ≈ well past
# any realistic single-utterance transcript.
MAX_INPUT_CHARS = 2000

# Chunk budget. XLM-R's position embedding limit is 512; we use 256 to
# stay safely under and halve latency on single-chunk utterances. Each
# chunk reserves 2 slots for BOS + EOS.
MAX_CHUNK_TOKENS = 256
CHUNK_CONTENT = MAX_CHUNK_TOKENS - 2  # 254
CHUNK_OVERLAP = 32  # tokens shared between adjacent chunks


# ---------- Logging ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("punctuation")


# ---------- Models loaded in lifespan ---------------------------------------

_session: Optional[ort.InferenceSession] = None
_tokenizer: Optional[Tokenizer] = None
_id_to_label: dict[int, str] = {}   # {0: "0", 1: ".", 2: ",", ...}
_bos_id: int = 0
_eos_id: int = 0
_pad_id: int = 0
_itn: Optional[InverseNormalizer] = None
_model_ready: bool = False


def _load_model() -> None:
    global _session, _tokenizer, _id_to_label, _bos_id, _eos_id, _pad_id, _model_ready

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            f"Run `python export_model.py` first to download + quantize."
        )
    if not TOKENIZER_PATH.exists():
        raise RuntimeError(f"Tokenizer not found at {TOKENIZER_PATH}")
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found at {CONFIG_PATH}")

    log.info("loading tokenizer from %s", TOKENIZER_PATH)
    _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    # XLM-R special tokens. Fall back to hardcoded IDs if the tokenizer
    # doesn't expose them for some reason — the values have been stable
    # across XLM-R releases.
    _bos_id = _tokenizer.token_to_id("<s>") or 0
    _eos_id = _tokenizer.token_to_id("</s>") or 2
    _pad_id = _tokenizer.token_to_id("<pad>") or 1

    log.info("loading label map from %s", CONFIG_PATH)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    raw_id2label = cfg.get("id2label") or {}
    _id_to_label = {int(k): v for k, v in raw_id2label.items()}
    if not _id_to_label:
        raise RuntimeError("config.json is missing id2label — check the export")
    log.info("labels: %s", _id_to_label)

    log.info("loading ONNX session from %s (%d threads)", MODEL_PATH, NUM_THREADS)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = NUM_THREADS
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    _model_ready = True
    log.info("model ready")


def _load_itn() -> None:
    """Construct the NeMo ITN normalizer. First construction compiles WFST
    grammars; with `ITN_CACHE_DIR` set the compiled grammars persist
    across restarts so this drops from ~10-30 s to ~1-2 s. We also run a
    warm-up call so the first real request doesn't eat the lazy init
    cost hidden inside the first `inverse_normalize` invocation."""
    global _itn

    ITN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("loading ITN (lang=%s, cache=%s)", ITN_LANG, ITN_CACHE_DIR)
    _itn = InverseNormalizer(lang=ITN_LANG, cache_dir=str(ITN_CACHE_DIR))

    # Warm-up: exercises the full path once so first real request has a
    # hot cache for common WFST branches.
    _ = _itn.inverse_normalize("one two three", verbose=False)
    log.info("ITN ready")


# ---------- FastAPI app ------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    _load_itn()
    yield


app = FastAPI(lifespan=lifespan)


class PunctuateRequest(BaseModel):
    text: str


class PunctuateResponse(BaseModel):
    text: str


class PunctuateStreamRequest(BaseModel):
    """Request for the V2 streaming endpoint. See the V2 plan for the full
    contract. `utterance_id` is provenance (which Moonshine utterance this
    came from — used for retries). `segment_id` is the client's current
    running rendering segment; the server echoes it back on every stage
    event. `silence_before_ms` is an optional client-supplied VAD-derived
    signal; reserved for Stage 1 paragraph-break heuristics, accepted but
    not yet used. `dictation_mode` controls Stage 1's spoken-command
    interpretation: when False, command words ("comma", "period", "new
    paragraph") pass through as literal text — useful when the client is
    capturing conversational speech rather than dictation. Default True
    so existing callers keep current behavior; conservative client UIs
    typically default the toggle to off."""
    text: str
    utterance_id: str
    segment_id: str
    silence_before_ms: int | None = None
    dictation_mode: bool = True


def _require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/healthz")
def healthz():
    if not _model_ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"ok": True}


@app.post("/punctuate", response_model=PunctuateResponse)
def punctuate(req: PunctuateRequest, _: None = Depends(_require_api_key)) -> PunctuateResponse:
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="input too long")
    return PunctuateResponse(text=_run_v1_pipeline(req.text))


def _run_v1_pipeline(text: str) -> str:
    """V1 punct + case pipeline on already-length-validated text. Shared
    between the non-streaming `/punctuate` endpoint and the streaming
    `/punctuate/stream` endpoint so both paths produce identical output
    for the same input."""
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    words = stripped.split()
    labeled = _infer_labels(words)
    joined = _apply_punctuation(labeled)
    return _apply_casing(joined)


# Matches standalone digit runs (optionally a decimal). The `\b` edges
# keep us from touching digits embedded in URLs, codes, or tokens that
# happen to contain numbers.
_BARE_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _digits_to_words(text: str) -> str:
    """Convert bare digit runs to their English word form before ITN
    sees the text. Moonshine emits numeric amounts as digits in many
    cases ("10 dollars") and NeMo ITN's grammar only triggers on word
    form ("ten dollars" → "$10") — so without this pre-pass, currency
    and similar entities don't get normalized for digit-form input.

    Hyphens in `num2words` output ("twenty-seven") are replaced with
    spaces so the ITN grammar reliably matches number sequences. On any
    conversion failure (overflow, malformed decimal) the original token
    passes through unchanged."""
    def _replace(m: re.Match) -> str:
        token = m.group(0)
        try:
            value = float(token) if "." in token else int(token)
            return num2words(value).replace("-", " ")
        except (ValueError, OverflowError):
            return token
    return _BARE_DIGIT_RE.sub(_replace, text)


# Two WFST artifacts worth fixing in NeMo ITN's output:
# 1. Spaces before trailing punctuation — "2027 .", "100 %", "word ,".
#    Reads as broken. Conservative regex: does not touch internal dots
#    in abbreviations like "U.S." or "p.m." because those have no space.
# 2. Doubled periods when an abbreviation ("p.m.", "a.m.") lands at end
#    of a sentence that already had a terminal period from Stage 2:
#    "at 03:00 p.m..". Safe against ellipsis because our Stage 2 label
#    set (`. , ? - :`) can't emit "..." in the first place.
_ITN_POST_CLEANUP_SPACE = re.compile(r" +([.,?!;:%])")
_ITN_POST_CLEANUP_DOUBLE = re.compile(r"\.\.(?=\s|$)")


def _stage_itn(text: str) -> str:
    """Stage 3: inverse text normalization. Converts spoken-form numbers
    and entities ("twenty twenty seven", "ten dollars") to written form
    ("2027", "$10"). Operates on already-punctuated/cased text from
    Stage 2. Graceful fallback — if the WFST raises on a pathological
    input, return the text unchanged rather than failing the whole
    stream; Stage 2's output is still useful.

    We pre-pass the input through `_digits_to_words` so ASR-emitted
    digits ("10 dollars") reach the WFST in the word form it expects.
    Without this the NeMo ITN grammar would only convert utterances
    where the ASR happened to spell numbers out."""
    if not text or _itn is None:
        return text
    pre = _digits_to_words(text)
    if pre != text:
        log.info("d2w: %r -> %r", text, pre)
    try:
        normalized = _itn.inverse_normalize(pre, verbose=False)
    except Exception:
        log.exception("ITN failed on %r; passing through unchanged", text)
        return text
    normalized = _ITN_POST_CLEANUP_SPACE.sub(r"\1", normalized)
    normalized = _ITN_POST_CLEANUP_DOUBLE.sub(".", normalized)
    return normalized


def _sse(event: str, payload: dict) -> bytes:
    """One SSE frame: `event:` line, `data:` line with JSON payload, blank
    terminator. Bytes (not str) so the streaming generator can yield
    without re-encoding."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _run_stage1(text: str, dictation_mode: bool) -> list[str]:
    """Stage 1 entry point shared by the streaming endpoint and tests.
    ASR punctuation strip is unconditional — Stage 2's model is trained
    on lowercased, unpunctuated input, so any ASR-emitted marks would
    put it out of distribution. Spoken-command substitution and
    paragraph splitting only run in dictation mode; in conversational
    mode `comma`/`period`/`new paragraph` land as literal words rather
    than being interpreted as formatting."""
    if dictation_mode:
        return apply_spoken_commands(text)
    return [strip_asr_punctuation(text)]


def _mint_segment_ids(utterance_id: str, incoming_segment_id: str, n: int) -> list[str]:
    """Deterministic segment ids for an n-paragraph split. Paragraph 0
    reuses the client's `incoming_segment_id` so its stage events
    overwrite the in-place row that was already rendered for the
    finalized utterance. Paragraphs 1..n-1 get
    `uuidv5(namespace=utterance_id, name=str(split_index))` — stable
    across retries so the client's upsert doesn't accumulate orphan
    rows on a retried stream. See v2 plan § Deterministic segment IDs."""
    if n <= 0:
        return []
    ids = [incoming_segment_id]
    if n == 1:
        return ids
    try:
        namespace = uuid.UUID(utterance_id)
    except (ValueError, TypeError):
        # Caller handed us a non-UUID utterance id. Fall back to deriving
        # the namespace from the raw string so we still produce stable,
        # reproducible ids — just not ones a client could recompute from
        # the UUID alone.
        namespace = uuid.uuid5(uuid.NAMESPACE_URL, utterance_id)
    for idx in range(1, n):
        ids.append(str(uuid.uuid5(namespace, str(idx))))
    return ids


@app.post("/punctuate/stream")
async def punctuate_stream(
    req: PunctuateStreamRequest,
    _: None = Depends(_require_api_key),
):
    """Streaming refinement endpoint. Emits one `event: stage` frame per
    pipeline stage as each completes; the final event carries `final:
    true`. Stages in order:

    - `commands` (Stage 1) — strip ASR-supplied punctuation then
      substitute spoken punctuation commands ("comma" → ",", etc.).
      Pure function of the input text; see `commands.py`.
    - `punct_case` (Stage 2) — XLM-R token-classification punctuation +
      regex sentence casing.
    - `itn` (Stage 3) — NeMo WFST-based inverse text normalization
      ("ten dollars" → "$10").

    Clients should key on `stage` (not event order) and treat each event
    as the *current* text for the given segment — each stage overwrites
    the prior stage's output for that segmentID.

    Segment splitting: Stage 1 may split the utterance into multiple
    paragraphs on `new paragraph` commands. Paragraph 0 reuses the
    client's incoming `segment_id`; paragraphs 1..n-1 get deterministic
    `uuidv5(utterance_id, split_index)` so retries reproduce identical
    ids. Stages are emitted per-segment: `commands` for all segments
    first, then `punct_case`, then `itn`. Only the very last event
    (last segment's `itn`) carries `final: true` — clients use that as
    the stream terminator regardless of segment count.
    """
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="input too long")

    # Short prefix of the utterance id for log correlation — full UUIDs
    # are noisy and the prefix is unique enough within a session.
    utter_tag = (req.utterance_id or "")[:8]
    log.info("stream utter=%s in=%r", utter_tag, req.text)

    async def gen():
        try:
            paragraphs = await asyncio.to_thread(_run_stage1, req.text, req.dictation_mode)
            segment_ids = _mint_segment_ids(
                req.utterance_id, req.segment_id, len(paragraphs)
            )
            log.info(
                "stream utter=%s stage=commands segments=%d texts=%r",
                utter_tag, len(paragraphs), paragraphs,
            )
            for seg_id, para in zip(segment_ids, paragraphs):
                yield _sse("stage", {
                    "utterance_id": req.utterance_id,
                    "segment_id": seg_id,
                    "stage": "commands",
                    "text": para,
                    "final": False,
                })

            # Stage 2: punct + case per paragraph. Blocking ONNX call →
            # asyncio.to_thread so the event loop can flush bytes between
            # segments. Without offloading, the whole response would
            # arrive together at EOF, defeating the streaming.
            refined_segs: list[str] = []
            for seg_id, para in zip(segment_ids, paragraphs):
                refined = await asyncio.to_thread(_run_v1_pipeline, para)
                refined_segs.append(refined)
                log.info(
                    "stream utter=%s seg=%s stage=punct_case text=%r",
                    utter_tag, seg_id[:8], refined,
                )
                yield _sse("stage", {
                    "utterance_id": req.utterance_id,
                    "segment_id": seg_id,
                    "stage": "punct_case",
                    "text": refined,
                    "final": False,
                })

            # Stage 3: ITN per paragraph. Pynini's WFST traversal is fast
            # (~1-5 ms for typical utterances) but still CPU-bound;
            # offload so it doesn't block the event loop for concurrent
            # streams. Only the last segment's itn is the stream
            # terminator (`final: true`).
            last_idx = len(segment_ids) - 1
            for idx, (seg_id, refined) in enumerate(zip(segment_ids, refined_segs)):
                normalized = await asyncio.to_thread(_stage_itn, refined)
                log.info(
                    "stream utter=%s seg=%s stage=itn text=%r",
                    utter_tag, seg_id[:8], normalized,
                )
                yield _sse("stage", {
                    "utterance_id": req.utterance_id,
                    "segment_id": seg_id,
                    "stage": "itn",
                    "text": normalized,
                    "final": idx == last_idx,
                })
        except Exception as e:
            log.exception("stream pipeline failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            # Defeat proxy/ALB response buffering — without these the
            # full body only arrives at EOF, which defeats streaming.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- Inference pipeline ----------------------------------------------

def _infer_labels(words: list[str]) -> list[tuple[str, str]]:
    """Tokenize, run chunked inference, return (word, label_string) pairs."""
    assert _tokenizer is not None

    # Token IDs for each word, individually, so we know which subwords
    # belong to which source word. XLM-R adds a leading ▁ for word-start
    # when tokenized in isolation — that's what we want for alignment.
    token_ids: list[int] = []
    word_last_token_idx: list[int] = []   # -1 means "word produced no tokens"
    for w in words:
        enc = _tokenizer.encode(w, add_special_tokens=False)
        if not enc.ids:
            word_last_token_idx.append(-1)
            continue
        token_ids.extend(enc.ids)
        word_last_token_idx.append(len(token_ids) - 1)

    if not token_ids:
        return [(w, "0") for w in words]

    token_labels = _run_model_chunked(token_ids)

    result: list[tuple[str, str]] = []
    for word, idx in zip(words, word_last_token_idx):
        if idx < 0:
            result.append((word, "0"))
        else:
            label_id = int(token_labels[idx])
            result.append((word, _id_to_label.get(label_id, "0")))
    return result


def _run_model_chunked(token_ids: list[int]) -> np.ndarray:
    """Run the model on token_ids with sliding-window chunking for inputs
    longer than `CHUNK_CONTENT`. Returns an int array of label ids, one
    per input token. For tokens covered by multiple overlapping windows,
    we pick the label from the window where the token sits closest to
    the center (more surrounding context → more confident label)."""
    assert _session is not None
    n = len(token_ids)
    # Per-token output label, initialized to 0 (no punctuation).
    out_labels = np.zeros(n, dtype=np.int64)
    # How close to its window's center was each token when its label
    # was assigned? Lower = more central; we overwrite only when a new
    # window puts the token more central than a previous window did.
    best_distance = np.full(n, np.inf, dtype=np.float64)

    start = 0
    while True:
        end = min(start + CHUNK_CONTENT, n)
        chunk = token_ids[start:end]

        input_ids = [_bos_id] + chunk + [_eos_id]
        attention = [1] * len(input_ids)
        # Pad to MAX_CHUNK_TOKENS for a stable input shape (simpler for
        # ORT; could skip and run variable-length, but fixed shape lets
        # us optimize later if needed).
        pad_len = MAX_CHUNK_TOKENS - len(input_ids)
        if pad_len > 0:
            input_ids.extend([_pad_id] * pad_len)
            attention.extend([0] * pad_len)

        np_input_ids = np.array([input_ids], dtype=np.int64)
        np_attention = np.array([attention], dtype=np.int64)

        logits = _session.run(
            None,
            {"input_ids": np_input_ids, "attention_mask": np_attention},
        )[0][0]   # [seq_len, num_labels]

        # Argmax over the content positions only. Content sits at
        # positions 1..len(chunk), skipping BOS at 0 and EOS after.
        preds = np.argmax(logits[1:1 + len(chunk)], axis=-1)

        center = (len(chunk) - 1) / 2.0
        for i, label_id in enumerate(preds):
            abs_pos = start + i
            distance = abs(i - center)
            if distance < best_distance[abs_pos]:
                best_distance[abs_pos] = distance
                out_labels[abs_pos] = int(label_id)

        if end >= n:
            break
        start = end - CHUNK_OVERLAP
        if start <= 0:
            break  # degenerate guard; shouldn't hit if inputs are sane

    return out_labels


# ---------- Text assembly ----------------------------------------------------

def _apply_punctuation(labeled: list[tuple[str, str]]) -> str:
    """Join words, appending each word's label if it isn't the
    no-punctuation marker. Label strings are the punctuation glyphs
    themselves as stored in the model's id2label."""
    out: list[str] = []
    for word, label in labeled:
        if label and label != "0":
            out.append(word + label)
        else:
            out.append(word)
    return " ".join(out)


_SENT_END = re.compile(r"([.?!])\s+([a-z])")


def _apply_casing(text: str) -> str:
    """Deterministic casing: capitalize first character, then first
    character after each sentence-ending punctuation + whitespace. Skips
    proper nouns and acronyms — acceptable limitation per the scope doc."""
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return _SENT_END.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", text)
