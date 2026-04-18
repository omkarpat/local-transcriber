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
    Today only `stage: "punct_case"` is emitted; Stages 1 (commands) and
    3 (ITN) will land as separate events as they ship.

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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

# ---------- Config (env vars) ------------------------------------------------

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model-onnx"))
MODEL_PATH = MODEL_DIR / os.environ.get("MODEL_FILENAME", "model.int8.onnx")
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CONFIG_PATH = MODEL_DIR / "config.json"

API_KEY = os.environ.get("PUNCTUATION_API_KEY", "dev-key-change-me")
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 2))

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


# ---------- FastAPI app ------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
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
    not yet used."""
    text: str
    utterance_id: str
    segment_id: str
    silence_before_ms: int | None = None


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


def _sse(event: str, payload: dict) -> bytes:
    """One SSE frame: `event:` line, `data:` line with JSON payload, blank
    terminator. Bytes (not str) so the streaming generator can yield
    without re-encoding."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


@app.post("/punctuate/stream")
async def punctuate_stream(
    req: PunctuateStreamRequest,
    _: None = Depends(_require_api_key),
):
    """Streaming refinement endpoint. Today emits a single `stage:
    "punct_case"` event wrapping the V1 output with `final: true`. As
    Stage 1 (commands) and Stage 3 (ITN) land, they'll add events before
    and after this one — clients should key on `stage` not event order.

    Stateless: `segment_id` is echoed back verbatim. No new segment ids
    are minted yet because splitting hasn't shipped; when it does, new
    ids will be deterministic `uuidv5(utterance_id, split_index)` so
    retries reproduce identical output.
    """
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="input too long")

    async def gen():
        try:
            # ONNX `session.run` is blocking; offload so the event loop
            # can flush bytes between stages when there's more than one
            # stage to emit (today still a single stage, but the pattern
            # needs to be in place for Stages 1 and 3).
            refined = await asyncio.to_thread(_run_v1_pipeline, req.text)
            yield _sse("stage", {
                "utterance_id": req.utterance_id,
                "segment_id": req.segment_id,
                "stage": "punct_case",
                "text": refined,
                "final": True,
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
