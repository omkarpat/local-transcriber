# Punctuation Service — Implementation Plan

Bare-minimum cloud service that takes an unpunctuated utterance from an on-device ASR client and returns punctuated, properly-cased text. No custom training, no GPUs, no LLMs.

## Scope

**In:**
- Punctuation restoration (`. , ? - :`)
- Sentence casing (capitalize after sentence-ending punctuation)

**Out:**
- Custom dictionary substitution
- Command recognition
- Style presets
- Disfluency removal
- Grammar correction

If any of the "out" items become product requirements later, escalate the architecture — don't bolt them onto this service.

## Architecture

```
[Client / on-device ASR]
        │ HTTPS POST {text}
        ▼
[ALB] → [ECS Service: FastAPI + ONNX Runtime] → response
              (auto-scaled on RequestCountPerTarget)
```

Stateless, single-endpoint, horizontally scalable. No Redis, no queue, no database.

## Model

**Primary:** `oliverguhr/fullstop-punctuation-multilingual-base`
- Backbone: XLM-RoBERTa base (~280 MB)
- Languages: EN, DE, FR, IT, NL
- Labels: `0`, `.`, `,`, `?`, `-`, `:`
- License: MIT

**Alternative if proper-noun truecasing matters:** NVIDIA NeMo joint Punctuation+Capitalization model (DistilBERT-based). Slightly heavier container, removes the need for the deterministic casing step.

## Build Steps

### 1. Export to ONNX

```bash
pip install optimum[onnxruntime] transformers
optimum-cli export onnx \
  --model oliverguhr/fullstop-punctuation-multilingual-base \
  --task token-classification \
  ./model-onnx/
```

### 2. INT8 dynamic quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "./model-onnx/model.onnx",
    "./model-onnx/model.int8.onnx",
    weight_type=QuantType.QInt8,
)
```

Ship `model.int8.onnx` + `tokenizer.json` + label config in the container. No PyTorch at runtime.

### 3. Server

FastAPI + `onnxruntime` + `tokenizers` (Rust, fast). Single worker process; threading is handled inside ORT.

**Critical config:**

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = NUM_VCPUS   # parallelism within one inference
opts.inter_op_num_threads = 1           # one inference at a time per process
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model.int8.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)
```

**Run uvicorn with `--workers 1`.** Multiple Python workers each load the model into memory and fight for the same cores — net loss.

### 4. Inference handler (sketch)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer
import numpy as np

app = FastAPI()
tokenizer = Tokenizer.from_file("tokenizer.json")
LABELS = ["0", ".", ",", "?", "-", ":"]
MAX_LEN = 256
OVERLAP = 32
MAX_INPUT_CHARS = 2000

class Req(BaseModel):
    text: str

@app.post("/punctuate")
def punctuate(req: Req):
    text = req.text.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        return {"text": text}
    words = text.split()
    labeled = run_model_chunked(words)        # see chunking section
    out = apply_punctuation(labeled)
    return {"text": apply_casing(out)}
```

### 5. Casing post-step

Deterministic, no model. Covers the visible majority of cases:

```python
import re

_SENT_END = re.compile(r'([.?!])\s+([a-z])')

def apply_casing(text: str) -> str:
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return _SENT_END.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", text)
```

Skips proper nouns and acronyms — accept this limitation, or swap in NeMo if it becomes a complaint.

### 6. Long-input chunking

XLM-R has a 512-token limit; keep chunks ≤256 with overlap so boundary tokens see context on both sides.

- Tokenize the whole input
- Slide a 256-token window with `OVERLAP=32`
- Run inference per window
- For overlapping regions, prefer labels from the window where the token sits in the middle (more context)
- Stitch labeled words back into output text

Build this from day one — production utterances will sometimes be long.

## Deployment (ECS)

| Component | Choice |
|-----------|--------|
| Compute | ECS on EC2, `c7g.xlarge` (4 vCPU Graviton3) |
| Container | Distroless or slim Python base, ~400 MB total |
| Load balancer | ALB, HTTP/2, TLS termination |
| Auto-scale | Target tracking on `RequestCountPerTarget` @ 60% of measured peak |
| Min capacity | 2 tasks per AZ across 2-3 AZs |
| Health check | `/healthz` returning 200 once ORT session is loaded |

**Cold start:** ~10-20s (container pull + model load + ORT init). Scale out *before* needing capacity — target 60% utilization, not 85%.

**Spot for cost savings:** mix on-demand (baseline) + spot (burst) via capacity providers. ~60-70% cost cut on spot portion.

**Start on Fargate** if ops simplicity matters more than cost; move to EC2 once sustained traffic justifies it (~50+ QPS).

## Performance Expectations

Per-instance, INT8 quantized XLM-R base, 30-60 token utterances:

| Instance | QPS | p50 latency | p99 latency |
|----------|-----|-------------|-------------|
| `c7g.large` (2 vCPU) | 40-80 | 50-90 ms | ~150 ms |
| `c7g.xlarge` (4 vCPU) | 80-150 | 50-90 ms | ~150 ms |
| `c7g.2xlarge` (8 vCPU) | 150-280 | 50-90 ms | ~150 ms |

Scales linearly with vCPUs (within a process) and linearly with task count (across the service). 10K sustained QPS ≈ 67 `c7g.xlarge` tasks.

**Cost at scale:** ~$0.25 per million requests on-demand, ~$0.10 per million on spot (compute + ALB).

## Gotchas to Watch

- **Tokenizer must be the Rust `tokenizers` library**, not Python `transformers` slow tokenizer. Verify with `tokenizer.is_fast` if you load via `AutoTokenizer`. At sub-100ms latencies the slow tokenizer becomes 20-40% of request time.
- **Reject inputs >2000 chars** at the edge. Prevents pathological requests from blocking workers.
- **One uvicorn worker per task.** Set `--workers 1`. Threading happens inside ORT.
- **Pin `OMP_NUM_THREADS` and `MKL_NUM_THREADS`** to match `intra_op_num_threads` to avoid thread oversubscription:
  ```bash
  ENV OMP_NUM_THREADS=4
  ENV MKL_NUM_THREADS=4
  ```
- **Model load on import, not on first request.** Otherwise the first request after scale-out gets a 5-10s spike.
- **Health check should fail until model is loaded.** Otherwise ALB sends traffic to a cold task.
- **Log p50/p95/p99 latency and tokens-per-request** from day one. You can't tune what you can't see.

## What's Explicitly Not Here

If product asks for any of these, this service does not handle them — they belong upstream/downstream or in a different service:

- Custom user dictionaries
- Voice commands ("delete that", "new line")
- Style adaptation (Slack vs email tone)
- Disfluency removal
- Grammar correction
- Per-user context or memory

## Container Skeleton

```dockerfile
FROM python:3.12-slim

ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] onnxruntime tokenizers pydantic

COPY model.int8.onnx tokenizer.json labels.json ./
COPY app.py ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

## Build Order

1. Export + quantize model locally, verify accuracy on a held-out set vs the original PyTorch model (should be within 1 F1 point).
2. Write FastAPI handler + chunking + casing post-step. Benchmark locally with `wrk` or `vegeta`.
3. Containerize, push to ECR.
4. Stand up ECS service with 2 tasks, ALB, basic CloudWatch metrics.
5. Load-test with realistic utterance distribution. Tune `intra_op_num_threads` and instance size.
6. Configure auto-scaling policy based on observed peak.
7. Ship.
