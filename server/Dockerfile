# Punctuation service container. Assumes `python export_model.py` has
# already produced `model-onnx/` in the build context (gitignored, so
# you run it locally before `docker build`). The image does NOT contain
# torch / transformers / optimum — those are only needed for export.

FROM python:3.12-slim

# Pin thread counts to avoid oversubscription: onnxruntime's
# intra_op_num_threads already parallelizes within an inference. Letting
# OpenMP / MKL spin up additional threads on top just thrashes the CPU.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Stage 3 ITN needs OpenFst (pynini's C++ dep) + a toolchain to build it
# from source. libgomp1 is onnxruntime's OpenMP runtime. build-essential
# is only required for the pynini build; it could be dropped in a
# multi-stage build later if image size becomes a concern.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libfst-dev \
        libfst-tools \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# --no-deps on nemo_text_processing because its strict `pynini==2.1.6.post1`
# pin fails to build against current openfst; we pin pynini==2.1.7
# ourselves (requirements.txt covers the rest of the deps).
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps nemo_text_processing==1.1.0

# Model artifacts first (changes rarely) then app code (changes often) —
# Docker layer cache friendly.
COPY model-onnx/ ./model-onnx/
COPY app.py ./

# Pre-compile ITN WFST grammars so cold-start readiness is fast. Without
# this the first `/punctuate/stream` request eats ~1-2 s of lazy init.
RUN mkdir -p itn_cache && \
    python -c "from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer; InverseNormalizer(lang='en', cache_dir='/app/itn_cache').inverse_normalize('one two three', verbose=False)"

EXPOSE 8000

# One uvicorn worker per task. Python-level concurrency doesn't help
# here; ONNX Runtime handles intra-inference threading.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
