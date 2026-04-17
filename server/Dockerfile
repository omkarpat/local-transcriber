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

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Model artifacts first (changes rarely) then app code (changes often) —
# Docker layer cache friendly.
COPY model-onnx/ ./model-onnx/
COPY app.py ./

EXPOSE 8000

# One uvicorn worker per task. Python-level concurrency doesn't help
# here; ONNX Runtime handles intra-inference threading.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
