"""
One-shot script: download `oliverguhr/fullstop-punctuation-multilingual-base`
from Hugging Face, export to ONNX via optimum, then dynamically quantize
to INT8. Artifacts land in `./model-onnx/` which app.py reads at startup.

Run once locally (or in the Docker build stage). The output directory is
gitignored; ship it by baking it into the container image.

    pip install -r requirements-export.txt
    python export_model.py

Takes ~3-5 minutes on first run: 1 GB model download + ONNX export +
quantization pass. Subsequent runs are quick (HF cache hits) if you
haven't cleared `~/.cache/huggingface/`.
"""

from pathlib import Path
import shutil
import subprocess
import sys

from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "oliverguhr/fullstop-punctuation-multilingual-base"
OUT_DIR = Path(__file__).parent / "model-onnx"
FP32_MODEL = OUT_DIR / "model.onnx"
INT8_MODEL = OUT_DIR / "model.int8.onnx"


def export_onnx() -> None:
    """Use optimum-cli to produce model.onnx + tokenizer/config JSONs."""
    if FP32_MODEL.exists():
        print(f"[export] {FP32_MODEL} already exists, skipping export")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[export] exporting {MODEL_ID} to {OUT_DIR} …")
    subprocess.check_call([
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL_ID,
        "--task", "token-classification",
        str(OUT_DIR),
    ])


def quantize_int8() -> None:
    """Dynamic INT8 quantization — roughly halves size + latency, <1 F1 point drop."""
    if INT8_MODEL.exists():
        print(f"[quant] {INT8_MODEL} already exists, skipping quantization")
        return
    print(f"[quant] quantizing {FP32_MODEL} → {INT8_MODEL} …")
    quantize_dynamic(
        model_input=str(FP32_MODEL),
        model_output=str(INT8_MODEL),
        weight_type=QuantType.QInt8,
    )
    print(f"[quant] done. sizes:")
    print(f"        fp32 {FP32_MODEL.stat().st_size / 1e6:.1f} MB")
    print(f"        int8 {INT8_MODEL.stat().st_size / 1e6:.1f} MB")


def clean_fp32() -> None:
    """Drop the fp32 .onnx + its external-data shards. We only ship int8."""
    removed = 0
    for pattern in ["model.onnx", "model.onnx_data", "model.onnx.data"]:
        p = OUT_DIR / pattern
        if p.exists():
            p.unlink()
            removed += 1
    if removed:
        print(f"[clean] removed {removed} fp32 artifact(s) (only int8 ships)")


if __name__ == "__main__":
    export_onnx()
    quantize_int8()
    clean_fp32()
    print("[done] artifacts in", OUT_DIR)
    print("       required files:", sorted(p.name for p in OUT_DIR.iterdir()))
