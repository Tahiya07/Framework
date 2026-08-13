#!/usr/bin/env python
"""Export the merged Bloom classifier to ONNX and dynamic INT8 for CPU tests.

This does not replace the deployed PyTorch classifier.  It creates candidate
artifacts and rejects them when their predictions disagree too much with the
merged FP32 checkpoint.  Dynamic ONNX quantization is usually the right first
CPU experiment: it compresses MatMul/Gemm weights without requiring a
calibration dataset.

Install the optional tooling once:
    pip install onnx==1.16.2 onnxruntime==1.18.1

Then export and validate:
    python export_bloom_onnx_cpu.py --model-size 0.5b
    python evaluate_cpu_deployment.py --model-size 0.5b --onnx-dir models/qwen_bloom_onnx_int8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from bloom_model_profiles import DEFAULT_MODEL_SIZE, get_profile
from predict_bloom import LABELS, _load_tokenizer, build_prompt, load_model, predict

QUESTIONS = [
    "Define a computer network.",
    "Explain how binary search works.",
    "Use Ohm's law to calculate current from voltage and resistance.",
    "Compare arrays and linked lists for a memory-constrained program.",
    "Critique the reliability of this experimental design.",
    "Design a study plan for learning data structures.",
]


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def output_size_mib(path: Path) -> float:
    return round(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 2**20, 2)


def onnx_predictions(model_path: Path, tokenizer, questions: list[str]) -> list[str]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    labels: list[str] = []
    for question in questions:
        encoded = tokenizer(build_prompt(question), truncation=True, max_length=256, return_tensors="np")
        feeds = {name: encoded[name].astype(np.int64) for name in input_names if name in encoded}
        logits = session.run(None, feeds)[0][0]
        labels.append(LABELS[int(np.argmax(logits))])
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an offline CPU ONNX/INT8 Bloom classifier candidate.")
    parser.add_argument("--model-size", choices=("0.5b", "1.5b"), default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--merged-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--min-agreement", type=float, default=0.98)
    parser.add_argument("--keep-fp32", action="store_true", help="Keep model_fp32.onnx alongside model_int8.onnx.")
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit("Install optional export tools first: pip install onnx==1.16.2 onnxruntime==1.18.1") from exc

    profile = get_profile(args.model_size)
    source = Path(args.merged_dir or profile.merged_dir)
    output = Path(args.output_dir or f"models/qwen_bloom_onnx_int8{profile.key}")
    if not (source / "config.json").is_file():
        raise SystemExit(f"Merged checkpoint not found: {source}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    tokenizer = _load_tokenizer(str(source), fallback_path=None)
    print(f"Loading merged FP32 checkpoint from {source}...")
    model = AutoModelForSequenceClassification.from_pretrained(str(source), torch_dtype=torch.float32, trust_remote_code=True).cpu().eval()
    example = tokenizer(build_prompt(QUESTIONS[0]), truncation=True, max_length=256, return_tensors="pt")
    fp32_path = output / "model_fp32.onnx"
    print("Exporting ONNX FP32...")
    torch.onnx.export(
        LogitsOnly(model),
        (example["input_ids"], example["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    int8_path = output / "model_int8.onnx"
    print("Quantizing ONNX weights to dynamic INT8...")
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, per_channel=True, extra_options={"MatMulConstBOnly": True})
    tokenizer.save_pretrained(output)

    reference_tokenizer, reference_model = load_model(str(source))
    reference = [predict(question, reference_tokenizer, reference_model)["prediction"] for question in QUESTIONS]
    candidate = onnx_predictions(int8_path, tokenizer, QUESTIONS)
    agreement = sum(a == b for a, b in zip(reference, candidate)) / len(QUESTIONS)
    metadata = {
        "format": "onnx_dynamic_int8",
        "source_merged_dir": str(source),
        "model_size": profile.key,
        "opset": args.opset,
        "agreement_questions": len(QUESTIONS),
        "fp32_vs_int8_prediction_agreement": round(agreement, 4),
        "fp32_predictions": reference,
        "int8_predictions": candidate,
        "deployable": agreement >= args.min_agreement,
        "minimum_agreement": args.min_agreement,
        "size_mib": output_size_mib(output),
        "notes": "Candidate only. Benchmark it and validate held-out accuracy before production deployment.",
    }
    (output / "onnx_deployment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.keep_fp32:
        fp32_path.unlink(missing_ok=True)
        metadata["size_mib"] = output_size_mib(output)
        (output / "onnx_deployment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if not metadata["deployable"]:
        raise SystemExit("INT8 agreement did not meet the threshold; do not deploy this artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
