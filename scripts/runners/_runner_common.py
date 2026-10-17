"""Helpers shared by the per-predictor runner scripts.

Runners execute inside an isolated environment holding the predictor's
dependencies, so this module stays limited to the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def parse_job() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    return json.loads(parser.parse_args().job.read_text())


def read_variants(job: dict) -> list[tuple[int, str, str]]:
    """Return (position, wt_aa, mut_aa) tuples."""
    rows: list[tuple[int, str, str]] = []
    with open(job["variants_csv"], newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["position"]), row["wt_aa"].strip(), row["mut_aa"].strip()))
    return rows


def write_scores(job: dict, rows: list[dict], extra_columns: list[str] | None = None) -> None:
    columns = ["position", "wt_aa", "mut_aa", "score"] + list(extra_columns or [])
    out = Path(job["out_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_meta(job: dict, meta: dict) -> None:
    Path(job["out_meta"]).write_text(json.dumps(meta, indent=2, default=str))


def fail(message: str, code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def pick_device(options: dict) -> str:
    """Choose cuda, mps or cpu, honouring an explicit option."""
    requested = options.get("device")
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(options: dict, device: str):
    """Resolve the compute dtype.

    Defaults to float32 on CPU, where half precision is slow or unsupported,
    and to bfloat16 on a CUDA device that supports it, which halves memory at
    no meaningful cost in accuracy for a log-ratio score. Override with the
    ``dtype`` option.
    """
    import torch

    requested = options.get("dtype")
    if requested:
        mapping = {"float32": torch.float32, "fp32": torch.float32,
                   "float16": torch.float16, "fp16": torch.float16,
                   "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
        if requested not in mapping:
            fail(f"unknown dtype {requested!r}; use float32, float16 or bfloat16")
        return mapping[requested]
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def default_batch_size(options: dict, device: str) -> int:
    if options.get("batch_size"):
        return int(options["batch_size"])
    return 16 if device == "cuda" else 4


def describe_hardware(device: str) -> dict:
    """Record what the run actually executed on, for metadata.json."""
    info: dict = {"device": device}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(index)
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(index).total_memory / 1e9, 1
        )
    return info


def report_oom_hint(device: str, model: str) -> str:
    if device != "cuda":
        return ""
    return (
        f"\n{model} ran out of GPU memory. Retry with --dtype float16, a smaller "
        f"--batch-size, or --device cpu."
    )


def hf_cache_note() -> None:
    """Hugging Face now serves large files through its Xet CDN by default.

    Some networks allow huggingface.co while blocking that CDN, which surfaces
    as a connection failure mid-download. The orchestrator sets
    HF_HUB_DISABLE_XET=1 for every runner; this only reports the state.
    """
    if os.environ.get("HF_HUB_DISABLE_XET") == "1":
        print("  (Hugging Face Xet CDN disabled; using the classic LFS endpoint)",
              file=sys.stderr)


def window_for_position(position: int, length: int, window: int) -> tuple[int, int]:
    """Window of ``window`` residues keeping ``position`` as central as possible.

    Positions and the returned bounds are 1-based and inclusive. Used for
    models with a hard context limit, such as ESM-1v's 1022 residues.
    """
    if length <= window:
        return 1, length
    half = window // 2
    start = max(1, position - half)
    end = start + window - 1
    if end > length:
        end = length
        start = end - window + 1
    return start, end


def apply_domain_window(job: dict, variants, options: dict):
    """Restrict scoring to the domain window when a predictor asks for it.

    Returns the sub-sequence, the variants renumbered into it, and the offset
    needed to put them back onto UniProt numbering.
    """
    sequence = job["sequence"]
    domain = job.get("domain")
    if not options.get("use_domain") or not domain:
        return sequence, variants, 0
    start, end = domain
    offset = start - 1
    kept = [(p - offset, w, m) for p, w, m in variants if start <= p <= end]
    if not kept:
        fail("no requested variants fall inside the chosen domain window")
    return sequence[start - 1 : end], kept, offset
