#!/usr/bin/env python3
"""METL scoring through metl-pretrained.

Gelman et al., Nature Methods 2025. The global source models predict Rosetta
biophysical attributes rather than assay fitness, so the score written here is
the predicted total Rosetta score, where higher energy is worse. The local
fine-tuned models predict assay-specific fitness instead and exist only for
proteins the authors simulated, none of them a membrane protein.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (
    describe_hardware,
    fail,
    parse_job,
    pick_device,
    read_variants,
    write_meta,
    write_scores,
)


def main() -> int:
    import torch

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    model_name = options.get("model", "metl-g-20m-3d")

    try:
        import metl
    except ImportError as exc:
        fail(
            f"metl-pretrained is not importable ({exc}). Install it with "
            f"python -m vepbench install-env metl"
        )

    if not job.get("structure_path") and model_name.endswith("3d"):
        fail(
            "the 3D METL models need a structure. The 1D models do not; select one "
            "with the model option, for example metl-g-20m-1d."
        )

    model, data_encoder = (
        metl.get_from_uuid(model_name) if len(model_name) == 8
        else metl.get_from_ident(model_name)
    )
    model = model.to(device).eval()

    target = options.get("target_attribute", "total_score")
    variant_strings = [f"{w}{p}{m}" for p, w, m in variants if m != w]
    encoded = data_encoder.encode_variants(sequence, variant_strings)
    with torch.no_grad():
        predictions = model(
            torch.tensor(encoded).to(device), pdb_fn=job.get("structure_path")
        ).cpu().numpy()

    names = list(getattr(model, "output_names", []) or [])
    if target in names:
        column = names.index(target)
    elif predictions.ndim == 1 or predictions.shape[1] == 1:
        column = 0
    else:
        fail(
            f"METL returned {predictions.shape[1]} outputs and {target!r} is not among "
            f"{names[:10]}. Set the target_attribute option to one of them."
        )

    lookup = {
        name: float(predictions[i][column] if predictions.ndim == 2 else predictions[i])
        for i, name in enumerate(variant_strings)
    }

    rows = []
    for position, wt, mut in variants:
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        value = lookup.get(f"{wt}{position}{mut}")
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                     "score": "" if value is None else value})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"METL {model_name} ({target})",
        "scoring": f"predicted Rosetta attribute {target}; higher energy is worse",
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
