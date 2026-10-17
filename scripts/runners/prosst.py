#!/usr/bin/env python3
"""ProSST scoring.

ProSST quantises local structure into a discrete vocabulary and attends over
sequence and structure tokens separately. Scoring follows the repository's own
zero-shot recipe: one forward pass over the wild-type sequence, then the
log-odds of the mutant against the wild type at each position.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (
    describe_hardware,
    fail,
    hf_cache_note,
    parse_job,
    pick_device,
    read_variants,
    write_meta,
    write_scores,
)
from _third_party import add_to_path, repo_path


def main() -> int:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    model_name = options.get("model", "AI4Protein/ProSST-2048")
    hf_cache_note()

    if not job.get("structure_path"):
        fail("ProSST needs a structure to build its quantised structure tokens")

    repo = repo_path(job, "ProSST", "PROSST_DIR",
                     "https://github.com/ai4protein/ProSST", "scripts/install_prosst.sh")
    add_to_path(repo)
    try:
        from prosst.structure.quantizer import PdbQuantizer
    except ImportError as exc:
        fail(
            f"could not import the ProSST structure quantizer from {repo} ({exc}); "
            f"check scripts/install_prosst.sh"
        )

    quantizer = PdbQuantizer(structure_vocab_size=int(options.get("vocab_size", 2048)))
    quantized = quantizer(job["structure_path"], return_residue_seq=False)
    if isinstance(quantized, dict):
        # Newer releases return a nested mapping keyed by file then vocab size.
        inner = next(iter(quantized.values()))
        quantized = next(iter(inner.values())) if isinstance(inner, dict) else inner
    structure_tokens = list(quantized)

    if len(structure_tokens) != len(sequence):
        fail(
            f"ProSST produced {len(structure_tokens)} structure tokens for a sequence "
            f"of {len(sequence)}; supply a structure covering the whole canonical "
            f"sequence"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(device).eval()

    encoded = tokenizer([sequence], return_tensors="pt").to(device)
    structure_input = torch.tensor(
        [[token + 3 for token in structure_tokens]], dtype=torch.long
    ).to(device)
    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            ss_input_ids=structure_input,
        ).logits
    log_probs = torch.log_softmax(logits[0].float(), dim=-1).cpu()

    vocabulary = tokenizer.get_vocab()
    rows = []
    for position, wt, mut in variants:
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        wt_id, mut_id = vocabulary.get(wt), vocabulary.get(mut)
        if wt_id is None or mut_id is None:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            continue
        vector = log_probs[position]        # +1 offset for the BOS token
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                     "score": float(vector[mut_id] - vector[wt_id])})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": model_name,
        "scoring": "wild-type marginals over sequence conditioned on quantised structure",
        "structure_vocab_size": options.get("vocab_size", 2048),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
