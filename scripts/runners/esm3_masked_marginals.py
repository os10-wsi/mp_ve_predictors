#!/usr/bin/env python3
"""Masked-marginal scoring for ESM3.

By default only the sequence track is used, which keeps the score comparable
with the other sequence language models. With ``structure_conditioned`` the
structure track is supplied as conditioning, which changes the score; that
configuration is registered separately as ``esm3_dg_masked_marginal`` so the
two are never confused.

This is a likelihood ratio, not a free energy. The published ESM3-based
stability model with a trained dG head is a different predictor, ``esm3_dg``.
"""

from __future__ import annotations

import os
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


def main() -> int:
    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.utils.constants.models import ESM3_OPEN_SMALL

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    structure_conditioned = bool(options.get("structure_conditioned"))
    hf_cache_note()

    if not os.environ.get("HF_TOKEN"):
        fail(
            "ESM3 weights require accepting the EvolutionaryScale Community licence at "
            "huggingface.co/EvolutionaryScale/esm3-sm-open-v1 and exporting HF_TOKEN."
        )

    model = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(device).eval()
    config = LogitsConfig(sequence=True)

    coordinates = None
    if structure_conditioned:
        if not job.get("structure_path"):
            fail("structure_conditioned was requested but no structure was supplied")
        from esm.utils.structure.protein_chain import ProteinChain

        chain = ProteinChain.from_pdb(
            job["structure_path"], chain_id=job.get("structure_chain")
        )
        coordinates = torch.tensor(chain.atom37_positions, dtype=torch.float32)
        if coordinates.shape[0] != len(sequence):
            fail(
                f"the structure has {coordinates.shape[0]} residues but the sequence "
                f"has {len(sequence)}; ESM3 structure conditioning needs them aligned. "
                f"Supply a structure covering the full canonical sequence, or run "
                f"esm3_open instead."
            )

    tokenizer = model.tokenizers.sequence
    log_probs = {}
    for position in sorted({p for p, _w, _m in variants}):
        masked = sequence[: position - 1] + "_" + sequence[position:]
        encoded = model.encode(ESMProtein(sequence=masked, coordinates=coordinates))
        with torch.no_grad():
            output = model.logits(encoded, config)
        log_probs[position] = torch.log_softmax(
            output.logits.sequence[0, position].float(), dim=-1
        ).cpu()

    rows = []
    for position, wt, mut in variants:
        vector = log_probs[position]
        score = 0.0 if mut == wt else float(
            vector[tokenizer.convert_tokens_to_ids(mut)]
            - vector[tokenizer.convert_tokens_to_ids(wt)]
        )
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": score})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": options.get("model", "esm3-sm-open-v1")
        + (" (structure conditioned)" if structure_conditioned else " (sequence only)"),
        "scoring": "masked marginals, log p(mut) - log p(wt)",
        "structure_conditioned": structure_conditioned,
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
