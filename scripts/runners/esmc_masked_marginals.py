#!/usr/bin/env python3
"""Masked-marginal scoring for ESM C (ESM Cambrian).

Uses the EvolutionaryScale ``esm`` package. The score is the same
likelihood-ratio convention as the rest of the ESM family.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (  # noqa: E402
    describe_hardware, fail, hf_cache_note, parse_job, pick_device, read_variants,
    write_meta, write_scores,
)


def main() -> int:
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    model_name = options.get("model", "esmc_600m")
    hf_cache_note()

    if not os.environ.get("HF_TOKEN"):
        fail(
            "ESM C is gated on Hugging Face. Accept the licence on the model page and "
            "export HF_TOKEN before running it."
        )

    model = ESMC.from_pretrained(model_name).to(device).eval()
    config = LogitsConfig(sequence=True)

    log_probs = {}
    for position in sorted({p for p, _w, _m in variants}):
        masked = sequence[: position - 1] + "_" + sequence[position:]
        encoded = model.encode(ESMProtein(sequence=masked))
        with torch.no_grad():
            output = model.logits(encoded, config)
        # index position, not position-1, because of the BOS token
        log_probs[position] = torch.log_softmax(
            output.logits.sequence[0, position].float(), dim=-1
        ).cpu()

    tokenizer = model.tokenizer
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
        "model_version": model_name,
        "scoring": "masked marginals, log p(mut) - log p(wt)",
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
