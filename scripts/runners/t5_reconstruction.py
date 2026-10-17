#!/usr/bin/env python3
"""Reconstruction-likelihood scoring for the T5-style models, ProtT5 and Ankh.

These are encoder-decoder models trained with span denoising, so they have no
masked-LM head. The substitute used here replaces the target residue with a
sentinel token and reads the decoder's distribution over the first
reconstructed token:

    s = log p(mut | sequence corrupted at i) - log p(wt | sequence corrupted at i)

That is a reasonable likelihood baseline, not a scheme proposed by the authors
of either model, and the predictor entries say so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (
    default_batch_size,
    describe_hardware,
    fail,
    hf_cache_note,
    parse_job,
    pick_device,
    pick_dtype,
    read_variants,
    report_oom_hint,
    write_meta,
    write_scores,
)


def main() -> int:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    dtype = pick_dtype(options, device)
    model_name = options["model"]
    hf_cache_note()

    # ProtT5 expects residues separated by spaces, with U, Z, O and B as X.
    residues = re.sub(r"[UZOB]", "X", sequence)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    try:
        try:
            model = T5ForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype
            )
    except Exception as exc:  # noqa: BLE001
        fail(
            f"could not load {model_name} as a conditional-generation model ({exc}). "
            f"Encoder-only checkpoints cannot produce reconstruction likelihoods; use "
            f"the full checkpoint, for example Rostlab/prot_t5_xl_uniref50."
        )
    model = model.to(device).eval()

    sentinel = options.get("sentinel", "<extra_id_0>")
    sentinel_id = tokenizer.convert_tokens_to_ids(sentinel)
    if sentinel_id is None or sentinel_id == tokenizer.unk_token_id:
        fail(f"the tokenizer for {model_name} has no sentinel token {sentinel!r}")

    positions = sorted({p for p, _w, _m in variants})
    batch_size = default_batch_size(options, device)
    log_probs: dict[int, object] = {}
    try:
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            texts = []
            for position in chunk:
                local = list(residues)
                local[position - 1] = sentinel
                texts.append(" ".join(local))
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            # Decoder input is the start token then the sentinel, so the next
            # predicted token is the reconstructed residue.
            decoder_input = torch.tensor(
                [[model.config.decoder_start_token_id, sentinel_id]] * len(chunk)
            ).to(device)
            with torch.no_grad():
                output = model(**encoded, decoder_input_ids=decoder_input)
            for row, position in enumerate(chunk):
                log_probs[position] = torch.log_softmax(
                    output.logits[row, -1].float(), dim=-1
                ).cpu()
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        fail(f"{exc}{report_oom_hint(device, model_name)}")

    rows = []
    for position, wt, mut in variants:
        vector = log_probs[position]
        wt_id = tokenizer.convert_tokens_to_ids(wt)
        mut_id = tokenizer.convert_tokens_to_ids(mut)
        if tokenizer.unk_token_id in (wt_id, mut_id):
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            continue
        score = 0.0 if mut == wt else float(vector[mut_id] - vector[wt_id])
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": score})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": model_name,
        "scoring": "span-denoising reconstruction likelihood ratio at a sentinel token",
        "dtype": str(dtype), **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
