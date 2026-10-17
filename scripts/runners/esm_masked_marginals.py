#!/usr/bin/env python3
"""Masked-marginal scoring for the ESM-1v and ESM-2 families.

Score, following Meier et al. 2021 (NeurIPS), is the masked marginal
log-likelihood ratio

    s(wt -> mut, i) = log p(x_i = mut | x_\\i) - log p(x_i = wt | x_\\i)

so a positive score means the substitution is better tolerated than the wild
type. The orchestrator flips the sign when it writes ``score_harmonized``.

Models are loaded through Hugging Face ``transformers`` rather than
``fair-esm``. The weights are the same, but this gives clean device and dtype
control for GPU runs, and keeps every ESM checkpoint on one host.

Proteins longer than the model's context are tiled: each position is scored in
the window where it is most central, which is the strategy the ESM-1v paper
describes for long sequences. The window used for every position is written to
the output so the choice is auditable.

Ensemble members are loaded one at a time and freed, so peak memory is one
model, not five.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (
    apply_domain_window,
    default_batch_size,
    describe_hardware,
    fail,
    hf_cache_note,
    parse_job,
    pick_device,
    pick_dtype,
    read_variants,
    report_oom_hint,
    window_for_position,
    write_meta,
    write_scores,
)

# fair-esm and the Hugging Face ports share a 1024-token positional embedding,
# two slots of which are the BOS and EOS tokens.
DEFAULT_CONTEXT_LIMIT = 1022


def masked_marginals(model, tokenizer, sequence, positions, device, limit, batch_size):
    """Log-softmax over the vocabulary at each requested position."""
    import torch

    length = len(sequence)
    results: dict[int, dict] = {}

    # Group positions by the window they will be scored in, so identical
    # windows share work.
    grouped: dict[tuple[int, int], list[int]] = {}
    for position in positions:
        grouped.setdefault(window_for_position(position, length, limit), []).append(position)

    for (start, end), window_positions in grouped.items():
        sub_sequence = sequence[start - 1 : end]
        for chunk_start in range(0, len(window_positions), batch_size):
            chunk = window_positions[chunk_start : chunk_start + batch_size]
            texts = []
            for position in chunk:
                offset = position - start          # 0-based index inside the window
                texts.append(
                    sub_sequence[:offset] + tokenizer.mask_token + sub_sequence[offset + 1 :]
                )
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, position in enumerate(chunk):
                offset = position - start
                results[position] = {
                    # +1 for the BOS token that the ESM tokenizer prepends
                    "log_probs": log_probs[row, offset + 1].cpu(),
                    "window": (start, end),
                }
    return results


def score_with_model(model_name, sequence, variants, options, device, dtype):
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = EsmForMaskedLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:                      # transformers < 4.56 spelled it torch_dtype
        model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()

    limit = int(options.get("context_limit") or DEFAULT_CONTEXT_LIMIT)
    batch_size = default_batch_size(options, device)
    positions = sorted({position for position, _wt, _mut in variants})

    try:
        marginals = masked_marginals(
            model, tokenizer, sequence, positions, device, limit, batch_size
        )
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        fail(f"{exc}{report_oom_hint(device, model_name)}")

    vocabulary = tokenizer.get_vocab()
    scores: dict[tuple[int, str], float] = {}
    windows: dict[int, tuple[int, int]] = {}
    for position, wt, mut in variants:
        entry = marginals[position]
        windows[position] = entry["window"]
        if mut == wt:
            scores[(position, mut)] = 0.0
            continue
        wt_id, mut_id = vocabulary.get(wt), vocabulary.get(mut)
        if wt_id is None or mut_id is None:
            scores[(position, mut)] = float("nan")
            continue
        scores[(position, mut)] = float(entry["log_probs"][mut_id] - entry["log_probs"][wt_id])

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores, windows


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    sequence, variants, offset = apply_domain_window(job, variants, options)

    device = pick_device(options)
    dtype = pick_dtype(options, device)
    hf_cache_note()

    model_names = options.get("models") or [options.get("model")]
    model_names = [m for m in model_names if m]
    if not model_names:
        fail("no model specified; set the 'model' or 'models' option")

    per_model = []
    windows: dict[int, tuple[int, int]] = {}
    for model_name in model_names:
        print(f"  scoring with {model_name} on {device} ({dtype})", file=sys.stderr)
        scores, windows = score_with_model(
            model_name, sequence, variants, options, device, dtype
        )
        per_model.append(scores)

    rows = []
    for position, wt, mut in variants:
        values = [scores[(position, mut)] for scores in per_model]
        start, end = windows[position]
        row = {
            "position": position + offset, "wt_aa": wt, "mut_aa": mut,
            "score": sum(values) / len(values),
            "window_start": start + offset, "window_end": end + offset,
        }
        for index, value in enumerate(values, start=1):
            row[f"score_member{index}"] = value
        rows.append(row)

    extra = ["window_start", "window_end"] + [
        f"score_member{i}" for i in range(1, len(per_model) + 1)
    ]
    write_scores(job, rows, extra_columns=extra)
    write_meta(job, {
        "model_version": "+".join(model_names),
        "models": model_names,
        "scoring": "masked marginals, log p(mut) - log p(wt)",
        "context_limit": int(options.get("context_limit") or DEFAULT_CONTEXT_LIMIT),
        "tiled": len({tuple(w) for w in windows.values()}) > 1,
        "domain_window": job.get("domain") if options.get("use_domain") else None,
        "dtype": str(dtype),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
