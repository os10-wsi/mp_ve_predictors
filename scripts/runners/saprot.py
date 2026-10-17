#!/usr/bin/env python3
"""SaProt masked-marginal scoring.

SaProt's vocabulary pairs each residue with its Foldseek 3Di structural
letter, so a token looks like ``Ad``: amino acid then structure letter.
Scoring masks the token at the target position and compares the logits of
``<mut><3Di>`` against ``<wt><3Di>``, holding the structural letter fixed,
which is the procedure in the SaProt repository's own scripts.

foldseek must be on PATH to derive the 3Di string.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (  # noqa: E402
    default_batch_size, describe_hardware, fail, hf_cache_note, parse_job, pick_device,
    pick_dtype, read_variants, report_oom_hint, write_meta, write_scores,
)


def foldseek_3di(structure_path: str, chain: str | None) -> str:
    """Run foldseek structureto3didescriptor and return the 3Di string."""
    executable = shutil.which("foldseek")
    if executable is None:
        fail(
            "foldseek is not on PATH. SaProt needs it for the 3Di structural alphabet. "
            "Install it with bash scripts/install_foldseek.sh"
        )
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out.tsv"
        completed = subprocess.run(
            [executable, "structureto3didescriptor", "-v", "0", "--threads", "1",
             "--chain-name-mode", "1", structure_path, str(output)],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            fail(f"foldseek failed:\n{completed.stderr[-1500:]}")
        wanted = (chain or "A").strip()
        fallback = None
        for line in output.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            name, descriptor = fields[0], fields[2]
            if fallback is None:
                fallback = descriptor
            if name.endswith((f"_{wanted}", f" {wanted}")):
                return descriptor
        if fallback is None:
            fail("foldseek produced no 3Di descriptor")
        return fallback


def main() -> int:
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    dtype = pick_dtype(options, device)
    model_name = options.get("model", "westlake-repl/SaProt_650M_AF2")
    hf_cache_note()

    if not job.get("structure_path"):
        fail("SaProt needs a structure to build its structural alphabet")

    structure_tokens = foldseek_3di(job["structure_path"], job.get("structure_chain"))
    if len(structure_tokens) != len(sequence):
        fail(
            f"the 3Di string has {len(structure_tokens)} letters but the sequence has "
            f"{len(sequence)}. SaProt needs a structure covering the whole canonical "
            f"sequence; supply one with --pdb."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = EsmForMaskedLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = EsmForMaskedLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()

    # SaProt lower-cases the 3Di letters in its combined vocabulary.
    tokens = [
        aa + letter.lower()
        for aa, letter in zip(sequence, structure_tokens, strict=True)
    ]

    positions = sorted({p for p, _w, _m in variants})
    batch_size = default_batch_size(options, device)
    log_probs: dict[int, object] = {}
    try:
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            texts = []
            for position in chunk:
                local = list(tokens)
                local[position - 1] = tokenizer.mask_token
                texts.append(" ".join(local))
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
            for row, position in enumerate(chunk):
                log_probs[position] = torch.log_softmax(
                    logits[row, position].float(), dim=-1
                ).cpu()
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        fail(f"{exc}{report_oom_hint(device, model_name)}")

    unknown = tokenizer.unk_token_id
    rows = []
    for position, wt, mut in variants:
        vector = log_probs[position]
        letter = structure_tokens[position - 1].lower()
        wt_id = tokenizer.convert_tokens_to_ids(wt + letter)
        mut_id = tokenizer.convert_tokens_to_ids(mut + letter)
        if unknown in (wt_id, mut_id):
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            continue
        score = 0.0 if mut == wt else float(vector[mut_id] - vector[wt_id])
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": score})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": model_name,
        "scoring": "masked marginals over the combined amino acid and 3Di vocabulary",
        "foldseek_3di_length": len(structure_tokens), "dtype": str(dtype),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
