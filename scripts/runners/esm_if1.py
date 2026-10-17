#!/usr/bin/env python3
"""ESM-IF1 inverse-folding scoring.

Two modes:

``conditional`` (default)
    One teacher-forced pass over the wild-type sequence given the backbone
    yields p(x_i | x_<i, structure) at every position, and the score is
    ``log p(mut_i | ...) - log p(wt_i | ...)``. A whole saturation scan costs
    one forward pass.

``full_sequence``
    Rescores the entire mutated sequence for each variant, which is what the
    ESM-IF1 paper does for individual mutants. Exact, but 19L forward passes.

ESM-IF1 is the one model here still distributed solely through fair-esm, so
its weights come from dl.fbaipublicfiles.com. Positions with no residue in the
structure are written empty, not zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (  # noqa: E402
    describe_hardware, fail, parse_job, pick_device, read_variants, write_meta,
    write_scores,
)


def conditional_log_probs(model, alphabet, coords, sequence, device):
    """Per-position log p(residue | prefix, backbone) from one forward pass."""
    import torch
    from esm.inverse_folding.util import CoordBatchConverter

    converter = CoordBatchConverter(alphabet)
    coords_t, confidence, _strs, tokens, padding_mask = converter(
        [(coords, None, sequence)], device=device
    )
    with torch.no_grad():
        logits, _extra = model.forward(coords_t, padding_mask, confidence, tokens[:, :-1])
    # logits: (batch, vocabulary, length)
    return torch.log_softmax(logits[0].float(), dim=0).cpu()


def main() -> int:
    import esm
    import esm.inverse_folding as inverse_folding
    import torch

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)
    mode = options.get("scoring", "conditional")

    if not job.get("structure_path"):
        fail("ESM-IF1 needs a structure")

    torch.hub.set_dir(str(Path(job["weights_dir"]) / "torch_hub"))
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)

    structure = inverse_folding.util.load_structure(
        job["structure_path"], job.get("structure_chain") or "A"
    )
    coords, native = inverse_folding.util.extract_coords_from_structure(structure)

    # Map UniProt positions onto structure indices. A contiguous offset covers
    # the common case; anything else is reported rather than guessed.
    offset = sequence.find(native)
    if offset == -1:
        if len(native) == len(sequence):
            offset = 0
        else:
            fail(
                "the structure's sequence is not a contiguous match to the UniProt "
                "sequence, so ESM-IF1 positions cannot be mapped safely. Supply a "
                "structure covering the canonical sequence, or trim it first."
            )

    def structure_index(position: int) -> int | None:
        index = position - 1 - offset
        return index if 0 <= index < len(native) else None

    rows = []
    if mode == "conditional":
        log_probs = conditional_log_probs(model, alphabet, coords, native, device)
        for position, wt, mut in variants:
            index = structure_index(position)
            if index is None:
                rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
                continue
            if native[index] != wt:
                fail(
                    f"wild-type mismatch at UniProt position {position}: the structure "
                    f"has {native[index]}, the sequence has {wt}"
                )
            column = log_probs[:, index]
            score = 0.0 if mut == wt else float(
                column[alphabet.get_idx(mut)] - column[alphabet.get_idx(wt)]
            )
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": score})
    elif mode == "full_sequence":
        wild_type_ll, _ = inverse_folding.util.score_sequence(
            model, alphabet, coords, native
        )
        for position, wt, mut in variants:
            index = structure_index(position)
            if index is None:
                rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
                continue
            mutant = native[:index] + mut + native[index + 1 :]
            mutant_ll, _ = inverse_folding.util.score_sequence(
                model, alphabet, coords, mutant
            )
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                         "score": float(mutant_ll - wild_type_ll)})
    else:
        fail(f"unknown scoring mode {mode!r}; use 'conditional' or 'full_sequence'")

    scored = sum(1 for r in rows if r["score"] != "")
    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"esm_if1_gvp4_t16_142M_UR50 ({mode})",
        "scoring": mode, "structure_offset": offset,
        "scored_variants": scored, "unscored_variants": len(rows) - scored,
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
