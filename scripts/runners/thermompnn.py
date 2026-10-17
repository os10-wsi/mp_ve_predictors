#!/usr/bin/env python3
"""ThermoMPNN saturation-scan driver.

ThermoMPNN scores every position in one pass over the structure, which is
what a saturation scan wants. This imports the upstream model and checkpoint
rather than reimplementing them.

Sign convention: the paper reports ddG with positive meaning destabilising,
but the raw Megascale training data uses the opposite sign and some forks flip
it. The predictor entry marks this unverified.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import (  # noqa: E402
    describe_hardware, fail, parse_job, pick_device, read_variants, write_meta,
    write_scores,
)
from _third_party import add_to_path, repo_path  # noqa: E402

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def main() -> int:
    import torch

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    device = pick_device(options)

    if not job.get("structure_path"):
        fail("ThermoMPNN needs a structure")

    repo = repo_path(job, "ThermoMPNN", "THERMOMPNN_DIR",
                     "https://github.com/Kuhlman-Lab/ThermoMPNN",
                     "scripts/install_proteinmpnn.sh")
    add_to_path(repo)
    try:
        from thermompnn.inference.inference_utils import get_trained_model
        from thermompnn.protein_mpnn_utils import parse_PDB, tied_featurize
    except ImportError as exc:
        fail(
            f"could not import ThermoMPNN from {repo} ({exc}). The upstream layout "
            f"changes between releases; check scripts/install_proteinmpnn.sh"
        )

    checkpoint = options.get("checkpoint") or str(repo / "models" / "thermoMPNN_default.pt")
    if not Path(checkpoint).exists():
        found = sorted(repo.rglob("*.pt")) + sorted(repo.rglob("*.ckpt"))
        fail(
            f"ThermoMPNN checkpoint not found at {checkpoint}. Checkpoints in the "
            f"checkout: {[str(p) for p in found[:5]] or 'none'}. Run "
            f"bash scripts/download_weights_thermompnn.sh"
        )

    model = get_trained_model(checkpoint, model_name="ThermoMPNN", device=device).eval()
    parsed = parse_PDB(job["structure_path"],
                       input_chain_list=[job.get("structure_chain") or "A"])
    features = tied_featurize(parsed, device, None, None, None, None, None, None,
                              ca_only=False)
    X, S, mask, _lengths, chain_M, chain_encoding, residue_idx = features[:7]

    structure_sequence = "".join(
        AA_ORDER[i] for i in S[0].cpu().numpy() if i < len(AA_ORDER)
    )
    offset = sequence.find(structure_sequence)
    if offset == -1:
        offset = 0

    with torch.no_grad():
        predictions = model(X, S, mask, chain_M, residue_idx, chain_encoding)
    ddg = predictions["ddG"] if isinstance(predictions, dict) else predictions
    ddg = ddg.squeeze().cpu().numpy()      # (length, 20)

    index_of = {aa: i for i, aa in enumerate(AA_ORDER)}
    rows = []
    for position, wt, mut in variants:
        index = position - 1 - offset
        if index < 0 or index >= ddg.shape[0]:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            continue
        value = 0.0 if mut == wt else float(ddg[index, index_of[mut]])
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": value})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"ThermoMPNN ({Path(checkpoint).name})",
        "scoring": "predicted ddG in kcal/mol; sign convention UNVERIFIED",
        "structure_offset": offset,
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
