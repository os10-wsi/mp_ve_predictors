#!/usr/bin/env python3
"""Generate PREDICTORS.md from the registry, so the table cannot drift.

Run it after adding or changing a predictor:

    python scripts/generate_predictors_md.py

A test asserts that the committed file matches what this produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vepbench.config import PATHS
from vepbench.envs import ENVIRONMENTS
from vepbench.predictors import REGISTRY

CATEGORY_TITLES = {
    "language_model": "Protein language models",
    "evolutionary": "Evolutionary models (need an alignment)",
    "inverse_folding": "Inverse folding",
    "structure_aware": "Newer structure-aware models",
    "stability": "Stability predictors (ddG)",
    "baseline": "Baselines implemented in this repository",
}
CATEGORY_ORDER = ["language_model", "evolutionary", "inverse_folding",
                  "structure_aware", "stability", "baseline"]

PREAMBLE = """# Predictors

Generated from the registry by `scripts/generate_predictors_md.py`. Do not
edit by hand; edit the entries in `vepbench/predictors/` and regenerate.

`vepbench list` prints the same information at the terminal, and
`vepbench list --json` emits it as JSON.

## How to read the sign conventions

Every predictor records which way its raw score runs. The pipeline writes both
the raw `score` and a `score_harmonized` column oriented so that **higher
always means more damaging or more destabilizing**, in the same units. Nothing
is rescaled; only the sign is flipped where needed.

A `no` in the **Confirmed** column means the convention was inferred rather
than read from the method's own documentation. Confirm those before trusting a
benchmark: an inverted sign turns the best predictor into the worst. Run
`vepbench check-conventions` for the current list.

## Data leakage

Two columns flag training-set overlap that matters when benchmarking against a
deep mutational scan:

- **Megascale**: trained on Tsuboyama et al. 2023 mega-scale stability data.
- **ProteinGym**: the method's own paper reports state-of-the-art ProteinGym
  results, so that benchmark likely influenced model selection.

**Supervised** marks methods fitted to experimental variant-effect or
stability measurements, as opposed to zero-shot likelihood models.

The ProteinGym column is deliberately conservative. Many other methods here
were also evaluated on ProteinGym, so check each paper before treating an
unflagged method as free of that influence.

## GPU memory

**VRAM** is the approximate peak for a single protein in float32. Half
precision roughly halves it, and `--dtype float16` or `--dtype bfloat16`
enables that. Ensemble members load one at a time, so ESM-1v peaks at one 650M
model rather than five.

## Membrane proteins

Every stability predictor here (FoldX, RaSP, ThermoMPNN and the two dG
heads) and both inverse-folding models were trained or parameterised largely
on soluble, globular proteins. For a membrane protein their scores are out of
distribution, and a lipid-facing residue looks to them like exposed
hydrophobic surface in water. The methods are not modified to compensate; the
caveat is recorded in each run's `metadata.json` and repeated in the README.
"""


def table(predictors) -> str:
    header = (
        "| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | "
        "Supervised | Megascale | ProteinGym | Environment |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for predictor in predictors:
        spec = predictor.spec
        needs = ", ".join(sorted(spec.requires - {"sequence"})) or "sequence only"
        gpu = "required" if spec.gpu_required else (
            "recommended" if spec.gpu_recommended else "no"
        )
        vram = f"{spec.approximate_vram_gb:.0f} GB" if spec.approximate_vram_gb else "-"
        rows.append(
            f"| `{spec.name}` | {needs} | {spec.convention.score_type} | "
            f"{spec.convention.higher_is.replace('_', ' ')} | "
            f"{'yes' if spec.convention.verified else '**no**'} | {gpu} | {vram} | "
            f"{'yes' if spec.supervised else 'no'} | "
            f"{'yes' if spec.trained_on_megascale else 'no'} | "
            f"{'yes' if spec.trained_on_proteingym else 'no'} | "
            f"`{spec.env or 'in-process'}` |"
        )
    return header + "\n".join(rows) + "\n"


def details(predictors) -> str:
    blocks = []
    for predictor in predictors:
        spec = predictor.spec
        lines = [f"#### `{spec.name}` — {spec.display_name}", "",
                 f"- **Paper**: {spec.citation}"]
        if spec.repo_url:
            lines.append(f"- **Code**: {spec.repo_url}")
        lines.append(f"- **Licence**: {spec.license}")
        lines.append(f"- **Model version**: {spec.model_version}")
        lines.append(
            f"- **Score**: {spec.convention.score_type}, higher is "
            f"{spec.convention.higher_is.replace('_', ' ')}"
        )
        lines.append(f"- **Convention source**: {spec.convention.source}")
        if spec.approximate_vram_gb:
            lines.append(f"- **Approximate VRAM**: {spec.approximate_vram_gb:.0f} GB")
        if spec.install_hint:
            lines.append(f"- **Install**: `{spec.install_hint}`")
        if spec.env and spec.env in ENVIRONMENTS:
            state = "installed" if ENVIRONMENTS[spec.env].is_installed(PATHS) else "not installed"
            lines.append(f"- **Status here**: environment `{spec.env}` is {state}")
        if spec.notes:
            lines.append(f"- **Notes**: {spec.notes}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


FOOTER = """
## Methods whose existence was checked

Six items were flagged as uncertain. All six are real; here is what was found
and what this repository does about each.

| Asked about | Status | Reference |
|---|---|---|
| **S3F** | Exists as described: sequence, structure and surface. | Zhang Z, Notin P, Huang Y, Lozano AC, Chenthamarakshan V, Marks D, Das P, Tang J. *Multi-Scale Representation Learning for Protein Fitness Prediction*, NeurIPS 2024, [arXiv:2412.01108](https://arxiv.org/abs/2412.01108). Code: [DeepGraphLearning/S3F](https://github.com/DeepGraphLearning/S3F) |
| **VenusREM** | Exists. Retrieval-enhanced; reported state of the art on 217 ProteinGym assays. | Tan Y et al., *Bioinformatics* 2025;41(Suppl 1):i401–i409, [doi:10.1093/bioinformatics/btaf189](https://doi.org/10.1093/bioinformatics/btaf189). Code: [ai4protein/VenusREM](https://github.com/ai4protein/VenusREM), mirror [tyang816/VenusREM](https://github.com/tyang816/VenusREM) |
| **ProMEP** | Exists. Alignment-free multimodal model over sequence and structure. | Cheng P et al., *Cell Research* 2024;34:630–647. Code: [wenjiegroup/ProMEP](https://github.com/wenjiegroup/ProMEP) |
| **METL** | Exists. Biophysics-pretrained on synthetic Rosetta data. | Gelman S et al., *Biophysics-based protein language models for protein engineering*, *Nature Methods* 2025. Code: [gitter-lab/metl](https://github.com/gitter-lab/metl), [gitter-lab/metl-pretrained](https://github.com/gitter-lab/metl-pretrained) |
| **SaProtΔG** | **Not a separate publication.** The real implementation is a fine-tuned stability head, not a masked-marginal score. | [yehlincho/absolute-stability-predictor](https://github.com/yehlincho/absolute-stability-predictor): SaProt 650M with LoRA r=4 plus a stability head, trained on experimental ΔG and ΔΔG, ensembled over 3 checkpoints. Weights: `Yehlin/absolute-stability` on Hugging Face |
| **ESM3ΔG** | **Not a separate publication.** Same repository, ESM3 backbone. | As above. The ESM3 half additionally requires accepting the EvolutionaryScale licence |

### How the two ΔG methods are handled

Because SaProtΔG and ESM3ΔG turned out to be supervised regression heads
rather than likelihood scores, each is registered **twice**, and the two can
never be confused in a merged table:

- `saprot_dg` and `esm3_dg` wrap the published implementation. They output
  ΔΔG in kcal/mol, are supervised, and are flagged for Megascale leakage.
- `saprot_dg_masked_marginal` and `esm3_dg_masked_marginal` are the standard
  masked-marginal log-likelihood ratios from the same backbones, the
  documented fallback. They are **not** free-energy estimates, their units are
  log-likelihood, and their notes say so.

## Assumptions worth knowing

- **ESM-1v and ESM-2 load through Hugging Face `transformers`**, not
  `fair-esm`. The weights are identical. This gives clean device and dtype
  control for GPU runs and keeps the checkpoints on one host. ESM-IF1 is the
  exception: it is distributed only through `fair-esm`, so its weights come
  from `dl.fbaipublicfiles.com`.
- **ESM-1v context limit.** ESM-1v accepts 1022 residues. Longer proteins are
  tiled: each position is scored in the window where it is most central, and
  the window used is written to `outputs/<id>/per_member/esm1v_full.csv`.
- **Domain windows.** For a membrane protein the default window spans the
  first to the last UniProt transmembrane segment, which is usually what a
  biogenesis assay probes. Override with `--domain START-END`. The choice and
  its source go into `metadata.json`.
- **ProtT5 and Ankh** have no masked-LM head, so they are scored with a
  span-denoising reconstruction likelihood at a sentinel token. That is a
  baseline of this repository's choosing, not a scheme either set of authors
  proposed.
- **ESM-IF1** is scored by default with a single teacher-forced pass, giving
  `log p(mut | prefix, backbone) - log p(wt | prefix, backbone)`. The exact
  full-sequence rescoring from the paper is available with the
  `scoring: full_sequence` option but costs 19L forward passes.
- **popEVE** uses the published proteome-wide precomputed scores by default.
  Running it locally needs EVE and ESM-1v scores for the protein first.
"""


def build() -> str:
    parts = [PREAMBLE]
    for category in CATEGORY_ORDER:
        members = sorted((p for p in REGISTRY.values() if p.spec.category == category),
                         key=lambda p: p.spec.name)
        if not members:
            continue
        parts.append(f"\n## {CATEGORY_TITLES[category]}\n")
        parts.append(table(members))
        parts.append("\n### Details\n")
        parts.append(details(members))
    parts.append(FOOTER)
    return "\n".join(parts)


def main() -> int:
    target = ROOT / "PREDICTORS.md"
    target.write_text(build())
    print(f"wrote {target} ({len(REGISTRY)} predictors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
