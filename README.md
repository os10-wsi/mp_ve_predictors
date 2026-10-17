# vepbench

Saturation variant-effect and stability prediction across many published
predictors, for one UniProt ID at a time, in a form that merges cleanly with
deep mutational scanning data.

For a protein of length *L* it produces every single amino acid substitution
at every position, *L* x 19 rows, scored by each predictor you ask for, plus
per-position structural and evolutionary features.

Built for membrane-protein biogenesis benchmarking, which shapes several
decisions below; see [Membrane proteins](#membrane-proteins).

## Quick start

```bash
pip install -e .

python -m vepbench list            # what exists
python -m vepbench status          # which environments are installed

python -m vepbench install-env esm # one environment (see envs/README.md)

# one protein, one predictor
python -m vepbench run --uniprot P26678 --predictors esm2_650m

# a subset, or everything
python -m vepbench run --uniprot P26678 --predictors esm1v_full saprot foldx
python -m vepbench run --uniprot P26678 --predictors all

# many proteins
python -m vepbench run --uniprot-list ids.txt --predictors all
```

The orchestrator needs only numpy, pandas and requests. Predictors live in
isolated environments and are reached over subprocess, so installing vepbench
never pulls in torch.

### Worked example

```bash
bash examples/run_demo.sh
```

Runs predict, merge and benchmark end to end on phospholamban (P26678) using
only the built-in baselines, so it needs no model weights and no GPU. It
includes two deliberately broken DMS rows to show how unmatched rows are
reported.

## GPU

Everything that can use a GPU does, automatically. Three flags override the
defaults:

| flag | default |
|---|---|
| `--device` | `cuda` when available, else `mps`, else `cpu` |
| `--dtype` | `bfloat16` on a capable CUDA card, `float16` otherwise, `float32` on CPU |
| `--batch-size` | 16 on GPU, 4 on CPU |

`vepbench list` shows approximate VRAM per predictor. If a model runs out of
memory the error names the flags that fix it. Ensemble members load one at a
time and are freed, so ESM-1v peaks at one 650M model rather than five.

Hardware settings are deliberately **excluded** from the cache key. Moving a
model from CPU to GPU should not change its score, and if it does that is a
bug worth seeing rather than hiding.

The pinned torch in `envs/*/requirements.txt` targets CUDA 13. For an older
driver, install the matching build into the environment yourself; see
[`envs/README.md`](envs/README.md).

## Output

`outputs/<uniprot>/` holds:

| file | contents |
|---|---|
| `predictions_long.csv` | one row per (variant, predictor) |
| `predictions_wide.csv` | one row per variant, one column per predictor, plus features |
| `metadata.json` | sequence, structure, MSA statistics, domain window, predictor versions, hardware, runtimes, failures, caveats |
| `status.json` | per-predictor outcome: ok, cached, unavailable or failed, with the reason |
| `per_member/` | individual replicate scores, for example each of the five ESM-1v models |

### Long format

| column | meaning |
|---|---|
| `uniprot_id` | accession |
| `position` | 1-based position in the canonical UniProt sequence |
| `wt_aa`, `mut_aa` | wild-type and mutant residue |
| `variant` | `A123G` |
| `predictor` | predictor name |
| `score` | the raw number the method produces, untouched |
| `score_type` | `log_likelihood_ratio`, `ddG_kcal_mol`, `evolutionary_index`, ... |
| `higher_is` | `more_deleterious` or `more_tolerated`, documenting the raw convention |
| `score_harmonized` | sign-flipped so **higher always means more damaging**, same units |
| `model_version` | checkpoint or version string |

### Wide format

Key columns (`uniprot_id, position, wt_aa, mut_aa, variant`), then one raw
column per predictor, then a matching block of `<predictor>__harmonized`
columns, then the feature columns.

## Merging with a DMS table

```bash
python -m vepbench merge --dms my_dms.csv --uniprot P26678 \
    --variant-col variant --out merged.csv

# or with separate columns, three-letter codes accepted
python -m vepbench merge --dms my_dms.csv --uniprot P26678 \
    --position-col pos --wt-col wt --mut-col mut --out merged.csv
```

**Nothing is dropped silently.** Every input row comes back with a
`match_status`:

| status | meaning |
|---|---|
| `matched` | joined to a prediction |
| `wt_mismatch` | the table's wild-type residue disagrees with the UniProt sequence |
| `out_of_range` | position beyond the sequence |
| `multi_mutant` | more than one substitution in the field |
| `unparseable` | notation not recognised |
| `nonsense` | a stop codon; no substitution score exists |
| `synonymous` | scored only when `--include-wt` was used |
| `no_prediction` | parsed and valid, but no predictor produced a score |

A `match_detail` column explains each failure, the unmatched rows go to
`<out>.unmatched.csv`, and a summary to `<out>.merge_report.json`.

## Benchmarking

```bash
python -m vepbench benchmark --merged merged.csv --score-col abundance \
    --group-col uniprot_id --plot summary.png
```

Reports Spearman, Pearson, AUROC, NDCG and top-*k* recall with bootstrap
confidence intervals, per protein and pooled, computed on the harmonized
scores so every predictor is oriented the same way.

`--assay-direction` says which way your measured column runs. The default,
`functional`, means a higher measured value is a more functional protein,
which is usual for an abundance or biogenesis assay. Correlations are then
reported so that a predictor which correctly identifies damaging variants
scores **positively**.

## Inputs, fetched automatically

| input | source | override |
|---|---|---|
| sequence | UniProt REST, canonical isoform | `--sequence`, `--fasta` |
| structure | AlphaFold DB | `--pdb` |
| MSA | EBI PSI-BLAST, or ColabFold, or local jackhmmer/HHblits | `--msa`, `--msa-backend` |
| domain | UniProt Domain, else Pfam/InterPro, else transmembrane span | `--domain START-END` |

Everything is cached under `data/cache/`, and `--offline` refuses to touch the
network, which is how the test suite runs.

### Alignments

Nine of the predictors need an MSA. Four backends:

- **`ebi`** (default) submits PSI-BLAST or BLAST to the EBI Job Dispatcher and
  converts the hits into a query-anchored A3M. No local database, nothing to
  install. Jobs are queued server-side and take minutes, and alignments are
  shallower than a profile HMM search. EBI requires a contact address, so pass
  `--email` or set `VEPBENCH_EMAIL`.
- **`colabfold`** uses the public MMseqs2 server against UniRef30 plus
  environmental databases. The deepest alignments for the least effort, if
  `api.colabfold.com` is reachable.
- **`jackhmmer`** and **`hhblits`** search a database you supply with
  `--msa-database`. The most sensitive route. UniRef90 is roughly 60 GB and
  UniRef30 roughly 90 GB, so neither is downloaded for you.
- **`none`** skips the alignment, and MSA-dependent predictors report why.

Neff and coverage are computed for every alignment. Below Neff 100, or 50
sequences, the run warns that EVE, GEMME, EVmutation and DeepSequence are
unreliable, and records the warning in `metadata.json`.

### Numbering

Mismatches between UniProt numbering and structure numbering are the most
common source of silent errors here, so they are errors, not warnings:

- Structures are mapped by direct match, then contiguous offset, then global
  alignment. A residue that cannot be mapped confidently is left unmapped
  rather than guessed.
- Every output row is asserted to have `wt_aa` equal to the UniProt residue at
  that position. A predictor that violates this fails rather than writing a row.
- The mapping route and coverage go into `metadata.json`.

## Features

Added as columns in the wide table. On by default: `rsasa`, `plddt`,
`physchem`, `blosum`, `grantham`, `membrane`. Behind `--features`: `entropy`,
`gaps`, `depth`, `contacts`, `secondary_structure`, `bilayer_depth`.

| feature | notes |
|---|---|
| `rsasa` | Shrake-Rupley, 1.4 A probe, normalised by Tien et al. 2013 maxima |
| `plddt` | per-residue, AlphaFold models only |
| `contact_number` | C-alpha neighbours within 10 A |
| `residue_depth` | distance from C-alpha to the nearest accessible surface point |
| `secondary_structure` | DSSP, if `mkdssp` is installed; omitted otherwise |
| `msa_entropy_bits`, `msa_gap_fraction` | per-position conservation |
| `blosum62`, `grantham` | substitution matrices |
| `delta_hydropathy`, `delta_volume`, `delta_charge` | plus `introduces_proline`, `removes_glycine`, `introduces_charge` |
| `tm_segment`, `in_transmembrane`, `topology`, `residues_to_nearest_tm` | from UniProt |
| `bilayer_depth` | signed distance along the membrane normal, for an OPM or PDBTM model |

A feature whose input is missing is **omitted**, never filled with a
placeholder.

## Membrane proteins

This repository exists to benchmark predictors against a membrane-protein
biogenesis dataset, which is a harder case than most of these methods were
built for. Nothing here changes any method to compensate. What it does:

- **Flags the mismatch.** Every stability predictor and both inverse-folding
  models were trained or parameterised largely on soluble, globular proteins.
  When the UniProt entry has transmembrane segments, that caveat is written
  into `metadata.json` and printed at the end of the run.
- **Flags the structure.** An AlphaFold model has no bilayer, so rSASA and
  residue depth describe a protein in water. Supply an OPM or PDBTM model with
  `--pdb` for `bilayer_depth` and more meaningful burial.
- **Defines the domain sensibly.** For the domain-level variants of EVE and
  ESM-1v, the default window for a membrane protein spans the first to the
  last UniProt transmembrane segment. Override with `--domain START-END`.
- **Annotates topology.** Transmembrane segment index, inside/outside
  topology and distance to the nearest segment are feature columns.

## Sign conventions

Conventions differ between methods and are easy to get wrong in a way that
silently inverts a benchmark. Every predictor records its direction and where
that direction was read from:

```bash
python -m vepbench check-conventions
```

Five currently have unconfirmed directions: `thermompnn`, `ddmut`,
`metl_global`, `saprot_dg`, `esm3_dg`. Confirm them against the method's own
documentation before trusting a benchmark. A unit test pins all 36 directions,
so a change is deliberate rather than accidental.

## Isolated environments

These tools pin incompatible versions of torch, numpy and transformers, and
one still targets Python 2.7, so they cannot share an interpreter. Each gets
its own `uv` environment under `envs/<name>/.venv`, driven over subprocess
with a JSON job file.

```bash
python -m vepbench install-env esm
python -m vepbench install-env all --dry-run   # see what each one is first
python -m vepbench status
```

See [`envs/README.md`](envs/README.md) for sizes and hardware notes. FoldX is
licensed and is not distributed here; set `FOLDX_BINARY` to your own copy.

### Gated models

ESM3 and ESM C need the EvolutionaryScale licence accepted on Hugging Face and
a token exported as `HF_TOKEN`. Create one at
<https://huggingface.co/settings/tokens> (a read token is enough), accept the
licence at <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1>, then
`export HF_TOKEN=hf_...`.

### A networking note

Hugging Face serves large files through its Xet CDN by default, which some
networks allow `huggingface.co` but block. Every runner subprocess therefore
gets `HF_HUB_DISABLE_XET=1`, falling back to the classic endpoint. Set
`VEPBENCH_ALLOW_XET=1` to leave the default alone. Similarly, AlphaFold DB
answers 403 to a bare client user agent, so every outbound request sends a
real `User-Agent`.

## Resumability and failure isolation

- Results are cached per predictor, per protein, against a fingerprint of
  exactly the inputs that predictor consumes. Re-running skips finished work;
  `--force` recomputes.
- A predictor that cannot run is marked `unavailable` with the specific reason
  and contributes no rows. One that crashes is marked `failed` with its
  traceback. Neither affects the others.
- A predictor never invents a score. A variant an upstream tool did not score
  is written empty, not zero.
- FoldX and DDMut cache per variant, so an interrupted saturation scan resumes.

## Reproducibility

`metadata.json` records the vepbench version and git hash, Python and
platform, the device and dtype actually used, the UniProt sequence and its
version, the structure source and mapping route, MSA backend, depth and Neff,
the domain window and where it came from, every predictor's checkpoint string,
per-predictor runtimes, and every failure. Environments are pinned in
`envs/*/requirements.txt`, and `scripts/_common.sh` appends the commit of
every cloned third-party repository to `data/weights/provenance.tsv`.

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
python scripts/generate_predictors_md.py   # after changing a predictor
```

The test suite runs entirely offline against cached fixtures. The test protein
is phospholamban (P26678), 52 residues with one transmembrane helix, giving
988 variants.

`tests/fixtures/*_TESTONLY.pdb` are **synthetic idealised helices** generated
by `scripts/make_test_fixture.py`, so the PDB reader, the renumbering and the
surface code can be tested deterministically. They are not structural models
and must never be used as a structure input for a real prediction.
`AF-P26678-F1-model_v6.pdb` beside them is the real AlphaFold model, used to
test pLDDT and numbering against genuine data.

## What is and is not verified

**Verified by running it**: the output schema and harmonization, saturation
enumeration, variant parsing, UniProt fetching, AlphaFold download and
parsing, UniProt renumbering through all three routes, Shrake-Rupley surface
areas, contact number, the EBI alignment conversion against a real EBI result,
MSA statistics and Neff, conservation entropy, all feature columns, Grantham
and BLOSUM62 against their published matrices, the DMS merge including every
failure mode, all benchmark metrics, caching and cache invalidation, failure
isolation and the CLI end to end.

**ESM-2 650M is verified against real downloaded weights, through the whole
pipeline**: environment creation, weight download, subprocess dispatch,
scoring, schema validation and caching. 988 phospholamban variants in 111
seconds on four CPU cores. The scores are also right, not merely present:

| substitution | transmembrane helix | cytoplasmic region |
|---|---|---|
| to a charged residue | -1.82 | +0.08 |
| to a hydrophobic residue | +0.15 | -0.48 |
| to proline | -1.58 | -0.60 |

The model insists the helix stays hydrophobic, penalises a helix-breaking
proline inside it, and ranks substitutions at the initiator methionine worst
of all 988.

**Not executed yet**: the predictors whose weights or upstream repositories
were not installed in the development container. Their wrappers, runners,
environment specifications and install scripts are written against each
method's documented interface and are syntax and registry checked, but expect
small interface drift the first time you run each one. The runners fail with
the upstream error text rather than swallowing it, and a missing dependency
is reported as `unavailable` with the fix rather than as a crash.

See [PREDICTORS.md](PREDICTORS.md) for the full table, per-method notes, and
what was found when checking the methods whose existence was uncertain.
