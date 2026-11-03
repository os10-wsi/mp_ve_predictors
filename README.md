# vepbench

Score protein variants with ten predictors, then measure how well each one
separates pathogenic from benign ClinVar variants — protein by protein.

Give it a list of UniProt accessions. It scores every possible single
substitution, collects ClinVar labels for the human ones, and reports an
ROC-AUC with an error for every predictor on every protein, plus the
curves.

## Quick start

```bash
pip install -e '.[plots]'

# once: the shared resources
python3 -m vepbench fetch --uniprot-list ids.txt

# the work
python3 -m vepbench score  --uniprot-list ids.txt
python3 -m vepbench report --uniprot-list ids.txt --out report/
```

`python3 -m vepbench status` shows which scores are ready and which
interpreter is which.

## The ten scores

| score | source | needs | raw direction |
|---|---|---|---|
| `blosum62` | built in | sequence | higher = tolerated |
| `delta_hydrophobicity` | built in | sequence | higher = damaging |
| `delta_volume` | built in | sequence | higher = damaging |
| `delta_charge` | built in | sequence | higher = damaging |
| `solvent_exposure` | built in | structure | higher = tolerated |
| `alphamissense` | published table | human only | higher = damaging |
| `thermompnn` | published table, else computed | structure | higher = damaging |
| `esm1v` | model (`esm`) | sequence | higher = tolerated |
| `saprot` | model (`huggingface`) | structure | higher = tolerated |
| `foldx` | model (`foldx`) | structure | higher = damaging |

Every score is written twice: the raw number as the method reported it,
and `<name>__damaging`, oriented so higher always means more likely
pathogenic. The ROC analysis reads the second; nothing is silently
flipped and the original is always recoverable.

**The physicochemical deltas are magnitudes.** A substitution that makes a
residue much larger and one that makes it much smaller are both
disruptive, so these score `abs(delta)`. A signed version would cancel one
against the other on a ROC curve.

**`solvent_exposure` is a property of the position**, so every
substitution at one residue shares a value. That is deliberate: it asks
whether buried positions carry more pathogenic variants, which is a real
effect worth measuring beside the predictors.

**AlphaMissense is not independent of this benchmark.** It is calibrated
against ClinVar, so its AUC here is not measuring what an unsupervised
predictor's is. Its licence is CC BY-NC-SA 4.0 — non-commercial, unlike
the rest.

### FoldX is scored at ClinVar positions only

FoldX costs tens of seconds per variant. At saturation that is around four
days for a single protein, and the ~8,800 scores per protein away from a
ClinVar position would never be read, because the ROC analysis only ever
uses labelled variants. So it scores the labelled positions and leaves
the rest blank. It also needs a licensed binary — set `FOLDX_BINARY`.

### ThermoMPNN prefers the published table

A precomputed table covers some proteins and not others, so each protein
asks whether it is in there. Download it from
<https://zenodo.org/records/18381534> and pass `--thermompnn-file`, or put
it at `data/weights/thermompnn/thermompnn_predictions.tsv.gz`.

Coverage is measured against the variants actually requested, and anything
below 50% is treated as absent and computed instead. A partially covering
table is the dangerous case: taken at face value it yields a column that
looks complete and silently stops wherever the table happened to end. The
`source` field in `status.json` records which happened for each protein.

## Output

```
<outputs>/
  <ACCESSION>/
    scores.tsv          every substitution x every score, plus the ClinVar label
    status.json         what ran, how long it took, and where each number came from
  report/
    auc_by_protein.tsv  one row per protein, an AUC and an error per predictor
    auc_long.tsv        one row per protein and predictor
    auc_summary.tsv     across proteins, per predictor
    roc_curves.pdf      one panel per protein, every predictor drawn in it
    auc_ranking.png     median AUC per predictor
```

`auc_by_protein.tsv` is the one asked for: a thousand accessions give a
thousand rows, with `<name>_auc` and `<name>_se` for each predictor. The
error is the Hanley–McNeil standard error, which is analytic — ten
thousand AUCs is too many to bootstrap.

A protein is scored only where ClinVar gives at least five pathogenic and
five benign variants. Below that the interval spans most of the possible
range and the number means nothing; `--min-per-class` moves the line.

## ClinVar labels

Pathogenic and likely pathogenic are the positive class, benign and likely
benign the negative. Uncertain and conflicting classifications are carried
through `scores.tsv` but never scored: they are the variants a predictor
is most useful for, and treating a guess about them as truth would measure
the guess.

The mapping is the part that can be wrong without looking wrong. ClinVar
is organised around RefSeq transcripts and this around UniProt canonical
sequences, and for some genes those disagree about which isoform is
canonical. Every row has to clear two gates: the gene symbol must resolve
to exactly one accession, and the wild-type residue ClinVar names must be
the one the UniProt sequence carries at that position. Failures are
counted and reported, never coerced.

## Caching

Every score is cached per protein under `data/cache/predictions/`, so a
re-run fills only the gaps. A scoring run writes **every** column it has
cached, not just the ones requested — `--scores esm1v` cannot drop the
other nine from the file.

Scores cached by the previous version of this pipeline are read through an
alias (`esm1v_full` → `esm1v`, `blosum62_baseline` → `blosum62`), so
results already computed are not computed again. Those are validated by
checking every cached wild-type residue against the current sequence,
rather than by a fingerprint this code cannot reproduce.

## Environments

Three scores need a model, and their dependencies conflict, so each gets
its own environment under `envs/<name>/.venv`, invoked over a subprocess.
You never activate these — the orchestrator calls them by absolute path.

```bash
python3 -m vepbench install-env esm huggingface thermompnn foldx
```

If CUDA wheels do not match the driver, torch loads and silently runs on
CPU. Set `VEPBENCH_TORCH_INDEX=https://download.pytorch.org/whl/cu126` for
any CUDA 12 driver before installing.

## Paths

| what | where | override |
|---|---|---|
| results | `outputs/` | `VEPBENCH_OUTPUTS` |
| caches | `data/cache/` | `VEPBENCH_DATA` |
| weights and tables | `data/weights/` | `VEPBENCH_WEIGHTS` |
| cloned repositories | `third_party/` | `VEPBENCH_THIRD_PARTY` |

Running two protein sets means two `VEPBENCH_OUTPUTS` trees. The caches
are deliberately shared, so a protein scored once is scored once.
