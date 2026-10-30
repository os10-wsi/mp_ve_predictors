# Proteome-wide run: step by step

Running the human membrane proteome against ClinVar, from a fresh `git pull`
to the figures. Roughly a week end to end, most of it waiting on the GPU
array.

Each stage is a separate command and each is resumable, so you can stop
anywhere and pick up later. Nothing after stage 1 has to be done in one
sitting.

Throughout, `$ROOT` is your checkout on the farm:

```bash
export ROOT=/lustre/scratch125/gengen/projects_v2/mp_stability/2026_paper/mp_ve_predictors
```

---

## 0. Update and install

```bash
cd "$ROOT"
git fetch origin
git checkout claude/zen-allen-agxiez
git pull origin claude/zen-allen-agxiez

pip install -e '.[proteome]'
```

The `[proteome]` extra adds **pyarrow** and **matplotlib**. Without pyarrow
the tables fall back to gzipped CSV, which is refused above five million
rows rather than written as a file nothing can read back — the proteome
table is 55.6 million rows, so this is not optional.

Check it took:

```bash
python3 -c "import pyarrow, matplotlib; print('ok')"
python3 -m vepbench proteome --help
```

## 1. Point the run at its own directory

Everything below writes under `VEPBENCH_OUTPUTS`. Give this analysis its own
tree so it cannot collide with the 57-protein run:

```bash
export VEPBENCH_OUTPUTS="$ROOT/outputs_proteome"
export VEPBENCH_LSF_GROUP=team354        # only if your submissions need -G
mkdir -p "$VEPBENCH_OUTPUTS"
```

Put those three lines somewhere you can re-source, because **every later
stage needs them**. A stage run without `VEPBENCH_OUTPUTS` set will look in
the default `outputs/` and report that it found nothing.

```bash
cat > "$ROOT/proteome-env.sh" <<'EOF'
export ROOT=/lustre/scratch125/gengen/projects_v2/mp_stability/2026_paper/mp_ve_predictors
export VEPBENCH_OUTPUTS="$ROOT/outputs_proteome"
export VEPBENCH_LSF_GROUP=team354
EOF
# then, in any new shell:  source "$ROOT/proteome-env.sh"
```

The per-predictor caches under `data/cache/predictions/` are **not** moved by
this and are shared with your other runs. That is deliberate: a protein
already scored by the 57-protein run will not be recomputed.

## 2. Check the inode budget before anything else

This is the thing most likely to kill the run halfway through. The proteome
needs roughly **150,000 files**; you hit a wall at about 38,000 on a single
torch environment.

```bash
lfs quota -u "$USER" /lustre/scratch125
```

Look at the **files** column, not the bytes. If the headroom is under
~200,000, clear space first:

```bash
python3 -m vepbench clean --list          # what each category costs to rebuild
python3 -m vepbench clean work pycache logs --dry-run
```

Bytes matter too — budget 250–350 GB — but inodes run out first.

## 3. Define the protein set

```bash
python3 -m vepbench proteome define --set human_membrane
```

Expect ~5,243 proteins, 2.9M residues, 55.6M substitutions. This writes
`proteome/human_membrane/manifest.tsv` and freezes the set: every later
stage reads it, so the answer cannot drift mid-analysis. Re-running reuses
the file unless you pass `--force`.

To see the size without downloading anything:

```bash
python3 -m vepbench proteome define --set human_membrane --count-only
```

**Optional:** `--max-length 2500` drops the 109 proteins over 2,000 residues
(titin is 14,507). Those few dominate wall-clock on the 3B-parameter models.
You can always run them separately afterwards.

## 4. Download the data

This is the long single-threaded stage: ~5,243 UniProt entries fetched
serially, a 1.2 GB AlphaMissense download sliced into one file per
accession, and the ClinVar release parsed and mapped. **Budget 1–3 hours**,
and do not run it on a login node.

```bash
bsub -J vepbench-fetch -q normal -n 1 -M 16000 \
     -R "select[mem>16000] rusage[mem=16000]" -W 8:00 \
     ${VEPBENCH_LSF_GROUP:+-G $VEPBENCH_LSF_GROUP} \
     -o "$VEPBENCH_OUTPUTS/fetch.%J.out" -e "$VEPBENCH_OUTPUTS/fetch.%J.err" \
     "cd $ROOT && source $ROOT/proteome-env.sh && \
      python3 -m vepbench proteome fetch-data --set human_membrane"
```

It needs a compute node with network, which the farm's do. The stage is
resumable: UniProt entries and AlphaMissense slices already on disk are
skipped, so a job that times out can simply be resubmitted.

UniProt throttles a long serial run, so entries that do not come back are
retried twice with backoff. Anything still missing is listed in
`data/cache/annotations/uniprot_fetch_failures.txt`, and those proteins will
be absent from the ClinVar mapping. If that file exists, re-run the stage
before trusting the mapping:

```bash
python3 -m vepbench proteome fetch-data --set human_membrane --only uniprot
python3 -m vepbench proteome fetch-data --set human_membrane --only clinvar
```

Not every protein gets an AlphaMissense slice either — its published table
is a 2023 canonical snapshot, so a few dozen accessions are simply absent
and score NaN. That is expected, not a fetch failure.

### Then read the mapping report. Really.

```bash
head -1 "$VEPBENCH_OUTPUTS/proteome/human_membrane/clinvar/mapping_report.tsv"
sort -k2 -rn "$VEPBENCH_OUTPUTS/proteome/human_membrane/clinvar/mapping_report.tsv" | head -20
```

The job's stdout prints the summary. ClinVar is organised around RefSeq
transcripts and vepbench around UniProt canonical sequences, and for some
genes those disagree about which isoform is canonical. Every row has to
clear two gates — a unique gene-symbol resolution, and the wild-type residue
matching the UniProt sequence at that position — and failures are counted
per gene rather than coerced.

**Read the percentage carefully.** It is measured against rows in genes the
protein set actually contains, not against all of ClinVar. ClinVar covers
the whole genome, so on the membrane proteome roughly three quarters of the
file is out of scope and is reported separately rather than as failure. On a
healthy run expect **above about 85%** of the in-scope rows to map; below
that, stop and look.

A gene with a high `wild_type_residue_disagrees` count is one where the two
databases disagree about the canonical isoform.

### Suspect genes

Check the `suspect` column of the report, and the genes the summary names.

A gene symbol can resolve to the wrong protein entirely. PRNP is the clear
case: the major prion protein is GPI-anchored and carries no transmembrane
keyword, so it is not in the set, while a 73-residue alternative reading
frame with the same gene symbol *is*. ClinVar's PRNP variants are numbered
against the 253-residue product, so nearly all of them fail the range check
— but a few can land on a plausible residue in the 73-residue protein and
pass both gates. Nothing downstream can tell those from real matches.

Any gene that mapped under half its attempts is flagged. Treat its mapped
rows as unusable and exclude them:

```bash
REPORT="$VEPBENCH_OUTPUTS/proteome/human_membrane/clinvar/mapping_report.tsv"
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i} NR>1 && $h["suspect"]=="True" {print $1}' \
  "$REPORT"
```

## 5. Calibrate on 20 proteins before committing 5,000 jobs

The runtime estimate is ~70 GPU-minutes per protein, which is extrapolated
from model size, not measured. `status.json` records `seconds` per predictor,
so a short run turns the estimate into a number.

```bash
cd "$ROOT"
MANIFEST="$VEPBENCH_OUTPUTS/proteome/human_membrane/manifest.tsv"

# 20 proteins evenly spaced through the length distribution. -F'\t' matters:
# the protein_name column contains spaces, so default splitting picks the
# wrong field.
N=$(( $(wc -l < "$MANIFEST") - 1 ))
STEP=$(( N / 20 )); [ "$STEP" -lt 1 ] && STEP=1
awk -F'\t' 'NR>1 {print $5"\t"$1}' "$MANIFEST" | sort -n > /tmp/by_length.$$

# 19 evenly spaced, plus the single longest protein. Even spacing by rank
# never reaches the tail, and the handful of proteins over 2,000 residues
# are what actually decide the wall-clock on the 3B-parameter models.
awk -v step="$STEP" 'NR % step == 1 {print $2}' /tmp/by_length.$$ | head -19 \
  > "$VEPBENCH_OUTPUTS/calibration.txt"
tail -1 /tmp/by_length.$$ | cut -f2 >> "$VEPBENCH_OUTPUTS/calibration.txt"
sort -u "$VEPBENCH_OUTPUTS/calibration.txt" -o "$VEPBENCH_OUTPUTS/calibration.txt"
rm -f /tmp/by_length.$$

wc -l "$VEPBENCH_OUTPUTS/calibration.txt"   # expect 20

bash scripts/submit_job.sh \
  --uniprot-list "$VEPBENCH_OUTPUTS/calibration.txt" \
  --predictors proteome --max-parallel 20
```

When it finishes, read the real timings:

```bash
python3 - <<'EOF'
import json, os, collections
from pathlib import Path
out = Path(os.environ["VEPBENCH_OUTPUTS"])
times = collections.defaultdict(list)
for path in out.glob("*/status.json"):
    for entry in json.loads(path.read_text()).get("predictors", []):
        if entry.get("seconds"):
            times[entry["predictor"]].append(entry["seconds"])
total = 0.0
for name in sorted(times, key=lambda n: -sum(times[n])/len(times[n])):
    mean = sum(times[name])/len(times[name])
    total += mean
    print(f"{name:<28} {mean:8.1f}s  (n={len(times[name])})")
print(f"\n{total/60:.1f} GPU-minutes per protein")
print(f"x 5243 proteins = {total*5243/3600:,.0f} GPU-hours")
print(f"at 57 in parallel = {total*5243/3600/57/24:.1f} days")
EOF
```

If that lands far above the estimate, drop the most expensive predictors
from the roster (`--predictors` takes explicit names instead of `proteome`)
before launching the full run.

This is also where **prosst** and **thermompnn** will show as failures — see
Known problems below.

## 6. Launch the full run

Start with a length cap. Of the 5,243 proteins, **5,051 are under 1,500
residues and 192 are at or above** — so capping there costs 3.7% of the set
and removes the worst of the runtime tail, where a single 14,507-residue
protein can occupy a GPU for the better part of a day.

```bash
python3 -m vepbench proteome run --set human_membrane --max-length 1499 --skip-done
```

Without `--submit` this writes the accession list and **prints** the submit
command for you to read first. `--skip-done` leaves out the calibration
proteins and anything earlier runs already covered.

When the printed command looks right, add `--submit`:

```bash
python3 -m vepbench proteome run --set human_membrane --max-length 1499 --skip-done --submit
```

That is ~5,000 array tasks at 57 concurrent. Expect several days.

The manifest still holds the whole proteome, so ClinVar mapping and the
analysis cover everything. The long proteins are only deferred: lift the cap
later and `--skip-done` submits just the 192 that are missing.

### After killing an array, drop `--skip-done`

`--skip-done` only checks whether `predictions_wide.csv` exists. A task
killed mid-flight may have written an incomplete set of predictors, and
`--skip-done` would then treat that protein as finished. After a `bkill`,
resubmit without it: proteins that genuinely completed come back from the
prediction cache in about a second each, and the half-done ones get
finished.

Monitor it:

```bash
bjobs -A                                    # array progress
ls "$VEPBENCH_OUTPUTS" | grep -c '^[A-Z0-9]' # proteins finished so far
grep -l Error "$VEPBENCH_OUTPUTS"/jobs/logs/*.err 2>/dev/null | head
```

To add more capacity or resume after an interruption, re-run the same
command: `--skip-done` means it only submits what is still missing.

## 7. Aggregate and analyse

Both are safe to run while the array is still going — they report how much
of the proteome they actually found rather than assuming it completed. Run
them early to see partial results.

```bash
python3 -m vepbench proteome aggregate --set human_membrane
python3 -m vepbench proteome analyse   --set human_membrane
```

`aggregate` prints per-predictor coverage. `analyse` joins ClinVar, scores
every predictor pooled and per protein, stratifies by transmembrane context,
burial, pLDDT, architecture and review status, and draws the figures.

Re-run both whenever more jobs land. Nothing is recomputed that does not
need to be.

**The bootstrap dominates `analyse`.** With ~50,000 labelled variants, 24
predictors and 13 strata, the default `--n-boot 1000` is about 28 minutes;
`--n-boot 200` is about 6 and the confidence intervals are no wider in any
way that matters at that sample size. Use the smaller value while you are
re-running against partial results, and the default once for the figures
you keep:

```bash
python3 -m vepbench proteome analyse --set human_membrane --n-boot 200
```

## Which interpreter is which

There are two layers, and only the first is something you "activate".

**The orchestrator** runs `vepbench` itself. It needs only numpy, pandas and
requests. On the farm it is the conda environment vepbench was installed
into:

```bash
conda activate /software/conda/users/os10/mpdms
# or skip activation entirely and use the absolute path:
/software/conda/users/os10/mpdms/bin/python3 -m vepbench status
```

If `conda activate` is not set up in your shell, the absolute path is the
reliable form, and it is what the error messages in the job logs print.
`scripts/submit_job.sh` finds an interpreter by trying `$VEPBENCH_PYTHON`,
then `$VIRTUAL_ENV/bin/python`, then `python3` and `python`, taking the first
that can `import vepbench.cli`. Set `VEPBENCH_PYTHON` to pin it:

```bash
export VEPBENCH_PYTHON=/software/conda/users/os10/mpdms/bin/python3
```

**The predictor environments** are never activated. Each lives at
`envs/<name>/.venv` and vepbench invokes it by absolute path over a
subprocess with a JSON job file, which is what keeps mutually incompatible
pins of torch and transformers apart. To debug one, run its interpreter
directly:

```bash
$ROOT/envs/proteinmpnn/.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
```

`vepbench status` prints both paths at the top, so you never have to
remember them.

### Reproducing a predictor failure

The job log contains the exact command that failed. Copy it and run it as
it stands — it is a plain subprocess call with no wrapper:

```bash
$ROOT/envs/stability_dg/.venv/bin/python -s \
  $ROOT/third_party/absolute-stability-predictor/ESM3dG.py \
  --pdb $ROOT/data/cache/structures/AF-A6NCQ9-F1-model_v6.pdb --chain A \
  --mutations $ROOT/data/cache/work/A6NCQ9/esm3_dg/variants.csv \
  --model esm3 --ensemble 3 --output /tmp/out.csv
```

The `data/cache/work/<ACC>/<predictor>/` directory holds the inputs that
were handed to it, so the command is reproducible after the run has moved
on. Note `-s`, not `-I`: these upstream scripts import their own siblings,
and `-I` removes the script's directory from `sys.path`.

## Watching the run

```bash
python3 -m vepbench proteome progress --set human_membrane --max-length 1499
```

Reports how many proteins are finished, which predictors are costing the
most, which are failing everywhere, and the remaining time projected from
observed throughput rather than from an estimate. Pass the same
`--max-length` the array was submitted with, or the denominator counts
proteins that were never queued.

Run it an hour after launching: the first completed tasks are the only
honest source for how long the rest will take.

## 8. Read the results

```
$VEPBENCH_OUTPUTS/proteome/human_membrane/
  analysis/metrics_summary.tsv       <- start here
  analysis/metrics_stratified.tsv
  analysis/metrics_per_protein.tsv
  analysis/metrics_pooled.tsv
  analysis/coverage.tsv
  analysis/labelled_variants.parquet
  clinvar/mapping_report.tsv
  plots/auroc_ranking.png
  plots/stratified_heatmap.png
  plots/pooled_vs_per_protein.png
  plots/score_distributions.png
  plots/coverage.png
  plots/per_protein/<ACC>.png
```

**Lead with `metrics_summary.tsv`** — AUROC *within* each protein. Pooled
AUROC across the proteome rewards a predictor that merely separates
well-studied genes from the rest, because pathogenic variants are not spread
evenly across genes. `plots/pooled_vs_per_protein.png` shows how big that
gap is for each predictor.

**AlphaMissense is not comparable to the rest.** It is read from its
published table rather than run, and it is calibrated against ClinVar, so
its number here is not measuring what an unsupervised predictor's is. It is
flagged `trained_on_clinvar` in every table and drawn in a separate colour in
every figure. Its licence is CC BY-NC-SA 4.0 — non-commercial.

Check `coverage.tsv` before quoting any ranking: a predictor that only ran on
a third of the proteome is not being compared on the same footing.

---

## Known problems

| what | status |
|---|---|
| `prosst` | fails with `'prosst' is not a package`. Undiagnosed. Will fail on all 5,243. |
| `thermompnn` | runner points at the wrong entry point; needs rewiring to `analysis/custom_inference.py`. |
| `rasp` | fails fast by design. The bundled upstream code has no prediction entry point — `src/run_pipeline.py` is the paper's reproduction script and ignores its arguments. |
| `foldx` | excluded from the `proteome` group: tens of seconds per variant is thousands of CPU-years here. |
| MSA predictors | excluded: each needs an alignment per protein, and the ColabFold API rate-limits far below this volume. |

The first two are worth fixing before stage 6 — otherwise they burn a slot
in every one of 5,200 tasks. Neither blocks the run.

## If something goes wrong

| symptom | cause |
|---|---|
| `analyse` says "no ClinVar table" | stage 4 never completed. It is a separate batch job; run it and re-run `analyse` |
| a stage says it found nothing | `VEPBENCH_OUTPUTS` not set in this shell — `source "$ROOT/proteome-env.sh"` |
| `Disk quota exceeded` | inodes, not bytes. `lfs quota -u $USER`, then `vepbench clean` |
| tables written as `.csv.gz` | pyarrow missing: `pip install -e '.[proteome]'` |
| download fails in stage 4 | login node without a route, or a transient FTP failure. Resubmit; it resumes |
| `Request aborted by esub` | submitting to a non-GPU queue with `-gpu`. The default queue is `gpu` |
| submission rejected for a group | `export VEPBENCH_LSF_GROUP=team354` |
| everything silently on CPU | CUDA wheels against an older driver: `python3 -m vepbench doctor` reports it |
