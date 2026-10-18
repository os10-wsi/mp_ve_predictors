# Runbook

Getting from a fresh clone to merged predictions on your own GPU machine.
[README.md](README.md) is the reference; this is the order to do things in.

## 1. Prerequisites

| | why | check |
|---|---|---|
| Python 3.10+ | the orchestrator | `python3 --version` |
| uv **or** conda | builds the isolated environments | `uv --version` or `conda --version` |
| git | install scripts clone upstream repositories | `git --version` |
| NVIDIA driver supporting CUDA 13 | the pinned torch build | `nvidia-smi` |
| ~80 GB free disk | environments and model weights | `df -h .` |

If `nvidia-smi` reports a CUDA version below 13, the pinned torch will not
load. Install a matching build into the environment after creating it, for
example:

```bash
uv pip install --python envs/esm/.venv/bin/python \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

Optional, each needed only by specific predictors: Java for GEMME,
`foldseek` for SaProt, `mkdssp` for secondary-structure features, Docker for
DeepSequence, HMMER for the local MSA backend. With the conda backend most of
these are installed for you; see below.

### uv or conda

Either works, and both put the environment in the same place, so nothing else
changes. uv is used when present because it is much faster. On a cluster where
your compiled dependencies already come from conda, prefer conda:

```bash
python3 -m vepbench install-env esm --backend conda     # or mamba, micromamba
export VEPBENCH_ENV_BACKEND=conda                      # to make it the default
```

Conda creates each environment with `--prefix envs/<name>/.venv` rather than
a named environment, so the interpreter lands exactly where a virtual
environment's would. Nothing downstream can tell the difference, and you can
build different environments with different backends.

**Conda does more for you.** Four environments need binaries that have no
PyPI wheel, and the conda backend installs them from bioconda automatically:

| environment | conda packages | with uv instead |
|---|---|---|
| `rasp` | reduce, openmm, pdbfixer | install by hand; RaSP cannot prepare structures without them |
| `gemme` | openjdk | install a JRE yourself |
| `huggingface` | foldseek | `bash scripts/install_foldseek.sh` |
| `evcouplings` | hmmer | needed for `--msa-backend jackhmmer` |

`install-env` prints these and says whether they were installed, so you are
never left guessing. With uv it warns rather than failing silently.

Conda does not relax the version pins: `requirements.txt` is still installed
with pip inside the environment, so the resolved versions are the same either
way.

## 2. Install

```bash
git clone https://github.com/os10-wsi/mp_ve_predictors.git
cd mp_ve_predictors
pip install -e .

vepbench list      # 36 predictors, what each needs, approximate VRAM
vepbench status    # which environments exist (none yet)
```

That installs only numpy, pandas and requests. Predictors live in separate
environments, so nothing here pulls in torch.

### Credentials

```bash
export HF_TOKEN=hf_...          # read token from huggingface.co/settings/tokens
export VEPBENCH_EMAIL=you@example.org   # only for the EBI alignment backend
export FOLDX_BINARY=/path/to/foldx      # only if you licensed FoldX
```

`HF_TOKEN` is needed only for ESM3 and ESM C, which are gated. Accept the
licence at <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1> as well
as setting the token. Never put any of these in a tracked file.

## 3. One command for all of it

```bash
bash scripts/bootstrap.sh                    # the verified core
bash scripts/bootstrap.sh --tier full        # every environment and weight
bash scripts/bootstrap.sh --backend conda    # build with conda instead of uv
bash scripts/bootstrap.sh --dry-run          # print the steps, change nothing
```

It installs the package, builds the environments, fetches the weights, runs
the smoke test and then runs a real predictor, printing each command as it
goes. `--tier core` covers ESM, the Hugging Face language models and the
ProteinMPNN pair, which is most of what you will use; `--tier full` adds the
wrappers that drive upstream research repositories. Add `--yes` to skip the
download confirmations.

Everything below is what that script does, in case you would rather do it a
piece at a time or something fails partway.

## 3a. Smoke test, two minutes, no weights

```bash
bash examples/run_demo.sh
```

Runs predict, merge and benchmark end to end on phospholamban using only the
built-in baselines. If this works, the pipeline, the UniProt fetch and the
AlphaFold download are all fine.

## 3b. Pre-staging weights

Weights otherwise download on first use. Fetch them deliberately when you want
the cost paid once, or when the machine that runs the predictions has no
outbound network:

```bash
bash scripts/download_weights_esm.sh esm1v          # 12.5 GB, five replicates
bash scripts/download_weights_esm.sh esm2           # 2.5 GB
bash scripts/download_weights_esm.sh esm2-3b        # 11 GB
bash scripts/download_weights_esm.sh esm-if1        # 0.5 GB

bash scripts/download_weights_huggingface.sh        # lists what is available
bash scripts/download_weights_huggingface.sh all    # SaProt, ProSST, ProtT5, Ankh, ESM3
bash scripts/download_weights_tranception.sh        # 3.1 GB
```

Note that `all` means all of *that script's* catalogue. The ESM checkpoints
come from `download_weights_esm.sh`, which runs inside the `esm` environment,
and the rest from `download_weights_huggingface.sh`, which runs inside
`huggingface`; neither can fetch the other's models. `doctor` prints the
right command for each missing checkpoint, with its size, rather than one
blanket command that only covers some of them.

Each states its size and asks before downloading; `VEPBENCH_ASSUME_YES=1`
skips the prompt. Hugging Face weights land in `HF_HOME`, so set that if you
want them somewhere specific, and set the same value when you run.

## 3c. Check before you run

Every install step above can half-succeed. Before committing a batch to a
queue, ask what is actually in place:

```bash
python3 -m vepbench doctor                                   # everything
python3 -m vepbench doctor --predictors esm2_650m saprot     # just these
python3 -m vepbench doctor --predictors esm2_650m -v         # also list what passes
python3 -m vepbench doctor --msa-backend jackhmmer           # include the local backend
python3 -m vepbench doctor --features secondary_structure    # include what features need
python3 -m vepbench doctor --json                           # for scripts
```

`--json` reports `ready`, `needs_download`, `blocked`, `problems` and
`warnings` separately, because "something is missing" and "nothing can run"
are different answers and an exit code cannot carry both.

It checks the environment backend, each environment the request needs, the
Hugging Face cache, external binaries (foldseek, java, jackhmmer, mkdssp,
docker, FoldX), credentials and free disk, then asks each predictor the same
availability question the runner asks, so the two cannot drift apart. Each
failure is printed with the command that fixes it. Nothing is downloaded or
installed, so it is safe to run repeatedly on a login node.

Predictors land in one of three groups:

| group | meaning |
|---|---|
| ready | will run now, with or without a network |
| will download weights on first use | installed, but the checkpoint is not cached, so `--offline` would fail |
| blocked | something is missing; the reason names it |

Exit status is 0 when nothing is missing, 1 when something is, 2 for a
mistyped predictor name — so it can gate a submission script:

```bash
python3 -m vepbench doctor --predictors esm2_650m gemme || exit 1
```

Warnings do not fail it. The common one is weights left in the old
`~/.cache/huggingface`: runs set `HF_HOME` to `data/weights/huggingface`, so
anything still in the home cache is invisible to them. What to do depends on
what the new cache already holds, and `doctor` distinguishes the two cases
rather than making you guess:

- **old HF cache** — these models exist *only* in the home cache. It prints a
  `mv`, never an `rm`: this is the only copy.
- **duplicate weights** — these are in both caches, so the home copy is space
  paid for twice. It prints an `rm -rf` naming exactly those directories, and
  how much you get back.

## 4. First real predictor

```bash
python3 -m vepbench install-env esm     # ~3 GB; add --backend conda if you prefer
# several at once: python3 -m vepbench install-env esm huggingface proteinmpnn
vepbench run --uniprot P26678 --predictors esm2_650m --msa-backend none
```

First call downloads 2.5 GB of weights. This exact path is verified: 988
variants in 111 seconds on four CPU cores, and far quicker on a GPU.

Check it landed:

```bash
vepbench status --uniprot P26678
head -3 outputs/P26678/predictions_long.csv
```

## 5. Install what you actually need

Work outward in order of how likely each is to need attention.

```bash
# Language models: same verified runner, different checkpoints
python3 -m vepbench install-env huggingface    # SaProt, ProSST, ProtT5, Ankh
bash scripts/install_foldseek.sh              # skip this if you used conda

# Structure-based
python3 -m vepbench install-env proteinmpnn    # ProteinMPNN, ThermoMPNN
python3 -m vepbench install-env esm_if1        # separate: needs torch-scatter

# Alignment-based
python3 -m vepbench install-env gemme          # needs Java
python3 -m vepbench install-env evcouplings    # builds plmc
python3 -m vepbench install-env eve            # GPU hours per protein
```

`python3 -m vepbench install-env all --dry-run` prints what each one is
without installing. `vepbench status` shows what is installed.

## 6. A real run

```bash
vepbench run --uniprot Q9NRX5 \
  --predictors esm1v_full esm2_650m saprot proteinmpnn thermompnn gemme \
  --features rsasa plddt physchem blosum grantham membrane entropy gaps
```

Sequence, AlphaFold structure and alignment are fetched and cached
automatically. Useful flags:

| flag | effect |
|---|---|
| `--predictors all` | everything registered; unavailable ones are skipped with a reason |
| `--predictors no_msa` | skip the nine alignment-based methods |
| `--uniprot-list ids.txt` | one accession per line |
| `--pdb model.pdb` | your own structure instead of the AlphaFold model |
| `--domain 40-447` | override the domain window for the domain-level models |
| `--dtype float16` | halve VRAM |
| `--force` | ignore caches |
| `--offline` | use only what is cached |

Re-running is cheap: finished predictors are cached per protein against a
fingerprint of their inputs.

## 6a. Submitting to LSF

```bash
bash scripts/submit_job.sh --uniprot P26678
bash scripts/submit_job.sh --uniprot P26678 --predictors esm1v_full esm2_650m saprot
bash scripts/submit_job.sh --uniprot-list ids.txt                # one task per protein
bash scripts/submit_job.sh --uniprot-list ids.txt --dry-run      # print, submit nothing
```

A list of more than one accession is submitted as an LSF job array: one task
per protein, so the scheduler runs them in parallel and one protein failing
costs that protein rather than the batch. At most 10 run at once, which
`--max-parallel` changes; `--one-job` runs the whole list serially in a
single job instead.

The list is copied beside the job script at submission, because a queued
array reads it as each task starts, which may be hours later. Editing the
original afterwards cannot change what runs. Blank lines and `#` comments
are dropped.

Logs are per task: `<name>-<stamp>.<jobid>.<task>.{out,err}`.

It writes a self-contained job script under `outputs/jobs/`, prints it with
the `bsub` line, and submits. The script is kept, so you can read exactly what
ran, edit it and resubmit by hand without the wrapper.

Two things it does that a hand-written `bsub` line usually forgets. It checks
what can actually run before queueing, and it pins `VEPBENCH_*` and `HF_HOME`
to absolute paths resolved on the submit host, so the job cannot fall back to
`$HOME` on the compute node and fill a small quota.

The check prints what will run and what will be skipped, and refuses only when
*nothing* requested can run. Some predictors being unavailable is normal and
not a reason to hold a job: `--predictors all` always includes FoldX, which
needs a licence, and DeepSequence, which needs a container runtime, and the
runner records each skip in `status.json` with its reason. `--skip-doctor`
bypasses the check entirely.

The GPU queue defaults to `gpu`. On a cluster that calls it something else,
pass `--queue` or set `$VEPBENCH_QUEUE`; `bqueues -gpu` lists the GPU queues.

| flag | default | |
|---|---|---|
| `--predictors` | `all` | unavailable ones are skipped with a reason |
| `--features` | the default set | |
| `--queue` | `gpu` | or `$VEPBENCH_QUEUE` |
| `--no-gpu` | | uses `$VEPBENCH_CPU_QUEUE`, default `normal` |
| `--gpus` | 1 | |
| `--gpu-spec` | `num=%d:mode=shared` | the `bsub -gpu` string; `%d` is the GPU count |
| `--cores` | 4 | |
| `--mem` | 32000 | MB |
| `--time` | `4:00` | |
| `--name` | `vepbench-<accession>` | also names the logs |
| `--max-parallel` | 10 | array tasks running at once |
| `--one-job` | | run a list serially in one job instead of an array |
| `--dry-run` | | print the job script and the `bsub` line, submit nothing |
| `-- ...` | | everything after `--` is passed through to `vepbench run` |

If the compute nodes need `module load` or anything else before Python, put
those lines in a file and point `VEPBENCH_JOB_PRELUDE` at it; the contents are
inlined at the top of the job script:

```bash
printf 'module load cuda/12.1\n' > ~/vepbench-prelude.sh
VEPBENCH_JOB_PRELUDE=~/vepbench-prelude.sh bash scripts/submit_job.sh --uniprot P26678
```

Logs land in `outputs/jobs/logs/<name>-<stamp>.<jobid>.{out,err}`. The job
prints its host and GPU first, and ends with `vepbench status`, so the tail of
the log says what succeeded.

## 6b. On a cluster with no network on the compute nodes

If the compute nodes are firewalled, stage every input on the login node
first and run with `--offline`. `submit_job.sh` assumes they are not; add
`-- --offline` to its arguments once the inputs are staged.

```bash
# login node: fetches sequences, structures and alignments into data/cache/
vepbench prefetch --uniprot-list ids.txt --predictors all

# compute node: no network needed
vepbench run --uniprot-list ids.txt --predictors all --offline
```

`prefetch` works out which inputs the requested predictors and features
actually need, and reports the alignment depth for each protein so you can see
before committing compute which ones the evolutionary models cannot be trusted
on. An `--offline` run refuses to touch the network and fails loudly if
something was not staged, rather than hanging.

Point `VEPBENCH_DATA` at shared storage so both nodes see the same cache, and
set `HF_HOME` to a shared path too if the weights live there.

## 6c. Clearing up

A run leaves scratch in `data/cache/work/`, and every submission leaves a job
script and a pair of logs in `outputs/jobs/`. None of it is cleaned up on its
own:

```bash
python3 -m vepbench clean                      # the free-to-recreate things
python3 -m vepbench clean --dry-run            # show what would go
python3 -m vepbench clean --list               # categories and what each costs
python3 -m vepbench clean --older-than 30      # keep the recent logs
```

It prints what it found with sizes and asks before removing anything; `--yes`
skips the prompt. The categories are deliberately not interchangeable:

| category | default | costs to recreate |
|---|---|---|
| `work` | yes | nothing; the next run rebuilds it |
| `jobs`, `logs` | yes | the record of what was submitted and printed |
| `pycache` | yes | nothing |
| `predictions` | no | GPU time: every predictor reruns |
| `inputs` | no | network time; alignments take minutes each |
| `results` | no | **the actual output of every run** |
| `venvs` | no | an install each; ~38,000 files apiece |
| `uv-cache` | no | re-downloading, and uv's ability to hardlink |

`--what all` covers everything except `results` and `venvs`, which have to be
named. Model weights, the environment *definitions* in git, `third_party/`
and the sequence databases are never touched: for duplicate weights, `doctor`
prints an exact `rm` it has checked.

## 7. Alignments

The default backend is the ColabFold MMseqs2 server, which needs nothing
installed. For a large batch, or for maximum sensitivity, use a local
UniRef90:

```bash
bash scripts/setup_uniref90.sh    # ~65 GB download, ~180 GB unpacked
vepbench run --uniprot Q9NRX5 --predictors msa_based --msa-backend jackhmmer
```

It is written to `data/databases/`, which is gitignored, and found there
automatically. `VEPBENCH_DATABASES` moves where it is written and
`VEPBENCH_MSA_DATABASE` names the file directly. It cannot go in git: GitHub
rejects files over 100 MB and Git LFS caps at 2 GB.

Every alignment is reported with its effective sequence count. Below Neff 100
the run warns that EVE, GEMME, EVmutation and DeepSequence are unreliable, and
records that in `metadata.json`. Take the warning seriously rather than
reading the numbers anyway.

## 8. Merge and benchmark

```bash
vepbench merge --dms my_dms.csv --uniprot Q9NRX5 \
  --variant-col variant --out merged.csv

vepbench benchmark --merged merged.csv --score-col abundance \
  --group-col uniprot_id --plot summary.png
```

`--assay-direction functional` (the default) means a higher measured value is
a more functional protein, which is usual for a biogenesis or abundance
readout. Correlations then come out **positive** for a predictor that is
right. Use `--assay-direction damaging` if your column runs the other way.

## 9. What you get

```
outputs/<uniprot>/
├── predictions_long.csv     one row per (variant, predictor)
├── predictions_wide.csv     one row per variant, one column per predictor, plus features
├── metadata.json            provenance: sequence, structure, MSA depth, versions, hardware, caveats
├── status.json              per-predictor outcome and reason
└── per_member/              individual replicate scores, e.g. the five ESM-1v models
```

For a protein of length *L* both tables have *L* x 19 variants. The wide table
is the one to merge against; `<predictor>__harmonized` columns are all
oriented so **higher means more damaging**, whatever the raw convention.

From the merge: `merged.csv`, plus `merged.unmatched.csv` and
`merged.merge_report.json` listing every row that did not join and why. From
the benchmark: a summary CSV with Spearman, Pearson, AUROC, NDCG and top-*k*
recall with bootstrap intervals, and optionally a plot.

## 10. Roughly how long

Per protein of a few hundred residues, on one modern GPU:

| | time |
|---|---|
| ESM-2 650M, ESM C, SaProt, ProSST | under a minute each |
| ESM-1v ensemble (five models) | a few minutes |
| ESM-2 3B | a few minutes |
| ProteinMPNN, ThermoMPNN, RaSP | seconds to a minute |
| ColabFold alignment | seconds to minutes, server-side |
| jackhmmer against UniRef90 | tens of minutes, CPU-bound |
| EVE | hours, it trains a VAE per protein |
| FoldX | hours, it is per-variant |

Start with the fast ones on one protein, confirm the numbers look sane, then
scale out.

## 11. When something fails

Nothing stops the run. Each predictor is either `ok`, `cached`, `unavailable`
with a reason, or `failed` with its traceback, all recorded in `status.json`:

```bash
vepbench status --uniprot Q9NRX5
```

Most failures are install-time rather than run-time, so start with the
pre-flight check (section 3c), which names the fix for each:

```bash
python3 -m vepbench doctor --predictors <the ones you asked for>
```

| symptom | fix |
|---|---|
| `OSError: [Errno 28] No space left on device` | the cache is on a small filesystem; see below |
| `build backend is missing the 'build_editable' hook` | old pip or setuptools; see below |
| `environment 'x' is not installed` | `python3 -m vepbench install-env x` |
| `install did not finish` / `No module named 'filelock'` | `python3 -m vepbench install-env x` tops it up |
| ESM-IF1 wants torch-scatter | `bash envs/esm_if1/install.sh`, needs `data.pyg.org` |
| SaProt cannot find foldseek | `bash scripts/install_foldseek.sh`; it lands in `third_party/` where vepbench looks, no PATH export needed |
| gated model 401 | `export HF_TOKEN=...` and accept the licence on the model page |
| CUDA out of memory | `--dtype float16`, a smaller `--batch-size`, or `--device cpu` |
| a weight download stalls | `export HF_HUB_DISABLE_XET=1` is already set for runners |
| an upstream repo moved | the runner prints the upstream error; the install script names the repo |

### If an environment install hits "Disk quota exceeded"

Usually the inode quota rather than the block quota: torch and its
dependencies are tens of thousands of small files, so the message names
something tiny like a timezone file while `df` still shows free space. On
Lustre, check both with:

```bash
lfs quota -h -u $USER /lustre/scratch125
```

The column that matters is `files`, not `kbytes`. Two things help most:

- **Keep uv's cache on the same filesystem as the environments.** vepbench
  sets `UV_CACHE_DIR` to `<repo>/.uv-cache` for this reason. uv hardlinks
  from its cache into each environment when they share a filesystem and
  copies when they do not, so a cache under `$HOME` on a cluster writes
  every file twice and counts it twice. If you see uv warn "Failed to
  hardlink files", that is what is happening; `doctor` reports it too.
- **Remove environments you are not going to use.** Each torch-based one is
  about 6 GB and 38,000 files. `rm -rf envs/<name>/.venv` for one, keeping
  its definition; `python3 -m vepbench clean --what venvs` for all of them.

`clean` reports file counts beside sizes, because the count is what runs
out:

```bash
python3 -m vepbench clean --what venvs uv-cache --dry-run
```

Neither `venvs` nor `results` is included in `--what all`: rebuilding an
environment is an install, and results are the point of the run. Both have
to be named.

Budget roughly 38,000 files per torch environment. All eighteen is about
690,000, which is more than many project quotas allow, so install the ones
you need rather than everything.

### If a download runs out of space

Hugging Face defaults its cache to `~/.cache/huggingface`, which on a cluster
is usually a small home quota that two or three models will fill, while the
project filesystem has terabytes free.

This repository therefore defaults the cache to `data/weights/huggingface`,
beside the other weights, and passes that to every runner so predictions find
them with no extra configuration. An explicit `HF_HOME` always wins:

```bash
export HF_HOME=/lustre/.../hf-cache      # if you want it somewhere else
```

Downloads now check free space before starting rather than dying partway. If
an earlier run filled your home directory, reclaim it:

```bash
du -sh ~/.cache/huggingface
rm -rf ~/.cache/huggingface
```

A download killed by ENOSPC leaves partial blobs behind, so clearing the
cache is worth doing before retrying.

### If an environment half-installed

`vepbench status` reports one of four states per environment: ready, not
installed, **incomplete** (the install started but did not finish) or
**stale** (the requirements changed since it was built). Re-running
`install-env` fixes the last two; it tops up what is missing rather than
rebuilding, and `--force` rebuilds from scratch.

An incomplete environment is the nastiest failure mode on a shared
filesystem, because the interpreter is created in the first step, before a
single requirement is installed. An install that dies partway leaves a
working `python` and a half-filled `site-packages`.

Worse, it can leave a package *recorded* as installed with its files missing.
Both uv and pip compare the requirements against installed metadata, so they
report "no changes" and skip, and re-running the install appears to succeed
while fixing nothing. Every install therefore finishes by importing a few key
modules; if any fail it reinstalls with the files rewritten and checks again,
and only then writes the completion marker. If that second check also fails
the environment is damaged rather than incomplete, and the error says to
rebuild with `--force`.

### If the editable install is refused

```
ERROR: Project file:///... has a 'pyproject.toml' and its build backend is
missing the 'build_editable' hook.
```

Editable installs need PEP 660, which means pip 21.3 or newer and setuptools
64 or newer. On a cluster this usually means one of two things: the tooling
predates those versions, or there is no outbound network, so pip cannot fetch
the setuptools that `pyproject.toml` asks for and quietly falls back to an
older one already present.

In order of preference:

```bash
python3 -m pip install --upgrade pip setuptools wheel   # the real fix
python3 -m pip install -e .

python3 -m pip install .        # not editable, but fine if you are not editing

cd /path/to/mp_ve_predictors    # no installation at all
python3 -m vepbench list
```

The last one always works, because the package is pure Python with no
compiled parts. Every `vepbench ...` command becomes `python3 -m vepbench ...`
run from the repository directory. `scripts/bootstrap.sh` already tries these
in order and tells you which one it used.

A `setup.py` shim sits beside `pyproject.toml` so that an older pip falls back
to the legacy path instead of refusing outright. It holds no configuration;
everything is still in `pyproject.toml`.

## 12. Before you trust a benchmark

```bash
vepbench check-conventions
```

Four sign conventions are unconfirmed: `thermompnn`, `metl_global`,
`saprot_dg`, `esm3_dg`. An inverted sign turns the best predictor into the
worst, so confirm them against each method's own documentation first.

Also check the leakage columns in [PREDICTORS.md](PREDICTORS.md). ThermoMPNN,
SaProt-dG and ESM3-dG were trained on the Tsuboyama 2023 mega-scale data, and
VenusREM reports state of the art on ProteinGym.

## 13. How much of this has actually been run

**Verified end to end with real weights**: ESM-2 650M, through environment
creation, weight download, subprocess dispatch, scoring, schema validation and
caching. The scores are also right, not merely present: inside the
transmembrane helix of phospholamban, substitutions to a charged residue score
-1.82 and to a hydrophobic residue +0.15, against +0.08 and -0.48 in the
cytoplasmic region.

**Verified without weights**: the whole surrounding pipeline. UniProt fetch,
AlphaFold download and parsing, UniProt renumbering, rSASA and contact number,
ColabFold and EBI alignments against real servers, every feature column, the
DMS merge across all failure modes, the benchmark metrics, caching, failure
isolation and the CLI. 370 tests, all offline.

**Not yet run**: every other predictor's weights and upstream repository.
ESM-1v, ESM-2 3B and ESM-1v-domain use the identical verified runner with a
different checkpoint, so they should work first time. The language models in
the `huggingface` environment share the same shape of code. The wrappers that
drive an upstream research repository are the ones most likely to need a small
fix on first contact; they fail with the upstream error text rather than
swallowing it.
