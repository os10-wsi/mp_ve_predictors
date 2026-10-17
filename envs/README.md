# Isolated predictor environments

Many of these tools pin mutually incompatible versions of torch, numpy or
transformers, and one still targets Python 2.7. They therefore cannot share an
interpreter. Each environment here is created separately, and the orchestrator
talks to it over a subprocess with a JSON job file. The `vepbench` package
itself depends only on numpy, pandas and requests.

## uv or conda

Both are supported and both produce an environment at `envs/<name>/.venv`.
uv is chosen automatically when it is on PATH, because it is much faster;
otherwise micromamba, mamba or conda, in that order. Override per command with
`--backend`, or globally with `VEPBENCH_ENV_BACKEND`.

Conda uses `--prefix envs/<name>/.venv` rather than a named environment, so
the interpreter path is identical and nothing downstream changes. Version pins
are identical too: `requirements.txt` is installed with pip inside the
environment either way.

Where an environment needs a binary with no PyPI wheel, it lists it in
`conda-packages.txt`. The conda backend installs those from conda-forge and
bioconda; uv warns that it cannot and names them. That covers `reduce`,
`openmm` and `pdbfixer` for RaSP, `openjdk` for GEMME, `foldseek` for SaProt,
and `hmmer` for the local alignment backend.

```
python3 -m vepbench install-env esm            # one environment
python3 -m vepbench install-env all            # everything (slow, large)
python3 -m vepbench install-env all --dry-run  # just print what each one is
python3 -m vepbench status                     # which are installed
```

Each directory holds `requirements.txt` with pinned dependencies, optionally
`conda-packages.txt` for things pip cannot supply, and, where neither can do
the whole job, an `install.sh` that clones repositories or builds binaries. Repositories land in `third_party/` and weights in
`data/weights/`, both gitignored. Override with `VEPBENCH_THIRD_PARTY` and
`VEPBENCH_WEIGHTS`, or point at an existing checkout with the per-predictor
variables such as `PROTEINMPNN_DIR` or `EVE_DIR`.

## GPU

The torch pins target **CUDA 13**, matching a torch 2.14.x install from PyPI.
If your driver is older, install the matching build into the environment
yourself before running anything else, for example:

```
uv pip install --python envs/esm/.venv/bin/python \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

Several environments (`esm_if1`, `eve`, `tranception`, `rasp`, `promep`, `s3f`) pin
older torch versions because the upstream code requires them. They pull their
own CUDA runtime, which may not match your driver, and are the most likely
installs to need hand-holding. `esm_if1` is pinned specifically because
torch-scatter and torch-sparse publish wheels only for older torch builds; if
that install fails, ESM-IF1 reports itself unavailable and everything else
still runs.

Runtime behaviour is controlled from the command line, not from these files:
`--device`, `--dtype` and `--batch-size`. The default is CUDA with bfloat16
where the card supports it, falling back to float16, and float32 on CPU.

## Download sizes

Approximate, before model weights:

| environment | install | weights | VRAM at run time |
|---|---|---|---|
| esm | ~3 GB | ESM-1v 5 x 2.5 GB, ESM-2 650M 2.5 GB, ESM-2 3B 11 GB | 3 to 13 GB |
| esm_if1 | ~2 GB | ESM-IF1 0.5 GB | ~2 GB |
| huggingface | ~3 GB | SaProt 650M 2.6 GB, SaProt 1.3B 5 GB, ProSST 0.5 GB, ProtT5 11 GB, Ankh 7 GB | 2 to 6 GB |
| proteinmpnn | ~3 GB | bundled, < 0.1 GB | ~1 GB |
| eve, tranception | ~3 GB each | Tranception Large 3 GB | 6 to 8 GB |
| rasp, gemme, vespa | ~2 GB each | small | 2 to 6 GB |
| deepsequence | container | small | optional |

Running every predictor pulls roughly 60 to 80 GB of weights in total.

Nothing here downloads a large sequence database. The default MSA backend is
the ColabFold MMseqs2 server, which needs none, and the EBI backend needs none
either. The local backends do: UniRef90 for jackhmmer is roughly 65 GB
compressed and 180 GB unpacked, UniRef30 for HHblits about 90 GB. Fetch
UniRef90 with `bash scripts/setup_uniref90.sh`. It goes in `data/databases/`,
which is gitignored, and must never be committed.
