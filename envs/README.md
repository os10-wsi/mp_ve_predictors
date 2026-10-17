# Isolated predictor environments

Many of these tools pin mutually incompatible versions of torch, numpy or
transformers, and one still targets Python 2.7. They therefore cannot share an
interpreter. Each environment here is created separately with `uv`, and the
orchestrator talks to it over a subprocess with a JSON job file. The
`vepbench` package itself depends only on numpy, pandas and requests.

```
python -m vepbench install-env esm            # one environment
python -m vepbench install-env all            # everything (slow, large)
python -m vepbench install-env all --dry-run  # just print what each one is
python -m vepbench status                     # which are installed
```

Each directory holds `requirements.txt` with pinned dependencies and, where
pip cannot do the whole job, an `install.sh` that clones repositories or
builds binaries. Repositories land in `third_party/` and weights in
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

Three older environments (`eve`, `tranception`, `rasp`, `promep`, `s3f`) pin
much older torch versions because the upstream code requires them. Those will
pull their own CUDA runtime and may not match your driver; they are the most
likely installs to need hand-holding.

Runtime behaviour is controlled from the command line, not from these files:
`--device`, `--dtype` and `--batch-size`. The default is CUDA with bfloat16
where the card supports it, falling back to float16, and float32 on CPU.

## Download sizes

Approximate, before model weights:

| environment | install | weights | VRAM at run time |
|---|---|---|---|
| esm | ~3 GB | ESM-1v 5 x 2.5 GB, ESM-2 650M 2.5 GB, ESM-2 3B 11 GB, ESM-IF1 0.5 GB | 3 to 13 GB |
| huggingface | ~3 GB | SaProt 650M 2.6 GB, SaProt 1.3B 5 GB, ProSST 0.5 GB, ProtT5 11 GB, Ankh 7 GB | 2 to 6 GB |
| proteinmpnn | ~3 GB | bundled, < 0.1 GB | ~1 GB |
| eve, tranception | ~3 GB each | Tranception Large 3 GB | 6 to 8 GB |
| rasp, gemme, vespa | ~2 GB each | small | 2 to 6 GB |
| deepsequence | container | small | optional |

Running every predictor pulls roughly 60 to 80 GB of weights in total.

Nothing here downloads a large sequence database. The default MSA backend is
EBI BLAST or PSI-BLAST, which needs none. `jackhmmer` and `hhblits` need a
database you supply: UniRef90 is roughly 60 GB and UniRef30 roughly 90 GB.
