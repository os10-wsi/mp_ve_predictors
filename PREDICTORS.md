# Predictors

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


## Protein language models

| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | Supervised | Megascale | ProteinGym | Environment |
|---|---|---|---|---|---|---|---|---|---|---|
| `ankh` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 5 GB | no | no | no | `huggingface` |
| `esm1v_domain` | domain | log_likelihood_ratio | more tolerated | yes | recommended | 3 GB | no | no | no | `esm` |
| `esm1v_full` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 3 GB | no | no | no | `esm` |
| `esm2_3b` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 13 GB | no | no | no | `esm` |
| `esm2_650m` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 3 GB | no | no | no | `esm` |
| `esm3_open` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 7 GB | no | no | no | `esm` |
| `esmc_600m` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 3 GB | no | no | no | `esm` |
| `prosst` | structure | log_likelihood_ratio | more tolerated | yes | recommended | 2 GB | no | no | no | `huggingface` |
| `prott5` | sequence only | log_likelihood_ratio | more tolerated | yes | recommended | 6 GB | no | no | no | `huggingface` |
| `saprot` | structure | log_likelihood_ratio | more tolerated | yes | recommended | 3 GB | no | no | no | `huggingface` |
| `saprot_1p3b` | structure | log_likelihood_ratio | more tolerated | yes | recommended | 6 GB | no | no | no | `huggingface` |


### Details

#### `ankh` — Ankh large

- **Paper**: Elnaggar A et al. Ankh: optimized protein language model unlocks general-purpose modelling. arXiv 2023:2301.06568
- **Code**: https://github.com/agemagician/Ankh
- **Licence**: CC-BY-NC-SA 4.0
- **Model version**: ElnaggarLab/ankh-large
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 5 GB
- **Install**: `python3 -m vepbench install-env huggingface`
- **Environment**: `huggingface`
- **Notes**: Same caveat as ProtT5: a denoising reconstruction likelihood used as a baseline rather than an authored variant-effect score.

#### `esm1v_domain` — ESM-1v (domain only, 5-model ensemble)

- **Paper**: Meier J et al. NeurIPS 2021
- **Code**: https://github.com/facebookresearch/esm
- **Licence**: MIT
- **Model version**: esm1v_t33_650M_UR90S_1..5 (domain window)
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 3 GB
- **Install**: `python3 -m vepbench install-env esm`
- **Environment**: `esm`
- **Notes**: Scores only the chosen domain window, with the model seeing the domain sequence alone. For a membrane protein the window defaults to the span of the UniProt transmembrane segments; override with --domain.

#### `esm1v_full` — ESM-1v (full length, 5-model ensemble)

- **Paper**: Meier J et al. Language models enable zero-shot prediction of the effects of mutations on protein function. NeurIPS 2021
- **Code**: https://github.com/facebookresearch/esm
- **Licence**: MIT
- **Model version**: esm1v_t33_650M_UR90S_1..5
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 3 GB
- **Install**: `python3 -m vepbench install-env esm`
- **Environment**: `esm`
- **Notes**: ESM-1v has a 1022-residue context. Longer proteins are tiled, each position scored in the window where it is most central, and the window used is written to the per-member CSV. Replicates are loaded one at a time, so peak memory is one 650M model, not five.

#### `esm2_3b` — ESM-2 3B

- **Paper**: Lin Z et al. Science 2023;379:1123-1130
- **Code**: https://github.com/facebookresearch/esm
- **Licence**: MIT
- **Model version**: facebook/esm2_t36_3B_UR50D
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 13 GB
- **Install**: `python3 -m vepbench install-env esm`
- **Environment**: `esm`
- **Notes**: Needs roughly 13 GB of VRAM in float32 and about 7 GB in float16. Pass --dtype float16 on a smaller card, or --device cpu to trade speed for memory.

#### `esm2_650m` — ESM-2 650M

- **Paper**: Lin Z et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 2023;379:1123-1130
- **Code**: https://github.com/facebookresearch/esm
- **Licence**: MIT
- **Model version**: facebook/esm2_t33_650M_UR50D
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 3 GB
- **Install**: `python3 -m vepbench install-env esm`
- **Environment**: `esm`
- **Notes**: Likelihood-ratio baseline; the same 1022-residue tiling applies.

#### `esm3_open` — ESM3 (open small, 1.4B)

- **Paper**: Hayes T et al. Simulating 500 million years of evolution with a language model. Science 2025;387:850-858
- **Code**: https://github.com/evolutionaryscale/esm
- **Licence**: EvolutionaryScale Community licence; non-commercial, accept on Hugging Face
- **Model version**: esm3-sm-open-v1
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 7 GB
- **Install**: `Accept the licence at huggingface.co/EvolutionaryScale/esm3-sm-open-v1, export HF_TOKEN, then install-env esm`
- **Environment**: `esm`
- **Notes**: Sequence-track masked marginals. The structure track can be added as conditioning, which changes the score, so it is off here and exposed separately as esm3_dg_masked_marginal.

#### `esmc_600m` — ESM C 600M

- **Paper**: ESM Team. ESM Cambrian: revealing the mysteries of proteins with unsupervised learning. EvolutionaryScale 2024
- **Code**: https://github.com/evolutionaryscale/esm
- **Licence**: Cambrian Open licence; non-commercial terms apply
- **Model version**: esmc_600m
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 3 GB
- **Install**: `python3 -m vepbench install-env esm, then export HF_TOKEN`
- **Environment**: `esm`
- **Notes**: Likelihood-ratio score from the ESM Cambrian encoder. Gated on Hugging Face, so HF_TOKEN must be set.

#### `prosst` — ProSST 2048

- **Paper**: Li M et al. ProSST: protein language modeling with quantized structure and disentangled attention. NeurIPS 2024
- **Code**: https://github.com/ai4protein/ProSST
- **Licence**: MIT
- **Model version**: AI4Protein/ProSST-2048
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 2 GB
- **Install**: `python3 -m vepbench install-env huggingface`
- **Environment**: `huggingface`
- **Notes**: Quantises local structure into a discrete vocabulary, then scores with disentangled attention over sequence and structure tokens.

#### `prott5` — ProtT5-XL UniRef50

- **Paper**: Elnaggar A et al. ProtTrans: towards cracking the language of life's code. IEEE TPAMI 2022;44:7112-7127
- **Code**: https://github.com/agemagician/ProtTrans
- **Licence**: Academic use per the model card
- **Model version**: Rostlab/prot_t5_xl_uniref50
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 6 GB
- **Install**: `python3 -m vepbench install-env huggingface`
- **Environment**: `huggingface`
- **Notes**: An encoder-decoder trained with span denoising, so a reconstruction likelihood at a sentinel token stands in for a masked-LM head. That is a baseline of this repository's choosing, not a scheme the ProtT5 authors proposed for variant effects.

#### `saprot` — SaProt 650M (AF2 structural alphabet)

- **Paper**: Su J et al. SaProt: protein language modeling with structure-aware vocabulary. ICLR 2024
- **Code**: https://github.com/westlake-repl/SaProt
- **Licence**: MIT
- **Model version**: SaProt_650M_AF2
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 3 GB
- **Install**: `python3 -m vepbench install-env huggingface && bash scripts/install_foldseek.sh`
- **Environment**: `huggingface`
- **Notes**: Each token pairs a residue with its Foldseek 3Di structural letter, so a structure is mandatory and foldseek must be on PATH. The 3Di alphabet was derived from soluble-protein structures.

#### `saprot_1p3b` — SaProt 1.3B

- **Paper**: Su J et al. ICLR 2024
- **Code**: https://github.com/westlake-repl/SaProt
- **Licence**: MIT
- **Model version**: SaProt_1.3B_AF2
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: masked marginals: log p(mut) - log p(wt)
- **Approximate VRAM**: 6 GB
- **Install**: `python3 -m vepbench install-env huggingface && bash scripts/install_foldseek.sh`
- **Environment**: `huggingface`
- **Notes**: The larger SaProt. Same scoring and the same foldseek requirement; enable it when you have the VRAM.


## Evolutionary models (need an alignment)

| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | Supervised | Megascale | ProteinGym | Environment |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepsequence` | msa | elbo_difference | more tolerated | yes | recommended | 4 GB | no | no | no | `deepsequence` |
| `eve_domain` | domain, msa | evolutionary_index | more deleterious | yes | required | 8 GB | no | no | no | `eve` |
| `eve_full` | msa | evolutionary_index | more deleterious | yes | required | 8 GB | no | no | no | `eve` |
| `evmutation` | msa | potts_energy_difference | more tolerated | yes | no | - | no | no | no | `evcouplings` |
| `gemme` | msa | gemme_score | more tolerated | yes | no | - | no | no | no | `gemme` |
| `popeve` | msa | popeve_score | more tolerated | yes | required | 4 GB | no | no | no | `eve` |
| `trancepteve` | msa | log_likelihood_ratio | more tolerated | yes | required | 8 GB | no | no | no | `tranception` |
| `tranception` | msa | log_likelihood_ratio | more tolerated | yes | recommended | 6 GB | no | no | no | `tranception` |
| `vespa` | sequence only | effect_probability | more deleterious | yes | recommended | 6 GB | yes | no | no | `vespa` |


### Details

#### `deepsequence` — DeepSequence

- **Paper**: Riesselman AJ, Ingraham JB, Marks DS. Deep generative models of genetic variation capture the effects of mutations. Nat Methods 2018;15:816-822
- **Code**: https://github.com/debbiemarkslab/DeepSequence
- **Licence**: MIT
- **Model version**: DeepSequence VAE, 2000 ELBO samples
- **Score**: elbo_difference, higher is more tolerated
- **Convention source**: Riesselman et al. 2018: the effect is the ELBO of the mutant minus the ELBO of the wild type, so higher is more tolerated
- **Approximate VRAM**: 4 GB
- **Install**: `docker build -t vepbench-deepsequence envs/deepsequence`
- **Environment**: `deepsequence`
- **Notes**: Published against Python 2.7 and Theano. The container in envs/deepsequence/Dockerfile is the practical way to run it today.

#### `eve_domain` — EVE (domain only)

- **Paper**: Frazer J et al. Nature 2021;599:91-95
- **Code**: https://github.com/OATML-Markslab/EVE
- **Licence**: MIT
- **Model version**: EVE VAE, trained per domain
- **Score**: evolutionary_index, higher is more deleterious
- **Convention source**: Frazer et al. 2021 Nature: the evolutionary index is the negative of the mutant-to-wild-type log-likelihood ratio, so higher is more pathogenic
- **Approximate VRAM**: 8 GB
- **Install**: `python3 -m vepbench install-env eve`
- **Environment**: `eve`
- **Notes**: The domain-level variant EVE was published with. The alignment is trimmed to the domain window before training, which usually deepens it.

#### `eve_full` — EVE (full length)

- **Paper**: Frazer J et al. Disease variant prediction with deep generative models of evolutionary data. Nature 2021;599:91-95
- **Code**: https://github.com/OATML-Markslab/EVE
- **Licence**: MIT
- **Model version**: EVE VAE, trained per protein
- **Score**: evolutionary_index, higher is more deleterious
- **Convention source**: Frazer et al. 2021 Nature: the evolutionary index is the negative of the mutant-to-wild-type log-likelihood ratio, so higher is more pathogenic
- **Approximate VRAM**: 8 GB
- **Install**: `python3 -m vepbench install-env eve`
- **Environment**: `eve`
- **Notes**: A VAE is trained from scratch on this protein's alignment, so the cost is GPU hours per protein rather than per variant. Unreliable on a shallow alignment; the pipeline warns below Neff 100.

#### `evmutation` — EVmutation (epistatic Potts model)

- **Paper**: Hopf TA et al. Mutation effects predicted from sequence co-variation. Nat Biotechnol 2017;35:128-135
- **Code**: https://github.com/debbiemarkslab/EVcouplings
- **Licence**: MIT
- **Model version**: plmc asymmetric, lambda_J scaled by (N-1)
- **Score**: potts_energy_difference, higher is more tolerated
- **Convention source**: Hopf et al. 2017: the epistatic score is E(mutant) - E(wild type) in the statistical energy, where more negative is more deleterious
- **Install**: `python3 -m vepbench install-env evcouplings`
- **Environment**: `evcouplings`
- **Notes**: Classic baseline. Fits a pairwise undirected graphical model with plmc and needs a deep alignment to fit couplings reliably.

#### `gemme` — GEMME

- **Paper**: Laine E, Karami Y, Carbone A. GEMME: a simple and fast global epistatic model predicting mutational effects. Mol Biol Evol 2019;36:2604-2619
- **Code**: https://github.com/elolaine/GEMME
- **Licence**: Free for academic use
- **Model version**: GEMME 1.1 with JET2
- **Score**: gemme_score, higher is more tolerated
- **Convention source**: Laine et al. 2019: GEMME predicted effects are negative, with more negative meaning more deleterious
- **Install**: `python3 -m vepbench install-env gemme`
- **Environment**: `gemme`
- **Notes**: Needs Java. Fast, no per-protein training, and one of the stronger alignment-only methods on ProteinGym.

#### `popeve` — popEVE

- **Paper**: Orenbuch R et al. popEVE. Marks lab, 2023 preprint
- **Code**: https://github.com/debbiemarkslab/popEVE
- **Licence**: MIT
- **Model version**: popEVE (ESM-1v + EVE Gaussian process)
- **Score**: popeve_score, higher is more tolerated
- **Convention source**: Orenbuch et al. 2023: popEVE scores are calibrated so that more negative is more damaging
- **Approximate VRAM**: 4 GB
- **Install**: `python3 -m vepbench install-env eve`
- **Environment**: `eve`
- **Notes**: Proteome-wide precomputed scores exist for human proteins and are used when available. A local run needs both EVE and ESM-1v scores first.

#### `trancepteve` — TranceptEVE L

- **Paper**: Notin P et al. TranceptEVE: combining family-specific and family-agnostic models of protein sequences. NeurIPS 2022 (LMRL)
- **Code**: https://github.com/OATML-Markslab/Tranception
- **Licence**: MIT
- **Model version**: TranceptEVE_L
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: log-likelihood ratio of mutant to wild type; positive favours the mutant
- **Approximate VRAM**: 8 GB
- **Install**: `python3 -m vepbench install-env tranception, after install-env eve`
- **Environment**: `tranception`
- **Notes**: Needs EVE trained for the same protein first: it mixes the EVE family model with Tranception's autoregressive scores.

#### `tranception` — Tranception L (retrieval)

- **Paper**: Notin P et al. Tranception: protein fitness prediction with autoregressive transformers and inference-time retrieval. ICML 2022
- **Code**: https://github.com/OATML-Markslab/Tranception
- **Licence**: MIT
- **Model version**: Tranception_Large
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: log-likelihood ratio of mutant to wild type; positive favours the mutant
- **Approximate VRAM**: 6 GB
- **Install**: `python3 -m vepbench install-env tranception`
- **Environment**: `tranception`
- **Notes**: Run with inference-time retrieval from the alignment, the configuration the paper benchmarks. Without retrieval it degrades to the autoregressive model alone.

#### `vespa` — VESPA

- **Paper**: Marquet C et al. Embeddings from protein language models predict conservation and variant effects. Hum Genet 2022;141:1629-1647
- **Code**: https://github.com/Rostlab/VESPA
- **Licence**: AFL-3.0
- **Model version**: VESPA (ProtT5 embeddings + conservation)
- **Score**: effect_probability, higher is more deleterious
- **Convention source**: Marquet et al. 2022: VESPA outputs a probability in [0,1] that the substitution has an effect, so higher is more deleterious
- **Approximate VRAM**: 6 GB
- **Install**: `python3 -m vepbench install-env vespa`
- **Environment**: `vespa`
- **Notes**: Supervised on variant-effect data. VESPA-light is the faster variant, selected with the light option.


## Inverse folding

| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | Supervised | Megascale | ProteinGym | Environment |
|---|---|---|---|---|---|---|---|---|---|---|
| `esm_if1` | structure | log_likelihood_ratio | more tolerated | yes | recommended | 2 GB | no | no | no | `esm_if1` |
| `proteinmpnn` | structure | log_likelihood_ratio | more tolerated | yes | no | 1 GB | no | no | no | `proteinmpnn` |


### Details

#### `esm_if1` — ESM-IF1

- **Paper**: Hsu C et al. Learning inverse folding from millions of predicted structures. ICML 2022
- **Code**: https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding
- **Licence**: MIT code; CC-BY-NC 4.0 weights
- **Model version**: esm_if1_gvp4_t16_142M_UR50
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: conditional log-likelihood difference log p(mut|backbone) - log p(wt|backbone)
- **Approximate VRAM**: 2 GB
- **Install**: `python3 -m vepbench install-env esm_if1`
- **Environment**: `esm_if1`
- **Notes**: The only model here still distributed solely through fair-esm, so its weights come from dl.fbaipublicfiles.com rather than Hugging Face. Weights are CC-BY-NC, so commercial use is restricted. It lives in its own environment because torch-scatter and torch-sparse, which it requires, publish wheels only for older torch releases.

#### `proteinmpnn` — ProteinMPNN

- **Paper**: Dauparas J et al. Robust deep learning-based protein sequence design with ProteinMPNN. Science 2022;378:49-56
- **Code**: https://github.com/dauparas/ProteinMPNN
- **Licence**: MIT
- **Model version**: v_48_020
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: conditional log-likelihood difference log p(mut|backbone) - log p(wt|backbone)
- **Approximate VRAM**: 1 GB
- **Install**: `python3 -m vepbench install-env proteinmpnn`
- **Environment**: `proteinmpnn`
- **Notes**: Scored with conditional probabilities averaged over random decoding orders, the inverse-folding analogue of a masked marginal. Small enough to run comfortably on CPU. A soluble-weights variant is available through the use_soluble_weights option.


## Newer structure-aware models

| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | Supervised | Megascale | ProteinGym | Environment |
|---|---|---|---|---|---|---|---|---|---|---|
| `metl_global` | structure | rosetta_total_score_delta | more deleterious | **no** | recommended | 2 GB | no | no | no | `metl` |
| `promep` | structure | log_likelihood_ratio | more tolerated | yes | required | 8 GB | no | no | no | `promep` |
| `s3f` | structure | log_likelihood_ratio | more tolerated | yes | required | 8 GB | no | no | no | `s3f` |
| `venusrem` | msa, structure | log_likelihood_ratio | more tolerated | yes | recommended | 4 GB | no | no | yes | `venusrem` |


### Details

#### `metl_global` — METL-Global

- **Paper**: Gelman S et al. Biophysics-based protein language models for protein engineering. Nature Methods 2025
- **Code**: https://github.com/gitter-lab/metl-pretrained
- **Licence**: MIT
- **Model version**: METL-G-20M-3D
- **Score**: rosetta_total_score_delta, higher is more deleterious
- **Convention source**: METL-Global predicts Rosetta biophysical attributes, and the total score follows Rosetta's convention where higher energy is worse. The fine-tuned METL-Local models instead predict assay-specific fitness, whose direction is whatever the training assay used. Confirm before benchmarking.
- **Approximate VRAM**: 2 GB
- **Install**: `python3 -m vepbench install-env metl`
- **Environment**: `metl`
- **Notes**: Pretrained on synthetic Rosetta data. Zero-shot in that it needs no assay data, but it predicts biophysical attributes rather than fitness. Local models exist only for proteins the authors simulated, none of them a membrane protein.

#### `promep` — ProMEP

- **Paper**: Cheng P et al. Zero-shot prediction of mutation effects with multimodal deep representation learning guides protein engineering. Cell Research 2024;34:630-647
- **Code**: https://github.com/wenjiegroup/ProMEP
- **Licence**: Check the repository
- **Model version**: ProMEP
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: fitness log-likelihood ratio of mutant to wild type; positive favours the mutant
- **Approximate VRAM**: 8 GB
- **Install**: `python3 -m vepbench install-env promep`
- **Environment**: `promep`
- **Notes**: Alignment-free multimodal model trained on sequence and structure from roughly 160 million AlphaFold DB entries. Needs SE(3)-Transformer and DGL, the most fragile install here.

#### `s3f` — S3F (sequence-structure-surface)

- **Paper**: Zhang Z, Notin P, Huang Y, Lozano AC, Chenthamarakshan V, Marks D, Das P, Tang J. Multi-Scale Representation Learning for Protein Fitness Prediction. NeurIPS 2024. arXiv:2412.01108
- **Code**: https://github.com/DeepGraphLearning/S3F
- **Licence**: Check the repository; not stated in the paper
- **Model version**: S3F
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: fitness log-likelihood ratio of mutant to wild type; positive favours the mutant
- **Approximate VRAM**: 8 GB
- **Install**: `python3 -m vepbench install-env s3f`
- **Environment**: `s3f`
- **Notes**: Combines a protein language model with residue-graph and molecular-surface encoders. The S3F-MSA variant also consumes an alignment; select it with the use_msa option.

#### `venusrem` — VenusREM

- **Paper**: Tan Y et al. From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model. Bioinformatics 2025;41(Suppl 1):i401-i409
- **Code**: https://github.com/ai4protein/VenusREM
- **Licence**: Check the repository
- **Model version**: VenusREM
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: fitness log-likelihood ratio of mutant to wild type; positive favours the mutant
- **Approximate VRAM**: 4 GB
- **Install**: `python3 -m vepbench install-env venusrem`
- **Environment**: `venusrem`
- **Notes**: Retrieval-enhanced: a structure-aware language model combined with alignment-derived residue preferences. Reported state of the art on 217 ProteinGym assays, so treat ProteinGym overlap as leakage.


## Stability predictors (ddG)

| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | Supervised | Megascale | ProteinGym | Environment |
|---|---|---|---|---|---|---|---|---|---|---|
| `esm3_dg` | structure | ddG_kcal_mol | more deleterious | **no** | recommended | 8 GB | yes | yes | no | `stability_dg` |
| `esm3_dg_masked_marginal` | structure | log_likelihood_ratio | more tolerated | yes | recommended | 7 GB | no | no | no | `esm` |
| `foldx` | structure | ddG_kcal_mol | more deleterious | yes | no | - | no | no | no | `foldx` |
| `rasp` | structure | ddG_kcal_mol | more deleterious | yes | no | 2 GB | yes | no | no | `rasp` |
| `saprot_dg` | structure | ddG_kcal_mol | more deleterious | **no** | recommended | 4 GB | yes | yes | no | `stability_dg` |
| `saprot_dg_masked_marginal` | structure | log_likelihood_ratio | more tolerated | yes | recommended | 3 GB | no | no | no | `huggingface` |
| `thermompnn` | structure | ddG_kcal_mol | more deleterious | **no** | no | 1 GB | yes | yes | no | `proteinmpnn` |


### Details

#### `esm3_dg` — ESM3-dG (absolute stability predictor)

- **Paper**: Cho Y et al. absolute-stability-predictor. github.com/yehlincho/absolute-stability-predictor
- **Code**: https://github.com/yehlincho/absolute-stability-predictor
- **Licence**: Check the repository; ESM3 weights are under the EvolutionaryScale Community licence
- **Model version**: ESM3 + LoRA stability head
- **Score**: ddG_kcal_mol, higher is more deleterious
- **Convention source**: Same repository and same caveat as saprot_dg.
- **Approximate VRAM**: 8 GB
- **Install**: `python3 -m vepbench install-env stability_dg, then export HF_TOKEN`
- **Environment**: `stability_dg`
- **Notes**: Requires accepting the ESM3 licence on Hugging Face. See esm3_dg_masked_marginal for the likelihood alternative.

#### `esm3_dg_masked_marginal` — ESM3 masked-marginal (labelled dG fallback)

- **Paper**: Hayes T et al. Science 2025 (scoring from Meier et al. 2021)
- **Code**: https://github.com/evolutionaryscale/esm
- **Licence**: EvolutionaryScale Community licence
- **Model version**: esm3-sm-open-v1 masked marginals, structure conditioned
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: standard masked marginal log p(mut) - log p(wt) from ESM3
- **Approximate VRAM**: 7 GB
- **Install**: `python3 -m vepbench install-env esm, then export HF_TOKEN`
- **Environment**: `esm`
- **Notes**: NOT a free-energy estimate. Same caveat as saprot_dg_masked_marginal; here the structure track is used as conditioning, which is what distinguishes it from esm3_open.

#### `foldx` — FoldX BuildModel

- **Paper**: Schymkowitz J et al. The FoldX web server: an online force field. Nucleic Acids Res 2005;33:W382-W388
- **Code**: https://foldxsuite.crg.eu/
- **Licence**: Proprietary; free academic licence on request
- **Model version**: set by FOLDX_BINARY at run time
- **Score**: ddG_kcal_mol, higher is more deleterious
- **Convention source**: ddG = G(mutant) - G(wild type); positive is destabilising, the Rosetta and FoldX convention
- **Install**: `export FOLDX_BINARY=/path/to/foldx`
- **Environment**: `foldx`
- **Notes**: RepairPDB runs once per structure and is cached, then BuildModel runs per variant. This is the slow path: seconds to minutes per variant, so a saturation scan of a large protein is expensive. Results are cached per variant, so an interrupted scan resumes. The empirical force field was parameterised on soluble proteins. Not proteome-scalable: at tens of seconds per variant, the human membrane proteome would take thousands of CPU-years.

#### `rasp` — RaSP

- **Paper**: Blaabjerg LM et al. Rapid protein stability prediction using deep learning representations. eLife 2023;12:e82593
- **Code**: https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg
- **Licence**: Apache-2.0
- **Model version**: RaSP cavity model + downstream ddG head
- **Score**: ddG_kcal_mol, higher is more deleterious
- **Convention source**: ddG = G(mutant) - G(wild type); positive is destabilising, the Rosetta and FoldX convention
- **Approximate VRAM**: 2 GB
- **Install**: `python3 -m vepbench install-env rasp`
- **Environment**: `rasp`
- **Notes**: Trained to reproduce Rosetta cartesian_ddg, so it inherits Rosetta's sign convention and its soluble-protein bias. Fast enough for saturation scans in principle, which is its advantage over FoldX and Rosetta. In practice the bundled upstream code has no prediction entry point: src/run_pipeline.py is the paper's reproduction script, which retrains the cavity model and ignores the arguments passed to it. The only inference path upstream ships is RaSPLab.ipynb, which upstream marks non-functional. This predictor therefore fails fast and reports itself unavailable rather than returning a number nobody can trust.

#### `saprot_dg` — SaProt-dG (absolute stability predictor)

- **Paper**: Cho Y et al. absolute-stability-predictor: fine-tuned SaProt and ESM3 for absolute protein stability. github.com/yehlincho/absolute-stability-predictor
- **Code**: https://github.com/yehlincho/absolute-stability-predictor
- **Licence**: Check the repository; SaProt itself is MIT
- **Model version**: SaProt 650M + LoRA r=4 stability head, ensemble of 3
- **Score**: ddG_kcal_mol, higher is more deleterious
- **Convention source**: The absolute-stability-predictor repository predicts dG and ddG with a LoRA stability head. Its sign convention is not stated in the sources used to build this entry; confirm with `vepbench check-conventions` before benchmarking.
- **Approximate VRAM**: 4 GB
- **Install**: `python3 -m vepbench install-env stability_dg`
- **Environment**: `stability_dg`
- **Notes**: Supervised on experimental dG and ddG, so not zero-shot, and overlapping benchmark data is leakage. This is the published implementation; see saprot_dg_masked_marginal for the likelihood alternative.

#### `saprot_dg_masked_marginal` — SaProt masked-marginal (labelled dG fallback)

- **Paper**: Su J et al. SaProt. ICLR 2024 (scoring from Meier et al. 2021)
- **Code**: https://github.com/westlake-repl/SaProt
- **Licence**: MIT
- **Model version**: SaProt_650M_AF2 masked marginals
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: standard masked marginal log p(mut) - log p(wt) from SaProt
- **Approximate VRAM**: 3 GB
- **Install**: `python3 -m vepbench install-env huggingface`
- **Environment**: `huggingface`
- **Notes**: NOT a free-energy estimate and NOT the published SaProt-dG. This is the standard likelihood-ratio score, provided as the documented fallback. Units are log-likelihood, not kcal/mol.

#### `thermompnn` — ThermoMPNN

- **Paper**: Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. Transfer learning to leverage larger datasets for improved prediction of protein stability changes. PNAS 2024;121:e2314853121
- **Code**: https://github.com/Kuhlman-Lab/ThermoMPNN
- **Licence**: MIT
- **Model version**: ThermoMPNN v1 (transfer-learned on Megascale)
- **Score**: ddG_kcal_mol, higher is more deleterious
- **Convention source**: Dieckhaus et al. 2024 report ddG with positive meaning destabilising, but the raw Megascale data uses the opposite sign and some forks flip it. Confirm with `vepbench check-conventions` before benchmarking.
- **Approximate VRAM**: 1 GB
- **Install**: `python3 -m vepbench install-env proteinmpnn`
- **Environment**: `proteinmpnn`
- **Notes**: Trained on Tsuboyama et al. 2023, so any benchmark overlapping that dataset is leaked. Built on ProteinMPNN features, hence the shared environment. Scores a whole saturation scan in one pass.


## Baselines implemented in this repository

| Predictor | Needs | Output | Direction | Confirmed | GPU | VRAM | Supervised | Megascale | ProteinGym | Environment |
|---|---|---|---|---|---|---|---|---|---|---|
| `blosum62_baseline` | sequence only | substitution_matrix_score | more tolerated | yes | no | - | no | no | no | `in-process` |
| `pssm_independent` | msa | log_likelihood_ratio | more tolerated | yes | no | - | no | no | no | `in-process` |


### Details

#### `blosum62_baseline` — BLOSUM62 baseline

- **Paper**: Henikoff S, Henikoff JG. PNAS 1992;89:10915-10919
- **Licence**: public domain
- **Model version**: BLOSUM62
- **Score**: substitution_matrix_score, higher is more tolerated
- **Convention source**: Henikoff & Henikoff 1992: positive scores are favoured substitutions
- **Notes**: Naive reference implemented in this repository, not a published predictor.

#### `pssm_independent` — Independent-sites MSA baseline

- **Paper**: Hopf TA et al. Nat Biotechnol 2017;35:128-135 (independent model baseline)
- **Licence**: MIT (this implementation)
- **Model version**: independent-sites
- **Score**: log_likelihood_ratio, higher is more tolerated
- **Convention source**: log f(mut) - log f(wt); positive means the mutant is the commoner residue
- **Notes**: Naive reference implemented in this repository, not a published predictor. Useful for checking whether an alignment is deep enough to be informative.


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
