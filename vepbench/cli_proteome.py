"""The ``vepbench proteome`` subcommands.

These chain the stages of the proteome-wide analysis. Each is separately
runnable and separately resumable, because the whole thing takes days and
nothing that long should have to start over. The stages share one
convention: they report what they found rather than assuming it, so a run
that only half finished says so in its own output instead of producing a
table that looks complete.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from vepbench.config import PATHS

log = logging.getLogger(__name__)

DEFAULT_SET = "human_membrane"
#: Passes over the accessions that did not come back. UniProt throttles a
#: long serial run, and those failures are transient.
UNIPROT_ATTEMPTS = 3
UNIPROT_BACKOFF_SECONDS = 20
#: Predictors to draw the per-protein tracks for, if they ran.
TRACK_PREFERENCE = ("alphamissense", "esm1v_full", "esm2_650m", "esm2_3b", "saprot")


def _directory(name: str):
    from vepbench.proteome import manifest_directory

    return manifest_directory(name, paths=PATHS)


def command_define(args: argparse.Namespace) -> int:
    from vepbench.proteome import count_matches, fetch_manifest, resolve_query

    if args.count_only:
        _, query = resolve_query(args.protein_set)
        print(f"{count_matches(query):,} entries match {query}")
        return 0

    frame, info = fetch_manifest(
        args.protein_set, paths=PATHS, force=args.force,
        min_length=args.min_length, max_length=args.max_length,
    )
    print(info.summary())
    print(f"manifest: {info.path}")
    if len(frame):
        multipass = int((frame["n_tm_segments"] > 1).sum())
        print(f"  {multipass:,} multipass, "
              f"{int((frame['n_tm_segments'] == 1).sum()):,} single-pass")
        print(f"  longest: {frame.loc[frame['length'].idxmax(), 'uniprot_id']} "
              f"({int(frame['length'].max()):,} residues)")
    return 0


def command_fetch_data(args: argparse.Namespace) -> int:
    from vepbench.proteome import load_manifest

    manifest = load_manifest(args.protein_set, paths=PATHS)
    wanted = set(args.only) if args.only else {"uniprot", "clinvar", "alphamissense"}
    failures = []

    if "uniprot" in wanted:
        failures += _fetch_uniprot(manifest, offline=False)
    if "alphamissense" in wanted:
        failures += _fetch_alphamissense(manifest, args)
    if "clinvar" in wanted:
        failures += _fetch_clinvar(manifest, args)

    for message in failures:
        print(f"error: {message}", file=sys.stderr)
    return 1 if failures else 0


def _fetch_uniprot(manifest, *, offline: bool) -> list[str]:
    """Cache the canonical sequence and topology for every protein.

    ClinVar mapping cannot start without these: the wild-type residue check
    needs the sequence. They are cached per accession, so an interrupted
    fetch resumes where it stopped.
    """
    from vepbench.uniprot import UniProtError, fetch_uniprot

    accessions = manifest["uniprot_id"].tolist()
    missing = [a for a in accessions
               if not (PATHS.annotations / f"{a}.json").exists()]
    if not missing:
        log.info("all %d UniProt entries are already cached", len(accessions))
        return []

    log.info("fetching %d UniProt entries (%d already cached)",
             len(missing), len(accessions) - len(missing))

    # Thousands of requests in a tight loop draw throttling, and a protein
    # dropped here is silently absent from the ClinVar mapping later. Retry
    # the stragglers with backoff rather than letting a blip cost a gene.
    pending = list(missing)
    failed: list[str] = []
    for attempt in range(1, UNIPROT_ATTEMPTS + 1):
        failed = []
        for index, accession in enumerate(pending, start=1):
            try:
                fetch_uniprot(accession, paths=PATHS, offline=offline)
            except (UniProtError, OSError) as exc:
                failed.append(accession)
                if attempt == UNIPROT_ATTEMPTS:
                    log.warning("%s: %s", accession, exc)
            if index % 250 == 0:
                log.info("  %d/%d", index, len(pending))
        if not failed or offline:
            break
        if attempt < UNIPROT_ATTEMPTS:
            delay = UNIPROT_BACKOFF_SECONDS * attempt
            log.info("%d entries did not come back; retrying in %ds (attempt %d of %d)",
                     len(failed), delay, attempt + 1, UNIPROT_ATTEMPTS)
            time.sleep(delay)
            pending = failed

    if failed:
        record = _directory_for_failures()
        record.write_text("\n".join(failed) + "\n")
        log.warning(
            "%d UniProt entries could not be fetched after %d attempts. They are "
            "listed in %s, and the proteins they name will be missing from the "
            "ClinVar mapping. Re-run `proteome fetch-data --only uniprot` to try "
            "again; cached entries are skipped.",
            len(failed), UNIPROT_ATTEMPTS, record,
        )
    return []


def _directory_for_failures() -> Path:
    PATHS.annotations.mkdir(parents=True, exist_ok=True)
    return PATHS.annotations / "uniprot_fetch_failures.txt"


def _fetch_alphamissense(manifest, args) -> list[str]:
    from vepbench.alphamissense import AlphaMissenseError, build_slices, download

    try:
        table = Path(args.alphamissense_file) if args.alphamissense_file else download(
            paths=PATHS, force=args.force
        )
        counts = build_slices(table, manifest["uniprot_id"].tolist(),
                              paths=PATHS, force=args.force)
    except AlphaMissenseError as exc:
        return [str(exc)]
    scored = sum(1 for value in counts.values() if value != 0)
    print(f"AlphaMissense: slices for {scored:,} of {len(manifest):,} proteins")
    return []


def _fetch_clinvar(manifest, args) -> list[str]:
    from vepbench import tables
    from vepbench.clinvar import (
        ClinVarError,
        download,
        gene_index,
        label_counts,
        parse,
        sequences_from_cache,
    )

    directory = _directory(args.protein_set) / "clinvar"
    try:
        path = Path(args.clinvar_file) if args.clinvar_file else download(
            directory / "variant_summary.txt.gz", force=args.force
        )
    except ClinVarError as exc:
        return [str(exc)]

    accessions = manifest["uniprot_id"].tolist()
    sequences, missing = sequences_from_cache(accessions, paths=PATHS)
    if missing:
        log.warning(
            "%d of %d proteins have no cached UniProt entry and cannot be mapped; "
            "run `vepbench proteome fetch-data --only uniprot` first",
            len(missing), len(accessions),
        )
    if not sequences:
        return ["no UniProt sequences are cached, so nothing can be mapped"]

    gene_to_accession, ambiguous = gene_index(manifest)
    frame, report = parse(
        path, sequences, gene_to_accession, ambiguous_genes=ambiguous,
        germline_only=not args.include_somatic,
    )

    written = tables.write_table(directory / "clinvar_missense", frame)
    report_path = directory / "mapping_report.tsv"
    report.as_frame().to_csv(report_path, sep="\t", index=False)

    print(report.summary())
    counts = label_counts(frame)
    print(f"ClinVar: {counts['pathogenic']:,} pathogenic, {counts['benign']:,} benign, "
          f"{counts['unlabelled']:,} unlabelled, across {counts['proteins']:,} proteins")
    print(f"  table:  {written}")
    print(f"  report: {report_path}")
    return []


def command_run(args: argparse.Namespace) -> int:
    """Write the accession list and hand it to the LSF submitter."""
    from vepbench.proteome import load_manifest

    manifest = load_manifest(args.protein_set, paths=PATHS)
    if args.max_length:
        manifest = manifest.loc[manifest["length"] <= args.max_length]
    if args.skip_done:
        done = [
            a for a in manifest["uniprot_id"]
            if (PATHS.output_dir(a) / "predictions_wide.csv").exists()
        ]
        manifest = manifest.loc[~manifest["uniprot_id"].isin(done)]
        log.info("%d proteins already have output and were skipped", len(done))

    if manifest.empty:
        print("nothing left to run")
        return 0

    directory = _directory(args.protein_set)
    directory.mkdir(parents=True, exist_ok=True)
    list_path = directory / "to_run.txt"
    list_path.write_text("\n".join(manifest["uniprot_id"]) + "\n")
    print(f"{len(manifest):,} proteins -> {list_path}")

    script = PATHS.root / "scripts" / "submit_job.sh"
    command = [
        "bash", str(script), "--uniprot-list", str(list_path),
        "--predictors", *args.predictors,
        "--max-parallel", str(args.max_parallel),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.extra:
        command += ["--", *args.extra]

    if not args.submit:
        print("\nreview, then submit with:\n  " + " ".join(command))
        return 0
    return subprocess.call(command)


def command_aggregate(args: argparse.Namespace) -> int:
    from vepbench.aggregate import build_dataset, dataset_root
    from vepbench.proteome import load_manifest

    manifest = load_manifest(args.protein_set, paths=PATHS)
    coverage = build_dataset(manifest["uniprot_id"].tolist(),
                             name=args.protein_set, paths=PATHS, force=args.force)
    print(coverage.summary())
    directory = _directory(args.protein_set) / "analysis"
    directory.mkdir(parents=True, exist_ok=True)
    coverage.as_frame().to_csv(directory / "coverage.tsv", sep="\t", index=False)
    print(f"dataset: {dataset_root(args.protein_set, paths=PATHS)}")
    print(f"coverage: {directory / 'coverage.tsv'}")
    if coverage.failed:
        print(f"  {len(coverage.failed):,} proteins had at least one failed predictor")
    return 0


def command_analyse(args: argparse.Namespace) -> int:
    from vepbench import plots, tables
    from vepbench.aggregate import drop_empty_predictors, join_clinvar
    from vepbench.proteome import load_manifest
    from vepbench.proteome_stats import (
        StatsError,
        evaluate_labelled,
        evaluate_per_protein,
        summarise_per_protein,
        supervised_predictors,
        write_results,
    )

    directory = _directory(args.protein_set)
    manifest = load_manifest(args.protein_set, paths=PATHS)
    try:
        clinvar = tables.read_table(directory / "clinvar" / "clinvar_missense")
    except tables.TableError:
        print("error: no ClinVar table; run `vepbench proteome fetch-data` first",
              file=sys.stderr)
        return 1

    joined, coverage = join_clinvar(clinvar, name=args.protein_set, paths=PATHS,
                                    labelled_only=False)
    joined = joined.merge(
        manifest[["uniprot_id", "membrane_topology", "n_tm_segments"]],
        on="uniprot_id", how="left",
    )
    analysis = directory / "analysis"
    written = tables.write_table(analysis / "labelled_variants", joined)

    labelled = joined.loc[joined["label"].notna()]
    if labelled.empty:
        print("error: no labelled ClinVar variants joined to any prediction",
              file=sys.stderr)
        return 1
    labelled, dropped = drop_empty_predictors(labelled, min_rows=args.min_rows)

    supervised = supervised_predictors()
    try:
        pooled = evaluate_labelled(labelled, n_boot=args.n_boot, seed=args.seed,
                                   min_per_class=args.min_per_class,
                                   supervised=supervised)
    except StatsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    per_protein = evaluate_per_protein(labelled, min_per_class=args.min_per_class,
                                       supervised=supervised)
    summary = summarise_per_protein(per_protein)

    write_results(analysis, {
        "metrics_pooled": pooled.loc[pooled["stratum"] == "all"],
        "metrics_stratified": pooled,
        "metrics_per_protein": per_protein,
        "metrics_summary": summary,
    })

    plot_dir = directory / "plots"
    plots.auroc_ranking(summary, plot_dir / "auroc_ranking.png")
    plots.stratified_heatmap(pooled, plot_dir / "stratified_heatmap.png")
    plots.pooled_versus_per_protein(pooled, summary,
                                    plot_dir / "pooled_vs_per_protein.png")
    plots.score_distributions(labelled, list(summary["predictor"][:4]),
                              plot_dir / "score_distributions.png")
    plots.coverage_bars(coverage.as_frame().rename(columns={"proteins": "proteins"}),
                        plot_dir / "coverage.png")
    _protein_tracks(args, manifest, labelled, summary, plot_dir / "per_protein")

    print(f"{len(labelled):,} labelled variants across "
          f"{labelled['uniprot_id'].nunique():,} proteins")
    if dropped:
        print(f"  too sparse to score: {', '.join(dropped)}")
    if not summary.empty:
        best = summary.iloc[0]
        print(f"  best within-protein median AUROC: {best['predictor']} "
              f"{best['median_auroc']:.3f} over {int(best['n_proteins'])} proteins"
              + ("  (calibrated on ClinVar)" if best["trained_on_clinvar"] else ""))
    print(f"  tables: {analysis}")
    print(f"  plots:  {plot_dir}")
    print(f"  joined: {written}")
    return 0


def _protein_tracks(args, manifest, labelled, summary, directory: Path) -> None:
    """Per-protein figures for the proteins with the most labelled variants."""
    if args.tracks <= 0 or summary.empty:
        return
    from vepbench import plots
    from vepbench.aggregate import protein_predictions
    from vepbench.uniprot import fetch_uniprot

    available = set(summary["predictor"])
    chosen = next((p for p in TRACK_PREFERENCE if p in available), summary.iloc[0]["predictor"])
    counts = labelled["uniprot_id"].value_counts().head(args.tracks)

    for accession in counts.index:
        predictions = protein_predictions(accession, paths=PATHS)
        if predictions is None:
            continue
        try:
            record = fetch_uniprot(accession, paths=PATHS, offline=True)
            segments = record.transmembrane_segments
        except Exception:  # noqa: BLE001 - a missing cache must not stop the figures
            segments = []
        plots.protein_track(
            predictions, directory / f"{accession}.png", uniprot_id=accession,
            predictor=chosen,
            clinvar=labelled.loc[labelled["uniprot_id"] == accession],
            tm_segments=segments,
        )


def command_progress(args: argparse.Namespace) -> int:
    """How far the run has got, and how long the rest will take.

    The runtime of this pipeline cannot be predicted from the outside: it
    depends on the GPUs you were given, how many tasks are actually
    resident, and which predictors turn out to dominate. So this measures
    rather than estimates. Throughput is taken from the completed tasks
    themselves, which folds in queueing, startup and failures -- everything
    a per-predictor sum would leave out.
    """
    import time

    from vepbench.aggregate import protein_status
    from vepbench.proteome import load_manifest

    manifest = load_manifest(args.protein_set, paths=PATHS)
    if args.max_length:
        manifest = manifest.loc[manifest["length"] <= args.max_length]
    expected = manifest["uniprot_id"].tolist()

    finished: list[float] = []
    seconds: dict[str, list[float]] = {}
    failures: dict[str, int] = {}
    done = 0
    for accession in expected:
        if not (PATHS.output_dir(accession) / "predictions_wide.csv").exists():
            continue
        done += 1
        status = protein_status(accession, paths=PATHS)
        if status.get("generated"):
            finished.append(float(status["generated"]))
        for entry in status.get("predictors", []):
            name = entry.get("predictor", "?")
            if entry.get("status") == "failed":
                failures[name] = failures.get(name, 0) + 1
            value = entry.get("seconds")
            if value and entry.get("status") == "ok":
                seconds.setdefault(name, []).append(float(value))

    remaining = len(expected) - done
    print(f"{done:,} of {len(expected):,} proteins complete "
          f"({100 * done / max(len(expected), 1):.1f}%), {remaining:,} to go")

    if seconds:
        print("\ncompute per protein, by predictor (excludes job startup):")
        totals = {n: sum(v) / len(v) for n, v in seconds.items()}
        width = max(len(n) for n in totals)
        for name, mean in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<{width}}  {mean:8.1f}s   (n={len(seconds[name])})")
        print(f"  {'total':<{width}}  {sum(totals.values()):8.1f}s "
              f"= {sum(totals.values()) / 60:.1f} min")

    if failures:
        print("\nfailing on every protein they ran against:")
        for name, count in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {count:,}")

    # Throughput from the completions themselves. The earliest and latest
    # finish bracket a window during which `concurrency` tasks were running,
    # so the rate already accounts for everything a timing sum cannot see.
    if len(finished) >= 2 and remaining:
        span = max(finished) - min(finished)
        if span > 0:
            per_hour = (len(finished) - 1) / (span / 3600)
            hours = remaining / per_hour
            print(f"\nobserved throughput: {per_hour:.1f} proteins/hour "
                  f"over the last {span / 3600:.1f}h")
            print(f"projected: {hours:.1f} hours ({hours / 24:.1f} days) remaining")
            finish = time.strftime("%a %d %b %H:%M",
                                   time.localtime(time.time() + hours * 3600))
            print(f"           done around {finish}")
        else:
            print("\nnot enough spread in the completion times to project yet")
    elif remaining:
        print("\nnothing has finished yet, so there is nothing to project from")
    else:
        print("\nrun complete")
    return 0


def command_all(args: argparse.Namespace) -> int:
    """Run every stage in order, stopping at the first that fails."""
    for step in (command_define, command_fetch_data, command_run,
                 command_aggregate, command_analyse):
        print(f"\n=== {step.__name__.removeprefix('command_')} ===")
        code = step(args)
        if code:
            return code
        if step is command_run and not args.submit:
            print("\nstopping here: nothing has been submitted yet. Rerun with "
                  "--submit, or submit the command above and then run "
                  "`vepbench proteome aggregate` and `... analyse`.")
            return 0
    return 0


def add_parser(sub, common) -> None:
    """Attach ``proteome`` and its stages to the top-level parser."""
    proteome = sub.add_parser(
        "proteome",
        help="proteome-wide ClinVar, structure and predictor analysis",
        description="Run every stage of the proteome-wide analysis. Each stage is "
                    "separately runnable and resumable.",
    )
    stages = proteome.add_subparsers(dest="stage", required=True, parser_class=(
        lambda **kwargs: argparse.ArgumentParser(parents=[common], **kwargs)
    ))

    def stage(name, help_text, **kwargs):
        parser = stages.add_parser(name, help=help_text, **kwargs)
        parser.add_argument("--set", dest="protein_set", default=DEFAULT_SET,
                            help="a named protein set or a UniProt query "
                                 "(default: %(default)s)")
        return parser

    define = stage("define", "fetch the protein set and freeze it as a manifest")
    define.add_argument("--force", action="store_true",
                        help="refetch even though a manifest exists")
    define.add_argument("--count-only", action="store_true",
                        help="report how many entries match, and download nothing")
    define.add_argument("--min-length", type=int, default=1)
    define.add_argument("--max-length", type=int,
                        help="skip proteins longer than this; the largest few "
                             "dominate runtime on the big models")
    define.set_defaults(func=command_define)

    fetch = stage("fetch-data", "download UniProt, ClinVar and AlphaMissense")
    fetch.add_argument("--only", nargs="*",
                       choices=["uniprot", "clinvar", "alphamissense"],
                       help="run only these downloads (default: all three)")
    fetch.add_argument("--clinvar-file",
                       help="a variant_summary.txt.gz already on disk")
    fetch.add_argument("--alphamissense-file",
                       help="an AlphaMissense_aa_substitutions.tsv.gz already on disk")
    fetch.add_argument("--include-somatic", action="store_true",
                       help="keep somatic ClinVar records as well as germline")
    fetch.add_argument("--force", action="store_true", help="redownload and rebuild")
    fetch.set_defaults(func=command_fetch_data)

    run = stage("run", "submit the predictions as an LSF job array")
    run.add_argument("--predictors", nargs="+", default=["proteome"],
                     help="predictor names or a group (default: proteome, which is "
                          "everything that can run at this scale)")
    run.add_argument("--max-parallel", type=int, default=57,
                     help="array tasks running at once (default: %(default)s)")
    run.add_argument("--max-length", type=int,
                     help="skip proteins longer than this")
    run.add_argument("--skip-done", action="store_true",
                     help="leave out proteins that already have predictions")
    run.add_argument("--submit", action="store_true",
                     help="actually submit; without it the command is printed")
    run.add_argument("--dry-run", action="store_true",
                     help="pass --dry-run to the submitter")
    run.add_argument("--extra", nargs=argparse.REMAINDER,
                     help="further arguments for `vepbench run`")
    run.set_defaults(func=command_run)

    aggregated = stage("aggregate", "collect per-protein outputs into one dataset")
    aggregated.add_argument("--force", action="store_true",
                            help="rewrite partitions that already exist")
    aggregated.set_defaults(func=command_aggregate)

    analyse = stage("analyse", "join ClinVar, score every predictor, draw the figures")
    analyse.add_argument("--min-per-class", type=int, default=5,
                         help="pathogenic and benign variants needed before a group "
                              "is scored (default: %(default)s)")
    analyse.add_argument("--min-rows", type=int, default=50,
                         help="scored variants a predictor needs to be reported")
    analyse.add_argument("--n-boot", type=int, default=1000)
    analyse.add_argument("--seed", type=int, default=0)
    analyse.add_argument("--tracks", type=int, default=25,
                         help="per-protein figures to draw (default: %(default)s)")
    analyse.set_defaults(func=command_analyse)

    progress = stage("progress", "how far the run has got, and the time left")
    progress.add_argument("--max-length", type=int,
                          help="the cap the array was submitted with, so the "
                               "denominator matches what is actually running")
    progress.set_defaults(func=command_progress)

    everything = stage("all", "run every stage in order")
    # `all` has to satisfy every stage it calls, so it carries the union of
    # their options. They are listed again rather than copied out of the
    # other parsers: reaching into argparse internals to share them saves a
    # few lines and makes --help for this subcommand a lottery.
    everything.add_argument("--force", action="store_true")
    everything.add_argument("--min-length", type=int, default=1)
    everything.add_argument("--max-length", type=int)
    everything.add_argument("--count-only", action="store_true",
                            help=argparse.SUPPRESS)
    everything.add_argument("--only", nargs="*",
                            choices=["uniprot", "clinvar", "alphamissense"])
    everything.add_argument("--clinvar-file")
    everything.add_argument("--alphamissense-file")
    everything.add_argument("--include-somatic", action="store_true")
    everything.add_argument("--predictors", nargs="+", default=["proteome"])
    everything.add_argument("--max-parallel", type=int, default=57)
    everything.add_argument("--skip-done", action="store_true")
    everything.add_argument("--submit", action="store_true",
                            help="submit the array; without it `all` stops before "
                                 "aggregating and prints the command")
    everything.add_argument("--dry-run", action="store_true")
    everything.add_argument("--extra", nargs=argparse.REMAINDER)
    everything.add_argument("--min-per-class", type=int, default=5)
    everything.add_argument("--min-rows", type=int, default=50)
    everything.add_argument("--n-boot", type=int, default=1000)
    everything.add_argument("--seed", type=int, default=0)
    everything.add_argument("--tracks", type=int, default=25)
    everything.set_defaults(func=command_all)
