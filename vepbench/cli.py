"""Command line: fetch the data, score the proteins, report the result.

Three verbs do the work. ``fetch`` downloads the shared resources once.
``score`` runs the predictors over a list of accessions and writes one
table per protein. ``report`` turns those tables into the AUC tables and
the ROC panels. Each is separately runnable and resumable, because a
thousand proteins is not a thing you do in one sitting.

``install-env`` and ``status`` remain for the three environments that hold
a model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from vepbench import VEPBENCH
from vepbench.config import PATHS

log = logging.getLogger(__name__)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else (
        logging.DEBUG if verbosity > 0 else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)s: %(message)s")


def read_accessions(args: argparse.Namespace) -> list[str]:
    """Accessions from --uniprot, --uniprot-list, or both."""
    accessions: list[str] = list(args.uniprot or [])
    if args.uniprot_list:
        path = Path(args.uniprot_list)
        if not path.exists():
            raise SystemExit(f"error: no such file: {path}")
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            accessions.append(line.split()[0])
    if not accessions:
        raise SystemExit("error: give --uniprot ACCESSION... or --uniprot-list FILE")
    seen: set[str] = set()
    return [a for a in accessions if not (a in seen or seen.add(a))]


def command_score(args: argparse.Namespace) -> int:
    from vepbench.runner import score_protein, write_protein
    from vepbench.scores import ScoreError, resolve

    try:
        requested = resolve(args.scores)
    except ScoreError as exc:
        raise SystemExit(f"error: {exc}") from exc

    accessions = read_accessions(args)
    options: dict = {}
    if any(s.name == "thermompnn" for s in requested):
        options["thermompnn_index"] = _thermompnn_index(args)
    if args.device or args.batch_size:
        runtime = {}
        if args.device:
            runtime["device"] = args.device
        if args.batch_size:
            runtime["batch_size"] = args.batch_size
        options["runtime"] = runtime

    failures = 0
    for accession in accessions:
        print(f"\n=== {accession} ===")
        try:
            result = score_protein(accession, requested, paths=PATHS,
                                   force=args.force, offline=args.offline,
                                   options=options)
        except Exception as exc:  # noqa: BLE001 - one protein must not stop a list
            log.error("%s failed: %s", accession, exc)
            failures += 1
            continue
        path = write_protein(result, paths=PATHS)
        print(f"  {result.summary()}")
        print(f"  scored: {', '.join(result.scored) or 'nothing'}")
        print(f"  wrote {path}")
    if failures:
        print(f"\n{failures} of {len(accessions)} proteins failed", file=sys.stderr)
    return 1 if failures == len(accessions) else 0


def _thermompnn_index(args: argparse.Namespace) -> dict:
    from vepbench.thermompnn_table import (
        ThermoMPNNTableError,
        archive_path,
        index_by_protein,
        read_table,
    )

    path = Path(args.thermompnn_table) if args.thermompnn_table else archive_path(PATHS)
    try:
        return index_by_protein(read_table(path))
    except ThermoMPNNTableError as exc:
        log.warning("%s; ThermoMPNN will be computed for every protein", exc)
        return {}


def command_report(args: argparse.Namespace) -> int:
    from vepbench import report, roc
    from vepbench.scores import resolve

    accessions = read_accessions(args)
    predictors = [s.name for s in resolve(args.scores)]
    out = Path(args.out) if args.out else PATHS.outputs / "report"

    results: list[roc.RocResult] = []
    labelled = skipped = 0
    for accession in accessions:
        path = PATHS.output_dir(accession) / "scores.tsv"
        if not path.exists():
            skipped += 1
            continue
        frame = pd.read_csv(path, sep="\t")
        if "label" not in frame.columns or frame["label"].notna().sum() == 0:
            continue
        labelled += 1
        # Score on the damage-oriented column, not the raw one. Build a fresh
        # frame rather than renaming in place: the table holds both `x` and
        # `x__damaging`, so renaming would leave two columns called `x` and
        # every later lookup would get a DataFrame instead of a Series.
        present = [p for p in predictors if f"{p}__damaging" in frame.columns]
        oriented = pd.DataFrame({"label": frame["label"]})
        for name in present:
            oriented[name] = frame[f"{name}__damaging"]
        results += roc.evaluate(oriented, present, uniprot_id=accession,
                                min_per_class=args.min_per_class)

    if not results:
        print("error: no protein had enough labelled ClinVar variants to score. "
              "Run `vepbench fetch clinvar` and check the proteins are human.",
              file=sys.stderr)
        return 1

    written = report.write_tables(results, predictors, out)
    summary = pd.read_csv(written["auc_summary"], sep="\t")
    if not args.no_plots:
        report.roc_pdf(results, predictors, out / "roc_curves.pdf",
                       panels_per_page=args.panels_per_page)
        report.ranking_plot(summary, out / "auc_ranking.png")

    proteins = len({r.uniprot_id for r in results})
    print(f"{proteins:,} proteins scored, {len(results):,} protein-predictor AUCs")
    if skipped:
        print(f"  {skipped:,} accessions had no scores.tsv and were skipped")
    if labelled != proteins:
        print(f"  {labelled - proteins:,} had labels but too few of either class")
    print(f"\n{summary.to_string(index=False)}")
    print(f"\ntables and plots: {out}")
    return 0


def command_family(args: argparse.Namespace) -> int:
    from vepbench.families import FAMILIES, FamilyError, build

    pfam_ids = args.pfam or list(FAMILIES)
    if args.list:
        print(f"{'pfam':10} family")
        for pfam_id, name in FAMILIES.items():
            print(f"{pfam_id:10} {name}")
        return 0

    combined: list[str] = []
    for pfam_id in pfam_ids:
        try:
            result = build(pfam_id, n=args.n, identity=args.identity, seed=args.seed,
                           paths=PATHS, force=args.force, threads=args.threads)
        except FamilyError as exc:
            print(f"error: {pfam_id}: {exc}", file=sys.stderr)
            return 1
        print(result.summary())
        combined += result.accessions

    listing = PATHS.outputs / "families" / "all_accessions.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(combined))
    listing.write_text("\n".join(unique) + "\n")
    print(f"\n{len(unique):,} accessions across {len(pfam_ids)} families -> {listing}")
    print(f"  score them with: --uniprot-list {listing}")
    return 0


def command_family_report(args: argparse.Namespace) -> int:
    from vepbench import family_report
    from vepbench.families import membership

    owners = membership(PATHS)
    if not owners:
        print("error: no families have been built; run `vepbench family` first",
              file=sys.stderr)
        return 1
    frame = family_report.load_scored(sorted(owners), owners, outputs=PATHS.outputs,
                                      columns=args.scores)
    if frame.empty:
        print("error: none of the family members have been scored yet",
              file=sys.stderr)
        return 1

    predictors = family_report.predictors_in(frame)
    out = Path(args.out) if args.out else PATHS.outputs / "families" / "report"
    written = family_report.write_all(frame, predictors, out)

    counts = frame.groupby("family")["uniprot_id"].nunique()
    print(f"{len(frame):,} variant rows from {counts.sum():,} proteins "
          f"across {len(counts)} families")
    thin = counts[counts < family_report.MIN_PROTEINS]
    if len(thin):
        print(f"  {len(thin)} famil(y/ies) rest on fewer than "
              f"{family_report.MIN_PROTEINS} proteins: {', '.join(thin.index)}")
    for stem, path in written.items():
        print(f"  {stem}: {path}")
    if "family_auc" not in written:
        print("  no family had enough ClinVar-labelled variants for an AUC, "
              "which is expected: family members are mostly unreviewed")
    return 0


def command_check(args: argparse.Namespace) -> int:
    from vepbench.check import check, write_missing

    accessions = read_accessions(args)
    report = check(accessions, paths=PATHS)
    print("\n".join(report.lines()))

    if report.missing:
        destination = Path(args.write_missing) if args.write_missing else (
            PATHS.outputs / "missing.txt")
        path = write_missing(report, destination)
        print(f"\nthe {len(report.missing):,} with no table are listed in {path}")
        print(f"  resubmit with: --uniprot-list {path}")
    return 0 if report.complete else 1


def command_fetch(args: argparse.Namespace) -> int:
    from vepbench.fetch import fetch_all

    return fetch_all(args, paths=PATHS)


def command_status(args: argparse.Namespace) -> int:
    from vepbench.clinvar_store import available
    from vepbench.envs import ENVIRONMENTS
    from vepbench.scores import SCORES

    print(f"orchestrator : {sys.executable}")
    print(f"root         : {PATHS.root}")
    print(f"outputs      : {PATHS.outputs}")
    print(f"environments : {PATHS.envs}/<name>/.venv/bin/python")
    print()
    print(f"{'score':24s} {'kind':9s} {'needs':24s} state")
    print("-" * 78)
    for score in SCORES.values():
        needs = ", ".join(sorted(score.needs))
        if score.env:
            spec = ENVIRONMENTS.get(score.env)
            state = spec.state(PATHS) if spec else "no environment"
            state = f"env {score.env}: {state}"
        else:
            state = "built in"
        print(f"{score.name:24s} {score.kind:9s} {needs:24s} {state}")
    print()
    print(f"ClinVar slices: {available(PATHS):,} proteins")
    return 0


def command_install_env(args: argparse.Namespace) -> int:
    from vepbench.envs import ENVIRONMENTS, create_env

    names = args.names or sorted(ENVIRONMENTS)
    unknown = [n for n in names if n not in ENVIRONMENTS]
    if unknown:
        raise SystemExit(f"error: unknown environment(s) {', '.join(unknown)}; "
                         f"available: {', '.join(sorted(ENVIRONMENTS))}")
    for name in names:
        create_env(ENVIRONMENTS[name], paths=PATHS, backend=args.backend,
                   dry_run=args.dry_run, force=args.force)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from vepbench.scores import all_names

    parser = argparse.ArgumentParser(
        prog=VEPBENCH,
        description="Score protein variants with several predictors and measure "
                    "how well each separates pathogenic from benign ClinVar "
                    "variants, protein by protein.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0)
    common.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=(
        lambda **kwargs: argparse.ArgumentParser(parents=[common], **kwargs)))

    def targets(p):
        p.add_argument("--uniprot", nargs="*", help="one or more accessions")
        p.add_argument("--uniprot-list", help="a file of accessions, one per line")
        return p

    def selection(p):
        p.add_argument("--scores", nargs="+", default=None,
                       help=f"which scores to use (default: all). "
                            f"Available: {', '.join(all_names())}")
        return p

    score = selection(targets(sub.add_parser(
        "score", help="score every substitution in each protein")))
    score.add_argument("--force", action="store_true",
                       help="ignore cached scores and recompute")
    score.add_argument("--offline", action="store_true",
                       help="use only cached inputs; never reach the network")
    score.add_argument("--thermompnn-table",
                       help="a published ThermoMPNN table already on disk")
    score.add_argument("--device", choices=["cuda", "cpu", "mps"])
    score.add_argument("--batch-size", type=int,
                       help="masked positions per forward pass")
    score.set_defaults(func=command_score)

    rep = selection(targets(sub.add_parser(
        "report", help="AUC tables and ROC panels from scored proteins")))
    rep.add_argument("--out", help="output directory (default: <outputs>/report)")
    rep.add_argument("--min-per-class", type=int, default=5,
                     help="pathogenic and benign variants needed to score a "
                          "protein (default: %(default)s)")
    rep.add_argument("--panels-per-page", type=int, default=6)
    rep.add_argument("--no-plots", action="store_true",
                     help="write the tables only")
    rep.set_defaults(func=command_report)

    fetch = targets(sub.add_parser(
        "fetch", help="download ClinVar, AlphaMissense and the ThermoMPNN table"))
    fetch.add_argument("what", nargs="*",
                       choices=["clinvar", "alphamissense", "thermompnn", "uniprot"],
                       help="which resources (default: all)")
    fetch.add_argument("--clinvar-file")
    fetch.add_argument("--alphamissense-file")
    fetch.add_argument("--thermompnn-file")
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--include-somatic", action="store_true")
    fetch.set_defaults(func=command_fetch)

    checked = targets(sub.add_parser(
        "check", help="did every protein run, and did every score succeed"))
    checked.add_argument("--write-missing",
                         help="where to write the accessions with no table "
                              "(default: <outputs>/missing.txt)")
    checked.set_defaults(func=command_check)

    family = sub.add_parser(
        "family", help="build a scoreable sample from Pfam families")
    family.add_argument("--pfam", nargs="*",
                        help="Pfam accessions (default: the twenty built in)")
    family.add_argument("--list", action="store_true",
                        help="show the built-in families and exit")
    family.add_argument("-n", type=int, default=1000,
                        help="representatives to sample per family "
                             "(default: %(default)s)")
    family.add_argument("--identity", type=float, default=0.5,
                        help="clustering identity (default: %(default)s)")
    family.add_argument("--threads", type=int, default=8)
    family.add_argument("--seed", type=int, default=0)
    family.add_argument("--force", action="store_true")
    family.set_defaults(func=command_family)

    family_rep = selection(sub.add_parser(
        "family-report",
        help="compare predictor behaviour across the families"))
    family_rep.add_argument("--out", help="output directory")
    family_rep.set_defaults(func=command_family_report)

    status = sub.add_parser("status", help="what is installed and available")
    status.set_defaults(func=command_status)

    install = sub.add_parser("install-env", help="create a model environment")
    install.add_argument("names", nargs="*")
    install.add_argument("--backend", choices=["uv", "conda", "mamba", "micromamba"])
    install.add_argument("--dry-run", action="store_true")
    install.add_argument("--force", action="store_true")
    install.set_defaults(func=command_install_env)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(-1 if args.quiet else args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
