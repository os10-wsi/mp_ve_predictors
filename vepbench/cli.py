"""Command line interface: run, merge, list, status, benchmark, install-env."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from vepbench import VEPBENCH
from vepbench.config import PATHS, RunConfig
from vepbench.features import ALL_FEATURES, DEFAULT_FEATURES

log = logging.getLogger("vepbench")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else (
        logging.INFO if verbosity == 0 else logging.DEBUG
    )
    logging.basicConfig(level=level, format="%(levelname)-7s %(name)s: %(message)s")


def _parse_domain(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        start, end = text.replace(":", "-").split("-")
        return int(start), int(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--domain expects START-END, for example 32-52; got {text!r}"
        ) from exc


def _uniprot_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = list(args.uniprot or [])
    if args.uniprot_list:
        for raw_line in Path(args.uniprot_list).read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split()[0])
    if not ids:
        raise SystemExit("give at least one --uniprot, or a --uniprot-list file")
    seen, ordered = set(), []
    for identifier in ids:
        if identifier not in seen:
            seen.add(identifier)
            ordered.append(identifier)
    return ordered


def command_run(args: argparse.Namespace) -> int:
    from vepbench.runner import run_protein

    sequence_override = None
    if args.fasta:
        from vepbench.uniprot import read_fasta
        sequence_override = read_fasta(Path(args.fasta))
    elif args.sequence:
        sequence_override = args.sequence
    if args.email:
        os.environ["VEPBENCH_EMAIL"] = args.email

    config = RunConfig(
        predictors=args.predictors, include_wt=args.include_wt, force=args.force,
        structure_override=Path(args.pdb) if args.pdb else None,
        sequence_override=sequence_override,
        msa_override=Path(args.msa) if args.msa else None,
        domain=_parse_domain(args.domain), features=args.features,
        msa_backend=args.msa_backend, msa_database=args.msa_database,
        offline=args.offline, device=args.device, dtype=args.dtype,
        batch_size=args.batch_size,
    )

    exit_code = 0
    for uniprot_id in _uniprot_ids(args):
        print(f"\n=== {uniprot_id} ===")
        try:
            result = run_protein(uniprot_id, config)
        except Exception as exc:
            log.error("%s failed entirely: %s", uniprot_id, exc)
            if args.verbose > 0:
                raise
            exit_code = 1
            continue
        print(f"wrote {result.output_dir}/predictions_long.csv "
              f"({len(result.long_frame)} rows)")
        print(f"wrote {result.output_dir}/predictions_wide.csv "
              f"({len(result.wide_frame)} variants)")
        ok, unavailable, failed = result.succeeded, result.unavailable, result.failed
        print(f"  succeeded ({len(ok)}): {', '.join(ok) if ok else 'none'}")
        if unavailable:
            print(f"  unavailable ({len(unavailable)}): {', '.join(unavailable)}")
        if failed:
            print(f"  failed ({len(failed)}): {', '.join(failed)}")
            exit_code = 1
        for caveat in result.metadata.get("caveats", []):
            print(f"  note: {caveat}")
    return exit_code


def command_doctor(args: argparse.Namespace) -> int:
    """Report whether everything needed for a run is actually in place."""
    from vepbench.readiness import (
        MISSING,
        OK,
        WARNING,
        assess,
        download_command,
    )

    try:
        result = assess(args.predictors, msa_backend=args.msa_backend,
                        features=args.features, paths=PATHS)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.json:
        # For scripts deciding whether a run is worth submitting. The exit
        # code cannot carry that: "something is missing" and "nothing can
        # run" are different answers and only the second should stop a job.
        print(json.dumps({
            "ready": result.ready,
            "needs_download": result.needs_download,
            "blocked": result.blocked,
            "problems": [c.name for c in result.problems],
            "warnings": [c.name for c in result.warnings],
        }, indent=2))
        return 0 if not result.problems else 1

    groups = {OK: "ready", WARNING: "warnings", MISSING: "missing"}
    for status in (MISSING, WARNING, OK):
        selected = [c for c in result.checks if c.status == status]
        if not selected or (status == OK and not args.verbose):
            continue
        print(f"\n--- {groups[status]} ---")
        for check in selected:
            print(f"  {check.symbol:8s} {check.name}: {check.detail}")
            if check.fix:
                print(f"           -> {check.fix}")

    print(f"\n{len(result.ready)} predictor(s) ready to run")
    if result.ready:
        print("  " + ", ".join(result.ready))

    if result.needs_download:
        print(f"\n{len(result.needs_download)} installed but will download weights "
              f"on first use (so they need a network, and will fail under --offline):")
        commands = []
        for name, repos in sorted(result.needs_download.items()):
            print(f"  {name:28s} {', '.join(repos)}")
            for repo in repos:
                command = download_command(repo, PATHS)
                if command not in commands:
                    commands.append(command)
        print("  fetch them ahead of time:")
        for command in commands:
            print(f"    {command}")

    if result.blocked:
        print(f"\n{len(result.blocked)} blocked:")
        for name, reason in sorted(result.blocked.items()):
            print(f"  {name:28s} {reason[:110]}")

    if result.problems:
        print(f"\n{len(result.problems)} thing(s) need attention before a full run.")
        return 1
    print("\nEverything requested is in place.")
    return 0


def command_clean(args: argparse.Namespace) -> int:
    """Remove generated files, by category, after showing what and how much."""
    from vepbench.cleanup import (
        CATEGORIES,
        DEFAULT_CATEGORIES,
        PROTECTED,
        describe,
        human,
        remove,
        survey,
    )

    if args.list:
        print("categories (* = included by default):")
        for name in CATEGORIES:
            description, cost = describe(name)
            mark = "*" if name in DEFAULT_CATEGORIES else " "
            print(f"  {mark} {name:12s} {description}")
            print(f"    {'':12s} costs to recreate: {cost}")
        print(f"\nnever touched: {', '.join(PROTECTED)}")
        return 0

    wanted = args.what or list(DEFAULT_CATEGORIES)
    if "all" in wanted:
        # Still not results: losing a finished run to a tidy-up is not a
        # tidy-up. It has to be asked for by name.
        wanted = [n for n in CATEGORIES if n not in ("results", "venvs")]
    unknown = [n for n in wanted if n not in CATEGORIES]
    if unknown:
        print(f"ERROR: unknown categor(y/ies): {', '.join(unknown)}. "
              f"Try `{VEPBENCH} clean --list`.")
        return 2

    found = survey(wanted, paths=PATHS, older_than_days=args.older_than)
    total = sum(f.nbytes for f in found)
    count = sum(f.count for f in found)
    if not count:
        print("nothing to clean")
        return 0

    window = f" older than {args.older_than} day(s)" if args.older_than else ""
    print(f"would remove{window}:")
    for entry in found:
        if not entry.count:
            continue
        description, cost = describe(entry.category)
        print(f"  {entry.category:12s} {entry.count:4d} item(s)  "
              f"{human(entry.nbytes):>9s}  {entry.inodes:>9,d} files  {description}")
        if entry.category not in DEFAULT_CATEGORIES:
            print(f"  {'':12s} costs to recreate: {cost}")
    files = sum(f.inodes for f in found)
    print(f"  {'total':12s} {count:4d} item(s)  {human(total):>9s}  {files:>9,d} files")

    if args.dry_run:
        print("\n--dry-run: nothing removed.")
        return 0
    if not args.yes:
        try:
            answer = input("\nremove these? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("nothing removed.")
            return 0
    removed = remove(found)
    print(f"removed {removed} item(s), freeing {human(total)}")
    return 0


def command_prefetch(args: argparse.Namespace) -> int:
    """Download and cache every input a later run will need.

    Written for clusters, where the login node has network access and the
    compute nodes do not. Run this where there is a network, then run the
    predictions with ``--offline`` on the node.
    """
    from vepbench.msa import MSAError, build_msa
    from vepbench.predictors import REGISTRY, resolve_names
    from vepbench.structure import StructureError, load_structure
    from vepbench.uniprot import choose_domain, fetch_uniprot

    if args.email:
        os.environ["VEPBENCH_EMAIL"] = args.email

    names = resolve_names(args.predictors)
    needed = {
        requirement for name in names if name in REGISTRY
        for requirement in REGISTRY[name].spec.requires
    }
    from vepbench.features import DEFAULT_FEATURES, inputs_for_features
    needed |= inputs_for_features(args.features or list(DEFAULT_FEATURES))

    want_structure = "structure" in needed
    want_msa = "msa" in needed and args.msa_backend != "none"
    print(f"caching inputs for {len(_uniprot_ids(args))} protein(s): sequence"
          + (", structure" if want_structure else "")
          + (f", alignment via {args.msa_backend}" if want_msa else ""))

    failures = 0
    for uniprot_id in _uniprot_ids(args):
        print(f"\n=== {uniprot_id} ===")
        try:
            record = fetch_uniprot(uniprot_id, paths=PATHS, force=args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"  sequence FAILED: {exc}")
            failures += 1
            continue
        print(f"  sequence  {record.length} residues, {record.entry_name}")

        if "domain" in needed:
            prefer = "transmembrane" if record.is_membrane_protein else "auto"
            window, note = choose_domain(record, prefer=prefer, paths=PATHS)
            print(f"  domain    {window[0]}-{window[1]} ({note})")

        if want_structure:
            try:
                structure = load_structure(uniprot_id, record.sequence, paths=PATHS,
                                           force=args.force)
                print(f"  structure {structure.path.name}, coverage "
                      f"{structure.coverage:.0%}")
            except (StructureError, Exception) as exc:  # noqa: BLE001
                print(f"  structure FAILED: {exc}")
                failures += 1

        if want_msa:
            try:
                msa = build_msa(uniprot_id, record.sequence, backend=args.msa_backend,
                                paths=PATHS, force=args.force,
                                database=args.msa_database,
                                email=os.environ.get("VEPBENCH_EMAIL", ""))
                flag = " (SHALLOW)" if msa.is_shallow else ""
                print(f"  alignment {msa.n_sequences} sequences, "
                      f"Neff {msa.n_effective:.0f}{flag}")
            except (MSAError, Exception) as exc:  # noqa: BLE001
                print(f"  alignment FAILED: {exc}")
                failures += 1

    print(f"\ncache is at {PATHS.cache}")
    print("Runs can now use --offline, including on a node with no network.")
    return 1 if failures else 0


def command_merge(args: argparse.Namespace) -> int:
    from vepbench.merge import load_table, merge_with_dms
    from vepbench.uniprot import fetch_uniprot

    dms = load_table(Path(args.dms))
    wide_path = (Path(args.predictions) if args.predictions
                 else PATHS.output_dir(args.uniprot) / "predictions_wide.csv")
    if not wide_path.exists():
        raise SystemExit(
            f"no predictions at {wide_path}. Run `{VEPBENCH} run --uniprot "
            f"{args.uniprot}` first, or pass --predictions."
        )
    wide = pd.read_csv(wide_path)
    record = fetch_uniprot(args.uniprot, paths=PATHS, offline=args.offline)

    merged, report = merge_with_dms(
        dms, wide, record.sequence, variant_col=args.variant_col,
        position_col=args.position_col, wt_col=args.wt_col, mut_col=args.mut_col,
        uniprot_id=args.uniprot, keep_unmatched=not args.drop_unmatched,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"wrote {out} ({len(merged)} rows)")
    print(report.summary())

    report_path = out.with_name(out.stem + ".merge_report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2))
    print(f"wrote {report_path}")

    if report.unmatched:
        unmatched = out.with_name(out.stem + ".unmatched.csv")
        merged.loc[merged["match_status"] != "matched"].to_csv(unmatched, index=False)
        print(f"wrote {unmatched} ({report.unmatched} rows that did not match)")
    return 0


def command_list(args: argparse.Namespace) -> int:
    from vepbench.predictors import REGISTRY

    if args.json:
        print(json.dumps({n: p.spec.as_dict() for n, p in sorted(REGISTRY.items())},
                         indent=2))
        return 0

    header = f"{'name':30s} {'category':16s} {'needs':22s} {'env':14s} {'VRAM':>6s} verified"
    print(header)
    print("-" * len(header))
    for name, predictor in sorted(REGISTRY.items()):
        spec = predictor.spec
        if args.category and spec.category != args.category:
            continue
        needs = ",".join(sorted(spec.requires))
        vram = f"{spec.approximate_vram_gb:.0f}G" if spec.approximate_vram_gb else "-"
        flag = "yes" if spec.convention.verified else "NO - check"
        print(f"{name:30s} {spec.category:16s} {needs:22s} "
              f"{spec.env or '-':14s} {vram:>6s} {flag}")
    print(f"\n{len(REGISTRY)} predictors. Groups accepted by --predictors: all, "
          f"sequence_only, structure_based, msa_based, stability, no_msa.")
    return 0


def command_status(args: argparse.Namespace) -> int:
    from vepbench.envs import ENVIRONMENTS
    from vepbench.predictors import REGISTRY

    if args.uniprot:
        path = PATHS.output_dir(args.uniprot) / "status.json"
        if not path.exists():
            raise SystemExit(f"no status for {args.uniprot}; expected {path}")
        payload = json.loads(path.read_text())
        print(f"{args.uniprot}: " + ", ".join(f"{k}={v}"
                                              for k, v in payload["counts"].items()))
        for entry in payload["predictors"]:
            line = f"  {entry['predictor']:30s} {entry['status']:12s}"
            if entry.get("message"):
                line += f" {entry['message'][:90]}"
            print(line)
        return 0

    explanation = {
        "ok": "ready",
        "missing": "not installed",
        "incomplete": "INCOMPLETE - install did not finish",
        "stale": "stale - requirements changed since it was built",
    }
    print(f"{'environment':16s} {'state':44s} predictors")
    print("-" * 100)
    needs_attention = []
    for name, spec in sorted(ENVIRONMENTS.items()):
        members = sorted(n for n, p in REGISTRY.items() if p.spec.env == name)
        state = spec.state(PATHS)
        if state in ("incomplete", "stale"):
            needs_attention.append(name)
        print(f"{name:16s} {explanation[state]:44s} "
              f"{', '.join(members) if members else '-'}")
    if needs_attention:
        print(f"\nRe-run: {VEPBENCH} install-env "
              f"{' '.join(needs_attention) if len(needs_attention) == 1 else '<name>'}")
        for name in needs_attention:
            print(f"  {VEPBENCH} install-env {name}")
    return 0


def command_check_conventions(args: argparse.Namespace) -> int:
    """Report which sign conventions still need confirming.

    Harmonization is also verified arithmetically for every predictor, because
    getting a sign wrong silently inverts a benchmark.
    """
    from vepbench.predictors import REGISTRY
    from vepbench.schema import MORE_DELETERIOUS, harmonize

    unverified = []
    print(f"{'predictor':30s} {'score_type':26s} {'higher_is':18s} status")
    print("-" * 100)
    for name, predictor in sorted(REGISTRY.items()):
        convention = predictor.spec.convention
        probe = 2.5
        expected = probe if convention.higher_is == MORE_DELETERIOUS else -probe
        assert harmonize(probe, convention.higher_is) == expected, name
        status = "verified" if convention.verified else "NEEDS CONFIRMATION"
        if not convention.verified:
            unverified.append((name, convention))
        print(f"{name:30s} {convention.score_type:26s} {convention.higher_is:18s} {status}")

    if unverified:
        print("\nThese conventions were not confirmed against the method's own "
              "documentation. Confirm each before trusting a benchmark, because an "
              "inverted sign turns a good predictor into a bad one:\n")
        for name, convention in unverified:
            print(f"  {name}\n    {convention.source}\n")
    else:
        print("\nAll conventions are marked verified.")
    return 0


def command_install_env(args: argparse.Namespace) -> int:
    from vepbench.envs import ENVIRONMENTS, EnvError, create_env, get_env, resolve_backend

    try:
        backend = resolve_backend(args.backend)
    except EnvError as exc:
        raise SystemExit(str(exc)) from exc

    # Several names in one call: building one environment at a time is the
    # common case, but installing a group should not need a loop in the shell.
    names: list[str] = []
    for requested in args.names:
        for name in (sorted(ENVIRONMENTS) if requested == "all" else [requested]):
            if name not in names:
                names.append(name)
    unknown = [n for n in names if n not in ENVIRONMENTS]
    if unknown:
        # Check all of them before building any, so a typo in the last name
        # is not discovered after twenty minutes of downloads.
        raise SystemExit(
            f"unknown environment(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(ENVIRONMENTS))}"
        )

    print(f"building {len(names)} environment(s) with {backend}")
    failed: list[str] = []
    for name in names:
        spec = get_env(name)
        print(f"--- {name}: {spec.description}")
        if spec.notes:
            print(f"    note: {spec.notes}")
        extra = spec.conda_packages(PATHS)
        if extra.exists():
            wanted = [ln.strip() for ln in extra.read_text().splitlines()
                      if ln.strip() and not ln.startswith("#")]
            how = "installed from conda" if backend != "uv" else (
                "NOT installed: uv cannot supply these, use --backend conda"
            )
            print(f"    conda packages: {', '.join(wanted)} ({how})")
        state = spec.state(PATHS)
        if state == "ok" and not args.force:
            print("    already installed and verified")
            continue
        if state == "incomplete":
            print("    previous install did not finish; topping it up")
        elif state == "stale":
            print("    requirements changed since it was built; updating")
        if args.dry_run:
            continue
        try:
            venv = create_env(spec, paths=PATHS, force=args.force, backend=backend)
            print(f"    installed and verified at {venv}")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} environment(s) failed: {', '.join(failed)}")
        print("Nothing downstream will use them; they report as unavailable "
              "rather than producing wrong answers.")
        return 1
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    from vepbench.benchmark import (
        HIGHER_IS_MORE_DAMAGING,
        HIGHER_IS_MORE_FUNCTIONAL,
        evaluate_by_group,
        plot_summary,
    )
    from vepbench.merge import load_table

    merged = load_table(Path(args.merged))
    direction = (HIGHER_IS_MORE_DAMAGING if args.assay_direction == "damaging"
                 else HIGHER_IS_MORE_FUNCTIONAL)
    summary = evaluate_by_group(
        merged, args.score_col, group_col=args.group_col, assay_direction=direction,
        binarize_threshold=args.binarize_threshold,
        binarize_quantile=args.binarize_quantile, n_boot=args.n_boot,
        seed=args.seed, top_k=args.top_k,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"wrote {out}")

    pooled = summary.loc[summary["group"] == "pooled"].sort_values(
        "spearman", ascending=False
    )
    print(f"\n{'predictor':30s} {'n':>6s} {'spearman':>9s} {'95% CI':>18s} "
          f"{'auroc':>7s} {'ndcg':>6s}")
    for row in pooled.itertuples():
        ci = f"[{row.spearman_ci_low:+.3f},{row.spearman_ci_high:+.3f}]"
        print(f"{row.predictor:30s} {row.n_variants:6d} {row.spearman:+9.3f} "
              f"{ci:>18s} {row.auroc:7.3f} {row.ndcg:6.3f}")

    if args.plot:
        path = plot_summary(summary, Path(args.plot))
        if path:
            print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vepbench",
        description="Saturation variant-effect and stability prediction across many "
                    "predictors, for merging with deep mutational scanning data.",
    )
    # Shared flags go on every subparser as well as the top level, so
    # `vepbench run -q ...` works as readily as `vepbench -q run ...`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0)
    common.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=(
        lambda **kwargs: argparse.ArgumentParser(parents=[common], **kwargs)
    ))

    run = sub.add_parser("run", help="predict every substitution for one or more proteins")
    run.add_argument("--uniprot", nargs="*", help="one or more UniProt accessions")
    run.add_argument("--uniprot-list", help="file with one accession per line")
    run.add_argument("--predictors", nargs="+", default=["all"],
                     help="predictor names, or a group: all, sequence_only, "
                          "structure_based, msa_based, stability, no_msa")
    run.add_argument("--include-wt", action="store_true",
                     help="also emit synonymous rows, scored 0")
    run.add_argument("--force", action="store_true", help="ignore caches and recompute")
    run.add_argument("--pdb", help="structure file to use instead of the AlphaFold model")
    run.add_argument("--sequence", help="override the UniProt sequence")
    run.add_argument("--fasta", help="override the UniProt sequence from a FASTA file")
    run.add_argument("--msa", help="alignment file to use instead of generating one")
    run.add_argument("--msa-backend", default="colabfold",
                     choices=["colabfold", "ebi", "jackhmmer", "hhblits", "none"],
                     help="colabfold (default) is deepest for least effort; ebi needs "
                          "no local database but is shallower; jackhmmer and hhblits "
                          "search a database you supply and are the most sensitive")
    run.add_argument("--msa-database",
                     help="for jackhmmer or hhblits, the database path (or set "
                          "VEPBENCH_MSA_DATABASE, or leave it in data/databases/); "
                          "for ebi, 'psiblast:uniprotkb' or 'ncbiblast:uniprotkb'")
    run.add_argument("--email", help="contact address for EBI job submission "
                                     "(or set VEPBENCH_EMAIL)")
    run.add_argument("--domain", help="domain window as START-END, 1-based inclusive")
    run.add_argument("--features", nargs="*", default=list(DEFAULT_FEATURES),
                     choices=list(ALL_FEATURES))
    run.add_argument("--device", choices=["cuda", "cpu", "mps"],
                     help="override device selection (default: cuda when available)")
    run.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                     help="compute dtype (default: bfloat16 on a capable GPU, "
                          "float32 on CPU)")
    run.add_argument("--batch-size", type=int,
                     help="masked positions per forward pass (default: 16 on GPU, 4 on CPU)")
    run.add_argument("--offline", action="store_true",
                     help="use only cached inputs; never reach the network")
    run.set_defaults(func=command_run)

    doctor = sub.add_parser(
        "doctor",
        help="check environments, weights, binaries and credentials before a run",
    )
    doctor.add_argument("--predictors", nargs="+", default=["all"],
                        help="only check what these need")
    doctor.add_argument("--msa-backend", default="colabfold",
                        choices=["colabfold", "ebi", "jackhmmer", "hhblits", "none"])
    doctor.add_argument("--features", nargs="+",
                        help="also check what these features need")
    doctor.add_argument("--json", action="store_true",
                        help="machine-readable report, for scripts that need to "
                             "tell 'nothing can run' from 'something is missing'")
    # -v/--verbose is inherited from the shared parent parser; passing it
    # also lists the checks that already pass, not just the failures.
    doctor.set_defaults(func=command_doctor)

    clean = sub.add_parser(
        "clean",
        help="remove generated scratch, job scripts and logs",
    )
    clean.add_argument("--what", nargs="+", metavar="CATEGORY",
                       help="categories to remove, or 'all' (never results); "
                            "default: the ones that cost nothing to recreate")
    clean.add_argument("--older-than", type=int, default=0, metavar="DAYS",
                       help="only remove items last modified over DAYS ago")
    clean.add_argument("--list", action="store_true",
                       help="show the categories and what each costs to recreate")
    clean.add_argument("--dry-run", action="store_true",
                       help="show what would go, remove nothing")
    clean.add_argument("--yes", "-y", action="store_true", help="skip the prompt")
    clean.set_defaults(func=command_clean)

    prefetch = sub.add_parser(
        "prefetch",
        help="download and cache sequences, structures and alignments ahead of a run",
    )
    prefetch.add_argument("--uniprot", nargs="*")
    prefetch.add_argument("--uniprot-list")
    prefetch.add_argument("--predictors", nargs="+", default=["all"],
                          help="decides which inputs are needed")
    prefetch.add_argument("--features", nargs="*", default=list(DEFAULT_FEATURES),
                          choices=list(ALL_FEATURES))
    prefetch.add_argument("--msa-backend", default="colabfold",
                          choices=["colabfold", "ebi", "jackhmmer", "hhblits", "none"])
    prefetch.add_argument("--msa-database")
    prefetch.add_argument("--email")
    prefetch.add_argument("--force", action="store_true")
    prefetch.set_defaults(func=command_prefetch)

    merge = sub.add_parser("merge", help="join predictions onto a DMS table")
    merge.add_argument("--dms", required=True)
    merge.add_argument("--uniprot", required=True)
    merge.add_argument("--predictions", help="wide predictions CSV (default: outputs/<id>/)")
    merge.add_argument("--variant-col", help="column holding A123G or p.Ala123Gly")
    merge.add_argument("--position-col")
    merge.add_argument("--wt-col")
    merge.add_argument("--mut-col")
    merge.add_argument("--out", required=True)
    merge.add_argument("--drop-unmatched", action="store_true",
                       help="discard rows that did not match (kept by default)")
    merge.add_argument("--offline", action="store_true")
    merge.set_defaults(func=command_merge)

    listing = sub.add_parser("list", help="list the registered predictors")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--category")
    listing.set_defaults(func=command_list)

    status = sub.add_parser("status", help="show environment or per-protein run status")
    status.add_argument("--uniprot")
    status.set_defaults(func=command_status)

    conventions = sub.add_parser("check-conventions",
                                 help="audit the sign convention of every predictor")
    conventions.set_defaults(func=command_check_conventions)

    install = sub.add_parser("install-env", help="create an isolated predictor environment")
    install.add_argument("names", nargs="+", metavar="NAME",
                         help="one or more environment names, or 'all'")
    install.add_argument("--backend", choices=["uv", "conda", "mamba", "micromamba"],
                         help="how to build the environment (default: uv if present, "
                              "otherwise micromamba, mamba or conda; or set "
                              "VEPBENCH_ENV_BACKEND)")
    install.add_argument("--force", action="store_true")
    install.add_argument("--dry-run", action="store_true")
    install.set_defaults(func=command_install_env)

    bench = sub.add_parser("benchmark", help="score predictions against measured values")
    bench.add_argument("--merged", required=True)
    bench.add_argument("--score-col", required=True, help="column of measured values")
    bench.add_argument("--group-col",
                       help="column to report per-group metrics for, usually uniprot_id")
    bench.add_argument("--assay-direction", default="functional",
                       choices=["functional", "damaging"],
                       help="whether a higher measured value means more functional "
                            "(default) or more damaging")
    bench.add_argument("--binarize-threshold", type=float,
                       help="measured-value cutoff defining a damaging variant")
    bench.add_argument("--binarize-quantile", type=float, default=0.2,
                       help="without a threshold, call the worst this fraction damaging")
    bench.add_argument("--n-boot", type=int, default=1000)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--top-k", type=int, default=100)
    bench.add_argument("--out", default="benchmark_summary.csv")
    bench.add_argument("--plot", help="path for a summary plot (PNG)")
    bench.set_defaults(func=command_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(-1 if args.quiet else args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
