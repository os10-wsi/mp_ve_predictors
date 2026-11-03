"""Downloading the shared resources, once.

ClinVar, AlphaMissense and the ThermoMPNN table are large, change on their
own schedules and are read by every protein. They are fetched here, sliced
per accession, and never touched again by a scoring run. Each step skips
what is already on disk, so an interrupted fetch resumes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)

ALL = ("uniprot", "clinvar", "alphamissense", "thermompnn")


def fetch_all(args, *, paths: Paths = PATHS) -> int:
    wanted = list(args.what) if args.what else list(ALL)
    accessions = _accessions(args)
    failures: list[str] = []

    if "uniprot" in wanted:
        failures += _uniprot(accessions, paths=paths)
    if "alphamissense" in wanted:
        failures += _alphamissense(accessions, args, paths=paths)
    if "thermompnn" in wanted:
        failures += _thermompnn(args, paths=paths)
    if "clinvar" in wanted:
        failures += _clinvar(accessions, args, paths=paths)

    for message in failures:
        print(f"error: {message}")
    return 1 if failures else 0


def _accessions(args) -> list[str]:
    from vepbench.cli import read_accessions

    try:
        return read_accessions(args)
    except SystemExit:
        return []


def _uniprot(accessions: list[str], *, paths: Paths) -> list[str]:
    """Cache the sequence for each accession; ClinVar mapping needs them."""
    from vepbench.uniprot import UniProtError, fetch_uniprot

    if not accessions:
        return []
    missing = [a for a in accessions
               if not (paths.annotations / f"{a}.json").exists()]
    if not missing:
        log.info("all %d UniProt entries are cached", len(accessions))
        return []
    log.info("fetching %d UniProt entries", len(missing))
    failed = []
    for index, accession in enumerate(missing, start=1):
        try:
            fetch_uniprot(accession, paths=paths)
        except (UniProtError, OSError) as exc:
            failed.append(accession)
            log.warning("%s: %s", accession, exc)
        if index % 250 == 0:
            log.info("  %d/%d", index, len(missing))
    if failed:
        log.warning("%d UniProt entries could not be fetched", len(failed))
    return []


def _alphamissense(accessions: list[str], args, *, paths: Paths) -> list[str]:
    from vepbench.alphamissense import AlphaMissenseError, build_slices, download

    try:
        table = (Path(args.alphamissense_file) if args.alphamissense_file
                 else download(paths=paths, force=args.force))
        counts = build_slices(table, accessions or None, paths=paths,
                              force=args.force)
    except AlphaMissenseError as exc:
        return [str(exc)]
    print(f"AlphaMissense: slices for {len(counts):,} proteins")
    return []


def _thermompnn(args, *, paths: Paths) -> list[str]:
    """Place the published ThermoMPNN table where scoring will look for it."""
    from vepbench.thermompnn_table import (
        ZENODO_RECORD,
        ThermoMPNNTableError,
        archive_path,
        read_table,
    )

    destination = archive_path(paths)
    if args.thermompnn_file:
        source = Path(args.thermompnn_file)
        if not source.exists():
            return [f"no ThermoMPNN table at {source}"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            destination.write_bytes(source.read_bytes())
    elif not destination.exists():
        return [
            f"the ThermoMPNN table is not distributed automatically. Download it "
            f"from {ZENODO_RECORD} and pass --thermompnn-file, or put it at "
            f"{destination}."
        ]
    try:
        table = read_table(destination)
    except ThermoMPNNTableError as exc:
        return [str(exc)]
    print(f"ThermoMPNN: {len(table):,} published predictions covering "
          f"{table['uniprot_id'].nunique():,} proteins")
    return []


def _clinvar(accessions: list[str], args, *, paths: Paths) -> list[str]:
    from vepbench.clinvar import (
        ClinVarError,
        download,
        gene_index,
        label_counts,
        parse,
        sequences_from_cache,
    )
    from vepbench.clinvar_store import write_slices
    from vepbench.config import read_json

    try:
        path = (Path(args.clinvar_file) if args.clinvar_file
                else download(paths.cache / "clinvar" / "variant_summary.txt.gz",
                              force=args.force))
    except ClinVarError as exc:
        return [str(exc)]

    if not accessions:
        accessions = sorted(p.stem for p in paths.annotations.glob("*.json"))
    if not accessions:
        return ["no accessions given and none cached; pass --uniprot-list"]

    sequences, missing = sequences_from_cache(accessions, paths=paths)
    if missing:
        log.warning("%d of %d proteins have no cached UniProt entry and cannot "
                    "be mapped; run `vepbench fetch uniprot` first",
                    len(missing), len(accessions))
    if not sequences:
        return ["no UniProt sequences are cached, so nothing can be mapped"]

    genes = []
    for accession in sequences:
        payload = read_json(paths.annotations / f"{accession}.json")
        entry = payload.get("entry_name", "") or ""
        genes.append({"uniprot_id": accession,
                      "gene": entry.split("_")[0] if entry else ""})
    import pandas as pd

    manifest = pd.DataFrame(genes)
    gene_to_accession, ambiguous = gene_index(manifest)
    frame, report = parse(path, sequences, gene_to_accession,
                          ambiguous_genes=ambiguous,
                          germline_only=not args.include_somatic)
    write_slices(frame, paths=paths)
    print(report.summary())
    counts = label_counts(frame)
    print(f"ClinVar: {counts['pathogenic']:,} pathogenic, {counts['benign']:,} "
          f"benign, {counts['unlabelled']:,} unlabelled, across "
          f"{counts['proteins']:,} proteins")
    return []
