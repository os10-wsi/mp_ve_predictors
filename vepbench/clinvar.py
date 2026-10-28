"""ClinVar missense variants, mapped onto UniProt canonical sequences.

This is the one join in the analysis that can be wrong without looking
wrong. ClinVar is organised around RefSeq transcripts; vepbench is organised
around UniProt canonical sequences, and for a meaningful minority of genes
those disagree about which isoform is canonical. A gene-symbol join alone
would then line up the right protein at the wrong coordinates and produce a
table that passes every shape check while being quietly nonsense.

So every row has to clear two gates: the gene symbol has to resolve to one
reviewed accession in the manifest, and the wild-type residue ClinVar names
has to be the residue the UniProt sequence actually carries at that
position. Rows that fail are dropped and counted, never silently coerced,
and ``mapping_report.tsv`` records what happened per gene so the failures
can be inspected rather than assumed away.
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vepbench.config import PATHS, Paths, read_json
from vepbench.constants import AA_SET, THREE_TO_ONE

log = logging.getLogger(__name__)

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
DEFAULT_TIMEOUT = 1800

# `NM_000314.8(PTEN):c.389G>A (p.Arg130Gln)` -> ("Arg", "130", "Gln").
# Anything with fs, del, ins, dup, ext or a stop is not a single substitution
# and is excluded by the pattern itself rather than filtered afterwards.
_PROTEIN_CHANGE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")

# `Ter` is a stop codon and `Xaa` an unspecified residue. Both match the
# pattern above but neither is a substitution, so they are skipped the same
# way a frameshift is rather than reported as a malformed row.
_NOT_A_SUBSTITUTION = {"TER", "XAA"}

#: ClinVar review status to the familiar star rating.
REVIEW_STARS = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no assertion provided": 0,
    "no classifications from unflagged records": 0,
}

PATHOGENIC = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN = {"benign", "likely benign", "benign/likely benign"}

# Mapping outcomes, recorded per row and summarised per gene.
MAPPED = "mapped"
NO_PROTEIN_CHANGE = "no_protein_change"
UNKNOWN_GENE = "gene_not_in_manifest"
AMBIGUOUS_GENE = "gene_maps_to_several_accessions"
NO_SEQUENCE = "sequence_unavailable"
OUT_OF_RANGE = "position_past_sequence_end"
WT_MISMATCH = "wild_type_residue_disagrees"
NON_STANDARD = "non_standard_residue"

OUTPUT_COLUMNS = [
    "uniprot_id", "gene", "position", "wt_aa", "mut_aa", "variant",
    "clinical_significance", "label", "review_status", "review_stars",
    "n_submitters", "phenotypes", "variation_id", "origin",
]


class ClinVarError(RuntimeError):
    pass


@dataclass
class MappingReport:
    """What became of every ClinVar row we looked at."""

    counts: Counter = field(default_factory=Counter)
    per_gene: dict[str, Counter] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def record(self, gene: str, outcome: str, detail: str = "") -> None:
        self.counts[outcome] += 1
        self.per_gene.setdefault(gene, Counter())[outcome] += 1
        if detail and len(self.examples.setdefault(outcome, [])) < 5:
            self.examples[outcome].append(detail)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> str:
        mapped = self.counts[MAPPED]
        lines = [
            f"ClinVar mapping: {mapped:,} of {self.total:,} missense rows "
            f"({100 * mapped / self.total:.1f}%) mapped onto UniProt coordinates"
        ]
        for outcome, count in self.counts.most_common():
            if outcome == MAPPED:
                continue
            lines.append(f"  {outcome}: {count:,}")
            for example in self.examples.get(outcome, [])[:2]:
                lines.append(f"      e.g. {example}")
        return "\n".join(lines)

    def as_frame(self) -> pd.DataFrame:
        outcomes = sorted({o for counter in self.per_gene.values() for o in counter})
        rows = []
        for gene, counter in sorted(self.per_gene.items()):
            row = {"gene": gene, "total": sum(counter.values())}
            row.update({outcome: counter.get(outcome, 0) for outcome in outcomes})
            rows.append(row)
        return pd.DataFrame(rows)


def download(destination: Path, *, url: str = CLINVAR_URL, force: bool = False) -> Path:
    """Fetch variant_summary.txt.gz, skipping the download if it is already here.

    The file is around 100 MB compressed and changes weekly, so it is cached
    and only refetched on request. Which release was used is part of the
    result, so the path keeps its download date in ``.source.txt``.
    """
    destination = Path(destination)
    if destination.exists() and not force:
        log.info("using the cached ClinVar release at %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading ClinVar from %s (about 100 MB)", url)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT) as response, \
                tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ClinVarError(
            f"could not download ClinVar from {url}: {exc}. If this machine has no "
            f"route to ftp.ncbi.nlm.nih.gov, download the file elsewhere and pass "
            f"--clinvar-file."
        ) from exc
    tmp.replace(destination)
    destination.with_suffix(destination.suffix + ".source.txt").write_text(
        f"{url}\n{pd.Timestamp.utcnow().isoformat()}\n"
    )
    return destination


def _stars(review_status: str) -> int:
    key = (review_status or "").strip().lower()
    if key in REVIEW_STARS:
        return REVIEW_STARS[key]
    for prefix, stars in REVIEW_STARS.items():
        if key.startswith(prefix):
            return stars
    return 0


def _label(significance: str) -> int | None:
    """1 for pathogenic, 0 for benign, None for anything we will not train on.

    Conflicting and uncertain classifications are deliberately not folded
    into either class. They are the rows a predictor is most likely to be
    useful for, and scoring against a guess about them would measure the
    guess.
    """
    key = (significance or "").strip().lower()
    if key in PATHOGENIC:
        return 1
    if key in BENIGN:
        return 0
    return None


def _iter_chunks(path: Path, chunksize: int):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from pd.read_csv(
            handle, sep="\t", dtype=str, chunksize=chunksize,
            na_filter=False, low_memory=False,
        )


def parse(
    path: Path, sequences: dict[str, str], gene_to_accession: dict[str, str], *,
    assembly: str = "GRCh38", germline_only: bool = True,
    ambiguous_genes: frozenset[str] = frozenset(), chunksize: int = 200_000,
) -> tuple[pd.DataFrame, MappingReport]:
    """Read variant_summary and return the mapped missense rows.

    ``sequences`` and ``gene_to_accession`` come from the manifest and the
    UniProt annotation cache. Nothing is fetched here, so this is pure and
    testable.
    """
    report = MappingReport()
    rows: list[dict] = []

    for raw in _iter_chunks(Path(path), chunksize):
        chunk = raw
        if "Type" in chunk:
            chunk = chunk.loc[chunk["Type"].str.lower() == "single nucleotide variant"]
        if assembly and "Assembly" in chunk:
            chunk = chunk.loc[chunk["Assembly"] == assembly]
        if germline_only and "OriginSimple" in chunk:
            chunk = chunk.loc[chunk["OriginSimple"].str.contains("germline", case=False)]
        if chunk.empty:
            continue

        for record in chunk.to_dict("records"):
            gene = (record.get("GeneSymbol") or "").strip()
            name = record.get("Name") or ""
            match = _PROTEIN_CHANGE.search(name)
            if match is None:
                # Synonymous, frameshift, indel, nonsense: not a substitution.
                continue

            wt3, position_text, mut3 = match.groups()
            if wt3.upper() in _NOT_A_SUBSTITUTION or mut3.upper() in _NOT_A_SUBSTITUTION:
                continue
            wt = THREE_TO_ONE.get(wt3.upper(), "")
            mut = THREE_TO_ONE.get(mut3.upper(), "")
            if wt not in AA_SET or mut not in AA_SET:
                report.record(gene, NON_STANDARD, f"{gene} {name}")
                continue
            if wt == mut:
                continue

            if gene in ambiguous_genes:
                report.record(gene, AMBIGUOUS_GENE, f"{gene} {name}")
                continue
            accession = gene_to_accession.get(gene)
            if accession is None:
                report.record(gene, UNKNOWN_GENE, f"{gene} {name}")
                continue
            sequence = sequences.get(accession)
            if not sequence:
                report.record(gene, NO_SEQUENCE, f"{gene} ({accession})")
                continue

            position = int(position_text)
            if position < 1 or position > len(sequence):
                report.record(gene, OUT_OF_RANGE,
                              f"{gene} {name}: {accession} is {len(sequence)} residues")
                continue
            if sequence[position - 1] != wt:
                report.record(
                    gene, WT_MISMATCH,
                    f"{gene} {name}: {accession} has "
                    f"{sequence[position - 1]} at {position}, ClinVar says {wt}",
                )
                continue

            significance = (record.get("ClinicalSignificance") or "").strip()
            review = (record.get("ReviewStatus") or "").strip()
            report.record(gene, MAPPED)
            rows.append({
                "uniprot_id": accession, "gene": gene, "position": position,
                "wt_aa": wt, "mut_aa": mut, "variant": f"{wt}{position}{mut}",
                "clinical_significance": significance,
                "label": _label(significance),
                "review_status": review, "review_stars": _stars(review),
                "n_submitters": pd.to_numeric(
                    record.get("NumberSubmitters"), errors="coerce"),
                "phenotypes": (record.get("PhenotypeList") or "")[:300],
                "variation_id": record.get("VariationID") or "",
                "origin": record.get("OriginSimple") or "",
            })

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return deduplicate(frame), report


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the rows that describe one protein substitution.

    One amino-acid change can appear many times: several nucleotide changes
    encode it, and each lands in several RCV records. Keeping them all would
    weight a variant by how many ways there are to spell it. We keep the
    best-reviewed row, and mark the substitution as conflicting when the
    surviving records disagree about pathogenicity.
    """
    if frame.empty:
        return frame
    keys = ["uniprot_id", "position", "wt_aa", "mut_aa"]
    labelled = frame.loc[frame["label"].notna()]
    disagreeing = {
        key for key, group in labelled.groupby(keys)["label"]
        if group.nunique() > 1
    }

    ordered = frame.sort_values(
        ["review_stars", "n_submitters"], ascending=False, na_position="last"
    )
    deduped = ordered.drop_duplicates(subset=keys, keep="first").copy()

    if disagreeing:
        index = pd.MultiIndex.from_frame(deduped[keys])
        conflicted = index.isin(disagreeing)
        deduped.loc[conflicted, "label"] = pd.NA
        deduped.loc[conflicted, "clinical_significance"] = "Conflicting (vepbench)"
        log.info("%d substitutions had records disagreeing on pathogenicity; "
                 "their labels were cleared", int(conflicted.sum()))

    deduped["label"] = deduped["label"].astype("Int64")
    return deduped.sort_values(["uniprot_id", "position", "mut_aa"]).reset_index(drop=True)


def gene_index(manifest: pd.DataFrame) -> tuple[dict[str, str], frozenset[str]]:
    """Map gene symbol to accession, and name the symbols that are ambiguous.

    A symbol shared by two reviewed accessions cannot be resolved from the
    symbol alone, so those are excluded rather than arbitrarily assigned.
    """
    by_gene: dict[str, list[str]] = {}
    for raw_gene, accession in zip(manifest["gene"], manifest["uniprot_id"], strict=True):
        if pd.isna(raw_gene):
            continue
        gene = str(raw_gene).strip()
        if gene:
            by_gene.setdefault(gene, []).append(accession)
    unique = {gene: accs[0] for gene, accs in by_gene.items() if len(accs) == 1}
    ambiguous = frozenset(gene for gene, accs in by_gene.items() if len(accs) > 1)
    if ambiguous:
        log.info("%d gene symbols map to more than one reviewed accession and are "
                 "excluded: %s", len(ambiguous), ", ".join(sorted(ambiguous)[:8]))
    return unique, ambiguous


def label_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Pathogenic, benign and unlabelled totals, for the run log."""
    if frame.empty:
        return {"pathogenic": 0, "benign": 0, "unlabelled": 0, "proteins": 0}
    labels = frame["label"]
    return {
        "pathogenic": int((labels == 1).sum()),
        "benign": int((labels == 0).sum()),
        "unlabelled": int(labels.isna().sum()),
        "proteins": int(frame["uniprot_id"].nunique()),
    }


def sequences_from_cache(
    accessions: list[str], *, paths: Paths = PATHS
) -> tuple[dict[str, str], list[str]]:
    """Canonical sequences for the manifest, from the UniProt annotation cache.

    Returns the sequences found and the accessions still missing, so the
    caller can decide whether to fetch them or carry on with a subset.
    """
    sequences: dict[str, str] = {}
    missing: list[str] = []
    for accession in accessions:
        path = paths.annotations / f"{accession}.json"
        if not path.exists():
            missing.append(accession)
            continue
        try:
            sequences[accession] = read_json(path)["sequence"]
        except (KeyError, ValueError):
            missing.append(accession)
    return sequences, missing
