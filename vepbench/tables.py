"""Reading and writing the large derived tables.

The per-protein outputs stay CSV: they are small and people open them by
hand. The proteome-wide tables are a different problem. Fifty-five million
variant rows across ninety-odd columns is roughly two hundred gigabytes as
CSV and cannot be loaded at all, so those go to Parquet, partitioned by
accession, and are read back a slice at a time.

Parquet needs pyarrow, which is not a core dependency because nothing else
in vepbench needs it. When it is missing we fall back to gzipped CSV so the
small cases still work, and say loudly that the fallback will not survive
proteome scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PARQUET_SUFFIX = ".parquet"
FALLBACK_SUFFIX = ".csv.gz"

# Above this, gzipped CSV stops being a slow option and becomes an unusable
# one. Writing it anyway produces a file nobody can read back.
FALLBACK_ROW_LIMIT = 5_000_000

_INSTALL_HINT = "pip install 'vepbench[proteome]'  (or: pip install pyarrow)"


class TableError(RuntimeError):
    pass


def have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write one frame atomically, in the format the path's suffix names."""
    fallback = path.name.endswith(FALLBACK_SUFFIX)
    if fallback and len(frame) > FALLBACK_ROW_LIMIT:
        raise TableError(
            f"{len(frame):,} rows is past what the gzipped-CSV fallback can usefully "
            f"store, and pyarrow is not installed. {_INSTALL_HINT}"
        )
    if fallback:
        log.warning("pyarrow is missing; writing %s as gzipped CSV. %s", path.name, _INSTALL_HINT)
    tmp = path.with_name(path.name + ".tmp")
    if fallback:
        frame.to_csv(tmp, index=False, compression="gzip")
    else:
        frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.name.endswith(FALLBACK_SUFFIX):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def table_path(stem: Path) -> Path:
    """The path a table will actually be written to, given a suffix-less stem."""
    stem = Path(stem)
    return stem.with_suffix(PARQUET_SUFFIX if have_parquet() else FALLBACK_SUFFIX)


def write_table(stem: Path, frame: pd.DataFrame) -> Path:
    """Write one table, atomically, picking the best available format.

    ``stem`` carries no suffix; the chosen one is appended. The written path
    is returned so callers can log and record it.
    """
    path = table_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(path, frame)
    return path


def read_table(stem: Path) -> pd.DataFrame:
    """Read a table written by :func:`write_table`, whichever format it used."""
    stem = Path(stem)
    for suffix in (PARQUET_SUFFIX, FALLBACK_SUFFIX):
        candidate = stem.with_name(stem.name + suffix)
        if candidate.exists():
            return _read_frame(candidate)
    raise TableError(f"no table found at {stem}{PARQUET_SUFFIX} or {stem}{FALLBACK_SUFFIX}")


def partition_path(root: Path, uniprot_id: str) -> Path:
    """Where one protein's slice of a partitioned dataset lives."""
    suffix = PARQUET_SUFFIX if have_parquet() else FALLBACK_SUFFIX
    return Path(root) / f"uniprot_id={uniprot_id}" / f"part-0{suffix}"


def write_partition(root: Path, uniprot_id: str, frame: pd.DataFrame) -> Path:
    """Write one protein's slice, replacing whatever was there.

    Hive-style partitioning keeps the accession out of every row and lets a
    reader touch one protein without scanning the rest.
    """
    path = partition_path(root, uniprot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(path, frame.drop(columns=["uniprot_id"], errors="ignore"))
    return path


def read_dataset(root: Path, uniprot_ids: list[str] | None = None) -> pd.DataFrame:
    """Read a partitioned dataset back, optionally only the named proteins.

    Reading the whole membrane proteome into one frame needs well over a
    hundred gigabytes of memory, so callers that do not pass ``uniprot_ids``
    are normally working on a subset that was already filtered on disk.
    """
    root = Path(root)
    if not root.exists():
        raise TableError(f"no dataset at {root}")
    wanted = set(uniprot_ids) if uniprot_ids is not None else None
    frames: list[pd.DataFrame] = []
    for directory in sorted(root.glob("uniprot_id=*")):
        accession = directory.name.split("=", 1)[1]
        if wanted is not None and accession not in wanted:
            continue
        for part in sorted(directory.iterdir()):
            if part.name.endswith(".tmp"):
                continue
            if not part.name.endswith((FALLBACK_SUFFIX, PARQUET_SUFFIX)):
                continue
            frame = _read_frame(part)
            frame.insert(0, "uniprot_id", accession)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def dataset_members(root: Path) -> list[str]:
    """Accessions already present in a partitioned dataset."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        d.name.split("=", 1)[1] for d in root.glob("uniprot_id=*") if any(
            p for p in d.iterdir() if not p.name.endswith(".tmp")
        )
    )
