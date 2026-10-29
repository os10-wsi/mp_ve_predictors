"""Write a run's output files without discarding what earlier runs wrote.

A run computes only the predictors it was asked for, but the files in the
output directory describe the protein, not the run. Writing them straight out
drops every column an earlier run produced, so a later ``--predictors rasp``
would leave a one-column table where fourteen columns used to be. Each write
here merges onto whatever is already on disk instead.

The per-predictor caches under ``data/cache/predictions/`` are the durable
record; these files are derived from them. That is what makes the merge safe
to redo: a column recovered from disk is a column some run already computed
and cached.

Two runs for the same protein can overlap, so the read-modify-write below is
serialised by a lock file in the output directory. Without it both would read
the same table and the second write would drop the first one's column.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from vepbench.config import unique_temp
from vepbench.schema import LONG_COLUMNS, SchemaError, empty_long_frame, validate_long_frame

log = logging.getLogger(__name__)

LOCK_NAME = ".outputs.lock"


@contextmanager
def output_lock(output_dir: Path, timeout: float = 600.0):
    """Hold an exclusive lock on one protein's output directory.

    Not every shared filesystem implements ``flock`` — some Lustre mounts
    refuse it. A refusal downgrades to a warning rather than failing the run:
    the merge is still worth doing, and losing a column to a simultaneous
    write is a narrower risk than losing the whole run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LOCK_NAME
    handle = None
    try:
        handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        _flock_with_timeout(handle, timeout, path)
    except OSError as exc:
        if handle is not None:
            os.close(handle)
            handle = None
        if exc.errno in (errno.ENOSYS, errno.ENOLCK, errno.EINVAL, errno.EOPNOTSUPP):
            log.warning("%s does not support file locking (%s); writing without a "
                        "lock. Concurrent runs on this protein could race.",
                        output_dir, exc.strerror)
        else:
            raise
    try:
        yield
    finally:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)


def _flock_with_timeout(handle: int, timeout: float, path: Path) -> None:
    """Block for the lock, but say what we are waiting on rather than hanging mute."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"waited {timeout:.0f}s for {path}. Another run holds it; if no "
                    f"run is active, delete the file."
                ) from None
            if not announced:
                log.info("another run holds %s, waiting for it to finish writing", path)
                announced = True
            time.sleep(1.0)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV that a reader never catches half-finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_temp(path)
    try:
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_existing_long(path: Path, sequence: str | None = None) -> pd.DataFrame:
    """Read a previous run's long table, or return an empty one.

    Anything unreadable or inconsistent with the current sequence is set aside
    rather than merged. A changed isoform renumbers every position, so mixing
    the two tables would produce rows that silently disagree about what the
    wild-type residue is.
    """
    if not path.exists():
        return empty_long_frame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not kill the run
        return _set_aside(path, f"could not be read ({exc})")

    missing = [c for c in LONG_COLUMNS if c not in frame.columns]
    if missing:
        return _set_aside(path, f"is missing columns {missing}")
    frame = frame.loc[:, LONG_COLUMNS]
    if sequence is not None:
        try:
            validate_long_frame(frame, sequence)
        except SchemaError as exc:
            return _set_aside(path, f"disagrees with the current sequence ({exc})")
    return frame


def _set_aside(path: Path, reason: str) -> pd.DataFrame:
    backup = path.with_suffix(path.suffix + ".superseded")
    path.replace(backup)
    log.warning("%s %s; moved it to %s and starting that table fresh. Re-run the "
                "affected predictors to restore their columns.", path.name, reason, backup.name)
    return empty_long_frame()


def merge_long(existing: pd.DataFrame, computed: pd.DataFrame) -> pd.DataFrame:
    """Combine this run's rows with earlier runs', newest wins per predictor.

    Only predictors that produced rows this time replace what is on disk. One
    that failed or was unavailable leaves the earlier result standing, which is
    what makes a narrow re-run of a single predictor safe to repeat.
    """
    if existing.empty:
        return computed
    recomputed = set(computed["predictor"].unique()) if len(computed) else set()
    kept = existing.loc[~existing["predictor"].isin(recomputed)]
    if not len(computed):
        return kept.reset_index(drop=True)
    return pd.concat([kept, computed], ignore_index=True).loc[:, LONG_COLUMNS]


def merge_status_entries(existing: list[dict], computed: list[dict]) -> list[dict]:
    """Keep the record of predictors this run did not touch.

    ``vepbench status`` tallies these across every protein. If a one-predictor
    run overwrote the list, the other fourteen would read as never attempted.
    """
    by_name = {e["predictor"]: e for e in existing if "predictor" in e}
    by_name.update({e["predictor"]: e for e in computed})
    return [by_name[name] for name in sorted(by_name)]
