#!/usr/bin/env python3
"""DDMut client.

DDMut is published as a web service rather than as distributable code, so this
submits jobs to the server and collects results. A saturation scan means many
submissions, so requests are throttled and every answered variant is cached;
an interrupted run resumes where it stopped.

Set ``DDMUT_ENDPOINT`` to use a local deployment.

Sign convention: the mCSM and DDMut family reports positive ddG as
stabilising, the opposite of FoldX and Rosetta. That was not confirmed against
DDMut's own documentation, so the predictor entry marks it unverified.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import fail, parse_job, read_variants, write_meta, write_scores

DEFAULT_ENDPOINT = "https://biosig.lab.uq.edu.au/ddmut/api"


def main() -> int:
    import requests

    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])
    endpoint = os.environ.get("DDMUT_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    chain = job.get("structure_chain") or "A"
    pause = float(options.get("rate_limit_seconds", 2.0))

    if not job.get("structure_path"):
        fail("DDMut needs a structure")

    cache_path = work_dir / "ddmut_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    session = requests.Session()
    rows, failures = [], 0
    for index, (position, wt, mut) in enumerate(variants, start=1):
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        key = f"{chain}:{wt}{position}{mut}"
        if key in cache:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                         "score": cache[key]})
            continue
        try:
            with open(job["structure_path"], "rb") as handle:
                response = session.post(
                    f"{endpoint}/prediction_single", files={"pdb_file": handle},
                    data={"mutation": f"{wt}{position}{mut}", "chain": chain},
                    timeout=180,
                )
            response.raise_for_status()
            payload = response.json()
            value = payload.get("ddG") or payload.get("prediction") or payload.get("ddg")
            if value is None:
                raise ValueError(f"no ddG field in the response: {list(payload)[:8]}")
            value = float(value)
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: {exc}", file=sys.stderr)
            failures += 1
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            time.sleep(pause)
            continue
        cache[key] = value
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": value})
        if index % 20 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"  DDMut: {index}/{len(variants)}", file=sys.stderr)
        time.sleep(pause)

    cache_path.write_text(json.dumps(cache))
    if failures == len(variants):
        fail(
            f"every DDMut request failed. The public server at {endpoint} may be "
            f"unreachable from this machine, or its API may have changed. Check "
            f"https://biosig.lab.uq.edu.au/ddmut/ and set DDMUT_ENDPOINT if you run "
            f"it locally."
        )

    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"DDMut web service ({endpoint})",
        "scoring": "ddG in kcal/mol; sign convention UNVERIFIED, see PREDICTORS.md",
        "failed_variants": failures,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
