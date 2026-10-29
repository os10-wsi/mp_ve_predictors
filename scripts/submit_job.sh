#!/usr/bin/env bash
# Submit a vepbench run to LSF.
#
#   bash scripts/submit_job.sh --uniprot P26678
#   bash scripts/submit_job.sh --uniprot P26678 --predictors esm2_650m saprot
#   bash scripts/submit_job.sh --uniprot-list ids.txt --predictors ready
#
# Set VEPBENCH_LSF_GROUP (or pass --group) if your cluster requires -G.
#   bash scripts/submit_job.sh --uniprot P26678 --dry-run   # print, do not submit
#   bash scripts/submit_job.sh --uniprot P26678 --queue gpu-huge
#
# The GPU queue defaults to `gpu`; --queue or $VEPBENCH_QUEUE overrides it.
#
# Writes a self-contained job script under \$VEPBENCH_OUTPUTS/jobs/ and submits it with
# bsub. The script is kept, so you can read what ran, edit it and resubmit by
# hand without going through this wrapper.
#
# Two things this does that a hand-written bsub line usually forgets:
#
#   - It pins VEPBENCH_* and HF_HOME to absolute paths resolved here. Without
#     that the job falls back to $HOME on the compute node, which on a cluster
#     is a small quota, and a couple of checkpoints fill it.
#   - It runs `vepbench doctor` before submitting, so a missing environment
#     costs you a second rather than a place in the queue.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ACCESSIONS=()
LIST=""
PREDICTORS=()
FEATURES=()
QUEUE="${VEPBENCH_QUEUE:-gpu}"
# Some clusters refuse a submission that names no fairshare group. LSF reads
# LSB_DEFAULTGROUP by itself, so -G is only added when one is given here.
LSF_GROUP="${VEPBENCH_LSF_GROUP:-}"       # --no-gpu switches to the CPU queue
QUEUE_SET=0                          # whether --queue or $VEPBENCH_QUEUE chose it
if [ -n "${VEPBENCH_QUEUE:-}" ]; then QUEUE_SET=1; fi
GPU_SPEC="${VEPBENCH_GPU_SPEC:-num=%d:mode=shared}"
CORES=4
MEM_MB=32000
WALLTIME="4:00"
GPUS=1
NAME=""
ARRAY=1                              # a list submits one task per accession
MAX_PARALLEL="${VEPBENCH_MAX_PARALLEL:-10}"
DRY_RUN=0
SKIP_DOCTOR=0
EXTRA=()

usage() { sed -n '2,20p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --uniprot)       shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ACCESSIONS+=("$1"); shift; done ;;
    --uniprot-list)  LIST="$2"; shift 2 ;;
    --predictors)    shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do PREDICTORS+=("$1"); shift; done ;;
    --features)      shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do FEATURES+=("$1"); shift; done ;;
    --queue)         QUEUE="$2"; QUEUE_SET=1; shift 2 ;;
    --gpu-spec)      GPU_SPEC="$2"; shift 2 ;;
    --cores)         CORES="$2"; shift 2 ;;
    --mem)           MEM_MB="$2"; shift 2 ;;
    --time)          WALLTIME="$2"; shift 2 ;;
    --no-gpu)        GPUS=0; shift ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --name)          NAME="$2"; shift 2 ;;
    --one-job)       ARRAY=0; shift ;;
    --max-parallel)  MAX_PARALLEL="$2"; shift 2 ;;
    --group)         LSF_GROUP="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --skip-doctor)   SKIP_DOCTOR=1; shift ;;
    --)              shift; EXTRA+=("$@"); break ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

if [ ${#ACCESSIONS[@]} -eq 0 ] && [ -z "$LIST" ]; then
  echo "error: give --uniprot ACCESSION... or --uniprot-list FILE" >&2
  exit 2
fi
if [ -n "$LIST" ] && [ ! -f "$LIST" ]; then
  echo "error: no such file: $LIST" >&2
  exit 2
fi
if [ ${#PREDICTORS[@]} -eq 0 ]; then PREDICTORS=(all); fi

# The default queue only makes sense with a GPU, so swap it for the CPU one
# unless a queue was named explicitly.
if [ "$GPUS" -eq 0 ] && [ "$QUEUE_SET" = "0" ]; then
  QUEUE="${VEPBENCH_CPU_QUEUE:-normal}"
fi

# The interpreter that can import vepbench here is the one the job should use.
# A bare `python` does not exist on many systems, and `python3` outside an
# active virtualenv may be a different one.
# `import vepbench` is not enough to prove an interpreter can run the job. We
# are sitting in $ROOT, so the package imports from the working directory on
# any interpreter at all, and vepbench/__init__.py pulls in nothing but shlex
# and sys. A bare python3 with no dependencies installed sails through it and
# then dies in the queue on `import pandas`. Import the CLI instead: that is
# what the job runs, and it reaches the third-party imports.
find_python() {
  local candidate
  for candidate in "${VEPBENCH_PYTHON:-}" "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
                   python3 python; do
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c 'import vepbench.cli' >/dev/null 2>&1 && { command -v "$candidate"; return 0; }
  done
  return 1
}
if ! PYTHON="$(find_python)"; then
  echo "error: no Python here can run vepbench." >&2
  echo >&2
  echo "An interpreter needs vepbench's dependencies, not just the package:" >&2
  for candidate in "${VEPBENCH_PYTHON:-}" "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
                   python3 python; do
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    echo "  $(command -v "$candidate"): $("$candidate" -c 'import vepbench.cli' 2>&1 | tail -1)" >&2
  done
  echo >&2
  echo "Activate the environment you installed into, or name it explicitly:" >&2
  echo "  export VEPBENCH_PYTHON=/path/to/venv/bin/python" >&2
  echo "Or install here: python3 -m pip install -e $ROOT" >&2
  exit 1
fi
echo "python  : $PYTHON"

if [ ${#ACCESSIONS[@]} -gt 0 ]; then
  TARGET_FLAG=(--uniprot "${ACCESSIONS[@]}")
  LABEL="${ACCESSIONS[0]}"
  if [ ${#ACCESSIONS[@]} -gt 1 ]; then LABEL="${LABEL}+$(( ${#ACCESSIONS[@]} - 1 ))"; fi
else
  LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
  TARGET_FLAG=(--uniprot-list "$LIST")
  LABEL="$(basename "${LIST%.*}")"
fi
if [ -n "$LIST" ]; then
  N_IDS="$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' "$LIST" || true)"
  if [ "${N_IDS:-0}" -lt 1 ]; then
    echo "error: $LIST has no accessions in it" >&2
    exit 2
  fi
fi
if [ -z "$NAME" ]; then NAME="vepbench-$LABEL"; fi

# Preflight on the submit host. A job that cannot run anything should not wait
# in a queue to find out. A job that will skip some predictors is normal:
# `--predictors all` always includes ones needing a licence (FoldX) or a
# container runtime (DeepSequence), and the runner skips them with a reason.
# Gating on "everything is installed" would refuse every `all` submission.
if [ "$SKIP_DOCTOR" = "0" ]; then
  echo "=== checking what can run ==="
  # `doctor --json` exits non-zero whenever anything is missing, which for
  # `--predictors all` is the normal case, so the exit code is deliberately
  # not an answer to "is this worth submitting" -- see command_doctor in
  # vepbench/cli.py. The report is. Keep stderr off stdout so log lines
  # cannot corrupt the JSON, and let the ready count below make the call.
  DOCTOR_ERR="$(mktemp)"
  trap 'rm -f "$DOCTOR_ERR"' EXIT
  REPORT="$("$PYTHON" -m vepbench doctor --predictors "${PREDICTORS[@]}" --json \
            2>"$DOCTOR_ERR")" || true
  SUMMARY="$(printf '%s' "$REPORT" | "$PYTHON" -c '
import json, sys
try:
    report = json.load(sys.stdin)
except ValueError:
    print("UNREADABLE"); raise SystemExit(0)
ready, blocked = report["ready"], report["blocked"]
pending = report["needs_download"]
print(f"{len(ready)}	{len(blocked)}	{len(pending)}")
print("	".join(ready))
print("	".join(sorted(blocked)))
')"
  if [ "$SUMMARY" = "UNREADABLE" ]; then
    echo "warning: could not read the doctor report; submitting anyway." >&2
    if [ -s "$DOCTOR_ERR" ]; then tail -3 "$DOCTOR_ERR" | sed 's/^/  /' >&2; fi
  else
    COUNTS="$(printf '%s' "$SUMMARY" | sed -n 1p)"
    READY_N="$(printf '%s' "$COUNTS" | cut -f1)"
    BLOCKED_N="$(printf '%s' "$COUNTS" | cut -f2)"
    PENDING_N="$(printf '%s' "$COUNTS" | cut -f3)"
    echo "  will run : $READY_N predictor(s)"
    printf '%s\n' "$SUMMARY" | sed -n 2p | tr '\t' ' ' | fold -s -w 76 | sed 's/^/             /'
    if [ "$PENDING_N" -gt 0 ]; then
      echo "  will download weights first: $PENDING_N"
    fi
    if [ "$BLOCKED_N" -gt 0 ]; then
      echo "  will skip: $BLOCKED_N (recorded in status.json with a reason)"
      printf '%s\n' "$SUMMARY" | sed -n 3p | tr '\t' ' ' | fold -s -w 76 | sed 's/^/             /'
      echo "             \`$PYTHON -m vepbench doctor\` explains each one"
    fi
    if [ "$READY_N" -eq 0 ]; then
      echo >&2
      echo "error: none of the requested predictors can run, so the job would do" >&2
      echo "nothing. Install what you need, widen --predictors, or pass" >&2
      echo "--skip-doctor to submit regardless." >&2
      exit 1
    fi
  fi
fi

# Job scripts and logs belong with the analysis they produced, so they
# follow VEPBENCH_OUTPUTS rather than always landing in the checkout. A run
# directed elsewhere would otherwise leave its logs behind in a tree that
# has nothing else to do with it.
JOB_DIR="${VEPBENCH_OUTPUTS:-$ROOT/outputs}/jobs"
LOG_DIR="$JOB_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_SCRIPT="$JOB_DIR/${NAME}-${STAMP}.sh"
USE_ARRAY=0
if [ -n "$LIST" ] && [ "$ARRAY" = "1" ] && [ "$N_IDS" -gt 1 ]; then
  USE_ARRAY=1
  # A queued array reads this file as each task starts, which can be hours
  # after submission. Copy it, so editing the original does not silently
  # change what runs.
  LIST_SNAPSHOT="$JOB_DIR/${NAME}-${STAMP}.ids"
  grep -ve '^[[:space:]]*$' -e '^[[:space:]]*#' "$LIST" > "$LIST_SNAPSHOT"
fi

# Resolve the data locations now, so the job cannot fall back to $HOME.
VEPBENCH_DATA_ABS="${VEPBENCH_DATA:-$ROOT/data}"
VEPBENCH_OUTPUTS_ABS="${VEPBENCH_OUTPUTS:-$ROOT/outputs}"
VEPBENCH_WEIGHTS_ABS="${VEPBENCH_WEIGHTS:-$VEPBENCH_DATA_ABS/weights}"
HF_HOME_ABS="${HF_HOME:-$VEPBENCH_WEIGHTS_ABS/huggingface}"

quote() { printf '%q ' "$@"; }

{
  echo "#!/usr/bin/env bash"
  echo "# Generated by scripts/submit_job.sh on $STAMP. Safe to edit and rerun."
  echo "set -euo pipefail"
  echo
  echo "cd $(printf '%q' "$ROOT")"
  echo
  echo "# Lines to run before vepbench, e.g. \`module load cuda\`. Set"
  echo "# VEPBENCH_JOB_PRELUDE to a file to have its contents inlined here."
  if [ -n "${VEPBENCH_JOB_PRELUDE:-}" ] && [ -f "${VEPBENCH_JOB_PRELUDE}" ]; then
    cat "$VEPBENCH_JOB_PRELUDE"
  fi
  echo
  echo "export VEPBENCH_DATA=$(printf '%q' "$VEPBENCH_DATA_ABS")"
  echo "export VEPBENCH_OUTPUTS=$(printf '%q' "$VEPBENCH_OUTPUTS_ABS")"
  echo "export VEPBENCH_WEIGHTS=$(printf '%q' "$VEPBENCH_WEIGHTS_ABS")"
  echo "export HF_HOME=$(printf '%q' "$HF_HOME_ABS")"
  echo "export HF_HUB_DISABLE_XET=1"
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "# HF_TOKEN is read from the environment; bsub forwards it."
  fi
  echo
  echo "PYTHON=$(printf '%q' "$PYTHON")"
  echo '# The submit host and the compute node may not see the same filesystem.'
  echo '# Fall back only to an interpreter that can actually run the CLI, and say'
  echo '# so plainly rather than failing on an import several frames deep.'
  echo '[ -x "$PYTHON" ] || PYTHON=python3'
  echo 'if ! "$PYTHON" -c "import vepbench.cli" >/dev/null 2>&1; then'
  echo '  echo "error: $PYTHON cannot import vepbench.cli on $(hostname)" >&2'
  echo '  "$PYTHON" -c "import vepbench.cli" 2>&1 | tail -3 >&2'
  echo '  echo "Re-submit with VEPBENCH_PYTHON set to an interpreter that can." >&2'
  echo '  exit 1'
  echo 'fi'
  echo
  echo 'echo "host    : $(hostname)"'
  echo 'echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"'
  echo 'command -v nvidia-smi >/dev/null 2>&1 \'
  echo '  && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \'
  echo '  || echo "gpu     : none visible, running on CPU"'
  echo
  if [ "$USE_ARRAY" = "1" ]; then
    echo "# One accession per array task, from the snapshot taken at submission."
    echo "ACCESSION=\"\$(sed -n \"\${LSB_JOBINDEX}p\" $(printf '%q' "$LIST_SNAPSHOT"))\""
    echo 'if [ -z "$ACCESSION" ]; then'
    echo '  echo "no accession at index $LSB_JOBINDEX" >&2; exit 1'
    echo 'fi'
    echo 'echo "accession: $ACCESSION"'
    echo
    RUN_TARGET=(--uniprot '$ACCESSION')
  else
    RUN_TARGET=("${TARGET_FLAG[@]}")
  fi
  echo '"$PYTHON" -m vepbench run \'
  if [ "$USE_ARRAY" = "1" ]; then
    echo '  --uniprot "$ACCESSION" \'
  else
    echo "  $(quote "${TARGET_FLAG[@]}")\\"
  fi
  echo "  --predictors $(quote "${PREDICTORS[@]}")\\"
  if [ ${#FEATURES[@]} -gt 0 ]; then echo "  --features $(quote "${FEATURES[@]}")\\"; fi
  if [ ${#EXTRA[@]} -gt 0 ]; then echo "  $(quote "${EXTRA[@]}")\\"; fi
  echo "  ;"
  echo
  echo 'echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"'
  if [ "$USE_ARRAY" = "1" ]; then
    echo '"$PYTHON" -m vepbench status --uniprot "$ACCESSION"'
  else
    echo "\"\$PYTHON\" -m vepbench status $(quote "${TARGET_FLAG[@]}")"
  fi
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

JOB_NAME="$NAME"
LOG_SUFFIX="%J"
if [ "$USE_ARRAY" = "1" ]; then
  JOB_NAME="${NAME}[1-${N_IDS}]%${MAX_PARALLEL}"
  LOG_SUFFIX="%J.%I"
fi
BSUB=(bsub -J "$JOB_NAME" -q "$QUEUE" -n "$CORES"
      -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB] span[hosts=1]"
      -W "$WALLTIME"
      -o "$LOG_DIR/${NAME}-${STAMP}.${LOG_SUFFIX}.out"
      -e "$LOG_DIR/${NAME}-${STAMP}.${LOG_SUFFIX}.err")
if [ -n "$LSF_GROUP" ]; then
  BSUB+=(-G "$LSF_GROUP")
fi
if [ "$GPUS" -gt 0 ]; then
  # shellcheck disable=SC2059  # the %d is the caller's, from --gpu-spec
  BSUB+=(-gpu "$(printf "$GPU_SPEC" "$GPUS")")
fi
BSUB+=(bash "$JOB_SCRIPT")

echo
echo "=== job script ==="
echo "$JOB_SCRIPT"
sed 's/^/  /' "$JOB_SCRIPT"
echo
echo "=== submit command ==="
printf '  '; printf '%q ' "${BSUB[@]}"; echo
echo
if [ "$USE_ARRAY" = "1" ]; then
  echo "array: $N_IDS task(s), at most $MAX_PARALLEL at once (--max-parallel)"
  echo "       one accession each, from $LIST_SNAPSHOT"
  echo "       --one-job instead runs them all in a single serial job"
fi
echo "logs: $LOG_DIR/${NAME}-${STAMP}.<jobid>${USE_ARRAY:+.<task>}.{out,err}"

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "--dry-run: nothing submitted."
  exit 0
fi
command -v bsub >/dev/null 2>&1 || {
  echo >&2
  echo "error: bsub is not on PATH, so this is not an LSF submit host." >&2
  echo "The job script above is complete; run it directly with:" >&2
  echo "  bash $JOB_SCRIPT" >&2
  exit 1
}
echo
"${BSUB[@]}"
