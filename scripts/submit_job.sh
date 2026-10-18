#!/usr/bin/env bash
# Submit a vepbench run to LSF.
#
#   bash scripts/submit_job.sh --uniprot P26678
#   bash scripts/submit_job.sh --uniprot P26678 --predictors esm2_650m saprot
#   bash scripts/submit_job.sh --uniprot-list ids.txt --predictors ready
#   bash scripts/submit_job.sh --uniprot P26678 --dry-run   # print, do not submit
#   bash scripts/submit_job.sh --uniprot P26678 --queue gpu-huge
#
# The GPU queue defaults to `gpu`; --queue or $VEPBENCH_QUEUE overrides it.
#
# Writes a self-contained job script under outputs/jobs/ and submits it with
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
QUEUE="${VEPBENCH_QUEUE:-gpu}"       # --no-gpu switches to the CPU queue
QUEUE_SET=0                          # whether --queue or $VEPBENCH_QUEUE chose it
if [ -n "${VEPBENCH_QUEUE:-}" ]; then QUEUE_SET=1; fi
GPU_SPEC="${VEPBENCH_GPU_SPEC:-num=%d:mode=shared}"
CORES=4
MEM_MB=32000
WALLTIME="4:00"
GPUS=1
NAME=""
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
find_python() {
  local candidate
  for candidate in "${VEPBENCH_PYTHON:-}" "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
                   python3 python; do
    [ -n "$candidate" ] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c 'import vepbench' >/dev/null 2>&1 && { command -v "$candidate"; return 0; }
  done
  return 1
}
if ! PYTHON="$(find_python)"; then
  echo "error: no Python here can import vepbench; run python3 -m pip install -e $ROOT" >&2
  exit 1
fi

if [ ${#ACCESSIONS[@]} -gt 0 ]; then
  TARGET_FLAG=(--uniprot "${ACCESSIONS[@]}")
  LABEL="${ACCESSIONS[0]}"
  if [ ${#ACCESSIONS[@]} -gt 1 ]; then LABEL="${LABEL}+$(( ${#ACCESSIONS[@]} - 1 ))"; fi
else
  LIST="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
  TARGET_FLAG=(--uniprot-list "$LIST")
  LABEL="$(basename "${LIST%.*}")"
fi
if [ -z "$NAME" ]; then NAME="vepbench-$LABEL"; fi

# Preflight on the submit host. A job that cannot run anything should not wait
# in a queue to find out. A job that will skip some predictors is normal:
# `--predictors all` always includes ones needing a licence (FoldX) or a
# container runtime (DeepSequence), and the runner skips them with a reason.
# Gating on "everything is installed" would refuse every `all` submission.
if [ "$SKIP_DOCTOR" = "0" ]; then
  echo "=== checking what can run ==="
  REPORT="$("$PYTHON" -m vepbench doctor --predictors "${PREDICTORS[@]}" --json)" || true
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

JOB_DIR="$ROOT/outputs/jobs"
LOG_DIR="$JOB_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_SCRIPT="$JOB_DIR/${NAME}-${STAMP}.sh"

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
  echo '[ -x "$PYTHON" ] || PYTHON=python3'
  echo
  echo 'echo "host    : $(hostname)"'
  echo 'echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"'
  echo 'command -v nvidia-smi >/dev/null 2>&1 \'
  echo '  && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \'
  echo '  || echo "gpu     : none visible, running on CPU"'
  echo
  echo '"$PYTHON" -m vepbench run \'
  echo "  $(quote "${TARGET_FLAG[@]}")\\"
  echo "  --predictors $(quote "${PREDICTORS[@]}")\\"
  if [ ${#FEATURES[@]} -gt 0 ]; then echo "  --features $(quote "${FEATURES[@]}")\\"; fi
  if [ ${#EXTRA[@]} -gt 0 ]; then echo "  $(quote "${EXTRA[@]}")\\"; fi
  echo "  ;"
  echo
  echo 'echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"'
  echo "\"\$PYTHON\" -m vepbench status $(quote "${TARGET_FLAG[@]}")"
} > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

BSUB=(bsub -J "$NAME" -q "$QUEUE" -n "$CORES"
      -M "$MEM_MB" -R "select[mem>$MEM_MB] rusage[mem=$MEM_MB] span[hosts=1]"
      -W "$WALLTIME"
      -o "$LOG_DIR/${NAME}-${STAMP}.%J.out"
      -e "$LOG_DIR/${NAME}-${STAMP}.%J.err")
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
echo "logs: $LOG_DIR/${NAME}-${STAMP}.<jobid>.{out,err}"

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
