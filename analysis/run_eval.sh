#!/usr/bin/env bash
# Usage: ./run_eval.sh [group] [expr_id] [n_tasksets] [extra experiments.py flags...]
#   group      - experiment group (default: 1)
#   expr_id    - experiment id    (default: 4)
#   n_tasksets - tasksets per point (default: 200)
#   extra flags - any args after the three positionals are forwarded verbatim to
#                 experiments.py, e.g. to inject measured per-dispatch overheads:
#                 ./run_eval.sh 1 4 200 --gcaps-overhead-us 1149 --seq-overhead-us 155
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

GROUP=${1:-1}
EXPR=${2:-4}
N=${3:-200}
# Consume the (up to three) positionals, forward the rest to experiments.py.
shift $(( $# > 3 ? 3 : $# ))
EXTRA=("$@")

# --- Column headers per group ------------------------------------------------
if   [ "$GROUP" -eq 1 ]; then
    COLS=("mpcp_sus" "mpcp_bsy" "fmlp_sus" "fmlp_bsy" \
          "ioctl_sus" "ioctl_bsy" "tsg_sus" "tsg_bsy" \
          "npfp_bsy" "npfp_sus")
elif [ "$GROUP" -eq 3 ]; then
    COLS=("ioctl_sus_base" "ioctl_bsy_base" "ioctl_sus_gprio" "ioctl_bsy_gprio")
else
    echo "Unknown group $GROUP" >&2; exit 1
fi

# --- Row labels per expr_id --------------------------------------------------
case "$EXPR" in
  1) ROWS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8);       ROW_HDR="be_task%" ;;
  2) ROWS=(2 4 6 8 10 12 14 16 18 20);                   ROW_HDR="num_tasks" ;;
  3) ROWS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0); ROW_HDR="g/c_ratio" ;;
  4) ROWS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0);   ROW_HDR="util/cpu" ;;
  5) ROWS=(1 2 3 4 5 6 7 8 9);                           ROW_HDR="num_cpus" ;;
  6) ROWS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0);   ROW_HDR="gpu_task%" ;;
  *) echo "Unknown expr_id $EXPR" >&2; exit 1 ;;
esac

# --- Run experiment, show live progress, capture output ----------------------
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

echo "Running: python3 experiments.py -g $GROUP -e $EXPR -n $N ${EXTRA[*]}"
echo
python3 experiments.py -g "$GROUP" -e "$EXPR" -n "$N" "${EXTRA[@]}" 2>&1 | tee "$TMPFILE"

DATA=$(grep -E '^[0-9]' "$TMPFILE")

# --- Print formatted table ---------------------------------------------------
echo
echo "=== Schedulability results (% of $N tasksets passing) ==="
echo

{
    # Header
    printf "%s" "$ROW_HDR"
    for col in "${COLS[@]}"; do printf "\t%s" "$col"; done
    printf "\n"

    # Separator
    printf "%s" "--------"
    for col in "${COLS[@]}"; do printf "\t%s" "--------"; done
    printf "\n"

    # Data rows
    i=0
    while IFS= read -r line; do
        label="${ROWS[$i]:-$i}"
        printf "%s" "$label"
        # split comma-separated values and print each
        IFS=',' read -ra vals <<< "$line"
        for v in "${vals[@]}"; do
            printf "\t%s" "$(echo "$v" | tr -d ' ')"
        done
        printf "\n"
        i=$(( i + 1 ))
    done <<< "$DATA"
} | column -t -s $'\t'
