#!/usr/bin/env bash
# =============================================================================
# run_gptvq_rbvt_topk_sweep_debug.sh
# -----------------------------------------------------------------------------
# Sweep RBVT_TOPK values using the 6-Linear debug runner, with RBVT_LAMBDA fixed
# at 1.0 by default. Each topk writes a full debug log + run_summary.json under
# its own directory, and this script appends Aggregate/SUMMARY/NET blocks to
# sweep_summary.txt.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-./runs_gptvq3_llama3_rbvt_topk_debug}"
TOPKS="${TOPKS:-256 512 1024 2048}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
DEBUG_LAYER_LIMIT="${DEBUG_LAYER_LIMIT:-6}"
DEBUG_MAX_TOKENS="${DEBUG_MAX_TOKENS:-4096}"

cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"

SWEEP_SUMMARY="$OUT_ROOT/sweep_summary.txt"
printf "GPTVQ-1D + RBVT debug topk sweep\n" > "$SWEEP_SUMMARY"
printf "topks: %s\n" "$TOPKS" >> "$SWEEP_SUMMARY"
printf "rbvt_lambda: %s\n" "$RBVT_LAMBDA" >> "$SWEEP_SUMMARY"
printf "debug layers: %s | max tokens: %s\n\n" "$DEBUG_LAYER_LIMIT" "$DEBUG_MAX_TOKENS" >> "$SWEEP_SUMMARY"

for topk in $TOPKS; do
  topk_root="$OUT_ROOT/topk_${topk}"

  echo ""
  echo "================================================================"
  echo ">>> RBVT_TOPK=$topk | RBVT_LAMBDA=$RBVT_LAMBDA | debug layers=$DEBUG_LAYER_LIMIT"
  echo ">>> output: $topk_root"
  echo "================================================================"

  RBVT_LAMBDA="$RBVT_LAMBDA" \
  RBVT_TOPK="$topk" \
  DEBUG_LAYER_LIMIT="$DEBUG_LAYER_LIMIT" \
  DEBUG_MAX_TOKENS="$DEBUG_MAX_TOKENS" \
  OUT_ROOT="$topk_root" \
  LM_EVAL=0 \
  bash run_gptvq_rbvt_debug.sh

  log_file="$(find "$topk_root" -maxdepth 1 -type f -name 'log_gptvq*b_g*_rbvt_debug.txt' | sort | tail -n 1)"
  if [[ -z "${log_file:-}" || ! -f "$log_file" ]]; then
    echo "!! missing debug log for topk=$topk under $topk_root"
    exit 1
  fi

  {
    echo ""
    echo "================================================================"
    echo "RBVT_TOPK=$topk | RBVT_LAMBDA=$RBVT_LAMBDA"
    echo "log: $log_file"
    echo "================================================================"
    awk '
      /^Aggregate RBVT:/ {capture=1}
      capture {print}
      /^=============================================$/ && seen_summary {capture=0}
      /^================== SUMMARY ==================$/ {seen_summary=1}
    ' "$log_file"
  } >> "$SWEEP_SUMMARY"

  echo ""
  echo ">>> Saved topk=$topk summary to $SWEEP_SUMMARY"
done

echo ""
echo "================================================================"
echo "RBVT topk debug sweep done."
echo "Full per-topk logs:"
find "$OUT_ROOT" -maxdepth 2 -type f -name 'log_gptvq*b_g*_rbvt_debug.txt' | sort
echo ""
echo "Combined summary:"
echo "  $SWEEP_SUMMARY"
echo "================================================================"
