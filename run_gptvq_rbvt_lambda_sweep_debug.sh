#!/usr/bin/env bash
# =============================================================================
# run_gptvq_rbvt_lambda_sweep_debug.sh
# -----------------------------------------------------------------------------
# Sweep RBVT_LAMBDA values using the 6-Linear debug runner. Each lambda writes a
# full debug log + run_summary.json under its own directory, and this script also
# appends the Aggregate/SUMMARY/NET block to sweep_summary.txt.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-./runs_gptvq3_llama3_rbvt_lambda_debug}"
LAMBDAS="${LAMBDAS:-3.0 5.0 10.0 20.0}"
DEBUG_LAYER_LIMIT="${DEBUG_LAYER_LIMIT:-6}"
DEBUG_MAX_TOKENS="${DEBUG_MAX_TOKENS:-4096}"

cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"

SWEEP_SUMMARY="$OUT_ROOT/sweep_summary.txt"
printf "GPTVQ-1D + RBVT debug lambda sweep\n" > "$SWEEP_SUMMARY"
printf "lambdas: %s\n" "$LAMBDAS" >> "$SWEEP_SUMMARY"
printf "debug layers: %s | max tokens: %s\n\n" "$DEBUG_LAYER_LIMIT" "$DEBUG_MAX_TOKENS" >> "$SWEEP_SUMMARY"

for lambda in $LAMBDAS; do
  lambda_slug="${lambda//./p}"
  lambda_root="$OUT_ROOT/lambda_${lambda_slug}"

  echo ""
  echo "================================================================"
  echo ">>> RBVT_LAMBDA=$lambda | debug layers=$DEBUG_LAYER_LIMIT"
  echo ">>> output: $lambda_root"
  echo "================================================================"

  RBVT_LAMBDA="$lambda" \
  DEBUG_LAYER_LIMIT="$DEBUG_LAYER_LIMIT" \
  DEBUG_MAX_TOKENS="$DEBUG_MAX_TOKENS" \
  OUT_ROOT="$lambda_root" \
  LM_EVAL=0 \
  bash run_gptvq_rbvt_debug.sh

  log_file="$(find "$lambda_root" -maxdepth 1 -type f -name 'log_gptvq*b_g*_rbvt_debug.txt' | sort | tail -n 1)"
  if [[ -z "${log_file:-}" || ! -f "$log_file" ]]; then
    echo "!! missing debug log for lambda=$lambda under $lambda_root"
    exit 1
  fi

  {
    echo ""
    echo "================================================================"
    echo "RBVT_LAMBDA=$lambda"
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
  echo ">>> Saved lambda=$lambda summary to $SWEEP_SUMMARY"
done

echo ""
echo "================================================================"
echo "RBVT lambda debug sweep done."
echo "Full per-lambda logs:"
find "$OUT_ROOT" -maxdepth 2 -type f -name 'log_gptvq*b_g*_rbvt_debug.txt' | sort
echo ""
echo "Combined summary:"
echo "  $SWEEP_SUMMARY"
echo "================================================================"
