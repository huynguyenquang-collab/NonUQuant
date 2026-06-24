#!/usr/bin/env bash
# =============================================================================
# run_gptvq_rbvt_budget_sweep_debug.sh
# -----------------------------------------------------------------------------
# Sweep RBVT_BUDGET_P values using the 6-Linear debug runner. This keeps lambda
# fixed at 1.0 by default and caps the per-row RBVT candidate prefix considered
# by the solver.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-./runs_gptvq3_llama3_rbvt_budget_debug}"
BUDGETS="${BUDGETS:-0.001 0.002 0.005 0.01 0.02}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
DEBUG_LAYER_LIMIT="${DEBUG_LAYER_LIMIT:-6}"
DEBUG_MAX_TOKENS="${DEBUG_MAX_TOKENS:-4096}"

cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"

SWEEP_SUMMARY="$OUT_ROOT/sweep_summary.txt"
printf "GPTVQ-1D + RBVT debug budget sweep\n" > "$SWEEP_SUMMARY"
printf "budgets: %s\n" "$BUDGETS" >> "$SWEEP_SUMMARY"
printf "rbvt_lambda: %s | rbvt_topk: %s\n" "$RBVT_LAMBDA" "$RBVT_TOPK" >> "$SWEEP_SUMMARY"
printf "debug layers: %s | max tokens: %s\n\n" "$DEBUG_LAYER_LIMIT" "$DEBUG_MAX_TOKENS" >> "$SWEEP_SUMMARY"

for budget in $BUDGETS; do
  budget_slug="${budget//./p}"
  budget_root="$OUT_ROOT/budget_${budget_slug}"

  echo ""
  echo "================================================================"
  echo ">>> RBVT_BUDGET_P=$budget | RBVT_LAMBDA=$RBVT_LAMBDA | RBVT_TOPK=$RBVT_TOPK"
  echo ">>> debug layers=$DEBUG_LAYER_LIMIT | output: $budget_root"
  echo "================================================================"

  RBVT_LAMBDA="$RBVT_LAMBDA" \
  RBVT_TOPK="$RBVT_TOPK" \
  RBVT_BUDGET_P="$budget" \
  DEBUG_LAYER_LIMIT="$DEBUG_LAYER_LIMIT" \
  DEBUG_MAX_TOKENS="$DEBUG_MAX_TOKENS" \
  OUT_ROOT="$budget_root" \
  LM_EVAL=0 \
  bash run_gptvq_rbvt_debug.sh

  log_file="$(find "$budget_root" -maxdepth 1 -type f -name 'log_gptvq*b_g*_rbvt_debug.txt' | sort | tail -n 1)"
  if [[ -z "${log_file:-}" || ! -f "$log_file" ]]; then
    echo "!! missing debug log for budget=$budget under $budget_root"
    exit 1
  fi

  {
    echo ""
    echo "================================================================"
    echo "RBVT_BUDGET_P=$budget | RBVT_LAMBDA=$RBVT_LAMBDA | RBVT_TOPK=$RBVT_TOPK"
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
  echo ">>> Saved budget=$budget summary to $SWEEP_SUMMARY"
done

echo ""
echo "================================================================"
echo "RBVT budget debug sweep done."
echo "Full per-budget logs:"
find "$OUT_ROOT" -maxdepth 2 -type f -name 'log_gptvq*b_g*_rbvt_debug.txt' | sort
echo ""
echo "Combined summary:"
echo "  $SWEEP_SUMMARY"
echo "================================================================"
