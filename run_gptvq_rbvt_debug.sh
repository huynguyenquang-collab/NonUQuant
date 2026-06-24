#!/usr/bin/env bash
# =============================================================================
# run_gptvq_rbvt_debug.sh
# -----------------------------------------------------------------------------
# Debug GPTVQ-1D + RBVT with the same quantization settings as run_gptvq_rbvt.sh,
# but collect per-layer activation-output MSE diagnostics for the first few Linear
# modules and print a compact before/after table.
# =============================================================================
set -euo pipefail

# ---- paths / model ----------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(pwd)}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-./runs_gptvq3_llama3_rbvt_debug}"

# ---- GPTVQ-1D codebook knobs ------------------------------------------------
WBITS="${WBITS:-3}"
GROUPSIZE="${GROUPSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-100}"
KMEANS_INIT="${KMEANS_INIT:-mahalanobis}"
INCLUDE_M_STEP="${INCLUDE_M_STEP:-1}"
HESSIAN_LOOKUPS="${HESSIAN_LOOKUPS:-1}"
TRUE_SEQUENTIAL="${TRUE_SEQUENTIAL:-1}"
KEEP_ON_DEVICE="${KEEP_ON_DEVICE:-0}"

# ---- calibration / RBVT knobs -----------------------------------------------
N_CALIB="${N_CALIB:-128}"
MAX_LEN="${MAX_LEN:-2048}"
CALIB_DS="${CALIB_DS:-c4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_BUDGET_P="${RBVT_BUDGET_P:-${BUDGET_P:-0.005}}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-0.2}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-1}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"
STRICT_DESCENT="${STRICT_DESCENT:-1}"
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"

# ---- debug / eval -----------------------------------------------------------
DEBUG_LAYER_LIMIT="${DEBUG_LAYER_LIMIT:-6}"
DEBUG_MAX_TOKENS="${DEBUG_MAX_TOKENS:-4096}"
LM_EVAL="${LM_EVAL:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"

cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"

if [[ ! -d "GPTVQ" ]]; then
  echo "[setup] GPTVQ not found -> cloning ..."
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
fi

SLUG="gptvq${WBITS}b_g${GROUPSIZE}"
TAG="rbvt_debug"
OUTDIR="$OUT_ROOT/${SLUG}_${TAG}"
LOG="$OUT_ROOT/log_${SLUG}_${TAG}.txt"

common_args=(
  --model-path "$MODEL"
  --method gptvq
  --device "$DEVICE"
  --wbits "$WBITS"
  --groupsize "$GROUPSIZE"
  --kmeans-iters "$KMEANS_ITERS"
  --kmeans-init-method "$KMEANS_INIT"
  --gptq-blocksize "$GPTQ_BLOCKSIZE"
  --gptq-percdamp "$GPTQ_PERCDAMP"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LEN"
  --calib-dataset "$CALIB_DS"
  --eval-samples "$EVAL_SAMPLES"
  --gptvq-diagnostic-layer-limit "$DEBUG_LAYER_LIMIT"
  --gptvq-diagnostic-max-tokens "$DEBUG_MAX_TOKENS"
  --gptvq-stop-after-linear-layers "$DEBUG_LAYER_LIMIT"
  --skip-save-eval
)
[[ "$INCLUDE_M_STEP"   == "0" ]] && common_args+=(--no-include-m-step)
[[ "$HESSIAN_LOOKUPS"  == "0" ]] && common_args+=(--no-hessian-weighted-lookups)
[[ "$TRUE_SEQUENTIAL"  == "0" ]] && common_args+=(--no-true-sequential)
[[ "$KEEP_ON_DEVICE"   == "1" ]] && common_args+=(--keep-model-on-device)
[[ "$LM_EVAL"          == "0" ]] && common_args+=(--no-lm-eval)
if [[ "$STRICT_DESCENT" == "1" ]]; then
  common_args+=(--strict-descent)
else
  common_args+=(--allow-overshoot)
fi

echo ""
echo "================================================================"
echo ">>> VARIANT: $TAG  ->  $OUTDIR"
echo ">>> Debug layers: first $DEBUG_LAYER_LIMIT Linear modules, max_tokens=$DEBUG_MAX_TOKENS"
echo ">>> RBVT: lambda=$RBVT_LAMBDA topk=$RBVT_TOPK budget_p=$RBVT_BUDGET_P target_ratio=$RBVT_TARGET_RATIO mse_guard=$RBVT_MSE_GUARD sort=rho"
echo "================================================================"

rbvt_args=(
  --gptvq-correction rbvt
  --rbvt-lambda "$RBVT_LAMBDA"
  --rbvt-topk "$RBVT_TOPK"
  --rbvt-budget-p "$RBVT_BUDGET_P"
  --rbvt-target-ratio "$RBVT_TARGET_RATIO"
  --gap-floor "$GAP_FLOOR"
)
[[ "$RBVT_MSE_GUARD" == "1" ]] && rbvt_args+=(--rbvt-mse-guard)

set +e
python main.py "${common_args[@]}" --output-dir "$OUTDIR" \
  "${rbvt_args[@]}" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ -d "$OUTDIR" ]]; then
  find "$OUTDIR" -type f \
    ! -name "run_summary.json" \
    \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \
       -o -name "*.json" -o -name "*.model" -o -name "*.txt" \) \
    ! -name "run_summary.json" -delete 2>/dev/null || true
fi

if [[ ! -f "$OUTDIR/run_summary.json" ]]; then
  echo "!! missing $OUTDIR/run_summary.json"
  exit "$rc"
fi

python - "$OUTDIR/run_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
q = summary.get("quantization", {})
diag = q.get("activation_error_diagnostics", [])
hist = {row["layer"]: row for row in q.get("rbvt_layer_history", [])}

by_layer = {}
for row in diag:
    by_layer.setdefault(row["layer"], {})[row["variant"]] = row

print("\nRBVT debug | XW original vs XWq activation-output error")
print(f"{'layer':<42} {'mse_gptvq':>12} {'mse_rbvt':>12} {'mse_delta':>12} {'bias_before':>13} {'bias_after':>13} {'bias_delta':>12} {'flips':>8}")
print("-" * 132)
checked = 0
bias_not_increased = 0
awmse_base_up = 0
awmse_orig_up = 0
total_flips = 0
total_bias_before = 0.0
total_bias_after = 0.0
total_mse_base_before = 0.0
total_mse_base_after = 0.0
total_mse_orig_before = 0.0
total_mse_orig_after = 0.0
for layer, rows in by_layer.items():
    base = rows.get("gptvq")
    rbvt = rows.get("gptvq_rbvt")
    h = hist.get(layer, {})
    if not base or not rbvt:
        continue
    checked += 1
    bias_before = float(h.get("bias_before", 0.0))
    bias_after = float(h.get("bias_after", 0.0))
    mse_before = float(base["mse"])
    mse_after = float(rbvt["mse"])
    flips = int(h.get("flips", 0))
    bias_not_increased += int(bias_after <= bias_before + 1e-12)
    awmse_base_up += int(mse_after > mse_before + 1e-12)
    awmse_orig_up += int(mse_after > mse_before + 1e-12)
    total_flips += flips
    total_bias_before += bias_before
    total_bias_after += bias_after
    total_mse_base_before += mse_before
    total_mse_base_after += mse_after
    total_mse_orig_before += mse_before
    total_mse_orig_after += mse_after
    print(
        f"{layer:<42} "
        f"{mse_before:>12.6e} {mse_after:>12.6e} {mse_after - mse_before:>+12.6e} "
        f"{bias_before:>13.6e} {bias_after:>13.6e} "
        f"{bias_after - bias_before:>+12.6e} {flips:>8}"
    )

print("\nAggregate RBVT:")
for key in ("rbvt_lambda", "rbvt_topk", "rbvt_budget_p", "rbvt_target_ratio", "rbvt_mse_guard", "flips", "candidates", "bias_before", "bias_after", "objective_before", "objective_after", "variance_increase"):
    if key in q:
        print(f"  {key}: {q[key]}")

def pct_delta(before, after):
    if before == 0.0:
        return "nan"
    return f"{100.0 * (after - before) / before:+.2f}%"

def verdict(before, after, lower_is_better=True):
    if lower_is_better:
        return "NET WIN" if after <= before else "NET UP"
    return "NET WIN" if after >= before else "NET DOWN"

print("\n================== SUMMARY ==================")
print(f"layers checked          : {checked}")
print(f"bias not increased      : {bias_not_increased}/{checked}   (RBVT target: expect {checked}/{checked})")
print(f"awMSE[base] up          : {awmse_base_up}/{checked}   (vs chosen baseline; target 0)")
print(f"awMSE[orig] up          : {awmse_orig_up}/{checked}   (vs ORIGINAL fp = true inference error)")
print("=============================================")
print()
print("--- NET (summed over checked layers) ---")
print(f"total flips             : {total_flips}")
print(
    f"total bias              : {total_bias_before:.4e} -> {total_bias_after:.4e} "
    f"({pct_delta(total_bias_before, total_bias_after)})"
)
print(
    f"total act-wMSE[base]    : {total_mse_base_before:.4e} -> {total_mse_base_after:.4e} "
    f"({pct_delta(total_mse_base_before, total_mse_base_after)})   "
    f"[{verdict(total_mse_base_before, total_mse_base_after)}]"
)
print(
    f"total act-wMSE[orig]    : {total_mse_orig_before:.4e} -> {total_mse_orig_after:.4e} "
    f"({pct_delta(total_mse_orig_before, total_mse_orig_after)})   "
    f"[{verdict(total_mse_orig_before, total_mse_orig_after)}]  <- TRUE inference error"
)
print("=============================================")
PY

if [[ "$rc" != "0" ]]; then
  echo "!! debug run exited with code $rc"
  exit "$rc"
fi
