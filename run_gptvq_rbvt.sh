#!/usr/bin/env bash
# =============================================================================
# run_gptvq_rbvt.sh
# -----------------------------------------------------------------------------
# Same run/eval harness as run_gptvq_ncc.sh, but the corrected variant replaces
# GPTVQ-1D + NCC post_module with GPTVQ-1D + RBVT post_module.
#
# The script intentionally keeps the same defaults, output layout, evaluation,
# logging, cleanup, and comparison-table parsing style as run_gptvq_ncc.sh.
# =============================================================================
set -euo pipefail

# ---- paths / model ----------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(pwd)}"               # dir containing main.py
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-./runs_gptvq_rbvt}"

# ---- GPTVQ-1D codebook knobs ------------------------------------------------
WBITS="${WBITS:-3}"                           # 3|4
GROUPSIZE="${GROUPSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-100}"
KMEANS_INIT="${KMEANS_INIT:-mahalanobis}"     # cdf|kpp|mahalanobis
INCLUDE_M_STEP="${INCLUDE_M_STEP:-1}"         # 1 -> M-step on, 0 -> --no-include-m-step
HESSIAN_LOOKUPS="${HESSIAN_LOOKUPS:-1}"       # 1 -> on, 0 -> --no-hessian-weighted-lookups
TRUE_SEQUENTIAL="${TRUE_SEQUENTIAL:-1}"       # 1 -> on, 0 -> --no-true-sequential
KEEP_ON_DEVICE="${KEEP_ON_DEVICE:-0}"         # 1 -> --keep-model-on-device

# ---- calibration / RBVT knobs -----------------------------------------------
N_CALIB="${N_CALIB:-128}"
MAX_LEN="${MAX_LEN:-2048}"
CALIB_DS="${CALIB_DS:-c4}"                    # c4|wikitext2
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-1.0}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"
STRICT_DESCENT="${STRICT_DESCENT:-1}"         # 1 -> --strict-descent, 0 -> --allow-overshoot
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"

# ---- eval toggles (set to 0 to skip the heavy lm-eval during debugging) -----
LM_EVAL="${LM_EVAL:-1}"                       # 1 -> include lm-eval, 0 -> skip
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"

# ---- which variants to run --------------------------------------------------
RUN_BASE="${RUN_BASE:-1}"                     # GPTVQ-1D, no correction
RUN_RBVT_POST_MODULE="${RUN_RBVT_POST_MODULE:-1}"
RUN_RBVT_POST_BLOCK="${RUN_RBVT_POST_BLOCK:-0}"

# -----------------------------------------------------------------------------
cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"
# fresh comparison table
printf "variant\tperplexity...\tquant_stats\n" > "$OUT_ROOT/perplexity_table.tsv"

# GPTVQ upstream is required for every variant.
if [[ ! -d "GPTVQ" ]]; then
  echo "[setup] GPTVQ not found -> cloning ..."
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
fi

if [[ "$RUN_RBVT_POST_BLOCK" == "1" ]]; then
  echo "!! RBVT post_block is not implemented; RBVT is applied post_module."
  exit 2
fi

# slug for output dirs / table rows
SLUG="gptvq${WBITS}b_g${GROUPSIZE}"

# common args shared by every run
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

run_variant () {
  local tag="$1"; shift
  local outdir="$OUT_ROOT/${SLUG}_${tag}"
  echo ""
  echo "================================================================"
  echo ">>> VARIANT: $tag  ->  $outdir"
  echo "================================================================"
  # main.py: quantize -> save_pretrained -> perplexity eval -> (lm-eval) ->
  # save run_summary.json -> cleanup_output_dir (deletes model, keeps summary).
  set +e
  python -u main.py "${common_args[@]}" --output-dir "$outdir" "$@" \
    2>&1 | tee "$OUT_ROOT/log_${SLUG}_${tag}.txt"
  local rc=${PIPESTATUS[0]}
  set -e

  # Safety net: if main.py crashed before its own cleanup, delete model shards
  # ourselves so repeated variants don't fill the disk. Keep run_summary.json.
  if [[ -d "$outdir" ]]; then
    find "$outdir" -type f \
      ! -name "run_summary.json" \
      \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \
         -o -name "*.json" -o -name "*.model" -o -name "*.txt" \) \
      ! -name "run_summary.json" -delete 2>/dev/null || true
  fi

  # Pull perplexity out of the summary into the comparison table.
  if [[ -f "$outdir/run_summary.json" ]]; then
    python - "$tag" "$outdir/run_summary.json" >> "$OUT_ROOT/perplexity_table.tsv" <<'PYEOF'
import json, sys
tag, path = sys.argv[1], sys.argv[2]
try:
    s = json.load(open(path))
    q = (s.get("evaluation", {}).get("quantized_model", {})
         or s.get("results", {}).get("quantized_model", {})
         or s.get("quantized_model", {}))
    # results layout: {dataset: {"perplexity": x, ...}}
    cols = []
    for ds, m in (q.items() if isinstance(q, dict) else []):
        if isinstance(m, dict) and "perplexity" in m:
            cols.append(f"{ds}={m['perplexity']:.4f}")
    qs = s.get("quantization", {})
    extra = []
    for k in ("method", "bits", "vq_dim", "flips", "bias_before", "bias_after", "rbvt_lambda", "rbvt_topk", "rbvt_target_ratio", "rbvt_mse_guard"):
        if k in qs:
            extra.append(f"{k}={qs[k]}")
    print(tag + "\t" + "\t".join(cols) + "\t" + " ".join(extra))
except Exception as e:
    print(f"{tag}\t<parse-error: {e}>")
PYEOF
  fi

  if [[ "$rc" != "0" ]]; then
    echo "!! variant $tag exited with code $rc (model cleaned up; summary kept if produced)."
  fi
}

# 1) GPTVQ-1D base ------------------------------------------------------------
if [[ "$RUN_BASE" == "1" ]]; then
  run_variant "base" \
    --gptvq-correction none
fi

# 2) GPTVQ-1D + RBVT, post_module --------------------------------------------
if [[ "$RUN_RBVT_POST_MODULE" == "1" ]]; then
  rbvt_args=(
    --gptvq-correction rbvt
    --rbvt-lambda "$RBVT_LAMBDA"
    --rbvt-topk "$RBVT_TOPK"
    --rbvt-target-ratio "$RBVT_TARGET_RATIO"
    --gap-floor "$GAP_FLOOR"
  )
  [[ "$RBVT_MSE_GUARD" == "1" ]] && rbvt_args+=(--rbvt-mse-guard)
  run_variant "rbvt_post_module" \
    "${rbvt_args[@]}"
fi

echo ""
echo "================================================================"
echo "All requested variants done. Models DELETED after eval; kept:"
echo "  $OUT_ROOT/${SLUG}_<tag>/run_summary.json   (has perplexity)"
echo "  $OUT_ROOT/log_${SLUG}_*.txt                (full logs)"
echo ""
echo "Perplexity comparison:"
echo "----------------------------------------------------------------"
column -t -s$'\t' "$OUT_ROOT/perplexity_table.tsv" 2>/dev/null || cat "$OUT_ROOT/perplexity_table.tsv"
echo "================================================================"
