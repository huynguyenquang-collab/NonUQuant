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
RBVT_BUDGET_P="${RBVT_BUDGET_P:-${BUDGET_P:-0.005}}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-0.2}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-1}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"
STRICT_DESCENT="${STRICT_DESCENT:-1}"         # 1 -> --strict-descent, 0 -> --allow-overshoot
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"

# ---- eval toggles (set to 0 to skip the heavy lm-eval during debugging) -----
LM_EVAL="${LM_EVAL:-1}"                       # 1 -> include lm-eval, 0 -> skip
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./dataset_cache}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy arc_challenge hellaswag piqa winogrande boolq rte openbookqa lambada_openai}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-./outputs/lm_eval}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

# ---- which variants to run --------------------------------------------------
RUN_BASE="${RUN_BASE:-1}"                     # GPTVQ-1D, no correction
RUN_RBVT_POST_MODULE="${RUN_RBVT_POST_MODULE:-1}"
RUN_RBVT_POST_BLOCK="${RUN_RBVT_POST_BLOCK:-0}"
USE_SINGLE_PASS_COMPARE="${USE_SINGLE_PASS_COMPARE:-1}"  # 1 -> share one GPTVQ pass for base+RBVT

# -----------------------------------------------------------------------------
cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"
# fresh comparison table
printf "variant\tperplexity\tlm_eval\tlm_eval_avg\tquant_stats\n" > "$OUT_ROOT/perplexity_table.tsv"

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
  --eval-stride "$EVAL_STRIDE"
  --eval-max-length "$EVAL_MAX_LENGTH"
  --eval-samples "$EVAL_SAMPLES"
  --eval-cache-dir "$EVAL_CACHE_DIR"
  --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
  --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
)
[[ "$INCLUDE_M_STEP"   == "0" ]] && common_args+=(--no-include-m-step)
[[ "$HESSIAN_LOOKUPS"  == "0" ]] && common_args+=(--no-hessian-weighted-lookups)
[[ "$TRUE_SEQUENTIAL"  == "0" ]] && common_args+=(--no-true-sequential)
[[ "$KEEP_ON_DEVICE"   == "1" ]] && common_args+=(--keep-model-on-device)
[[ "$LM_EVAL"          == "0" ]] && common_args+=(--no-lm-eval)
[[ -n "$LM_EVAL_LIMIT" ]] && common_args+=(--lm-eval-limit "$LM_EVAL_LIMIT")
read -r -a lm_eval_tasks_array <<< "$LM_EVAL_TASKS"
common_args+=(--lm-eval-tasks "${lm_eval_tasks_array[@]}")
if [[ "$STRICT_DESCENT" == "1" ]]; then
  common_args+=(--strict-descent)
else
  common_args+=(--allow-overshoot)
fi

append_summary_row () {
  local tag="$1"
  local summary_path="$2"
  python - "$tag" "$summary_path" >> "$OUT_ROOT/perplexity_table.tsv" <<'PYEOF'
import json
import sys

tag, path = sys.argv[1], sys.argv[2]
preferred = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "exact_match",
    "f1,none",
    "acc",
)

def first_number(metrics):
    if not isinstance(metrics, dict):
        return None, None
    for name in preferred:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    for name, value in metrics.items():
        if name.endswith("_stderr") or name == "alias":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    return None, None

def perplexity_payload(summary):
    eval_section = summary.get("evaluation", {})
    return (
        eval_section.get("perplexity")
        or eval_section.get("quantized_model")
        or summary.get("results", {}).get("quantized_model")
        or summary.get("quantized_model")
        or {}
    )

def lm_eval_payload(summary):
    lm_eval = summary.get("evaluation", {}).get("lm_eval", {})
    if not isinstance(lm_eval, dict) or not lm_eval:
        return {}, []
    payload = next(iter(lm_eval.values()), {})
    task_summary = {}
    if isinstance(payload, dict):
        for section in (
            payload.get("summary", {}),
            payload.get("raw", {}).get("results", {}),
            payload.get("raw", {}).get("groups", {}),
        ):
            if isinstance(section, dict):
                task_summary.update(section)
    tasks = summary.get("evaluation", {}).get("lm_eval_tasks") or payload.get("tasks") or []
    return task_summary, list(tasks)

try:
    s = json.load(open(path))
    ppl_cols = []
    ppl = perplexity_payload(s)
    for ds, metrics in (ppl.items() if isinstance(ppl, dict) else []):
        value = metrics.get("perplexity") if isinstance(metrics, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ppl_cols.append(f"{ds}={value:.4f}")

    task_summary, tasks = lm_eval_payload(s)
    if not tasks:
        tasks = list(task_summary.keys())
    lm_cols = []
    lm_values = []
    for task in tasks:
        metric_name, metric_value = first_number(task_summary.get(task, {}))
        if metric_value is None:
            lm_cols.append(f"{task}=MISSING")
        else:
            lm_cols.append(f"{task}/{metric_name}={metric_value:.4f}")
            lm_values.append(metric_value)
    avg = f"{(sum(lm_values) / len(lm_values)):.4f}" if lm_values else "MISSING"

    qs = s.get("quantization", {})
    extra = []
    for key in (
        "method",
        "bits",
        "vq_dim",
        "num_linear_layers",
        "shared_gptvq_pass",
        "flips",
        "candidates",
        "bias_before",
        "bias_after",
        "rbvt_lambda",
        "rbvt_topk",
        "rbvt_budget_p",
        "rbvt_target_ratio",
        "rbvt_mse_guard",
    ):
        if key in qs:
            extra.append(f"{key}={qs[key]}")

    print(
        tag
        + "\t" + " ".join(ppl_cols)
        + "\t" + " ".join(lm_cols)
        + "\t" + avg
        + "\t" + " ".join(extra)
    )
except Exception as exc:
    print(f"{tag}\t<parse-error: {exc}>\t\t\t")
PYEOF
}

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
    append_summary_row "$tag" "$outdir/run_summary.json"
  fi

  if [[ "$rc" != "0" ]]; then
    echo "!! variant $tag exited with code $rc (model cleaned up; summary kept if produced)."
  fi
}

if [[ "$USE_SINGLE_PASS_COMPARE" == "1" && "$RUN_BASE" == "1" && "$RUN_RBVT_POST_MODULE" == "1" ]]; then
  rbvt_args=(
    --rbvt-lambda "$RBVT_LAMBDA"
    --rbvt-topk "$RBVT_TOPK"
    --rbvt-budget-p "$RBVT_BUDGET_P"
    --rbvt-target-ratio "$RBVT_TARGET_RATIO"
    --gap-floor "$GAP_FLOOR"
  )
  [[ "$RBVT_MSE_GUARD" == "1" ]] && rbvt_args+=(--rbvt-mse-guard)
  single_args=(
    --single-pass-compare
    --correction rbvt
    --model-path "$MODEL"
    --output-root "$OUT_ROOT"
    --device "$DEVICE"
    --wbits "$WBITS"
    --groupsize "$GROUPSIZE"
    --kmeans-iters "$KMEANS_ITERS"
    --kmeans-init-method "$KMEANS_INIT"
    --gptq-blocksize "$GPTQ_BLOCKSIZE"
    --percdamp "$GPTQ_PERCDAMP"
    --n-calib "$N_CALIB"
    --max-length "$MAX_LEN"
    --calib-dataset "$CALIB_DS"
    --eval-stride "$EVAL_STRIDE"
    --eval-max-length "$EVAL_MAX_LENGTH"
    --eval-samples "$EVAL_SAMPLES"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
    --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
    --lm-eval-tasks
  )
  single_args+=("${lm_eval_tasks_array[@]}")
  [[ "$INCLUDE_M_STEP"   == "0" ]] && single_args+=(--no-include-m-step)
  [[ "$HESSIAN_LOOKUPS"  == "0" ]] && single_args+=(--no-hessian-weighted-lookups)
  [[ "$TRUE_SEQUENTIAL"  == "0" ]] && single_args+=(--no-true-sequential)
  [[ "$KEEP_ON_DEVICE"   == "1" ]] && single_args+=(--keep-model-on-device)
  [[ "$LM_EVAL"          == "0" ]] && single_args+=(--no-lm-eval)
  [[ -n "$LM_EVAL_LIMIT" ]] && single_args+=(--lm-eval-limit "$LM_EVAL_LIMIT")
  if [[ "$STRICT_DESCENT" == "1" ]]; then
    single_args+=(--strict-descent)
  else
    single_args+=(--allow-overshoot)
  fi

  echo ""
  echo "================================================================"
  echo ">>> SINGLE GPTVQ PASS: base + rbvt_post_module  ->  $OUT_ROOT"
  echo "================================================================"
  set +e
  python -u gptvq_rbvt_benchmark.py \
    "${single_args[@]}" \
    "${rbvt_args[@]}" \
    2>&1 | tee "$OUT_ROOT/log_${SLUG}_single_pass_compare.txt"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ -f "$OUT_ROOT/gptvq/run_summary.json" ]]; then
    append_summary_row "base" "$OUT_ROOT/gptvq/run_summary.json"
  fi
  if [[ -f "$OUT_ROOT/gptvq_rbvt/run_summary.json" ]]; then
    append_summary_row "rbvt_post_module" "$OUT_ROOT/gptvq_rbvt/run_summary.json"
  fi
  if [[ "$rc" != "0" ]]; then
    echo "!! single-pass compare exited with code $rc (model cleaned up; summary kept if produced)."
  fi
else
  # 1) GPTVQ-1D base ----------------------------------------------------------
  if [[ "$RUN_BASE" == "1" ]]; then
    run_variant "base" \
      --gptvq-correction none
  fi

  # 2) GPTVQ-1D + RBVT, post_module ------------------------------------------
  if [[ "$RUN_RBVT_POST_MODULE" == "1" ]]; then
    rbvt_args=(
      --gptvq-correction rbvt
      --rbvt-lambda "$RBVT_LAMBDA"
      --rbvt-topk "$RBVT_TOPK"
      --rbvt-budget-p "$RBVT_BUDGET_P"
      --rbvt-target-ratio "$RBVT_TARGET_RATIO"
      --gap-floor "$GAP_FLOOR"
    )
    [[ "$RBVT_MSE_GUARD" == "1" ]] && rbvt_args+=(--rbvt-mse-guard)
    run_variant "rbvt_post_module" \
      "${rbvt_args[@]}"
  fi
fi

echo ""
echo "================================================================"
echo "All requested variants done. Models DELETED after eval; kept:"
echo "  $OUT_ROOT/<variant>/run_summary.json or $OUT_ROOT/${SLUG}_<tag>/run_summary.json"
echo "  $OUT_ROOT/log_${SLUG}_*.txt                (full logs)"
echo ""
echo "Comparison:"
echo "----------------------------------------------------------------"
column -t -s$'\t' "$OUT_ROOT/perplexity_table.tsv" 2>/dev/null || cat "$OUT_ROOT/perplexity_table.tsv"
echo ""
echo "LM-eval detail:"
echo "----------------------------------------------------------------"
python - "$OUT_ROOT" "$SLUG" <<'PYEOF'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
slug = sys.argv[2]
summary_paths = [
    ("base", out_root / "gptvq" / "run_summary.json"),
    ("rbvt_post_module", out_root / "gptvq_rbvt" / "run_summary.json"),
    ("base", out_root / f"{slug}_base" / "run_summary.json"),
    ("rbvt_post_module", out_root / f"{slug}_rbvt_post_module" / "run_summary.json"),
]
preferred = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "exact_match",
    "f1,none",
    "acc",
)

def pick_metric(metrics):
    if not isinstance(metrics, dict):
        return None, None
    for name in preferred:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    for name, value in metrics.items():
        if name.endswith("_stderr") or name == "alias":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return name, float(value)
    return None, None

def collect_lm_eval(summary):
    lm_eval = summary.get("evaluation", {}).get("lm_eval", {})
    if not isinstance(lm_eval, dict) or not lm_eval:
        return [], {}
    payload = next(iter(lm_eval.values()), {})
    if not isinstance(payload, dict):
        return [], {}
    task_summary = {}
    for section in (
        payload.get("summary", {}),
        payload.get("raw", {}).get("results", {}),
        payload.get("raw", {}).get("groups", {}),
    ):
        if isinstance(section, dict):
            task_summary.update(section)
    tasks = summary.get("evaluation", {}).get("lm_eval_tasks") or payload.get("tasks") or list(task_summary)
    return list(tasks), task_summary

seen = set()
printed = False
for tag, path in summary_paths:
    if tag in seen or not path.exists():
        continue
    seen.add(tag)
    summary = json.load(open(path))
    tasks, task_summary = collect_lm_eval(summary)
    print(f"[{tag}]")
    if not tasks:
        print("  lm_eval: MISSING")
        printed = True
        continue
    values = []
    for task in tasks:
        metric_name, metric_value = pick_metric(task_summary.get(task, {}))
        if metric_value is None:
            print(f"  {task:<18} MISSING")
        else:
            print(f"  {task:<18} {metric_name:<16} {metric_value:.4f}")
            values.append(metric_value)
    if values:
        print(f"  {'avg':<18} {'':<16} {sum(values) / len(values):.4f}")
    printed = True

if not printed:
    print("No run_summary.json found for base/RBVT.")
PYEOF
echo "================================================================"
