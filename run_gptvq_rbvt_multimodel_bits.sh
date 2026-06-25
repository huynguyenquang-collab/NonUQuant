#!/usr/bin/env bash
set -euo pipefail

# Run GPTVQ-1D base + GPTVQ-1D/RBVT for 3-bit and 4-bit across several models.
# The per-run quantization/eval settings intentionally flow through
# run_gptvq_rbvt.sh so this sweep stays aligned with the single-model runner.

REPO_DIR="${REPO_DIR:-$(pwd)}"
cd "$REPO_DIR"

if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

MODEL_SPECS="${MODEL_SPECS:-Llama31=meta-llama/Llama-3.1-8B;Mistral7Bv03=mistralai/Mistral-7B-v0.3;Qwen25_7B=Qwen/Qwen2.5-7B}"
DENSE_DEVICE="${DENSE_DEVICE:-cuda:0}"
BITS="${BITS:-3 4}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-./runs_gptvq_rbvt_multimodel_bits}"
LOG_DIR="${LOG_DIR:-$SWEEP_OUTPUT_ROOT/logs}"

# Same GPTVQ/RBVT defaults as run_gptvq_rbvt.sh unless overridden.
N_CALIB="${N_CALIB:-128}"
GROUPSIZE="${GROUPSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-100}"
KMEANS_INIT="${KMEANS_INIT:-mahalanobis}"
INCLUDE_M_STEP="${INCLUDE_M_STEP:-1}"
HESSIAN_LOOKUPS="${HESSIAN_LOOKUPS:-1}"
TRUE_SEQUENTIAL="${TRUE_SEQUENTIAL:-1}"
KEEP_ON_DEVICE="${KEEP_ON_DEVICE:-0}"
MAX_LEN="${MAX_LEN:-2048}"
CALIB_DS="${CALIB_DS:-c4}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
RBVT_BUDGET_P="${RBVT_BUDGET_P:-${BUDGET_P:-0.005}}"
RBVT_TARGET_RATIO="${RBVT_TARGET_RATIO:-0.1}"
RBVT_MSE_GUARD="${RBVT_MSE_GUARD:-1}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"
STRICT_DESCENT="${STRICT_DESCENT:-1}"
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"

LM_EVAL="${LM_EVAL:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./dataset_cache}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy arc_challenge hellaswag piqa winogrande boolq rte openbookqa lambada_openai mmlu gsm8k}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-0}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_BASE="${RUN_BASE:-1}"
RUN_RBVT_POST_MODULE="${RUN_RBVT_POST_MODULE:-1}"
RUN_RBVT_POST_BLOCK="${RUN_RBVT_POST_BLOCK:-0}"
USE_SINGLE_PASS_COMPARE="${USE_SINGLE_PASS_COMPARE:-1}"

mkdir -p "$SWEEP_OUTPUT_ROOT/runs" "$LOG_DIR"

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="$LOG_DIR/gptvq_rbvt_multimodel_bits_${timestamp}.log"

IFS=';' read -r -a MODEL_ARRAY <<< "$MODEL_SPECS"
read -r -a BITS_ARRAY <<< "$BITS"

{
  echo "=== GPTVQ/RBVT multi-model multi-bit sweep ==="
  echo "Model specs: $MODEL_SPECS"
  echo "Device: $DENSE_DEVICE"
  echo "Bits: $BITS"
  echo "LM-eval tasks: $LM_EVAL_TASKS"
  echo "LM-eval fewshot: $LM_EVAL_NUM_FEWSHOT"
  echo "W&B logging: $USE_WANDB | project=$WANDB_PROJECT | entity=${WANDB_ENTITY:-default}"
  echo "Output: $SWEEP_OUTPUT_ROOT"
  echo "RBVT: lambda=$RBVT_LAMBDA topk=$RBVT_TOPK budget_p=$RBVT_BUDGET_P target_ratio=$RBVT_TARGET_RATIO mse_guard=$RBVT_MSE_GUARD"
} | tee -a "$log_file"

for spec in "${MODEL_ARRAY[@]}"; do
  if [[ "$spec" != *=* ]]; then
    echo "Error: MODEL_SPECS entries must be label=checkpoint; got $spec" >&2
    exit 1
  fi
  label="${spec%%=*}"
  model="${spec#*=}"

  for bits in "${BITS_ARRAY[@]}"; do
    run_output="$SWEEP_OUTPUT_ROOT/runs/${label}/${bits}bit"
    context="Model:${label} bit:${bits}"
    {
      echo
      echo "================================================================"
      echo ">>> ${context}"
      echo ">>> checkpoint: $model"
      echo ">>> output: $run_output"
      echo "================================================================"
      MODEL="$model" \
      DEVICE="$DENSE_DEVICE" \
      WBITS="$bits" \
      GROUPSIZE="$GROUPSIZE" \
      KMEANS_ITERS="$KMEANS_ITERS" \
      KMEANS_INIT="$KMEANS_INIT" \
      INCLUDE_M_STEP="$INCLUDE_M_STEP" \
      HESSIAN_LOOKUPS="$HESSIAN_LOOKUPS" \
      TRUE_SEQUENTIAL="$TRUE_SEQUENTIAL" \
      KEEP_ON_DEVICE="$KEEP_ON_DEVICE" \
      N_CALIB="$N_CALIB" \
      MAX_LEN="$MAX_LEN" \
      CALIB_DS="$CALIB_DS" \
      RBVT_LAMBDA="$RBVT_LAMBDA" \
      RBVT_TOPK="$RBVT_TOPK" \
      RBVT_BUDGET_P="$RBVT_BUDGET_P" \
      RBVT_TARGET_RATIO="$RBVT_TARGET_RATIO" \
      RBVT_MSE_GUARD="$RBVT_MSE_GUARD" \
      GAP_FLOOR="$GAP_FLOOR" \
      STRICT_DESCENT="$STRICT_DESCENT" \
      GPTQ_BLOCKSIZE="$GPTQ_BLOCKSIZE" \
      GPTQ_PERCDAMP="$GPTQ_PERCDAMP" \
      LM_EVAL="$LM_EVAL" \
      EVAL_SAMPLES="$EVAL_SAMPLES" \
      EVAL_STRIDE="$EVAL_STRIDE" \
      EVAL_MAX_LENGTH="$EVAL_MAX_LENGTH" \
      EVAL_CACHE_DIR="$EVAL_CACHE_DIR" \
      LM_EVAL_TASKS="$LM_EVAL_TASKS" \
      LM_EVAL_BATCH_SIZE="$LM_EVAL_BATCH_SIZE" \
      LM_EVAL_NUM_FEWSHOT="$LM_EVAL_NUM_FEWSHOT" \
      LM_EVAL_LIMIT="$LM_EVAL_LIMIT" \
      LM_EVAL_OUTPUT_DIR="$run_output/lm_eval" \
      RUN_CONTEXT="$context" \
      USE_WANDB="$USE_WANDB" \
      WANDB_PROJECT="$WANDB_PROJECT" \
      WANDB_ENTITY="$WANDB_ENTITY" \
      RUN_BASE="$RUN_BASE" \
      RUN_RBVT_POST_MODULE="$RUN_RBVT_POST_MODULE" \
      RUN_RBVT_POST_BLOCK="$RUN_RBVT_POST_BLOCK" \
      USE_SINGLE_PASS_COMPARE="$USE_SINGLE_PASS_COMPARE" \
      OUT_ROOT="$run_output" \
      bash run_gptvq_rbvt.sh
    } 2>&1 | tee -a "$log_file"
  done
done

python - "$SWEEP_OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted((root / "runs").glob("*/*bit/*/run_summary.json")):
    bits_dir = summary_path.parents[1]
    model_key = bits_dir.parent.name
    bits = bits_dir.name.replace("bit", "")
    variant = summary_path.parent.name
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    quant = summary.get("quantization", {})
    eval_section = summary.get("evaluation", {})
    row = {
        "model-key": model_key,
        "bits": bits,
        "variant": variant,
        "checkpoint": summary.get("model_path", ""),
        "ppl-wiki": "",
        "ppl-c4": "",
        "lm_eval_avg": "",
        "flips": quant.get("flips", ""),
        "bias_before": quant.get("bias_before", ""),
        "bias_after": quant.get("bias_after", ""),
    }
    ppl = eval_section.get("perplexity", {})
    if isinstance(ppl.get("WikiText-2"), dict):
        row["ppl-wiki"] = ppl["WikiText-2"].get("perplexity", "")
    if isinstance(ppl.get("C4"), dict):
        row["ppl-c4"] = ppl["C4"].get("perplexity", "")

    payload = next(iter(eval_section.get("lm_eval", {}).values()), {})
    task_summary = {}
    if isinstance(payload, dict):
        for section in (
            payload.get("summary", {}),
            payload.get("raw", {}).get("results", {}),
            payload.get("raw", {}).get("groups", {}),
        ):
            if isinstance(section, dict):
                task_summary.update(section)
    tasks = eval_section.get("lm_eval_tasks", []) or list(task_summary)
    vals = []
    for task in tasks:
        metrics = task_summary.get(task, {})
        if not isinstance(metrics, dict):
            continue
        keys = (
            ("exact_match,strict-match",) if task == "gsm8k" else
            ("acc_norm,none", "acc,none", "exact_match,none", "exact_match", "f1,none", "acc")
        )
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"lm_eval/{task}"] = value
                vals.append(float(value))
                break
        if task == "gsm8k":
            flex = metrics.get("exact_match,flexible-extract")
            if isinstance(flex, (int, float)) and not isinstance(flex, bool):
                row["lm_eval/gsm8k_flexible"] = flex
    if vals:
        row["lm_eval_avg"] = sum(vals) / len(vals)
    rows.append(row)

if not rows:
    raise SystemExit(f"No run_summary.json files found under {root / 'runs'}")

columns = []
for row in rows:
    for key in row:
        if key not in columns:
            columns.append(key)

(root / "benchmark_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
(root / "benchmark_results.tsv").write_text(
    "\t".join(columns) + "\n" +
    "\n".join("\t".join(str(row.get(col, "")) for col in columns) for row in rows) + "\n",
    encoding="utf-8",
)

print("\n=== GPTVQ/RBVT sweep summary ===")
print("\t".join(columns))
for row in rows:
    print("\t".join(str(row.get(col, "")) for col in columns))
PY

echo ""
echo "================================================================"
echo "Sweep done:"
echo "  log: $log_file"
echo "  summary: $SWEEP_OUTPUT_ROOT/benchmark_results.tsv"
echo "  json: $SWEEP_OUTPUT_ROOT/benchmark_results.json"
echo "================================================================"
