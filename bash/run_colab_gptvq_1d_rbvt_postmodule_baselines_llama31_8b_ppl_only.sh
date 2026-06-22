#!/usr/bin/env bash
# run_colab_gptvq_1d_rbvt_postmodule_baselines_llama31_8b_ppl_only.sh
#
# RBVT post-module PPL-only comparison using the same debug-matched GPTVQ-1D
# config as:
#   bash/run_colab_gptvq_1d_ncc_cov_debugmatch_llama31_8b_4bit_adjusted_ppl_only.sh
#
# Runs two RBVT baseline scenarios by default:
#   1. adjusted : residual is measured against GPTVQ's error-feedback-adjusted
#                 assignment input W_assigned.
#   2. original : residual is measured against the original FP weight W_fp.
#
# Each scenario uses single-pass compare:
#   one GPTVQ pass -> save/eval gptvq and gptvq_rbvt, PPL only.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

RUN_SETUP="${RUN_SETUP:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
WBITS="${WBITS:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/gptvq_1d_rbvt_postmodule_debugmatch_llama31_8b_${WBITS}bit_ppl_only}"
BASELINES="${BASELINES:-adjusted original}"

N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
GROUPSIZE="${GROUPSIZE:-128}"
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-20}"
ASSIGNMENT_CHUNK_SIZE="${ASSIGNMENT_CHUNK_SIZE:-4096}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
DIAGNOSTIC_LAYER_LIMIT="${DIAGNOSTIC_LAYER_LIMIT:-0}"
DIAGNOSTIC_MAX_TOKENS="${DIAGNOSTIC_MAX_TOKENS:-4096}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a BASELINE_ARRAY <<< "$BASELINES"
if [ "${#BASELINE_ARRAY[@]}" -eq 0 ]; then
  echo "BASELINES is empty. Use BASELINES='adjusted original' or one of those values." >&2
  exit 1
fi

echo "=== GPTVQ-1D vs GPTVQ-1D+RBVT post-module baselines PPL only | Llama-3.1-8B | ${WBITS}-bit ==="
echo "Model: $MODEL"
echo "Output root: $OUTPUT_ROOT"
echo "Baseline scenarios: ${BASELINE_ARRAY[*]}"
echo "Calibration: $CALIB_DATASET n=$N_CALIB len=$MAX_LENGTH"
echo "Bits: $WBITS"
echo "GPTQ blocksize: $GPTQ_BLOCKSIZE | groupsize=$GROUPSIZE"
echo "GPTVQ EM/k-means iterations: $KMEANS_ITERS"
echo "RBVT placement: post_module | lambda=$RBVT_LAMBDA | topk=$RBVT_TOPK"
echo "Compare mode: per-baseline single GPTVQ pass -> save/eval gptvq and gptvq_rbvt"
echo "Quantization/eval layers: full model"
echo "PPL only: WikiText-2/C4 eval_samples=$EVAL_SAMPLES len=$EVAL_MAX_LENGTH stride=$EVAL_STRIDE"
echo "LM-eval: disabled"
echo "Note: WBITS default follows the referenced script's actual value. Set WBITS=4 for a literal 4-bit run."

if [ "$RUN_SETUP" = "1" ]; then
  "$PYTHON_BIN" -m pip install -q -r requirements.txt
fi

if [[ ! -d GPTVQ ]]; then
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
elif [[ -d GPTVQ/.git ]]; then
  git -C GPTVQ pull --ff-only
else
  echo "Using vendored GPTVQ directory."
fi

"$PYTHON_BIN" - <<'PY'
import inspect
import sys
import transformers

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D

    transformers.Conv1D = Conv1D

sys.path.insert(0, "GPTVQ")
from gptq import GPTQ  # noqa: F401
from modelutils import find_layers  # noqa: F401
from vq_quant import VQQuantizer  # noqa: F401

if "capture_w_assigned" not in inspect.signature(GPTQ.fasterquant).parameters:
    raise SystemExit(
        "Patched GPTVQ/gptq.py is required for baseline=adjusted "
        "(missing fasterquant(..., capture_w_assigned=...))."
    )

print("GPTVQ import smoke check passed.")
print("GPTVQ adjusted-baseline capture support present.")
PY

COMMON_ARGS=(
  --model-path "$MODEL"
  --device "$DEVICE"
  --variants gptvq gptvq_rbvt
  --single-pass-compare
  --correction rbvt
  --keep-model-on-device
  --wbits "$WBITS"
  --groupsize "$GROUPSIZE"
  --gptq-blocksize "$GPTQ_BLOCKSIZE"
  --percdamp 0.01
  --kmeans-iters "$KMEANS_ITERS"
  --kmeans-init-method mahalanobis
  --assignment-chunk-size "$ASSIGNMENT_CHUNK_SIZE"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LENGTH"
  --calib-dataset "$CALIB_DATASET"
  --eval-samples "$EVAL_SAMPLES"
  --eval-max-length "$EVAL_MAX_LENGTH"
  --eval-stride "$EVAL_STRIDE"
  --no-lm-eval
  --rbvt-lambda "$RBVT_LAMBDA"
  --rbvt-topk "$RBVT_TOPK"
  --row-chunk "$ROW_CHUNK"
  --diagnostic-layer-limit "$DIAGNOSTIC_LAYER_LIMIT"
  --diagnostic-max-tokens "$DIAGNOSTIC_MAX_TOKENS"
  --cleanup-model-artifacts
)

for BASELINE in "${BASELINE_ARRAY[@]}"; do
  case "$BASELINE" in
    adjusted|original) ;;
    *)
      echo "Unsupported baseline: $BASELINE (expected adjusted or original)" >&2
      exit 1
      ;;
  esac

  SCENARIO_ROOT="$OUTPUT_ROOT/baseline_$BASELINE"
  echo
  echo "=== Running RBVT baseline=$BASELINE | output=$SCENARIO_ROOT ==="
  "$PYTHON_BIN" gptvq_rbvt_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --output-root "$SCENARIO_ROOT" \
    --baseline "$BASELINE"
done

export RBVT_COMPARE_OUTPUT_ROOT="$OUTPUT_ROOT"
export RBVT_COMPARE_BASELINES="${BASELINE_ARRAY[*]}"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RBVT_COMPARE_OUTPUT_ROOT"])
baselines = os.environ["RBVT_COMPARE_BASELINES"].split()
datasets = ("WikiText-2", "C4")


def load_summary(baseline: str, variant: str):
    path = root / f"baseline_{baseline}" / variant / "run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_ppl(summary, dataset: str):
    if not summary:
        return None
    value = (
        summary.get("evaluation", {})
        .get("perplexity", {})
        .get(dataset, {})
        .get("perplexity")
    )
    return float(value) if isinstance(value, (int, float)) else None


def fmt(value):
    return "MISSING" if value is None else f"{value:.4f}"


def fmt_delta(value, base):
    if value is None or base in (None, 0.0):
        return "MISSING"
    return f"{((value - base) / base) * 100.0:+.3f}%"


print()
print("=" * 80)
print("RBVT BASELINE PPL SUMMARY")
print("=" * 80)
print(f"{'baseline':<10} {'variant':<12} {'WikiText-2':>12} {'delta':>10} {'C4':>12} {'delta':>10}")
for baseline in baselines:
    gptvq = load_summary(baseline, "gptvq")
    gptvq_ppl = {dataset: get_ppl(gptvq, dataset) for dataset in datasets}
    for variant in ("gptvq", "gptvq_rbvt"):
        summary = load_summary(baseline, variant)
        values = {dataset: get_ppl(summary, dataset) for dataset in datasets}
        print(
            f"{baseline:<10} {variant:<12} "
            f"{fmt(values['WikiText-2']):>12} {fmt_delta(values['WikiText-2'], gptvq_ppl['WikiText-2']):>10} "
            f"{fmt(values['C4']):>12} {fmt_delta(values['C4'], gptvq_ppl['C4']):>10}"
        )
PY
