#!/usr/bin/env bash
# run_colab_gptvq_1d_ncc_cov_debugmatch_llama31_8b_4bit_adjusted_ppl_only.sh
#
# PPL-only comparison using the debug-matched NCC-Cov GPTVQ-1D config.
# Runs both:
#   1. gptvq      : plain GPTVQ-1D baseline
#   2. gptvq_ncc  : NCC post-module with baseline=adjusted
#
# "adjusted" baseline:  e_j = W_gptvq_j - W_adj_j   (residual w.r.t. the
#   adjusted weight, so |e| <= g/2 by construction; NCC flips minimise the
#   self-referential bias Σ(mu^T e_j)^2 on that residual).
# "original" baseline:  e_j = W_gptvq_j - W_fp_j   (true inference error;
#   end-to-end first-moment target matching the paper's stated objective).
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/gptvq_1d_ncc_cov_debugmatch_adjusted_llama31_8b_4bit}"

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
NCC_BUDGET_P="${NCC_BUDGET_P:-0.02}"
NCC_SWEEPS="${NCC_SWEEPS:-1}"
NCC_STOP_EPS="${NCC_STOP_EPS:-0.0}"
NCC_SCORE="${NCC_SCORE:-cov}"
NCC_COV_EPS="${NCC_COV_EPS:-1e-6}"
DIAGNOSTIC_LAYER_LIMIT="${DIAGNOSTIC_LAYER_LIMIT:-6}"
DIAGNOSTIC_MAX_TOKENS="${DIAGNOSTIC_MAX_TOKENS:-4096}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== GPTVQ-1D vs GPTVQ-1D+NCC-Cov adjusted-baseline PPL only | Llama-3.1-8B | 4-bit ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_ROOT"
echo "Calibration: $CALIB_DATASET n=$N_CALIB len=$MAX_LENGTH"
echo "GPTQ blocksize: $GPTQ_BLOCKSIZE | groupsize=$GROUPSIZE"
echo "GPTVQ EM/k-means iterations: $KMEANS_ITERS"
echo "NCC placement: post_module | score=$NCC_SCORE | budget_p=$NCC_BUDGET_P | sweeps=$NCC_SWEEPS"
echo "NCC baseline: adjusted"
echo "Variants: gptvq gptvq_ncc"
echo "PPL only: WikiText-2/C4 eval_samples=$EVAL_SAMPLES len=$EVAL_MAX_LENGTH stride=$EVAL_STRIDE"
echo "LM-eval: disabled"

if [ "$RUN_SETUP" = "1" ]; then
  "$PYTHON_BIN" -m pip install -q -r requirements.txt
fi

if [[ ! -d GPTVQ/.git ]]; then
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
else
  git -C GPTVQ pull --ff-only
fi

if [[ ! -d NCCQuant/.git ]]; then
  git clone https://github.com/anhnda/NCCQuant.git NCCQuant
else
  git -C NCCQuant pull --ff-only
fi

"$PYTHON_BIN" - <<'PY'
import sys
import transformers

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D

    transformers.Conv1D = Conv1D

sys.path.insert(0, "GPTVQ")
from gptq import GPTQ  # noqa: F401
from modelutils import find_layers  # noqa: F401
from vq_quant import VQQuantizer  # noqa: F401

print("GPTVQ import smoke check passed.")
print("NCCQuant source present: NCCQuant/quantizers/ncc.py")
PY

COMMON_ARGS=(
  --model-path "$MODEL"
  --device "$DEVICE"
  --output-root "$OUTPUT_ROOT"
  --variants gptvq gptvq_ncc
  --correction ncc
  --ncc-placement post_module
  --keep-model-on-device
  --wbits 4
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
  --ncc-budget-p "$NCC_BUDGET_P"
  --ncc-sweeps "$NCC_SWEEPS"
  --ncc-stop-eps "$NCC_STOP_EPS"
  --ncc-score "$NCC_SCORE"
  --ncc-cov-eps "$NCC_COV_EPS"
  --baseline adjusted
  --diagnostic-layer-limit "$DIAGNOSTIC_LAYER_LIMIT"
  --diagnostic-max-tokens "$DIAGNOSTIC_MAX_TOKENS"
  --cleanup-model-artifacts
)

"$PYTHON_BIN" gptvq_rbvt_benchmark.py "${COMMON_ARGS[@]}"
