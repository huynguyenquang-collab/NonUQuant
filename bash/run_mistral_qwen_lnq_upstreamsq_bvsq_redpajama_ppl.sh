#!/usr/bin/env bash
set -Eeuo pipefail

# Two-phase RedPajama 3-bit job:
#   1. Plain LNQ for all models, initialized from upstream SqueezeLLM nuq.py LUTs.
#   2. BV-SQ Greedy lambda=1 + RBVT for all models.
# Evaluation is NonUQuantFix-style PPL only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

JOB_NAME="${JOB_NAME:-mistral-qwen-lnq-upstreamsq-bvsq-redpajama-ppl}"
BITS="${BITS:-3}"
DATASET="${DATASET:-redpajama}"
NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-4096}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# label|model_type=model_id entries.
MODEL_SPECS_TYPED="${MODEL_SPECS_TYPED:-Mistral7Bv03|mistral=mistralai/Mistral-7B-v0.3;Qwen25_7B|qwen=Qwen/Qwen2.5-7B;Qwen3_8B|qwen=Qwen/Qwen3-8B}"

COMMON_ENV=(
  "JOB_NAME=${JOB_NAME}"
  "BITS=${BITS}"
  "DATASET=${DATASET}"
  "NSAMPLES=${NSAMPLES}"
  "SEQLEN=${SEQLEN}"
  "DEVICE=${DEVICE}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "RUN_NF=0"
  "RUN_LEAN_SQUEEZE=0"
  "RUN_GPTVQ=0"
  "RUN_LMEVAL=0"
  "RUN_PPL=1"
  "USE_WANDB=${USE_WANDB:-0}"
  "FORCE_EVAL=${FORCE_EVAL:-0}"
  "OVERWRITE=${OVERWRITE:-0}"
  "DISK_CLEAN_BEFORE_RUN=0"
  "DISK_CLEAN_LUT_INTERMEDIATES=0"
  "DISK_CLEAN_EVAL_CACHE_AFTER_MODEL=0"
  "SQUEEZELLM_NUQ_SCRIPT=SqueezeLLM/quantization/nuq.py"
  "EVAL_STRIDE=${EVAL_STRIDE:-512}"
  "EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-2048}"
  "EVAL_SAMPLES=${EVAL_SAMPLES:-2000}"
  "PPL_BATCH_SIZE=${PPL_BATCH_SIZE:-1}"
  "PPL_DENSE_DTYPE=${PPL_DENSE_DTYPE:-float16}"
  "LNQ_NUM_ITERATIONS=${LNQ_NUM_ITERATIONS:-2}"
  "LNQ_CD_CYCLES=${LNQ_CD_CYCLES:-4}"
  "LNQ_CPU_COUNT=${LNQ_CPU_COUNT:-16}"
  "LNQ_ACTIVATION_STORAGE=${LNQ_ACTIVATION_STORAGE:-disk}"
  "LNQ_HESSIAN_SAVE_DTYPE=${LNQ_HESSIAN_SAVE_DTYPE:-float16}"
  "BV_SQ_VARIANTS=${BV_SQ_VARIANTS:-greedy_l1_rbvt}"
  "BV_MODEL_DTYPE=${BV_MODEL_DTYPE:-float16}"
  "BV_N_CALIB=${BV_N_CALIB:-1024}"
  "BV_BATCH_SIZE=${BV_BATCH_SIZE:-1}"
  "BV_CPU_COUNT=${BV_CPU_COUNT:-16}"
  "BV_ROW_CHUNKSIZE=${BV_ROW_CHUNKSIZE:-8}"
  "BV_H_SOURCE=${BV_H_SOURCE:-variance}"
  "BV_H_FLOOR=${BV_H_FLOOR:-1e-8}"
  "RBVT_LAMBDA=${RBVT_LAMBDA:-1.0}"
  "RBVT_BUDGET_P=${RBVT_BUDGET_P:-1.0}"
  "RBVT_TARGET_RATIO=${RBVT_TARGET_RATIO:-1.0}"
  "RBVT_MSE_GUARD=${RBVT_MSE_GUARD:-0}"
  "RBVT_ROW_CHUNK=${RBVT_ROW_CHUNK:-1024}"
)

run_one() {
  local phase="$1"
  local label="$2"
  local model_type="$3"
  local model="$4"
  shift 4

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] ${phase}: ${label} (${model}) model_type=${model_type}"
  env "${COMMON_ENV[@]}" \
    "MODEL_TYPE=${model_type}" \
    "MODEL_SPECS=${label}=${model}" \
    "$@" \
    bash bash/run_llama2_llama3_redpajama_full_matrix.sh
}

IFS=';' read -r -a MODEL_ARRAY <<< "${MODEL_SPECS_TYPED}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] Phase 1: LNQ plain for all models, upstream SqueezeLLM init"
for spec in "${MODEL_ARRAY[@]}"; do
  label_type="${spec%%=*}"
  model="${spec#*=}"
  label="${label_type%%|*}"
  model_type="${label_type#*|}"
  run_one "LNQ" "${label}" "${model_type}" "${model}" "RUN_LNQ=1" "RUN_BVSQ=0" "DISK_KEEP_DENSE_LUTS=0"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] Phase 2: BV-SQ greedy lambda=1 + RBVT for all models"
for spec in "${MODEL_ARRAY[@]}"; do
  label_type="${spec%%=*}"
  model="${spec#*=}"
  label="${label_type%%|*}"
  model_type="${label_type#*|}"
  run_one "BVSQ+RBVT" "${label}" "${model_type}" "${model}" "RUN_LNQ=0" "RUN_BVSQ=1" "DISK_KEEP_DENSE_LUTS=0"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] Done. Results under outputs/${JOB_NAME}"
