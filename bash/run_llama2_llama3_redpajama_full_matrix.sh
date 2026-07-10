#!/usr/bin/env bash
set -Eeuo pipefail

# Full Llama2/Llama3 RedPajama matrix:
# - models: Llama-2-7B base and Meta-Llama-3-8B base by default
# - bits: 3 and 4
# - methods: NF, LeanQuant, SqueezeLLM, LNQ, BV-SQ, GPTVQ
# - 4-bit also runs NVFP4
# - lm-eval runs MMLU and GSM8K by default

JOB_NAME="${JOB_NAME:-llama2-llama3-redpajama-full-matrix}"
trap 'echo "[${JOB_NAME}] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/squeezellm${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LM_EVAL_GEN_KWARGS="${LM_EVAL_GEN_KWARGS:-max_gen_toks=64,do_sample=False,temperature=0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_SPECS="${MODEL_SPECS:-llama2_7b=meta-llama/Llama-2-7b-hf;llama3_8b=meta-llama/Meta-Llama-3-8B}"
BITS="${BITS:-3 4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${JOB_NAME}}"
CACHE_ROOT="${CACHE_ROOT:-cache/${JOB_NAME}}"

NSAMPLES="${NSAMPLES:-1024}"
SEQLEN="${SEQLEN:-4096}"
SEED="${SEED:-0}"
DATASET="${DATASET:-redpajama}"
MODEL_TYPE="${MODEL_TYPE:-llama}"

RUN_NF="${RUN_NF:-1}"
RUN_LEAN_SQUEEZE="${RUN_LEAN_SQUEEZE:-1}"
RUN_LNQ="${RUN_LNQ:-1}"
RUN_BVSQ="${RUN_BVSQ:-1}"
RUN_GPTVQ="${RUN_GPTVQ:-1}"
RUN_LMEVAL="${RUN_LMEVAL:-1}"
RUN_PPL="${RUN_PPL:-1}"

LM_EVAL_TASKS="${LM_EVAL_TASKS:-mmlu gsm8k}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-0}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"
PPL_DENSE_DTYPE="${PPL_DENSE_DTYPE:-float16}"

NF_METHOD="${NF_METHOD:-rtn}"
LEAN_SQUEEZE_METHODS="${LEAN_SQUEEZE_METHODS:-rtn}"
SQUEEZELLM_MODE="${SQUEEZELLM_MODE:-dense-only}"
LEANQUANT_EXPONENT="${LEANQUANT_EXPONENT:-4.0}"
LEANQUANT_PERCDAMP="${LEANQUANT_PERCDAMP:-0.1}"

LNQ_NUM_ITERATIONS="${LNQ_NUM_ITERATIONS:-2}"
LNQ_CD_CYCLES="${LNQ_CD_CYCLES:-4}"
LNQ_CPU_COUNT="${LNQ_CPU_COUNT:-16}"
LNQ_DEVICES="${LNQ_DEVICES:-}"
LNQ_ACTIVATION_STORAGE="${LNQ_ACTIVATION_STORAGE:-disk}"
LNQ_HESSIAN_SAVE_DTYPE="${LNQ_HESSIAN_SAVE_DTYPE:-float16}"

BV_SQ_VARIANTS="${BV_SQ_VARIANTS:-greedy_l1 hier_l1}"
BV_MODEL_DTYPE="${BV_MODEL_DTYPE:-float16}"
BV_N_CALIB="${BV_N_CALIB:-1024}"
BV_BATCH_SIZE="${BV_BATCH_SIZE:-1}"
BV_CPU_COUNT="${BV_CPU_COUNT:-16}"
BV_ROW_CHUNKSIZE="${BV_ROW_CHUNKSIZE:-8}"
BV_H_SOURCE="${BV_H_SOURCE:-variance}"
BV_H_FLOOR="${BV_H_FLOOR:-1e-8}"

GPTVQ_GROUPSIZE="${GPTVQ_GROUPSIZE:-128}"
GPTVQ_KMEANS_ITERS="${GPTVQ_KMEANS_ITERS:-100}"
GPTVQ_CALIB_N="${GPTVQ_CALIB_N:-${NSAMPLES}}"
GPTVQ_CALIB_LEN="${GPTVQ_CALIB_LEN:-${SEQLEN}}"
GPTVQ_EVAL_SAMPLES="${GPTVQ_EVAL_SAMPLES:-${EVAL_SAMPLES}}"

OVERWRITE="${OVERWRITE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}" cache/tokens

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${JOB_NAME}] $*"
}

model_basename() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
print(sys.argv[1].rstrip("/").split("/")[-1])
PY
}

token_url_for_basename() {
  case "$1" in
    Llama-2-7b-hf) echo "https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Llama-2-7b-hf-redpajama_s1024_blk4096.pt" ;;
    Meta-Llama-3-8B) echo "https://github.com/snu-mllab/GuidedQuant/releases/download/v1.0.0/Meta-Llama-3-8B-redpajama_s1024_blk4096.pt" ;;
    *) echo "" ;;
  esac
}

ensure_redpajama_cache() {
  local basename="$1"
  local token_path="cache/tokens/${basename}-redpajama_s${NSAMPLES}_blk${SEQLEN}.pt"
  if [[ -s "${token_path}" ]]; then
    echo "${token_path}"
    return
  fi
  local url
  url="$(token_url_for_basename "${basename}")"
  if [[ -n "${url}" ]]; then
    log "Downloading GuidedQuant RedPajama token cache for ${basename}"
    wget -O "${token_path}" "${url}"
    echo "${token_path}"
    return
  fi
  log "No release token URL for ${basename}; RedPajama mirror will be sampled by calibration_utils"
  echo ""
}

read_array() {
  local value="$1"
  read -r -a _result <<< "${value}"
  printf '%s\n' "${_result[@]}"
}

lm_eval_args() {
  if [[ "${RUN_LMEVAL}" == "1" ]]; then
    echo "--include-lm-eval"
    echo "--lm-eval-tasks"
    read_array "${LM_EVAL_TASKS}"
    echo "--lm-eval-batch-size"
    echo "${LM_EVAL_BATCH_SIZE}"
    echo "--lm-eval-num-fewshot"
    echo "${LM_EVAL_NUM_FEWSHOT}"
    if [[ -n "${LM_EVAL_LIMIT}" ]]; then
      echo "--lm-eval-limit"
      echo "${LM_EVAL_LIMIT}"
    fi
  else
    echo "--no-lm-eval"
  fi
}

run_nf_like() {
  local label="$1" model="$2" bits="$3" quantizer="$4" token_path="$5"
  local run_dir="${OUTPUT_ROOT}/${label}/${bits}bit/${quantizer}_${NF_METHOD}"
  if [[ -s "${run_dir}/run_summary.json" && "${OVERWRITE}" != "1" && "${FORCE_EVAL}" != "1" ]]; then
    log "Skipping existing NF/NVFP run: ${run_dir}"
    return
  fi
  log "Running ${quantizer} ${bits}-bit ${NF_METHOD}: ${label}"
  local -a args
  mapfile -t args < <(lm_eval_args)
  CALIB_TOKENS_PATH="${token_path}" "${PYTHON_BIN}" main.py \
    --model-path "${model}" \
    --device "${DEVICE}" \
    --method "${NF_METHOD}" \
    --quantizer "${quantizer}" \
    --output-dir "${run_dir}" \
    --calib-dataset "${DATASET}" \
    --n-calib "${NSAMPLES}" \
    --max-length "${SEQLEN}" \
    --seed "${SEED}" \
    --eval-stride "${EVAL_STRIDE}" \
    --eval-max-length "${EVAL_MAX_LENGTH}" \
    --eval-samples "${EVAL_SAMPLES}" \
    --eval-cache-dir "${CACHE_ROOT}/eval_cache" \
    --lm-eval-output-dir "${run_dir}/lm_eval" \
    --no-wandb \
    "${args[@]}"
}

run_lean_squeeze() {
  local label="$1" model="$2" bits="$3" token_path="$4"
  local run_dir="${OUTPUT_ROOT}/${label}/${bits}bit/lean_squeeze"
  if [[ -s "${run_dir}/benchmark_results.json" && "${OVERWRITE}" != "1" && "${FORCE_EVAL}" != "1" ]]; then
    log "Skipping existing Lean/Squeeze benchmark: ${run_dir}"
    return
  fi
  log "Running LeanQuant + SqueezeLLM ${bits}-bit: ${label}"
  local -a args
  mapfile -t args < <(lm_eval_args)
  CALIB_TOKENS_PATH="${token_path}" "${PYTHON_BIN}" codebook_benchmark.py \
    --model-path "${model}" \
    --device "${DEVICE}" \
    --output-root "${run_dir}" \
    --codebooks leanquant squeezellm \
    --bits "${bits}" \
    --methods ${LEAN_SQUEEZE_METHODS} \
    --resume \
    --calib-dataset "${DATASET}" \
    --n-calib "${NSAMPLES}" \
    --max-length "${SEQLEN}" \
    --squeezellm-mode "${SQUEEZELLM_MODE}" \
    --leanquant-exponent "${LEANQUANT_EXPONENT}" \
    --leanquant-percdamp "${LEANQUANT_PERCDAMP}" \
    --eval-stride "${EVAL_STRIDE}" \
    --eval-max-length "${EVAL_MAX_LENGTH}" \
    --eval-samples "${EVAL_SAMPLES}" \
    --eval-cache-dir "${CACHE_ROOT}/eval_cache" \
    --lm-eval-output-dir "${run_dir}/lm_eval" \
    --no-wandb \
    "${args[@]}"
}

variant_folder() {
  local root="$1" bits="$2" variant="$3"
  case "${variant}" in
    greedy_l1) echo "${root}/bv_sq_greedy_w${bits}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda1.0" ;;
    hier_l1) echo "${root}/bv_sq_hier_w${bits}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda1.0" ;;
    greedy_l0) echo "${root}/bv_sq_greedy_w${bits}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_lambda0.0" ;;
    *) echo "Unknown BV_SQ variant: ${variant}" >&2; exit 2 ;;
  esac
}

variant_solver() {
  case "$1" in
    greedy_l1|greedy_l0) echo "greedy" ;;
    hier_l1) echo "hier" ;;
    *) echo "Unknown BV_SQ variant: $1" >&2; exit 2 ;;
  esac
}

variant_lambda() {
  case "$1" in
    greedy_l0) echo "0.0" ;;
    greedy_l1|hier_l1) echo "1.0" ;;
    *) echo "Unknown BV_SQ variant: $1" >&2; exit 2 ;;
  esac
}

ensure_sqllm_lut_stack() {
  local label="$1" model="$2" bits="$3" token_path="$4" root="$5"
  local chunks="${root}/chunks"
  local fisher="${root}/fisher_${DATASET}_s${NSAMPLES}_blk${SEQLEN}"
  local sq="${root}/squeezellm_w${bits}"
  if [[ "${OVERWRITE}" == "1" || ! -d "${chunks}" || "$(find "${chunks}" -maxdepth 1 -name 'layer_*.pt' 2>/dev/null | wc -l | tr -d ' ')" == "0" ]]; then
    log "Chunking model for LUT stack: ${label}"
    local -a chunk_args=()
    [[ "${OVERWRITE}" == "1" ]] && chunk_args+=(--overwrite)
    "${PYTHON_BIN}" quantization/chunk_models.py --model "${model}" --model_type "${MODEL_TYPE}" --output_path "${chunks}" "${chunk_args[@]}"
  fi
  if [[ "${RUN_LNQ}" == "1" || "${RUN_BVSQ}" == "1" ]]; then
    if [[ "${OVERWRITE}" == "1" || ! -d "${fisher}" || "$(find "${fisher}" -maxdepth 1 -name 'layer_*.pt' 2>/dev/null | wc -l | tr -d ' ')" == "0" ]]; then
      log "Collecting Fisher chunks for Squeeze/LNQ init: ${label} ${bits}-bit"
      CALIB_TOKENS_PATH="${token_path}" "${PYTHON_BIN}" quantization/fisher.py \
        --model "${model}" \
        --output_path "${fisher}" \
        --dataset "${DATASET}" \
        --nsamples "${NSAMPLES}" \
        --seqlen "${SEQLEN}" \
        --seed "${SEED}" \
        --cache_dir "${CACHE_ROOT}/tokens" \
        --device "${DEVICE}" \
        --batch_size 1 \
        --model_dtype default \
        --attn_implementation "${ATTN_IMPLEMENTATION:-auto}"
    fi
    if [[ "${OVERWRITE}" == "1" || ! -d "${sq}/lut" ]]; then
      log "Building SqueezeLLM LUT init for LNQ/BVSQ stack: ${label} ${bits}-bit"
      "${PYTHON_BIN}" quantization/nuq.py \
        --model_type "${MODEL_TYPE}" \
        --model "${chunks}" \
        --gradient "${fisher}" \
        --bit "${bits}" \
        --output_folder "${sq}"
    fi
  fi
}

run_lnq() {
  local label="$1" model="$2" bits="$3" token_path="$4" root="$5"
  local chunks="${root}/chunks"
  local sq="${root}/squeezellm_w${bits}"
  local hess="${root}/lnq_hessians_${DATASET}_s${NSAMPLES}_blk${SEQLEN}"
  local out="${root}/lnq_plain_w${bits}_${DATASET}_s${NSAMPLES}_blk${SEQLEN}_iter${LNQ_NUM_ITERATIONS}_cd${LNQ_CD_CYCLES}"
  if [[ "${OVERWRITE}" == "1" || ! -d "${hess}" ]]; then
    log "Collecting LNQ Hessians: ${label} ${bits}-bit"
    local -a device_args=()
    [[ -n "${LNQ_DEVICES}" ]] && device_args+=(--devices "${LNQ_DEVICES}")
    CALIB_TOKENS_PATH="${token_path}" "${PYTHON_BIN}" quantization/lnq.py hessians \
      --model "${model}" \
      --dataset "${DATASET}" \
      --nsamples "${NSAMPLES}" \
      --seqlen "${SEQLEN}" \
      --seed "${SEED}" \
      --cache_dir "${CACHE_ROOT}/tokens" \
      --output_folder "${hess}" \
      --device "${DEVICE}" \
      --calib_batch_size 1 \
      --activation_storage "${LNQ_ACTIVATION_STORAGE}" \
      --hessian_save_dtype "${LNQ_HESSIAN_SAVE_DTYPE}" \
      --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
      "${device_args[@]}"
  fi
  if [[ "${OVERWRITE}" == "1" || ! -d "${out}/lut" ]]; then
    log "Running LNQ plain: ${label} ${bits}-bit"
    local -a overwrite_args=()
    [[ "${OVERWRITE}" == "1" ]] && overwrite_args+=(--overwrite)
    "${PYTHON_BIN}" quantization/lnq.py quantize \
      --model_chunks "${chunks}" \
      --hessians "${hess}" \
      --initial_lut "${sq}" \
      --output_folder "${out}" \
      --model_type "${MODEL_TYPE}" \
      --bit "${bits}" \
      --num_iterations "${LNQ_NUM_ITERATIONS}" \
      --cd_cycles "${LNQ_CD_CYCLES}" \
      --cpu_count "${LNQ_CPU_COUNT}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      "${overwrite_args[@]}"
  fi
  eval_lut_method "${label}" "${model}" "${bits}" "${out}" "lnq_plain" "${root}"
}

run_bvsq() {
  local label="$1" model="$2" bits="$3" token_path="$4" root="$5"
  local chunks="${root}/chunks"
  local stats="${root}/bv_stats_${DATASET}_s${NSAMPLES}_blk${SEQLEN}.pt"
  for variant in ${BV_SQ_VARIANTS}; do
    local out solver lambda
    out="$(variant_folder "${root}" "${bits}" "${variant}")"
    solver="$(variant_solver "${variant}")"
    lambda="$(variant_lambda "${variant}")"
    if [[ "${OVERWRITE}" == "1" || ! -d "${out}/lut" ]]; then
      log "Running BVSQ ${variant}: ${label} ${bits}-bit"
      local -a overwrite_args=()
      [[ "${OVERWRITE}" == "1" ]] && overwrite_args+=(--overwrite --overwrite_stats)
      CALIB_TOKENS_PATH="${token_path}" "${PYTHON_BIN}" quantization/bv_sq.py all \
        --model "${model}" \
        --model_chunks "${chunks}" \
        --output_folder "${out}" \
        --model_type "${MODEL_TYPE}" \
        --dataset "${DATASET}" \
        --nsamples "${NSAMPLES}" \
        --seqlen "${SEQLEN}" \
        --seed "${SEED}" \
        --cache_dir "${CACHE_ROOT}/tokens" \
        --stats_path "${stats}" \
        --device "${DEVICE}" \
        --model_dtype "${BV_MODEL_DTYPE}" \
        --n_calib "${BV_N_CALIB}" \
        --batch_size "${BV_BATCH_SIZE}" \
        --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
        --bit "${bits}" \
        --solver "${solver}" \
        --bias_lambda "${lambda}" \
        --h_source "${BV_H_SOURCE}" \
        --h_floor "${BV_H_FLOOR}" \
        --cpu_count "${BV_CPU_COUNT}" \
        --row_chunksize "${BV_ROW_CHUNKSIZE}" \
        "${overwrite_args[@]}"
    fi
    eval_lut_method "${label}" "${model}" "${bits}" "${out}" "bvsq_${variant}" "${root}"
  done
}

eval_lut_method() {
  local label="$1" model="$2" bits="$3" lut_dir="$4" method="$5" root="$6"
  local ppl_file="${root}/ppl/${method}_${bits}bit.json"
  local lm_file="${root}/lm_eval/${method}_${bits}bit.json"
  mkdir -p "${root}/ppl" "${root}/lm_eval"
  if [[ "${RUN_PPL}" == "1" && ( "${FORCE_EVAL}" == "1" || ! -s "${ppl_file}" ) ]]; then
    log "PPL dense LUT eval: ${label} ${bits}-bit ${method}"
    "${PYTHON_BIN}" quantization/eval_nonuquantfix_ppl.py \
      --model "${model}" \
      --checkpoint "${method}" \
      --wbits "${bits}" \
      --model_type "${MODEL_TYPE}" \
      --backend dense_lut \
      --lut_folder "${lut_dir}" \
      --dense_dtype "${PPL_DENSE_DTYPE}" \
      --datasets wikitext2 c4 \
      --device "${DEVICE}" \
      --stride "${EVAL_STRIDE}" \
      --max_length "${EVAL_MAX_LENGTH}" \
      --batch_size "${PPL_BATCH_SIZE}" \
      --c4_samples "${EVAL_SAMPLES}" \
      --cache_dir "${CACHE_ROOT}/eval_cache" \
      --output_file "${ppl_file}"
  fi
  if [[ "${RUN_LMEVAL}" == "1" && ( "${FORCE_EVAL}" == "1" || ! -s "${lm_file}" ) ]]; then
    log "lm-eval dense LUT: ${label} ${bits}-bit ${method}"
    local -a limit_args=()
    [[ -n "${LM_EVAL_LIMIT}" ]] && limit_args+=(--limit "${LM_EVAL_LIMIT}")
    "${PYTHON_BIN}" quantization/eval_dense_lut_lm_eval.py \
      --model "${model}" \
      --lut_folder "${lut_dir}" \
      --model_type "${MODEL_TYPE}" \
      --dense_dtype "${PPL_DENSE_DTYPE}" \
      --device "${DEVICE}" \
      --tasks ${LM_EVAL_TASKS} \
      --batch_size "${LM_EVAL_BATCH_SIZE}" \
      --num_fewshot "${LM_EVAL_NUM_FEWSHOT}" \
      --output_dir "${root}/lm_eval/raw" \
      --run_name "${label}_${method}_${bits}bit" \
      --gen_kwargs "${LM_EVAL_GEN_KWARGS}" \
      --output_file "${lm_file}" \
      "${limit_args[@]}"
  fi
}

run_gptvq() {
  local label="$1" model="$2" bits="$3" token_path="$4"
  local run_dir="${OUTPUT_ROOT}/${label}/${bits}bit/gptvq"
  if [[ -s "${run_dir}/gptvq/run_summary.json" && "${OVERWRITE}" != "1" && "${FORCE_EVAL}" != "1" ]]; then
    log "Skipping existing GPTVQ run: ${run_dir}"
    return
  fi
  log "Running GPTVQ ${bits}-bit: ${label}"
  local -a lmeval_args=()
  if [[ "${RUN_LMEVAL}" == "0" ]]; then
    lmeval_args+=(--no-lm-eval)
  else
    lmeval_args+=(--include-lm-eval --lm-eval-tasks)
    read -r -a _tasks <<< "${LM_EVAL_TASKS}"
    lmeval_args+=("${_tasks[@]}")
    lmeval_args+=(--lm-eval-batch-size "${LM_EVAL_BATCH_SIZE}" --lm-eval-num-fewshot "${LM_EVAL_NUM_FEWSHOT}")
    [[ -n "${LM_EVAL_LIMIT}" ]] && lmeval_args+=(--lm-eval-limit "${LM_EVAL_LIMIT}")
  fi
  CALIB_TOKENS_PATH="${token_path}" "${PYTHON_BIN}" gptvq_rbvt_benchmark.py \
    --model-path "${model}" \
    --device "${DEVICE}" \
    --output-root "${run_dir}" \
    --variants gptvq \
    --wbits "${bits}" \
    --groupsize "${GPTVQ_GROUPSIZE}" \
    --kmeans-iters "${GPTVQ_KMEANS_ITERS}" \
    --n-calib "${GPTVQ_CALIB_N}" \
    --max-length "${GPTVQ_CALIB_LEN}" \
    --calib-dataset "${DATASET}" \
    --calibration-cache-dir "${CACHE_ROOT}/calibration" \
    --seed "${SEED}" \
    --eval-stride "${EVAL_STRIDE}" \
    --eval-max-length "${EVAL_MAX_LENGTH}" \
    --eval-samples "${GPTVQ_EVAL_SAMPLES}" \
    --eval-cache-dir "${CACHE_ROOT}/eval_cache" \
    --lm-eval-output-dir "${run_dir}/lm_eval" \
    --no-wandb \
    "${lmeval_args[@]}"
}

IFS=';' read -r -a MODEL_ARRAY <<< "${MODEL_SPECS}"
read -r -a BIT_ARRAY <<< "${BITS}"

log "Output root: ${OUTPUT_ROOT}"
log "Models: ${MODEL_SPECS}"
log "Bits: ${BITS}"
log "lm-eval tasks: ${LM_EVAL_TASKS}; gen_kwargs=${LM_EVAL_GEN_KWARGS}"

for spec in "${MODEL_ARRAY[@]}"; do
  if [[ "${spec}" != *=* ]]; then
    echo "Bad MODEL_SPECS entry: ${spec}; expected label=model" >&2
    exit 2
  fi
  label="${spec%%=*}"
  model="${spec#*=}"
  basename="$(model_basename "${model}")"
  token_path="$(ensure_redpajama_cache "${basename}")"
  for bits in "${BIT_ARRAY[@]}"; do
    run_root="${OUTPUT_ROOT}/${label}/${bits}bit/lut_stack"
    if [[ "${RUN_NF}" == "1" ]]; then
      run_nf_like "${label}" "${model}" "${bits}" "nf${bits}" "${token_path}"
      if [[ "${bits}" == "4" ]]; then
        run_nf_like "${label}" "${model}" "${bits}" "nvfp4" "${token_path}"
      fi
    fi
    if [[ "${RUN_LEAN_SQUEEZE}" == "1" ]]; then
      run_lean_squeeze "${label}" "${model}" "${bits}" "${token_path}"
    fi
    if [[ "${RUN_LNQ}" == "1" || "${RUN_BVSQ}" == "1" ]]; then
      ensure_sqllm_lut_stack "${label}" "${model}" "${bits}" "${token_path}" "${run_root}"
    fi
    if [[ "${RUN_LNQ}" == "1" ]]; then
      run_lnq "${label}" "${model}" "${bits}" "${token_path}" "${run_root}"
    fi
    if [[ "${RUN_BVSQ}" == "1" ]]; then
      run_bvsq "${label}" "${model}" "${bits}" "${token_path}" "${run_root}"
    fi
    if [[ "${RUN_GPTVQ}" == "1" ]]; then
      run_gptvq "${label}" "${model}" "${bits}" "${token_path}"
    fi
  done
done

log "Done. Results under ${OUTPUT_ROOT}"
