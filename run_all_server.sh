#!/usr/bin/env bash
# =============================================================================
#  run_all.sh  –  Chạy trên server Vast.ai (tmux session)
#  Mục tiêu:
#    1. Setup repo (clone dev branch NonUQuant, NCCQuant, GPTVQ)
#    2. HF login + tải LLaMA-3.1-8B nếu chưa có
#    3. Run debug_ncc_bias_mae_mse.py  (adjusted + original baseline)
#    4. Run gptvq_rbvt_benchmark.py full eval (ppl wikitext2+c4, lm_eval trừ gsm8k)
# =============================================================================
set -e

MODEL_PATH="/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
# HF_TOKEN should be set via environment variable before running this script
# export HF_TOKEN="your_hf_token_here"
HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN not set. Run: export HF_TOKEN=your_token_here}"
WORKDIR="$HOME/NonUQuant_dev"
DEVICE="cuda:0"

# ─── 0. Môi trường ──────────────────────────────────────────────────────────
echo "===== 0. Python env check ====="
python3 -c "import torch; print('torch', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# ─── 1. Clone / update repo ─────────────────────────────────────────────────
echo "===== 1. Clone NonUQuant dev branch ====="
if [ ! -d "$WORKDIR" ]; then
    git clone -b dev https://github.com/huynguyenquang-collab/NonUQuant.git "$WORKDIR"
else
    cd "$WORKDIR" && git fetch origin && git checkout dev && git pull origin dev
fi
cd "$WORKDIR"

# GPTVQ submodule
if [ ! -d "GPTVQ/.git" ]; then
    rm -rf GPTVQ
    git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
fi

# NCCQuant submodule
if [ ! -d "NCCQuant/.git" ]; then
    rm -rf NCCQuant
    git clone https://github.com/anhnda/NCCQuant.git NCCQuant
fi

echo "===== 2. Install requirements ====="
pip install -q -r requirements-server.txt

# ─── 3. HF login + kiểm tra / tải model ────────────────────────────────────
echo "===== 3. HuggingFace login ====="
hf auth login --token "$HF_TOKEN"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found locally – downloading meta-llama/Meta-Llama-3.1-8B ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Meta-Llama-3.1-8B',
    token='$HF_TOKEN',
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'],
)
"
    # Sau khi tải, cập nhật MODEL_PATH theo snapshot mới nhất
    MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('meta-llama/Meta-Llama-3.1-8B', token='$HF_TOKEN')
print(p)
")
    echo "Model downloaded to: $MODEL_PATH"
fi

# ─── 4. debug_ncc_bias_mae_mse.py ─────────────────────────────────────────
echo ""
echo "===== 4a. Debug NCC  –  baseline=adjusted ====="
python3 debug_ncc_bias_mae_mse.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --max-layers 2 \
    --n-calib 128 \
    --max-length 512 \
    --groupsize 128 \
    --wbits 3 \
    --score cov \
    --ncc-budget-p 0.02 \
    --ncc-sweeps 1 \
    --baseline adjusted \
    2>&1 | tee outputs/debug_ncc_adjusted.log

echo ""
echo "===== 4b. Debug NCC  –  baseline=original ====="
python3 debug_ncc_bias_mae_mse.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --max-layers 2 \
    --n-calib 128 \
    --max-length 512 \
    --groupsize 128 \
    --wbits 3 \
    --score cov \
    --ncc-budget-p 0.02 \
    --ncc-sweeps 1 \
    --baseline original \
    2>&1 | tee outputs/debug_ncc_original.log

# ─── 5. Full eval  (ppl wikitext2+c4  +  lm_eval trừ gsm8k) ───────────────
echo ""
echo "===== 5. Full benchmark – LLaMA-3.1-8B  (wbits=3, groupsize=128) ====="
mkdir -p outputs/full_eval

python3 gptvq_rbvt_benchmark.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --groupsize 128 \
    --wbits 3 \
    --ncc-budget-p 0.02 \
    --ncc-sweeps 1 \
    --n-calib 128 \
    --max-length 512 \
    --eval-samples 64 \
    --include-lm-eval \
    --lm-eval-tasks arc_easy arc_challenge hellaswag winogrande piqa openbookqa boolq lambada_openai \
    --lm-eval-batch-size auto \
    --output-root outputs/full_eval \
    2>&1 | tee outputs/full_eval/full_eval.log

echo ""
echo "===== DONE ====="
echo "Logs:"
echo "  Debug adjusted : $WORKDIR/outputs/debug_ncc_adjusted.log"
echo "  Debug original : $WORKDIR/outputs/debug_ncc_original.log"
echo "  Full eval      : $WORKDIR/outputs/full_eval/full_eval.log"
