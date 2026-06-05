#!/usr/bin/env bash
# ============================================================
# LLM SFT Training Task Launcher
# Launches distributed or single-card SFT training via the
# chosen framework, while profile_train.py monitors from the side.
#
# Usage:
#   bash run_train.sh huggingface meta-llama/Llama-3-8B
#   bash run_train.sh deepspeed meta-llama/Llama-3-8B
#   torchrun --nproc_per_node=4 bash run_train.sh deepspeed ...
#   deepspeed --num_gpus=4 bash run_train.sh deepspeed ...
# ============================================================

set -euo pipefail

FRAMEWORK="${1:-huggingface}"
MODEL="${2:-meta-llama/Llama-3-8B}"
DATASET="${3:-}"
BATCH_SIZE="${4:-4}"
EPOCHS="${5:-3}"
LR="${6:-2e-5}"
MAX_SEQ_LEN="${7:-512}"
GPU_PER_NODE="${8:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/../../../out"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo " LLM SFT Training Task"
echo " Framework  : $FRAMEWORK"
echo " Model      : $MODEL"
echo " Batch Size : $BATCH_SIZE"
echo " Epochs     : $EPOCHS"
echo " LR         : $LR"
echo " GPUs       : $GPU_PER_NODE"
echo "=========================================="

run_huggingface() {
    python3 "$SCRIPT_DIR/profile_train.py" \
        --framework huggingface \
        --model "$MODEL" \
        --train-dataset "$DATASET" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --max-seq-length "$MAX_SEQ_LEN" \
        --out-dir "$OUT_DIR"
}

run_deepspeed() {
    # DeepSpeed single-node multi-GPU
    if [ "$GPU_PER_NODE" -gt 1 ]; then
        deepspeed --num_gpus="$GPU_PER_NODE" \
            "$SCRIPT_DIR/profile_train.py" \
            --framework deepspeed \
            --model "$MODEL" \
            --train-dataset "$DATASET" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --max-seq-length "$MAX_SEQ_LEN" \
            --out-dir "$OUT_DIR"
    else
        python3 "$SCRIPT_DIR/profile_train.py" \
            --framework deepspeed \
            --model "$MODEL" \
            --train-dataset "$DATASET" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --max-seq-length "$MAX_SEQ_LEN" \
            --out-dir "$OUT_DIR"
    fi
}

run_megatron() {
    echo "[Megatron-LM] Requires separate Megatron installation."
    echo "Launch with: python3 $SCRIPT_DIR/profile_train.py --framework megatron --task-script megatron_run.sh"
}

run_llama_factory() {
    echo "[LLaMA-Factory] Launching via LLaMA-Factory CLI ..."
    if [ -n "$DATASET" ]; then
        llamafactory-cli train \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --stage sft \
            --finetuning_type full \
            --output_dir "$OUT_DIR/checkpoint" \
            --per_device_train_batch_size "$BATCH_SIZE" \
            --gradient_accumulation_steps 4 \
            --learning_rate "$LR" \
            --num_train_epochs "$EPOCHS"
    else
        # Launch profiler to monitor while LLaMA-Factory runs
        python3 "$SCRIPT_DIR/profile_train.py" \
            --framework llama_factory \
            --model "$MODEL" \
            --task-script "${SCRIPT_DIR}/llama_factory_run.sh" \
            --out-dir "$OUT_DIR"
    fi
}

run_axolotl() {
    echo "[Axolotl] Launching via axolotl CLI ..."
    accelerate launch -m axolotl.cli.train "${DATASET:-config.yaml}"
}

run_colossalai() {
    echo "[Colossal-AI] Launching via colossalai CLI ..."
    colossalai run --nproc_per_node="$GPU_PER_NODE" \
        "$SCRIPT_DIR/profile_train.py" \
        --framework colossalai \
        --model "$MODEL" \
        --out-dir "$OUT_DIR"
}

run_lmdeploy() {
    echo "[LMDeploy] LMDeploy is primarily for inference, not training."
    echo "For training, use the inference profiler instead."
}

run_tgi() {
    echo "[TGI] Text Generation Inference is for serving, not training."
}

case "$FRAMEWORK" in
    huggingface)  run_huggingface ;;
    deepspeed)    run_deepspeed ;;
    megatron)     run_megatron ;;
    llama_factory) run_llama_factory ;;
    axolotl)      run_axolotl ;;
    colossalai)   run_colossalai ;;
    lmdeploy)     run_lmdeploy ;;
    tgi)          run_tgi ;;
    *)
        echo "Unknown framework: $FRAMEWORK"
        echo "Supported: huggingface, deepspeed, megatron, llama_factory, axolotl, colossalai, lmdeploy, tgi"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo " Training task complete."
echo " Results in: $OUT_DIR/result.md"
echo "=========================================="
