#!/bin/bash
# Flow Model Training Script for Step-Audio-EditX
# Adapted from CosyVoice training pipeline

set -e

# ============= Configuration =============
# Paths
STEP_AUDIO_ROOT="/Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX"
COSYVOICE_ROOT="/Users/weitao_li/CodeField/DCAI/Projects/CosyVoice"
TOKENIZER_PATH="${STEP_AUDIO_ROOT}/pretrained_models/Step-Audio-Tokenizer"

# Data paths
DATA_DIR="${STEP_AUDIO_ROOT}/data"
TRAIN_DATA="${DATA_DIR}/train"
DEV_DATA="${DATA_DIR}/dev"

# Model paths
MODEL_DIR="${STEP_AUDIO_ROOT}/exp/flow_model"
TENSORBOARD_DIR="${STEP_AUDIO_ROOT}/tensorboard/flow_model"
CHECKPOINT=""  # Set to checkpoint path for resuming training

# Training configuration
CONFIG="${STEP_AUDIO_ROOT}/training/configs/flow_model.yaml"
TRAIN_ENGINE="torch_ddp"  # or "deepspeed"
DIST_BACKEND="nccl"

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# Training hyperparameters
NUM_WORKERS=4
PREFETCH=100
USE_AMP=true

# Job configuration
JOB_ID=2026
RDZV_ENDPOINT="localhost:29500"

# ============= Stage Control =============
stage=-1
stop_stage=100

# ============= Helper Functions =============
function print_stage() {
    echo "=================================================="
    echo "Stage $1: $2"
    echo "=================================================="
}

# ============= Stage -1: Environment Check =============
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    print_stage -1 "Environment Check"
    
    # Check if CosyVoice is available
    if [ ! -d "$COSYVOICE_ROOT" ]; then
        echo "ERROR: CosyVoice not found at $COSYVOICE_ROOT"
        echo "Please clone CosyVoice repository first"
        exit 1
    fi
    
    # Check if tokenizer is available
    if [ ! -d "$TOKENIZER_PATH" ]; then
        echo "ERROR: Step-Audio-Tokenizer not found at $TOKENIZER_PATH"
        echo "Please download the tokenizer model first"
        exit 1
    fi
    
    # Add CosyVoice to Python path
    export PYTHONPATH="${COSYVOICE_ROOT}:${STEP_AUDIO_ROOT}:${PYTHONPATH}"
    
    echo "Environment check passed!"
    echo "  CosyVoice: $COSYVOICE_ROOT"
    echo "  Tokenizer: $TOKENIZER_PATH"
    echo "  GPUs: $NUM_GPUS"
fi

# ============= Stage 0: Data Preparation =============
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    print_stage 0 "Data Preparation"
    
    # Create data directories
    mkdir -p ${TRAIN_DATA} ${DEV_DATA}
    
    echo "Please prepare your data in the following format:"
    echo "  ${TRAIN_DATA}/wav.scp  - List of audio files (utt_id /path/to/audio.wav)"
    echo "  ${TRAIN_DATA}/text     - Transcriptions (utt_id text)"
    echo "  ${DEV_DATA}/wav.scp    - Validation audio files"
    echo "  ${DEV_DATA}/text       - Validation transcriptions"
    echo ""
    echo "Example wav.scp format:"
    echo "  utt001 /path/to/audio/utt001.wav"
    echo "  utt002 /path/to/audio/utt002.wav"
    echo ""
    echo "Press Enter when data is ready..."
    read
fi

# ============= Stage 1: Extract Speaker Embeddings =============
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    print_stage 1 "Extract Speaker Embeddings"
    
    for dataset in train dev; do
        data_dir="${DATA_DIR}/${dataset}"
        echo "Processing ${dataset} set..."
        
        python ${COSYVOICE_ROOT}/tools/extract_embedding.py \
            --dir ${data_dir} \
            --onnx_path ${TOKENIZER_PATH}/campplus.onnx
    done
fi

# ============= Stage 2: Extract Speech Tokens =============
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    print_stage 2 "Extract Speech Tokens"
    
    for dataset in train dev; do
        data_dir="${DATA_DIR}/${dataset}"
        echo "Processing ${dataset} set..."
        
        python ${STEP_AUDIO_ROOT}/training/data_processing/extract_step_audio_tokens.py \
            --data_dir ${data_dir} \
            --tokenizer_path ${TOKENIZER_PATH}
    done
fi

# ============= Stage 3: Create Parquet Files =============
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    print_stage 3 "Create Parquet Files"
    
    for dataset in train dev; do
        src_dir="${DATA_DIR}/${dataset}"
        des_dir="${DATA_DIR}/${dataset}/parquet"
        echo "Creating parquet files for ${dataset} set..."
        
        python ${STEP_AUDIO_ROOT}/training/data_processing/make_parquet.py \
            --src_dir ${src_dir} \
            --des_dir ${des_dir} \
            --num_utts_per_parquet 1000 \
            --num_processes 8
    done
    
    # Create data lists
    cat ${TRAIN_DATA}/parquet/data.list > ${DATA_DIR}/train.data.list
    cat ${DEV_DATA}/parquet/data.list > ${DATA_DIR}/dev.data.list
    
    echo "Data preparation complete!"
    echo "  Train data: ${DATA_DIR}/train.data.list"
    echo "  Dev data: ${DATA_DIR}/dev.data.list"
fi

# ============= Stage 4: Train Flow Model =============
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    print_stage 4 "Train Flow Model"
    
    # Create output directories
    mkdir -p ${MODEL_DIR} ${TENSORBOARD_DIR}
    
    # Prepare training command
    train_cmd="torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
        --rdzv_id=${JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${RDZV_ENDPOINT} \
        ${COSYVOICE_ROOT}/cosyvoice/bin/train.py \
        --train_engine ${TRAIN_ENGINE} \
        --config ${CONFIG} \
        --train_data ${DATA_DIR}/train.data.list \
        --cv_data ${DATA_DIR}/dev.data.list \
        --model flow \
        --model_dir ${MODEL_DIR} \
        --tensorboard_dir ${TENSORBOARD_DIR} \
        --ddp.dist_backend ${DIST_BACKEND} \
        --num_workers ${NUM_WORKERS} \
        --prefetch ${PREFETCH} \
        --pin_memory"
    
    # Add checkpoint if specified
    if [ -n "$CHECKPOINT" ]; then
        train_cmd="${train_cmd} --checkpoint ${CHECKPOINT}"
    fi
    
    # Add AMP if enabled
    if [ "$USE_AMP" = true ]; then
        train_cmd="${train_cmd} --use_amp"
    fi
    
    # Add DeepSpeed config if using DeepSpeed
    if [ "$TRAIN_ENGINE" = "deepspeed" ]; then
        train_cmd="${train_cmd} --deepspeed_config ${STEP_AUDIO_ROOT}/training/configs/ds_stage2.json"
        train_cmd="${train_cmd} --deepspeed.save_states model+optimizer"
    fi
    
    echo "Starting training with command:"
    echo "$train_cmd"
    echo ""
    
    eval $train_cmd
fi

# ============= Stage 5: Model Averaging =============
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    print_stage 5 "Model Averaging"
    
    AVERAGE_NUM=5
    FINAL_MODEL="${MODEL_DIR}/flow_averaged.pt"
    
    python ${COSYVOICE_ROOT}/cosyvoice/bin/average_model.py \
        --dst_model ${FINAL_MODEL} \
        --src_path ${MODEL_DIR} \
        --num ${AVERAGE_NUM} \
        --val_best
    
    echo "Averaged model saved to: ${FINAL_MODEL}"
fi

# ============= Stage 6: Export Model =============
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    print_stage 6 "Export Model for Inference"
    
    # Copy flow model to Step-Audio-EditX model directory
    EXPORT_DIR="${STEP_AUDIO_ROOT}/pretrained_models/Step-Audio-EditX/CosyVoice-300M-25Hz"
    mkdir -p ${EXPORT_DIR}
    
    cp ${MODEL_DIR}/flow_averaged.pt ${EXPORT_DIR}/flow.pt
    
    echo "Model exported to: ${EXPORT_DIR}/flow.pt"
    echo ""
    echo "You can now use this model for inference!"
fi

echo ""
echo "=================================================="
echo "Training pipeline completed!"
echo "=================================================="
