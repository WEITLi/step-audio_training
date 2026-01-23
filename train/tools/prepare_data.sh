#!/bin/bash
# 完整的数据准备流程脚本
# 包含从原始音频到 Parquet 的所有步骤
# 使用方法: bash train/tools/prepare_data.sh --help

set -e

# 默认参数
MODE="auto"
INPUT_DIR=""
OUTPUT_DIR=""
AUDIO_EXT="wav"
PRETRAIN_DIR="./pretrained_models"
SKIP_KALDI=false
SKIP_EMBEDDING=false
SKIP_TOKEN=false

# 显示帮助信息
show_help() {
    cat << EOF
Step-Audio-EditX 数据准备完整流程

用法：
    bash $0 [选项]

选项:
    -m, --mode MODE          数据准备模式 (auto|directory|jsonl)
                            - auto: 自动检测（默认）
                            - directory: 从音频目录准备
                            - jsonl: 从 JSONL 文件准备
    
    -i, --input DIR/FILE     输入目录或 JSONL 文件
    -o, --output DIR         输出目录
    -e, --audio-ext EXT      音频文件扩展名（默认: wav）
    -p, --pretrain DIR       预训练模型目录（默认: ./pretrained_models）
    
    --skip-kaldi            跳过 Kaldi 文件生成（假设已存在）
    --skip-embedding        跳过 embedding 提取
    --skip-token            跳过 speech token 提取
    
    -h, --help              显示此帮助信息

示例:
    # 从音频目录准备
    bash $0 -m directory -i data/raw_audio -o data/raw
    
    # 从 JSONL 文件准备
    bash $0 -m jsonl -i data/metadata.jsonl -o data/raw
    
    # 跳过某些步骤（假设已完成）
    bash $0 -i data/raw -o data/raw --skip-kaldi

目录结构 (directory 模式):
    input_dir/
    ├── speaker1/
    │   ├── audio1.wav
    │   ├── audio1.txt
    │   └── ...
    └── speaker2/
        └── ...

JSONL 格式 (jsonl 模式):
    {"audio_path": "/path/to/audio.wav", "text": "文本", "speaker_id": "spk001"}

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -e|--audio-ext)
            AUDIO_EXT="$2"
            shift 2
            ;;
        -p|--pretrain)
            PRETRAIN_DIR="$2"
            shift 2
            ;;
        --skip-kaldi)
            SKIP_KALDI=true
            shift
            ;;
        --skip-embedding)
            SKIP_EMBEDDING=true
            shift
            ;;
        --skip-token)
            SKIP_TOKEN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "错误: 缺少必需参数 --input 或 --output"
    show_help
    exit 1
fi

# 自动检测模式
if [ "$MODE" = "auto" ]; then
    if [ -f "$INPUT_DIR" ]; then
        MODE="jsonl"
        echo "自动检测: JSONL 模式"
    elif [ -d "$INPUT_DIR" ]; then
        MODE="directory"
        echo "自动检测: Directory 模式"
    else
        echo "错误: 输入路径不存在: $INPUT_DIR"
        exit 1
    fi
fi

echo "=========================================="
echo "Step-Audio-EditX 数据准备完整流程"
echo "=========================================="
echo "模式: $MODE"
echo "输入: $INPUT_DIR"
echo "输出: $OUTPUT_DIR"
echo "预训练模型: $PRETRAIN_DIR"
echo "=========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 步骤 0: 生成 Kaldi 格式文件 (如果需要)
if [ "$SKIP_KALDI" = false ]; then
    echo ""
    echo "步骤 0/3: 生成 Kaldi 格式文件"
    echo "----------------------------------------"
    
    python train/tools/prepare_kaldi_files.py \
        --mode "$MODE" \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --audio_ext "$AUDIO_EXT"
    
    echo "✓ Kaldi 文件生成完成"
else
    echo ""
    echo "步骤 0/3: 跳过 Kaldi 文件生成"
    echo "----------------------------------------"
fi

# 检查必需文件
echo ""
echo "检查必需文件..."
for file in wav.scp text utt2spk; do
    if [ ! -f "${OUTPUT_DIR}/${file}" ]; then
        echo "错误: ${OUTPUT_DIR}/${file} 不存在"
        exit 1
    fi
done
echo "✓ 所有必需文件存在"

# 步骤 1: 提取 speaker embedding
if [ "$SKIP_EMBEDDING" = false ]; then
    echo ""
    echo "步骤 1/3: 提取 Speaker Embedding"
    echo "----------------------------------------"
    
    if [ -f "${OUTPUT_DIR}/utt2embedding.pt" ] && [ -f "${OUTPUT_DIR}/spk2embedding.pt" ]; then
        echo "⚠ Embedding 文件已存在，跳过提取"
    else
        python train/tools/extract_embedding.py \
            --wav_scp "${OUTPUT_DIR}/wav.scp" \
            --utt2spk "${OUTPUT_DIR}/utt2spk" \
            --onnx_path "${PRETRAIN_DIR}/Step-Audio-EditX/CosyVoice-300M-25Hz/campplus.onnx" \
            --output_dir "${OUTPUT_DIR}"
        echo "✓ Speaker embedding 提取完成"
    fi
else
    echo ""
    echo "步骤 1/3: 跳过 Embedding 提取"
    echo "----------------------------------------"
fi

# 步骤 2: 提取 speech token
if [ "$SKIP_TOKEN" = false ]; then
    echo ""
    echo "步骤 2/3: 提取 Speech Token"
    echo "----------------------------------------"
    
    if [ -f "${OUTPUT_DIR}/utt2speech_token.pt" ]; then
        echo "⚠ Speech token 文件已存在，跳过提取"
    else
        python train/tools/extract_speech_token.py \
            --wav_scp "${OUTPUT_DIR}/wav.scp" \
            --tokenizer_path "${PRETRAIN_DIR}/Step-Audio-Tokenizer" \
            --output "${OUTPUT_DIR}/utt2speech_token.pt" \
            --model_source local
        echo "✓ Speech token 提取完成"
    fi
else
    echo ""
    echo "步骤 2/3: 跳过 Speech Token 提取"
    echo "----------------------------------------"
fi

# 步骤 3: 打包成 parquet
echo ""
echo "步骤 3/3: 创建 Parquet 文件"
echo "----------------------------------------"

PARQUET_DIR="${OUTPUT_DIR}/parquet"

python train/tools/make_parquet.py \
    --src_dir "${OUTPUT_DIR}" \
    --des_dir "${PARQUET_DIR}" \
    --num_utts_per_parquet 1000

echo ""
echo "=========================================="
echo "✅ 数据准备完成!"
echo "=========================================="
echo "Kaldi 文件:"
echo "  - ${OUTPUT_DIR}/wav.scp"
echo "  - ${OUTPUT_DIR}/text"
echo "  - ${OUTPUT_DIR}/utt2spk"
echo ""
echo "预处理特征:"
echo "  - ${OUTPUT_DIR}/utt2embedding.pt"
echo "  - ${OUTPUT_DIR}/spk2embedding.pt"
echo "  - ${OUTPUT_DIR}/utt2speech_token.pt"
echo ""
echo "Parquet 文件:"
echo "  - ${PARQUET_DIR}/data.list"
echo ""
echo "下一步:"
echo "  1. 查看 ${PARQUET_DIR}/data.list"
echo "  2. 在配置文件中设置:"
echo "     data:"
echo "       train_data: \"${PARQUET_DIR}/data.list\""
echo "  3. 开始训练:"
echo "     python finetune_demo.py --mode flow"
echo "=========================================="
