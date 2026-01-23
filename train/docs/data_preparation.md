# 数据准备指南

## 📋 概览

本指南介绍如何从原始音频文件准备训练数据。

---

## 🚀 快速开始

### 方式 1: 从音频目录准备

如果你有以下目录结构：

```
audio_dir/
├── speaker1/
│   ├── audio1.wav
│   ├── audio1.txt
│   ├── audio2.wav
│   ├── audio2.txt
│   └── ...
├── speaker2/
│   └── ...
```

运行：

```bash
bash train/tools/prepare_data.sh -m directory -i audio_dir -o data/raw
```

### 方式 2: 从 JSONL 文件准备

如果你有 JSONL 格式的元数据：

```jsonl
{"audio_path": "/path/to/audio1.wav", "text": "你好世界", "speaker_id": "spk001"}
{"audio_path": "/path/to/audio2.wav", "text": "Hello world", "speaker_id": "spk002"}
```

运行：

```bash
bash train/tools/prepare_data.sh -m jsonl -i metadata.jsonl -o data/raw
```

### 方式 3: 手动准备（如果已有 Kaldi 文件）

如果你已经有 `wav.scp`, `text`, `utt2spk` 文件：

```bash
bash train/tools/prepare_data.sh -i data/raw -o data/raw --skip-kaldi
```

---

## 📊 完整流程

### 步骤 0: 生成 Kaldi 格式文件（自动）

脚本会自动生成：
- `wav.scp`: 音频文件列表
- `text`: 文本转录
- `utt2spk`: utterance 到 speaker 映射
- `spk2utt`: speaker 到 utterance 映射

### 步骤 1: 提取 Speaker Embedding

使用 FunASR Campplus 提取 192-dim speaker embedding：

**输出**:
- `utt2embedding.pt`: utterance-level embedding
- `spk2embedding.pt`: speaker-level embedding (平均)

### 步骤 2: 提取 Speech Token

使用 Step-Audio-Tokenizer 提取离散语音 token：

**输出**:
- `utt2speech_token.pt`: 每个 utterance 的 speech token

> [!IMPORTANT]
> Speech token 提取较慢，建议使用 GPU 加速

### 步骤 3: 打包成 Parquet

将所有预处理结果打包成 Parquet 格式：

**输出**:
- `parquet/data.list`: parquet 文件列表
- `parquet/parquet_*.tar`: parquet 数据文件

---

## 🎯 支持的数据格式

### 1. 目录结构格式

#### 带 speaker 子目录（推荐）

```
audio_dir/
├── speaker1/
│   ├── audio1.wav
│   ├── audio1.txt        # 对应文本
│   └── audio2.wav
│       └── audio2.txt
├── speaker2/
│   └── ...
```

- 第一层目录名作为 speaker ID
- 文件名（不含扩展名）作为 utterance ID 的一部分
- 每个 `.wav` 文件需要有对应的 `.txt` 文本文件

#### 扁平目录结构

```
audio_dir/
├── spk1_utt1.wav
├── spk1_utt1.txt
├── spk2_utt1.wav
├── spk2_utt1.txt
└── ...
```

- 从文件名自动推断 speaker（下划线前部分）
- 或使用默认 speaker ID

### 2. JSONL 格式

```jsonl
{"audio_path": "/abs/path/to/audio.wav", "text": "文本内容", "speaker_id": "spk001", "utt_id": "utt001"}
{"audio_path": "/abs/path/to/audio.wav", "text": "文本内容", "speaker_id": "spk002"}
```

**必需字段**:
- `audio_path` (或 `wav`, `path`): 音频文件绝对路径
- `text` (或 `transcript`): 文本转录

**可选字段**:
- `speaker_id` (或 `spk`): 说话人 ID（默认: `default_speaker`）
- `utt_id` (或 `utt`): utterance ID（默认: 自动生成）

---

## ⚙️ 高级选项

### 跳过某些步骤

如果某些步骤已完成，可以跳过：

```bash
# 跳过 Kaldi 文件生成（已有 wav.scp, text, utt2spk）
bash train/tools/prepare_data.sh -i data/raw -o data/raw --skip-kaldi

# 跳过 embedding 提取（已有 utt2embedding.pt）
bash train/tools/prepare_data.sh -i data/raw -o data/raw --skip-embedding

# 跳过 token 提取（已有 utt2speech_token.pt）
bash train/tools/prepare_data.sh -i data/raw -o data/raw --skip-token
```

### 自定义音频扩展名

```bash
bash train/tools/prepare_data.sh -i audio_dir -o data/raw -e flac
```

### 指定预训练模型路径

```bash
bash train/tools/prepare_data.sh -i audio_dir -o data/raw -p /custom/pretrain/path
```

---

## 🔍 验证数据

### 检查 Kaldi 文件

```bash
# 检查生成的文件
head data/raw/wav.scp
head data/raw/text
head data/raw/utt2spk

# 统计
wc -l data/raw/wav.scp
wc -l data/raw/text
```

### 检查 Parquet 文件

```python
import pandas as pd

# 读取一个 parquet 文件
df = pd.read_parquet('data/raw/parquet/parquet_000000000.tar')

print(f"Columns: {df.columns.tolist()}")
print(f"Number of samples: {len(df)}")
print(f"\nFirst sample:")
print(df.iloc[0])
```

---

## 🐛 故障排除

### 找不到文本文件

**问题**: `No text file for xxx.wav, skipping`

**解决**:
- 确保每个 `.wav` 文件有对应的 `.txt` 文件
- 或使用 JSONL 格式

### Token 提取太慢

**问题**: extract_speech_token.py 运行很慢

**解决**:
- 使用 GPU: `CUDA_VISIBLE_DEVICES=0`
- 分批处理数据
- 确保使用正确的 tokenizer 路径

### Campplus 模型找不到

**问题**: `FileNotFoundError: campplus.onnx`

**解决**:
- 检查预训练模型路径
- 确保下载了完整的 CosyVoice-300M-25Hz 模型

### JSONL 中音频路径不存在

**问题**: `Audio file not found: ...`

**解决**:
- 使用绝对路径
- 检查文件确实存在

---

## 📝 完整示例

### 示例 1: LibriTTS 风格数据

```bash
# 目录结构
LibriTTS/
├── speaker1/
│   ├── chapter1/
│   │   ├── audio1.wav
│   │   ├── audio1.normalized.txt
│   │   └── ...
│   └── ...

# 准备数据
bash train/tools/prepare_data.sh \
    -m directory \
    -i LibriTTS \
    -o data/libritts
```

### 示例 2: 自定义数据集

```bash
# 1. 创建 JSONL
cat > data/metadata.jsonl << EOF
{"audio_path": "/data/audio/sample1.wav", "text": "你好世界", "speaker_id": "speaker1"}
{"audio_path": "/data/audio/sample2.wav", "text": "Hello world", "speaker_id": "speaker2"}
EOF

# 2. 准备数据
bash train/tools/prepare_data.sh \
    -m jsonl \
    -i data/metadata.jsonl \
    -o data/custom
```

### 示例 3: 分步执行

```bash
# 步骤 1: 生成 Kaldi 文件
python train/tools/prepare_kaldi_files.py \
    --mode directory \
    --input audio_dir \
    --output data/raw

# 步骤 2: 提取 embedding
python train/tools/extract_embedding.py \
    --wav_scp data/raw/wav.scp \
    --utt2spk data/raw/utt2spk \
    --onnx_path pretrained_models/Step-Audio-EditX/CosyVoice-300M-25Hz/campplus.onnx \
    --output_dir data/raw

# 步骤 3: 提取 token
python train/tools/extract_speech_token.py \
    --wav_scp data/raw/wav.scp \
    --tokenizer_path pretrained_models/Step-Audio-Tokenizer \
    --output data/raw/utt2speech_token.pt

# 步骤 4: 打包 parquet
python train/tools/make_parquet.py \
    --src_dir data/raw \
    --des_dir data/raw/parquet
```

---

## 📚 下一步

数据准备完成后：

1. **划分训练/验证集**

```bash
# 合并训练数据
cat data/train1/parquet/data.list data/train2/parquet/data.list > data/train.data.list

# 验证集
cp data/dev/parquet/data.list data/dev.data.list
```

2. **更新配置文件**

```yaml
# train/configs/finetune_llm_flow.yaml
data:
  train_data: "./data/train.data.list"
  cv_data: "./data/dev.data.list"
```

3. **开始训练**

```bash
python finetune_demo.py --mode flow
```
