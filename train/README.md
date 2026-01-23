# Step-Audio-EditX LLM + Flow 微调训练系统

> 轻量化微调模块，支持语音克隆和音频编辑任务的 LLM (LoRA) 和 Flow (解码器) 微调

## 📁 目录结构

```
train/
├── configs/              # 训练配置
│   └── finetune_llm_flow.yaml
├── dataset/              # 数据集模块
│   ├── processor.py      # 数据处理器
│   └── dataset.py        # Dataset 实现
├── trainer/              # 训练器
│   ├── model_adapter.py  # 模型适配器
│   └── train_loop.py     # 训练循环
├── utils/                # 工具函数
│   ├── config_utils.py
│   ├── train_utils.py
│   └── data_utils.py
├── tools/                # 预处理工具
│   ├── extract_speech_token.py
│   ├── extract_embedding.py
│   ├── make_parquet.py
│   └── prepare_data.sh
└── docs/                 # 文档
    └── data_preparation.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r train/requirements.txt
```

### 2. 准备数据

#### 2.1 准备基础文件

在源目录（例如 `data/raw`）创建以下文件：

**wav.scp** (音频列表):
```
utt001 /path/to/audio001.wav
utt002 /path/to/audio002.wav
```

**text** (文本转录):
```
utt001 你好世界
utt002 Hello world
```

**utt2spk** (说话人映射):
```
utt001 spk001
utt002 spk002
```

#### 2.2 运行预处理

```bash
# 一键预处理脚本
bash train/tools/prepare_data.sh data/raw data/parquet
```

或者手动执行：

```bash
# 步骤 1: 提取 speaker embedding
python train/tools/extract_embedding.py \
    --wav_scp data/raw/wav.scp \
    --utt2spk data/raw/utt2spk \
    --onnx_path pretrained_models/Step-Audio-EditX/CosyVoice-300M-25Hz/campplus.onnx \
    --output_dir data/raw

# 步骤 2: 提取 speech token
python train/tools/extract_speech_token.py \
    --wav_scp data/raw/wav.scp \
    --tokenizer_path pretrained_models/Step-Audio-Tokenizer \
    --output data/raw/utt2speech_token.pt \
    --model_source local

# 步骤 3: 打包成 parquet
python train/tools/make_parquet.py \
    --src_dir data/raw \
    --des_dir data/parquet \
    --num_utts_per_parquet 1000
```

**输出**：`data/parquet/data.list` (parquet 文件列表)

### 3. 配置训练参数

编辑 `train/configs/finetune_llm_flow.yaml`:

```yaml
# 数据路径
data:
  train_data: "./data/train/parquet/data.list"
  cv_data: "./data/dev/parquet/data.list"

# 训练模式
basic:
  train_mode: "both"  # "llm" | "flow" | "both"
```

### 4. 开始训练

```bash
# 训练模式 1: 仅 Flow 模型（推荐快速微调）
python finetune_demo.py --mode flow

# 训练模式 2: 仅 LLM 模型
python finetune_demo.py --mode llm

# 训练模式 3: 分阶段训练（完整训练）
python finetune_demo.py --mode both
```

---

## 📊 数据格式

### Parquet 数据结构

每个样本包含以下字段：

```python
{
    'utt': 'utterance_id',           # 样本 ID
    'wav': '/path/to/audio.wav',     # 音频路径
    'audio_data': b'...',             # 音频二进制数据
    'text': '你好世界',               # 文本内容
    'spk': 'speaker_001',             # 说话人 ID
    'utt_embedding': [...],           # 192-dim utterance embedding
    'spk_embedding': [...],           # 192-dim speaker embedding
    'speech_token': [...],            # 离散语音 token
}
```

### 关键点

- ✅ **仅需要单个音频** + 文本，无需配对数据
- ✅ **Speech token 预提取**（离线完成，提高训练速度）
- ✅ **Speaker embedding 预提取**（使用 FunASR Campplus）
- ✅ **Parquet 格式**（高效 I/O）

---

## 🎯 训练模式

### 模式 1: flow - 仅训练 Flow 解码器
```bash
python finetune_demo.py --mode flow
```
- 训练内容：Flow 解码器
- 训练时间：1-2 天
- 适用场景：改善音质、mel 生成

### 模式 2: llm - 仅训练 LLM
```bash
python finetune_demo.py --mode llm
```
- 训练内容：LLM LoRA
- 训练时间：1-2 天
- 适用场景：改善 token 生成

### 模式 3: both - 分阶段训练
```bash
python finetune_demo.py --mode both
```
- 阶段 1：LLM 单独微调 (25 epochs)
- 阶段 2：Flow 单独微调 (25 epochs)
- 阶段 3：联合微调 (5 epochs，可选)
- 训练时间：3-5 天
- 适用场景：端到端优化

---

## 💾 使用微调模型

```python
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from model_loader import ModelSource

# 初始化
tokenizer = StepAudioTokenizer(
    encoder_path="pretrained_models/Step-Audio-EditX",
    model_source=ModelSource.LOCAL
)

tts = StepAudioTTS(
    model_path="pretrained_models/Step-Audio-EditX",
    audio_tokenizer=tokenizer,
    model_source=ModelSource.LOCAL
)

# 加载微调权重
tts.load_finetuned_model(
    llm_ckpt_path="ckpt/finetune/llm_best.pt",
    flow_ckpt_path="ckpt/finetune/flow_best.pt"
)

# 推理
audio, sr = tts.clone(
    prompt_wav_path="prompt.wav",
    prompt_text="参考文本",
    target_text="目标文本"
)
```

---

## ⚙️ 高级配置

### 调整 LoRA 参数

```yaml
model:
  llm:
    lora_r: 8              # LoRA rank
    lora_alpha: 32
    lora_dropout: 0.05
```

### 调整训练阶段

```yaml
stage:
  stage1_epochs: 30      # LLM
  stage2_epochs: 30      # Flow
  stage3_epochs: 10      # 联合
```

### 减少显存占用

```yaml
optim:
  accum_grad: 8          # 梯度累积

data:
  max_frames_in_batch: 1000  # 减小 batch size
```

---

## 🐛 故障排除

### CUDA Out of Memory
- 增加 `accum_grad`
- 减小 `max_frames_in_batch`
- 减小 `lora_r`

### Speech Token 提取太慢
- 使用 GPU
- 分批处理

### Parquet 文件损坏
- 检查原始音频完整性
- 重新运行 `make_parquet.py`

---

## 📚 详细文档

- [数据准备指南](train/docs/data_preparation.md)
- [训练模式说明](train/TRAINING_MODES.md)

---

## ⚠️ 重要说明

1. **训练数据简单**：只需 (音频 + 文本 + 说话人ID)
2. **预处理是关键**：Token 和 embedding 必须预提取
3. **编辑能力来自 SFT/PPO**：基础训练只学习生成，不学习编辑
4. **保留推理逻辑**：微调权重完全兼容原推理 API

---

## 📝 License

本模块遵循 Step-Audio-EditX 主仓库的 License
