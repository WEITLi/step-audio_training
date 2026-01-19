# Step-Audio-EditX LLM + Flow 微调训练

> 轻量化微调模块，支持语音克隆和音频编辑任务的 LLM (LoRA) 和 Flow (解码器) 微调

## 📁 目录结构

```
train/
├── configs/              # 训练配置文件
│   └── finetune_llm_flow.yaml
├── dataset/              # 数据集加载模块
│   └── audio_edit_dataset.py
├── trainer/              # 训练器模块
│   ├── model_adapter.py  # 模型适配器 (LoRA + Flow)
│   └── train_loop.py     # 训练循环
├── utils/                # 工具函数
│   ├── config_utils.py   # 配置加载
│   ├── train_utils.py    # 训练工具 (checkpoint, EMA, etc)
│   └── data_utils.py     # 数据处理工具
├── requirements.txt      # 训练依赖
└── README.md            # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /path/to/Step-Audio-EditX
pip install -r train/requirements.txt
```

### 2. 准备数据集

创建 JSONL 格式的训练数据，每行一个样本：

```json
{
  "prompt_audio_path": "data/audio/prompt_001.wav",
  "prompt_text": "这是参考音频的文本内容",
  "target_text": "这是目标合成的文本内容",
  "target_audio_path": "data/audio/target_001.wav",
  "edit_type": "clone",
  "speaker_id": "spk_001"
}
```

**必需字段**:
- `prompt_audio_path`: 参考音频路径
- `target_audio_path`: 目标音频路径 (用于监督训练)

**可选字段**:
- `prompt_text`: 参考文本
- `target_text`: 目标文本
- `edit_type`: 任务类型 (clone/emotion/pronunciation/speed)
- `speaker_id`: 说话人 ID

将数据分为训练集和验证集：
- `data/train.jsonl`
- `data/val.jsonl`

### 3. 修改配置

编辑 `train/configs/finetune_llm_flow.yaml`:

```yaml
# 模型路径
model:
  llm:
    model_path: "./pretrained_models/Step-Audio-EditX"
  flow:
    model_path: "./pretrained_models/Step-Audio-EditX/CosyVoice-300M-25Hz"

# 数据路径
data:
  train_path: "./data/train.jsonl"
  val_path: "./data/val.jsonl"
  batch_size: 2  # 根据显存调整

# 训练超参数
optim:
  lr_llm: 1e-4   # LLM LoRA 学习率
  lr_flow: 1e-5  # Flow 学习率
```

### 4. 开始训练

#### 训练模式选择

本系统支持三种训练模式：

**模式 1: 仅训练 Flow 模型（推荐用于快速微调）**
```bash
python finetune_demo.py --mode flow
```

**模式 2: 仅训练 LLM 模型**
```bash
python finetune_demo.py --mode llm
```

**模式 3: 分阶段训练两者（完整训练）**
```bash
python finetune_demo.py --mode both
# 或直接使用
python finetune_demo.py --config train/configs/finetune_llm_flow.yaml
```

#### 训练模式对比

| 模式 | 训练内容 | 训练时间 | 适用场景 |
|------|---------|---------|---------|
| `flow` | Flow 解码器 | 1-2 天 | 改善音质、mel 生成 |
| `llm` | LLM LoRA | 1-2 天 | 改善 token 生成 |
| `both` | 分阶段训练 | 3-5 天 | 端到端优化 |

详见 [训练模式详细说明](TRAINING_MODES.md)

## 📋 训练阶段

训练分为三个阶段（自动执行）：

### 阶段 1: LLM 单独微调 (25 epochs)
- 仅训练 LLM 的 LoRA 层
- 冻结 LLM 主干和 Flow 模型
- 学习率: `1e-4`

### 阶段 2: Flow 单独微调 (25 epochs)
- 仅训练 Flow 解码器
- 冻结 LLM 和 Flow 编码器
- 学习率: `1e-5`

### 阶段 3: 联合微调 (5 epochs, 可选)
- 同时微调 LLM LoRA 和 Flow 解码器
- 使用较小的学习率 (原学习率 × 0.1)
- 联合损失: `0.4 * llm_loss + 0.6 * flow_loss`

## 💾 Checkpoint 管理

训练过程中会自动保存 checkpoint：

```
ckpt/finetune/
├── llm_best.pt          # 最佳 LLM checkpoint
├── flow_best.pt         # 最佳 Flow checkpoint
├── llm_epoch_5.pt       # 定期保存
└── flow_epoch_5.pt
```

## 🎯 使用微调后的模型

### 方式 1: 通过 finetune_demo.py 测试

```python
python finetune_demo.py --config train/configs/finetune_llm_flow.yaml
```

### 方式 2: 在推理代码中加载

```python
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from model_loader import ModelSource

# 初始化模型
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

# 使用微调后的模型进行推理
audio, sr = tts.clone(
    prompt_wav_path="path/to/prompt.wav",
    prompt_text="参考文本",
    target_text="目标文本"
)
```

## ⚙️ 高级配置

### 调整 LoRA 参数

```yaml
model:
  llm:
    lora_r: 8              # LoRA rank (越大参数越多)
    lora_alpha: 32         # LoRA alpha
    lora_dropout: 0.05     # Dropout
    lora_target_modules: ["q_proj", "v_proj", "k_proj"]  # 目标模块
```

### 调整训练阶段

```yaml
stage:
  stage1_epochs: 30      # LLM 单独训练轮数
  stage2_epochs: 30      # Flow 单独训练轮数
  stage3_epochs: 10      # 联合训练轮数
```

### 梯度累积（降低显存）

```yaml
optim:
  accum_grad: 8          # 梯度累积步数 (越大显存占用越小)
```

## 🐛 故障排除

### CUDA Out of Memory

**解决方案**:
1. 减小 `batch_size`（如改为 1）
2. 增加 `accum_grad`（如改为 8）
3. 降低 `lora_r`（如改为 4）

### 训练不收敛

**检查**:
1. 数据质量: 确保音频清晰、文本准确
2. 学习率: 尝试降低学习率
3. 数据量: 确保训练数据足够（建议 >100 条）

### Token 提取失败

**原因**: tokenizer 路径配置错误

**解决**: 检查配置文件中的 `model.llm.model_path` 是否正确

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir=ckpt/finetune
```

### 日志文件

```bash
tail -f ckpt/finetune/train.log
```

## 🔬 技术细节

### LLM 微调策略
- **方法**: LoRA (Low-Rank Adaptation)
- **目标**: 注意力层 (q_proj, v_proj, k_proj)
- **参数量**: ~0.5M (相比 3B 主干)
- **优势**: 显存占用低、训练速度快

### Flow 微调策略
- **方法**: 仅微调解码器
- **冻结**: 编码器、声码器
- **损失**: Mel MSE Loss
- **优势**: 保留预训练特征提取能力

## 📚 参考文档

- [CosyVoice 训练文档](https://github.com/FunAudioLLM/CosyVoice)
- [PEFT (LoRA) 文档](https://github.com/huggingface/peft)
- [Step-Audio-EditX 主仓库](https://github.com/stepfun-ai/Step-Audio-EditX)

## ⚠️ 注意事项

1. **严格保留推理逻辑**: 本模块仅新增训练代码，不修改原有推理逻辑
2. **兼容性**: 微调后的权重完全兼容原有推理 API
3. **数据隐私**: 请确保训练数据的合法性和隐私保护
4. **资源需求**: 建议使用至少 16GB 显存的 GPU

## 📝 License

本模块遵循 Step-Audio-EditX 主仓库的 License
