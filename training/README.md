# Flow Model Training for Step-Audio-EditX

完整的 Flow Model (token2mel) 训练脚本和配置文件。

## 📁 目录结构

```
training/
├── configs/
│   ├── flow_model.yaml      # Flow Model 训练配置
│   └── ds_stage2.json        # DeepSpeed 配置
├── scripts/
│   └── train_flow.sh         # 主训练脚本
├── data_processing/
│   ├── extract_step_audio_tokens.py  # Token 提取
│   └── make_parquet.py       # Parquet 数据生成
└── tools/
    └── test_setup.py         # 环境测试脚本
```

## 🚀 快速开始

### 1. 测试环境

```bash
cd /Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX
python training/tools/test_setup.py
```

### 2. 准备数据

创建数据目录并准备音频文件：

```bash
mkdir -p data/train data/dev

# 准备 wav.scp (格式: utt_id /path/to/audio.wav)
# 准备 text (格式: utt_id transcription)
```

### 3. 运行训练

```bash
cd /Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX

# 完整流程 (stage 0-6)
bash training/scripts/train_flow.sh

# 或分阶段运行
bash training/scripts/train_flow.sh --stage 0 --stop_stage 3  # 数据准备
bash training/scripts/train_flow.sh --stage 4 --stop_stage 4  # 训练
bash training/scripts/train_flow.sh --stage 5 --stop_stage 6  # 导出
```

## 📋 训练流程

### Stage 0: 数据准备
准备 `wav.scp` 和 `text` 文件

### Stage 1: 提取说话人嵌入
使用 Campplus 模型提取 speaker embedding

### Stage 2: 提取语音 Token
使用 Step-Audio Tokenizer 提取双码本 token

### Stage 3: 生成 Parquet 文件
转换为训练所需的 parquet 格式

### Stage 4: 训练 Flow Model
使用 DDP 或 DeepSpeed 进行分布式训练

### Stage 5: 模型平均
对最后 N 个 checkpoint 进行平均

### Stage 6: 导出模型
导出为可用于推理的格式

## ⚙️ 配置说明

### GPU 配置

编辑 `training/scripts/train_flow.sh`:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 使用的 GPU
```

### 训练引擎

```bash
TRAIN_ENGINE="torch_ddp"  # 或 "deepspeed"
```

### 超参数

编辑 `training/configs/flow_model.yaml`:

```yaml
train_conf:
    optim_conf:
        lr: 0.001  # 学习率
    max_epoch: 100  # 训练轮数
    accum_grad: 2   # 梯度累积
```

## 💾 资源需求

### 最小配置
- GPU: 4x V100 (16GB)
- 内存: 64GB
- 存储: 100GB

### 推荐配置
- GPU: 4x A100 (40GB)
- 内存: 128GB
- 存储: 500GB

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir=/Users/weitao_li/CodeField/DCAI/Projects/Step-Audio-EditX/tensorboard/flow_model
```

### 查看日志

```bash
tail -f exp/flow_model/train.log
```

## 🔧 故障排除

### 问题 1: CUDA Out of Memory

**解决方案**:
- 减小 batch size: 修改 `flow_model.yaml` 中的 `max_frames_in_batch`
- 增加梯度累积: 修改 `accum_grad`
- 使用 DeepSpeed: 设置 `TRAIN_ENGINE="deepspeed"`

### 问题 2: 数据加载慢

**解决方案**:
- 增加 workers: 修改 `NUM_WORKERS`
- 增加 prefetch: 修改 `PREFETCH`

### 问题 3: 训练不收敛

**解决方案**:
- 降低学习率
- 检查数据质量
- 使用预训练模型初始化

## 📚 参考文档

- [CosyVoice 训练分析](file:///Users/weitao_li/.gemini/antigravity/brain/126bd317-85df-410a-b87b-a967071b5f85/cosyvoice_training_analysis.md)
- [Token2Mel 训练指南](file:///Users/weitao_li/.gemini/antigravity/brain/126bd317-85df-410a-b87b-a967071b5f85/token2mel_training_guide.md)
