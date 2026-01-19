# 训练模式使用示例

## 1. 仅训练 Flow 模型

```bash
# 方式 1: 通过配置文件
# 编辑 train/configs/finetune_llm_flow.yaml
# 设置: train_mode: "flow"
python finetune_demo.py --config train/configs/finetune_llm_flow.yaml

# 方式 2: 通过命令行参数（推荐）
python finetune_demo.py --mode flow
```

## 2. 仅训练 LLM 模型

```bash
python finetune_demo.py --mode llm
```

## 3. 训练两者（分阶段）

```bash
python finetune_demo.py --mode both
```

## 训练模式说明

### flow 模式
- 仅微调 Flow 解码器
- 冻结 Flow 编码器和 LLM
- 适用场景：改善 mel 生成质量
- 训练时间：约 1-2 天

### llm 模式
- 仅微调 LLM 的 LoRA 层
- 冻结 LLM 主干和 Flow
- 适用场景：改善 token 生成质量
- 训练时间：约 1-2 天

### both 模式（默认）
- 分三阶段训练
  1. LLM 单独微调
  2. Flow 单独微调
  3. 联合微调（可选）
- 适用场景：端到端优化
- 训练时间：约 3-5 天

## 配置文件设置

在 `train/configs/finetune_llm_flow.yaml` 中：

```yaml
basic:
  train_mode: "flow"  # 可选: "llm" | "flow" | "both"
```

## 命令行参数优先级

命令行参数 `--mode` 会覆盖配置文件中的 `train_mode` 设置。
