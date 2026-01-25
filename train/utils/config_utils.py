"""
配置文件加载与校验工具
"""

import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM 模型配置"""
    model_path: str = "./pretrained_models/Step-Audio-EditX"
    lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj"])
    freeze_backbone: bool = True


@dataclass
class TokenizerConfig:
    """Tokenizer 模型配置"""
    model_path: str = "./pretrained_models/stepaudio_editx/Step-Audio-Tokenizer"


@dataclass
class FlowConfig:
    """Flow 模型配置"""
    model_path: str = "./pretrained_models/Step-Audio-EditX/CosyVoice-300M-25Hz"
    freeze_encoder: bool = True
    freeze_vocoder: bool = True
    n_timesteps: int = 10


@dataclass
class OptimConfig:
    """优化器配置"""
    lr_llm: float = 1e-4
    lr_flow: float = 1e-5
    warmup_steps: int = 3000
    weight_decay: float = 0.01
    grad_clip: float = 5.0
    accum_grad: int = 4


@dataclass
class DataConfig:
    """数据集配置"""
    train_data: str = "./data/train.data.list"  # parquet 文件列表
    cv_data: str = "./data/dev.data.list"
    sample_rate: int = 24000
    token_mel_ratio: int = 2  # token 和 mel 的比例
    
    # 数据过滤参数
    filter: dict = field(default_factory=lambda: {
        'max_length': 40960,
        'min_length': 100,
        'token_max_length': 200,
        'token_min_length': 1,
    })
    
    # DataLoader 参数
    num_workers: int = 2
    prefetch_factor: int = 100
    pin_memory: bool = True
    
    # Pipeline 参数
    shuffle_size: int = 1000
    sort_size: int = 500
    max_frames_in_batch: int = 2000
    use_spk_embedding: bool = False


@dataclass
class SaveConfig:
    """保存配置"""
    ckpt_dir: str = "./ckpt/finetune"
    save_best: bool = True
    save_interval: int = 5


@dataclass
class LossConfig:
    """损失权重配置"""
    llm_weight: float = 0.4
    flow_weight: float = 0.6


@dataclass
class StageConfig:
    """阶段训练配置"""
    stage1_epochs: int = 25
    stage2_epochs: int = 25
    stage3_epochs: int = 5
    joint_lr_scale: float = 0.1


@dataclass
class BasicConfig:
    """基础配置"""
    seed: int = 42
    device: str = "cuda"
    max_epoch: int = 50
    log_interval: int = 50
    val_interval: int = 1
    train_mode: str = "both"  # "llm" | "flow" | "both"


@dataclass
class TrainConfig:
    """完整训练配置"""
    basic: BasicConfig = field(default_factory=BasicConfig)
    model_llm: LLMConfig = field(default_factory=LLMConfig)
    model_tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model_flow: FlowConfig = field(default_factory=FlowConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stage: StageConfig = field(default_factory=StageConfig)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: 配置文件格式错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    校验配置文件有效性
    
    Args:
        config: 配置字典
        
    Raises:
        ValueError: 配置参数无效
        FileNotFoundError: 必需的路径不存在
    """
    # 校验模型路径
    llm_path = config.get('model', {}).get('llm', {}).get('model_path', '')
    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"LLM 模型路径不存在: {llm_path}")
    
    tokenizer_path = config.get('model', {}).get('tokenizer', {}).get('model_path', '')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer 模型路径不存在: {tokenizer_path}")
    
    flow_path = config.get('model', {}).get('flow', {}).get('model_path', '')
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow 模型路径不存在: {flow_path}")
    
    # 校验数据集路径（训练时必须存在）
    train_path = config.get('data', {}).get('train_path', '')
    if train_path and not os.path.exists(train_path):
        print(f"警告: 训练数据路径不存在: {train_path}")
    
    val_path = config.get('data', {}).get('val_path', '')
    if val_path and not os.path.exists(val_path):
        print(f"警告: 验证数据路径不存在: {val_path}")
    
    # 校验学习率范围
    lr_llm = config.get('optim', {}).get('lr_llm', 0)
    if isinstance(lr_llm, str):
        lr_llm = float(lr_llm)
    if lr_llm > 1e-3:
        print(f"警告: LLM 学习率较高: {lr_llm}，建议不超过 1e-3")
    
    lr_flow = config.get('optim', {}).get('lr_flow', 0)
    if isinstance(lr_flow, str):
        lr_flow = float(lr_flow)
    if lr_flow > 1e-3:
        print(f"警告: Flow 学习率较高: {lr_flow}，建议不超过 1e-3")
    
    # 校验梯度裁剪阈值
    grad_clip = config.get('optim', {}).get('grad_clip', 0)
    if grad_clip <= 0:
        raise ValueError(f"梯度裁剪阈值必须大于0: {grad_clip}")
    
    print("配置校验通过！")


def load_and_validate_config(config_path: str) -> TrainConfig:
    """
    加载并校验配置文件，返回类型化配置对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        TrainConfig 配置对象
    """
    # 加载 YAML
    raw_config = load_yaml_config(config_path)
    
    # 校验配置
    try:
        validate_config(raw_config)
    except FileNotFoundError as e:
        print(f"配置校验警告: {e}")
    
    # 构建配置对象
    config = TrainConfig()
    
    # 解析 basic
    if 'basic' in raw_config:
        config.basic = BasicConfig(**raw_config['basic'])
    
    # 解析 model
    if 'model' in raw_config:
        if 'llm' in raw_config['model']:
            config.model_llm = LLMConfig(**raw_config['model']['llm'])
        if 'tokenizer' in raw_config['model']:
            config.model_tokenizer = TokenizerConfig(**raw_config['model']['tokenizer'])
        if 'flow' in raw_config['model']:
            config.model_flow = FlowConfig(**raw_config['model']['flow'])
    
    # 解析 optim
    if 'optim' in raw_config:
        optim_data = raw_config['optim'].copy()
        print(f"原始 optim 配置: {optim_data}")
        
        # 确保所有数值参数都是正确的类型
        if 'lr_llm' in optim_data:
            optim_data['lr_llm'] = float(optim_data['lr_llm'])
            print(f"lr_llm 转换为: {optim_data['lr_llm']} (类型: {type(optim_data['lr_llm'])})")
        if 'lr_flow' in optim_data:
            optim_data['lr_flow'] = float(optim_data['lr_flow'])
            print(f"lr_flow 转换为: {optim_data['lr_flow']} (类型: {type(optim_data['lr_flow'])})")
        if 'weight_decay' in optim_data:
            optim_data['weight_decay'] = float(optim_data['weight_decay'])
        if 'grad_clip' in optim_data:
            optim_data['grad_clip'] = float(optim_data['grad_clip'])
        if 'warmup_steps' in optim_data:
            optim_data['warmup_steps'] = int(optim_data['warmup_steps'])
        if 'accum_grad' in optim_data:
            optim_data['accum_grad'] = int(optim_data['accum_grad'])
        config.optim = OptimConfig(**optim_data)
    
    # 解析 data
    if 'data' in raw_config:
        config.data = DataConfig(**raw_config['data'])
    
    # 解析 save
    if 'save' in raw_config:
        config.save = SaveConfig(**raw_config['save'])
    
    # 解析 loss
    if 'loss' in raw_config:
        config.loss = LossConfig(**raw_config['loss'])
    
    # 解析 stage
    if 'stage' in raw_config:
        config.stage = StageConfig(**raw_config['stage'])
    
    return config


if __name__ == '__main__':
    # 测试配置加载
    import sys
    if len(sys.argv) > 1:
        config = load_and_validate_config(sys.argv[1])
        print(f"配置加载成功: {config}")
    else:
        print("用法: python config_utils.py <config.yaml>")
