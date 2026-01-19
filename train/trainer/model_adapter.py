"""
模型适配器
用于初始化 LLM (LoRA 微调) 和 Flow (解码器微调) 模型
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.utils.train_utils import print_model_info


def init_llm_for_finetune(cfg) -> nn.Module:
    """
    初始化 LLM 模型用于 LoRA 微调
    
    Args:
        cfg: 训练配置对象
        
    Returns:
        带 LoRA 适配器的 LLM 模型
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("请安装 peft: pip install peft")
    
    # 加载 StepAudioEditX 原有 LLM
    from tts import StepAudioTTS
    from tokenizer import StepAudioTokenizer
    from model_loader import ModelSource
    
    print(f"加载 LLM 模型: {cfg.model_llm.model_path}")
    
    # 初始化 tokenizer（必需）
    tokenizer = StepAudioTokenizer(
        encoder_path=cfg.model_llm.model_path,
        model_source=ModelSource.LOCAL
    )
    
    # 初始化 TTS 模型获取 LLM
    tts_model = StepAudioTTS(
        model_path=cfg.model_llm.model_path,
        audio_tokenizer=tokenizer,
        model_source=ModelSource.LOCAL
    )
    
    llm = tts_model.llm
    
    # 配置 LoRA
    if cfg.model_llm.lora:
        print("配置 LoRA 适配器...")
        lora_config = LoraConfig(
            r=cfg.model_llm.lora_r,
            lora_alpha=cfg.model_llm.lora_alpha,
            lora_dropout=cfg.model_llm.lora_dropout,
            target_modules=cfg.model_llm.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        llm = get_peft_model(llm, lora_config)
        print("LoRA 适配器已添加")
    
    # 冻结主干参数，仅训练 LoRA
    if cfg.model_llm.freeze_backbone:
        for name, param in llm.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        print("LLM 主干参数已冻结，仅训练 LoRA 层")
    
    # 切换为训练态
    llm.train()
    
    # 打印模型信息
    print_model_info(llm, "LLM (with LoRA)")
    
    return llm


def init_flow_for_finetune(cfg) -> nn.Module:
    """
    初始化 Flow 模型用于解码器微调
    
    Args:
        cfg: 训练配置对象
        
    Returns:
        Flow 模型（解码器可训练）
    """
    from stepvocoder.cosyvoice2.cli.cosyvoice import CosyVoice
    
    print(f"加载 Flow 模型: {cfg.model_flow.model_path}")
    
    # 加载 CosyVoice 模型
    cosy_model = CosyVoice(cfg.model_flow.model_path)
    flow_model = cosy_model.cosy_impl.flow
    
    # 冻结编码器
    if cfg.model_flow.freeze_encoder:
        if hasattr(flow_model, 'encoder'):
            for param in flow_model.encoder.parameters():
                param.requires_grad = False
            print("Flow 编码器已冻结")
        
        if hasattr(flow_model, 'input_embedding'):
            for param in flow_model.input_embedding.parameters():
                param.requires_grad = False
            print("Flow input_embedding 已冻结")
        
        if hasattr(flow_model, 'encoder_proj'):
            for param in flow_model.encoder_proj.parameters():
                param.requires_grad = False
            print("Flow encoder_proj 已冻结")
    
    # 解冻解码器
    if hasattr(flow_model, 'decoder'):
        for param in flow_model.decoder.parameters():
            param.requires_grad = True
        print("Flow 解码器已解冻，可训练")
    
    # 切换为训练态
    flow_model.train()
    
    # 打印模型信息
    print_model_info(flow_model, "Flow Model")
    
    return flow_model, cosy_model


def init_vocoder_for_finetune(
    cfg, 
    freeze: bool = True
) -> Optional[nn.Module]:
    """
    初始化声码器（通常保持冻结）
    
    Args:
        cfg: 训练配置对象
        freeze: 是否冻结
        
    Returns:
        声码器模型（如果需要）
    """
    if cfg.model_flow.freeze_vocoder:
        print("声码器已冻结，不参与训练")
        return None
    
    # 如果需要微调声码器，可以在这里添加逻辑
    # 目前按照规范，声码器保持冻结
    return None


def load_llm_lora_weights(
    llm: nn.Module,
    lora_ckpt_path: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    加载 LLM LoRA 权重
    
    Args:
        llm: LLM 模型
        lora_ckpt_path: LoRA checkpoint 路径
        device: 设备
        
    Returns:
        加载权重后的 LLM 模型
    """
    try:
        from peft import PeftModel
        
        if hasattr(llm, 'base_model'):
            # 已经是 PEFT 模型，直接加载权重
            ckpt = torch.load(lora_ckpt_path, map_location=device)
            llm.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            # 从基础模型加载
            llm = PeftModel.from_pretrained(llm, lora_ckpt_path)
        
        print(f"LLM LoRA 权重加载成功: {lora_ckpt_path}")
        
    except Exception as e:
        print(f"LLM LoRA 权重加载失败: {e}")
        raise
    
    return llm


def load_flow_weights(
    flow: nn.Module,
    flow_ckpt_path: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    加载 Flow 模型权重
    
    Args:
        flow: Flow 模型
        flow_ckpt_path: checkpoint 路径
        device: 设备
        
    Returns:
        加载权重后的 Flow 模型
    """
    ckpt = torch.load(flow_ckpt_path, map_location=device)
    flow.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Flow 权重加载成功: {flow_ckpt_path}")
    return flow


def prepare_models_for_training(cfg) -> Tuple[nn.Module, nn.Module, object]:
    """
    准备所有模型用于训练
    
    Args:
        cfg: 训练配置
        
    Returns:
        (llm, flow, cosy_model) 元组
    """
    # 初始化 LLM
    llm = init_llm_for_finetune(cfg)
    llm = llm.to(cfg.basic.device)
    
    # 初始化 Flow
    flow, cosy_model = init_flow_for_finetune(cfg)
    flow = flow.to(cfg.basic.device)
    
    return llm, flow, cosy_model
