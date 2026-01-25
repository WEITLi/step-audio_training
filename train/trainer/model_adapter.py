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
        encoder_path=cfg.model_tokenizer.model_path,
        model_source=ModelSource.LOCAL
    )
    
    # 初始化 TTS 模型获取 LLM
    tts_model = StepAudioTTS(
        model_path=cfg.model_llm.model_path,
        audio_tokenizer=tokenizer,
        model_source=ModelSource.LOCAL
    )
    
    llm = tts_model.llm
    
    # 确保模型配置正确
    if hasattr(llm, 'config'):
        # 设置一些默认配置以避免运行时错误
        if not hasattr(llm.config, 'use_cache'):
            llm.config.use_cache = False
        else:
            llm.config.use_cache = False  # 训练时禁用缓存
        
        # 确保其他必要的配置存在
        if not hasattr(llm.config, 'output_hidden_states'):
            llm.config.output_hidden_states = False
        if not hasattr(llm.config, 'output_attentions'):
            llm.config.output_attentions = False
    
    print(f"LLM 配置: use_cache={getattr(llm.config, 'use_cache', 'N/A')}")
    
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
        
        # 确保 PEFT 模型也有正确的配置
        if hasattr(llm, 'config'):
            llm.config.use_cache = False
    
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
    
    return llm, tts_model


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
    
    # 全量微调策略 - 不冻结任何部分
    print("Flow 模型采用全量微调，所有参数保持可训练")
    
    # 确保所有参数都是可训练的
    for param in flow_model.parameters():
        param.requires_grad = True
    
    # 修复 Flow 模型的梯度问题
    print("修复 Flow 模型的梯度连接问题...")
    _patch_flow_forward(flow_model)
    
    # 统计参数
    total_params = sum(p.numel() for p in flow_model.parameters())
    trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"Flow 模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练比例: {trainable_params/total_params*100:.1f}%")
    
    # 切换为训练态
    flow_model.train()
    
    # 打印模型信息
    print_model_info(flow_model, "Flow Model")
    
    return flow_model, cosy_model


def _patch_flow_forward(flow_model):
    """修补 Flow 模型的 forward 方法以保持梯度连接"""
    
    from stepvocoder.cosyvoice2.utils.mask import make_pad_mask
    import torch.nn.functional as F
    
    # 保存原始方法的引用
    original_forward = flow_model.__class__.forward
    
    def patched_forward(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """修补后的 forward 方法，使用 compute_loss 进行训练"""
        assert token.shape[0] == 1
        
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        
        # concat text and prompt_text
        token_len = prompt_token_len + token_len
        token = torch.concat([prompt_token, token], dim=1)
        
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        
        # token encode
        h, _ = self.encoder.forward(token, token_len)
        h = self.encoder_proj(h)
        
        # condition
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - prompt_feat.shape[1]
        
        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2).contiguous()
        
        # 创建目标特征 (非 prompt 部分)
        target_feat = prompt_feat[:, mel_len1:]  # 跳过 prompt 部分
        
        # 修复：使用正确的 mel 长度创建 mask
        total_mel_len = torch.tensor([mel_len2], dtype=torch.int32, device=h.device)
        mask = (~make_pad_mask(total_mel_len)).to(h)
        
        # 使用 compute_loss 进行训练（这是正确的训练方式）
        loss, _ = self.decoder.compute_loss(
            target_feat.transpose(1, 2).contiguous(),  # (batch, time, freq) -> (batch, freq, time)
            mask.unsqueeze(1),
            h[:, mel_len1:].transpose(1, 2).contiguous(),  # 只使用非 prompt 部分
            embedding,
            cond=conds[:, :, mel_len1:],  # 只使用非 prompt 部分的条件
            streaming=False
        )
        
        return loss
    
    # 应用修补
    flow_model.__class__.forward = patched_forward
    print("✅ Flow 模型梯度连接已修复")


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


def prepare_models_for_training(cfg) -> Tuple[nn.Module, nn.Module, object, object]:
    """
    准备所有模型用于训练
    
    Args:
        cfg: 训练配置
        
    Returns:
        (llm, flow, cosy_model, text_tokenizer) 元组
    """
    # 初始化 LLM (现在返回 LLM 和 TTS 模型)
    llm, tts_model = init_llm_for_finetune(cfg)
    llm = llm.to(cfg.basic.device)
    
    # 初始化 Flow
    flow, cosy_model = init_flow_for_finetune(cfg)
    flow = flow.to(cfg.basic.device)
    
    # 从 TTS 模型中获取文本 tokenizer
    text_tokenizer = tts_model.tokenizer
    
    return llm, flow, cosy_model, text_tokenizer
