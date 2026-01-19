"""
训练循环核心逻辑
实现分阶段训练：LLM 单独微调 → Flow 单独微调 → 联合微调
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.utils.config_utils import TrainConfig
from train.utils.train_utils import (
    set_seed, save_ckpt, load_ckpt, clip_grad_norm,
    AverageMeter, get_lr, setup_logging, print_model_info
)
from train.dataset.audio_edit_dataset import build_dataloader
from train.trainer.model_adapter import prepare_models_for_training

logger = logging.getLogger(__name__)


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup 学习率调度器
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            scale = self._step_count / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        return self.base_lrs


def compute_llm_loss(
    llm: nn.Module,
    output_ids: torch.Tensor,
    target_tokens: torch.Tensor,
    target_token_len: torch.Tensor
) -> torch.Tensor:
    """
    计算 LLM 损失（token 交叉熵）
    
    Args:
        llm: LLM 模型
        output_ids: 模型输出 logits
        target_tokens: 目标 token
        target_token_len: 目标 token 长度
        
    Returns:
        损失值
    """
    # 展平为 (B*T, V) 和 (B*T,)
    batch_size, seq_len, vocab_size = output_ids.shape
    output_flat = output_ids.reshape(-1, vocab_size)
    target_flat = target_tokens.reshape(-1)
    
    # 忽略 padding 位置
    loss = torch.nn.functional.cross_entropy(
        output_flat, 
        target_flat,
        ignore_index=0,  # 假设 0 是 padding token
        reduction='mean'
    )
    
    return loss


def compute_flow_loss(
    flow_output: torch.Tensor,
    target_mel: torch.Tensor,
    target_mel_len: torch.Tensor
) -> torch.Tensor:
    """
    计算 Flow 损失（mel MSE）
    
    Args:
        flow_output: Flow 模型输出的 mel
        target_mel: 目标 mel
        target_mel_len: 目标 mel 长度
        
    Returns:
        损失值
    """
    # MSE 损失
    mse_loss = torch.nn.functional.mse_loss(
        flow_output, 
        target_mel[:, :, :flow_output.shape[-1]],
        reduction='mean'
    )
    
    return mse_loss


def train_stage1_llm(
    llm: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: TrainConfig,
    start_epoch: int = 0
) -> Tuple[nn.Module, float]:
    """
    阶段 1: 单独微调 LLM
    
    Args:
        llm: LLM 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        cfg: 训练配置
        start_epoch: 起始 epoch
        
    Returns:
        (训练后的 LLM, 最佳验证损失)
    """
    print("\n" + "="*60)
    print("阶段 1: 单独微调 LLM (LoRA)")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    device = cfg.basic.device
    
    for epoch in range(start_epoch, cfg.stage.stage1_epochs):
        llm.train()
        loss_meter = AverageMeter("LLM Loss")
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f"Epoch {epoch+1}/{cfg.stage.stage1_epochs}")
        
        for step, batch in pbar:
            # 数据移动到设备
            prompt_tokens = batch['prompt_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            target_token_len = batch['target_token_len'].to(device)
            
            # 前向传播
            output_ids = llm(prompt_tokens)
            
            # 处理输出格式（如果是 CausalLMOutput）
            if hasattr(output_ids, 'logits'):
                output_ids = output_ids.logits
            
            # 计算损失
            loss = compute_llm_loss(llm, output_ids, target_tokens, target_token_len)
            
            # 梯度累积
            loss = loss / cfg.optim.accum_grad
            loss.backward()
            
            if (step + 1) % cfg.optim.accum_grad == 0:
                # 梯度裁剪
                clip_grad_norm(llm.parameters(), cfg.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 更新统计
            loss_meter.update(loss.item() * cfg.optim.accum_grad)
            
            # 日志
            if (step + 1) % cfg.basic.log_interval == 0:
                lr = get_lr(optimizer)[0]
                pbar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    'lr': f"{lr:.2e}"
                })
        
        # 验证
        if (epoch + 1) % cfg.basic.val_interval == 0:
            val_loss = validate_llm(llm, val_loader, device)
            logger.info(f"Epoch {epoch+1}, Val LLM Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            save_ckpt(
                llm, optimizer, epoch, val_loss, 
                cfg.save.ckpt_dir, "llm", scheduler, is_best
            )
    
    print(f"\n阶段 1 完成！最佳验证损失: {best_val_loss:.4f}\n")
    return llm, best_val_loss


def train_stage2_flow(
    flow: nn.Module,
    llm: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: TrainConfig,
    start_epoch: int = 0
) -> Tuple[nn.Module, float]:
    """
    阶段 2: 单独微调 Flow（使用阶段 1 训练好的 LLM 生成 token）
    
    Args:
        flow: Flow 模型
        llm: LLM 模型（冻结）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        cfg: 训练配置
        start_epoch: 起始 epoch
        
    Returns:
        (训练后的 Flow, 最佳验证损失)
    """
    print("\n" + "="*60)
    print("阶段 2: 单独微调 Flow (解码器)")
    print("="*60 + "\n")
    
    # 冻结 LLM
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False
    
    best_val_loss = float('inf')
    device = cfg.basic.device
    
    for epoch in range(start_epoch, cfg.stage.stage2_epochs):
        flow.train()
        loss_meter = AverageMeter("Flow Loss")
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{cfg.stage.stage2_epochs}")
        
        for step, batch in pbar:
            # 数据移动到设备
            prompt_tokens = batch['prompt_tokens'].to(device)
            prompt_token_len = batch['prompt_token_len'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            target_token_len = batch['target_token_len'].to(device)
            prompt_mel = batch['prompt_mel'].to(device)
            prompt_mel_len = batch['prompt_mel_len'].to(device)
            target_mel = batch['target_mel'].to(device)
            target_mel_len = batch['target_mel_len'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            
            # 使用冻结的 LLM 生成 token（可选，也可直接使用 target_tokens）
            # 这里直接使用 target_tokens 作为监督信号
            
            # Flow 前向传播
            # 调用训练态 forward（需要在 flow.py 中实现）
            flow_output = flow.forward(
                token=target_tokens,
                token_len=target_token_len,
                prompt_token=prompt_tokens,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_mel.transpose(1, 2),  # (B, T, F) -> (B, T, F)
                prompt_feat_len=prompt_mel_len,
                embedding=speaker_embedding,
                n_timesteps=cfg.model_flow.n_timesteps
            )
            
            # 计算损失
            loss = compute_flow_loss(flow_output, target_mel, target_mel_len)
            
            # 梯度累积
            loss = loss / cfg.optim.accum_grad
            loss.backward()
            
            if (step + 1) % cfg.optim.accum_grad == 0:
                clip_grad_norm(flow.parameters(), cfg.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 更新统计
            loss_meter.update(loss.item() * cfg.optim.accum_grad)
            
            # 日志
            if (step + 1) % cfg.basic.log_interval == 0:
                lr = get_lr(optimizer)[0]
                pbar.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    'lr': f"{lr:.2e}"
                })
        
        # 验证
        if (epoch + 1) % cfg.basic.val_interval == 0:
            val_loss = validate_flow(flow, llm, val_loader, device, cfg)
            logger.info(f"Epoch {epoch+1}, Val Flow Loss: {val_loss:.4f}")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            save_ckpt(
                flow, optimizer, epoch, val_loss,
                cfg.save.ckpt_dir, "flow", scheduler, is_best
            )
    
    print(f"\n阶段 2 完成！最佳验证损失: {best_val_loss:.4f}\n")
    return flow, best_val_loss


def train_stage3_joint(
    llm: nn.Module,
    flow: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: TrainConfig,
) -> Tuple[nn.Module, nn.Module, float]:
    """
    阶段 3: LLM + Flow 联合微调（可选，使用较小学习率）
    
    Args:
        llm: LLM 模型
        flow: Flow 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        cfg: 训练配置
        
    Returns:
        (训练后的 LLM, 训练后的 Flow, 最佳验证损失)
    """
    print("\n" + "="*60)
    print("阶段 3: LLM + Flow 联合微调")
    print("="*60 + "\n")
    
    # 降低学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= cfg.stage.joint_lr_scale
    
    # 解冻 LLM LoRA 层
    for name, param in llm.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    best_val_loss = float('inf')
    device = cfg.basic.device
    
    for epoch in range(cfg.stage.stage3_epochs):
        llm.train()
        flow.train()
        
        llm_loss_meter = AverageMeter("LLM Loss")
        flow_loss_meter = AverageMeter("Flow Loss")
        total_loss_meter = AverageMeter("Total Loss")
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Joint Epoch {epoch+1}/{cfg.stage.stage3_epochs}")
        
        for step, batch in pbar:
            # 数据移动到设备
            prompt_tokens = batch['prompt_tokens'].to(device)
            prompt_token_len = batch['prompt_token_len'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            target_token_len = batch['target_token_len'].to(device)
            prompt_mel = batch['prompt_mel'].to(device)
            prompt_mel_len = batch['prompt_mel_len'].to(device)
            target_mel = batch['target_mel'].to(device)
            target_mel_len = batch['target_mel_len'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            
            # LLM 前向
            llm_output = llm(prompt_tokens)
            if hasattr(llm_output, 'logits'):
                llm_output = llm_output.logits
            llm_loss = compute_llm_loss(llm, llm_output, target_tokens, target_token_len)
            
            # Flow 前向
            flow_output = flow.forward(
                token=target_tokens,
                token_len=target_token_len,
                prompt_token=prompt_tokens,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_mel.transpose(1, 2),
                prompt_feat_len=prompt_mel_len,
                embedding=speaker_embedding,
                n_timesteps=cfg.model_flow.n_timesteps
            )
            flow_loss = compute_flow_loss(flow_output, target_mel, target_mel_len)
            
            # 总损失
            total_loss = (cfg.loss.llm_weight * llm_loss + 
                         cfg.loss.flow_weight * flow_loss)
            
            # 梯度累积
            total_loss = total_loss / cfg.optim.accum_grad
            total_loss.backward()
            
            if (step + 1) % cfg.optim.accum_grad == 0:
                clip_grad_norm(list(llm.parameters()) + list(flow.parameters()), 
                             cfg.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 更新统计
            llm_loss_meter.update(llm_loss.item())
            flow_loss_meter.update(flow_loss.item())
            total_loss_meter.update(total_loss.item() * cfg.optim.accum_grad)
            
            if (step + 1) % cfg.basic.log_interval == 0:
                pbar.set_postfix({
                    'llm': f"{llm_loss_meter.avg:.4f}",
                    'flow': f"{flow_loss_meter.avg:.4f}",
                    'total': f"{total_loss_meter.avg:.4f}"
                })
        
        # 验证
        val_loss = total_loss_meter.avg  # 简化，使用训练损失
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # 保存
        save_ckpt(llm, optimizer, epoch, val_loss, cfg.save.ckpt_dir, "llm_joint", is_best=is_best)
        save_ckpt(flow, optimizer, epoch, val_loss, cfg.save.ckpt_dir, "flow_joint", is_best=is_best)
    
    print(f"\n阶段 3 完成！最佳验证损失: {best_val_loss:.4f}\n")
    return llm, flow, best_val_loss


def validate_llm(
    llm: nn.Module, 
    val_loader: DataLoader, 
    device: str
) -> float:
    """LLM 验证"""
    llm.eval()
    loss_meter = AverageMeter("Val LLM Loss")
    
    with torch.no_grad():
        for batch in val_loader:
            prompt_tokens = batch['prompt_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            target_token_len = batch['target_token_len'].to(device)
            
            output_ids = llm(prompt_tokens)
            if hasattr(output_ids, 'logits'):
                output_ids = output_ids.logits
            
            loss = compute_llm_loss(llm, output_ids, target_tokens, target_token_len)
            loss_meter.update(loss.item())
    
    return loss_meter.avg


def validate_flow(
    flow: nn.Module, 
    llm: nn.Module,
    val_loader: DataLoader, 
    device: str,
    cfg: TrainConfig
) -> float:
    """Flow 验证"""
    flow.eval()
    loss_meter = AverageMeter("Val Flow Loss")
    
    with torch.no_grad():
        for batch in val_loader:
            prompt_tokens = batch['prompt_tokens'].to(device)
            prompt_token_len = batch['prompt_token_len'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            target_token_len = batch['target_token_len'].to(device)
            prompt_mel = batch['prompt_mel'].to(device)
            prompt_mel_len = batch['prompt_mel_len'].to(device)
            target_mel = batch['target_mel'].to(device)
            target_mel_len = batch['target_mel_len'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            
            flow_output = flow.forward(
                token=target_tokens,
                token_len=target_token_len,
                prompt_token=prompt_tokens,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_mel.transpose(1, 2),
                prompt_feat_len=prompt_mel_len,
                embedding=speaker_embedding,
                n_timesteps=cfg.model_flow.n_timesteps
            )
            
            loss = compute_flow_loss(flow_output, target_mel, target_mel_len)
            loss_meter.update(loss.item())
    
    return loss_meter.avg


def train_llm_flow(cfg: TrainConfig):
    """
    主训练函数：分阶段训练 LLM 和 Flow
    
    Args:
        cfg: 训练配置
        
    Returns:
        (llm, flow) 训练后的模型
    """
    # 设置随机种子
    set_seed(cfg.basic.seed)
    
    # 设置日志
    setup_logging(cfg.save.ckpt_dir)
    
    logger.info("="*60)
    logger.info("Step-Audio-EditX LLM + Flow 微调训练")
    logger.info("="*60)
    
    # 创建保存目录
    os.makedirs(cfg.save.ckpt_dir, exist_ok=True)
    
    # 准备模型
    logger.info("初始化模型...")
    llm, flow, cosy_model = prepare_models_for_training(cfg)
    
    # 构建数据加载器
    logger.info("构建数据加载器...")
    train_loader = build_dataloader(
        cfg.data.train_path,
        cfg.data.batch_size,
        cfg.data.sample_rate,
        cfg.data.max_audio_len,
        cfg.data.max_text_len,
        cfg.model_llm.model_path,
        cfg.data.num_workers,
        train=True
    )
    
    val_loader = build_dataloader(
        cfg.data.val_path,
        cfg.data.batch_size,
        cfg.data.sample_rate,
        cfg.data.max_audio_len,
        cfg.data.max_text_len,
        cfg.model_llm.model_path,
        cfg.data.num_workers,
        train=False
    )
    
    # 初始化优化器（分 LLM/Flow 设置不同学习率）
    optimizer = optim.AdamW([
        {"params": llm.parameters(), "lr": cfg.optim.lr_llm},
        {"params": flow.parameters(), "lr": cfg.optim.lr_flow}
    ], weight_decay=cfg.optim.weight_decay)
    
    # 学习率调度器
    scheduler = WarmupLR(optimizer, cfg.optim.warmup_steps)
    
    # 阶段 1: 单独微调 LLM
    llm, _ = train_stage1_llm(
        llm, train_loader, val_loader,
        optimizer, scheduler, cfg
    )
    
    # 阶段 2: 单独微调 Flow
    flow, _ = train_stage2_flow(
        flow, llm, train_loader, val_loader,
        optimizer, scheduler, cfg
    )
    
    # 阶段 3: 联合微调（可选）
    if cfg.stage.stage3_epochs > 0:
        llm, flow, _ = train_stage3_joint(
            llm, flow, train_loader, val_loader,
            optimizer, scheduler, cfg
        )
    
    logger.info("="*60)
    logger.info("训练完成！")
    logger.info(f"LLM checkpoint: {cfg.save.ckpt_dir}/llm_best.pt")
    logger.info(f"Flow checkpoint: {cfg.save.ckpt_dir}/flow_best.pt")
    logger.info("="*60)
    
    return llm, flow
