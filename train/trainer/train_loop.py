"""
简化的训练循环实现
使用新的 parquet 数据格式和 CosyVoice-style data pipeline
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.dataset.dataset import Dataset
from train.dataset import processor
from train.utils.train_utils import (
    set_seed, save_ckpt, load_ckpt, clip_grad_norm,
    AverageMeter, get_lr, setup_logging, count_parameters
)
from train.trainer.model_adapter import prepare_models_for_training
from tokenizer import StepAudioTokenizer
from model_loader import ModelSource

logger = logging.getLogger(__name__)


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup learning rate scheduler"""
    
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)
    
    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps 
                    for base_lr in self.base_lrs]
        return self.base_lrs


def build_dataloader(cfg, tokenizer, train=True):
    """构建 DataLoader
    
    Args:
        cfg: 配置对象
        tokenizer: Qwen tokenizer
        train: 是否为训练集
        
    Returns:
        DataLoader
    """
    data_file = cfg.data.train_data if train else cfg.data.cv_data
    
    # 构建 mel spectrogram 提取器
    import torchaudio
    feat_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.data.sample_rate,
        n_fft=1920,
        hop_length=480,
        win_length=1920,
        n_mels=80,
        f_min=0,
        f_max=8000,
        center=False
    )
    
    # Data pipeline
    data_pipeline = [
        processor.parquet_opener,
        lambda data, mode: processor.tokenize(data, tokenizer, mode),
        lambda data, mode: processor.filter(data, **cfg.data.filter, mode=mode),
        lambda data, mode: processor.resample(data, cfg.data.sample_rate, mode=mode),
        lambda data, mode: processor.compute_fbank(data, feat_extractor, cfg.data.token_mel_ratio, mode=mode),
        lambda data, mode: processor.parse_embedding(data, normalize=True, mode=mode),
        lambda data, mode: processor.shuffle(data, cfg.data.shuffle_size, mode=mode) if train else data,
        lambda data, mode: processor.sort(data, cfg.data.sort_size, mode=mode),
        lambda data, mode: processor.dynamic_batch(data, cfg.data.max_frames_in_batch, mode=mode),
        lambda data, mode: processor.padding(data, cfg.data.use_spk_embedding, mode=mode),
    ]
    
    # 创建 dataset
    dataset = Dataset(
        data_file,
        data_pipeline,
        mode='train' if train else 'eval',
        shuffle=train,
        partition=True
    )
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # batch 已经在 pipeline 中完成
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=cfg.data.pin_memory
    )
    
    return dataloader


def compute_llm_loss(llm, batch, device):
    """计算 LLM 损失
    
    Args:
        llm: LLM 模型
        batch: 数据 batch
        device: 设备
        
    Returns:
        loss: 交叉熵损失
    """
    text_token = batch['text_token'].to(device)
    text_token_len = batch['text_token_len'].to(device)
    speech_token = batch['speech_token'].to(device)
    speech_token_len = batch['speech_token_len'].to(device)
    embedding = batch['embedding'].to(device)
    
    # Forward pass
    # TODO: 这里需要根据实际的 LLM forward 接口调整
    # 假设 LLM 接受 (text_token, embedding) 并输出 logits
    logits = llm(
        input_ids=text_token,
        attention_mask=(text_token != 0).long(),
    ).logits
    
    # 计算 cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=0)
    
    # 只计算有效 token 的损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = speech_token[..., 1:].contiguous()
    
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def compute_flow_loss(flow, batch, device):
    """计算 Flow 损失
    
    Args:
        flow: Flow 模型
        batch: 数据 batch
        device: 设备
        
    Returns:
        loss: MSE 损失
    """
    speech_token = batch['speech_token'].to(device)
    speech_token_len = batch['speech_token_len'].to(device)
    speech_feat = batch['speech_feat'].to(device)
    speech_feat_len = batch['speech_feat_len'].to(device)
    embedding = batch['embedding'].to(device)
    
    # Forward pass
    # 使用 Flow 模型的 forward 方法（已添加）
    pred_feat = flow.forward(
        token=speech_token,
        token_len=speech_token_len,
        prompt_token=speech_token[:, :10],  # dummy prompt
        prompt_token_len=torch.full_like(speech_token_len, 10),
        prompt_feat=speech_feat[:, :20],  # dummy prompt
        prompt_feat_len=torch.full_like(speech_feat_len, 20),
        embedding=embedding,
        n_timesteps=10
    )
    
    # 计算 MSE loss
    loss = nn.functional.mse_loss(pred_feat, speech_feat)
    
    return loss


def validate(model, dataloader, compute_loss_fn, device, model_name):
    """验证函数
    
    Args:
        model: 模型
        dataloader: 验证数据加载器
        compute_loss_fn: 损失计算函数
        device: 设备
        model_name: 模型名称
        
    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    val_loss = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating {model_name}", leave=False):
            loss = compute_loss_fn(model, batch, device)
            val_loss.update(loss.item(), n=len(batch['utts']))
    
    model.train()
    return val_loss.avg


def train_llm_only(llm, train_loader, val_loader, optimizer, scheduler, cfg):
    """仅训练 LLM (LoRA)
    
    Args:
        llm: LLM 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        cfg: 配置
        
    Returns:
        llm: 训练后的 LLM
        best_val_loss: 最佳验证损失
    """
    logger.info("=" * 60)
    logger.info("开始训练 LLM (LoRA)")
    logger.info("=" * 60)
    
    device = cfg.basic.device
    best_val_loss = float('inf')
    
    for epoch in range(cfg.stage.stage1_epochs):
        llm.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.stage.stage1_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Forward
            loss = compute_llm_loss(llm, batch, device)
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg.optim.accum_grad == 0:
                clip_grad_norm(llm.parameters(), cfg.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss.update(loss.item(), n=len(batch['utts']))
            
            # Log
            if (batch_idx + 1) % cfg.basic.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{train_loss.avg:.4f}',
                    'lr': f'{get_lr(optimizer):.2e}'
                })
        
        # Validation
        val_loss = validate(llm, val_loader, compute_llm_loss, device, "LLM")
        
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss.avg:.4f}, val_loss={val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(
                llm, optimizer, scheduler, epoch, best_val_loss,
                os.path.join(cfg.save.ckpt_dir, "llm_best.pt")
            )
            logger.info(f"保存最佳 LLM checkpoint (val_loss={val_loss:.4f})")
        
        if (epoch + 1) % cfg.save.save_interval == 0:
            save_ckpt(
                llm, optimizer, scheduler, epoch, best_val_loss,
                os.path.join(cfg.save.ckpt_dir, f"llm_epoch_{epoch+1}.pt")
            )
    
    return llm, best_val_loss


def train_flow_only(flow, llm, train_loader, val_loader, optimizer, scheduler, cfg):
    """仅训练 Flow (解码器)
    
    Args:
        flow: Flow 模型
        llm: LLM 模型 (冻结，用于生成 token)
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        cfg: 配置
        
    Returns:
        flow: 训练后的 Flow
        best_val_loss: 最佳验证损失
    """
    logger.info("=" * 60)
    logger.info("开始训练 Flow (解码器)")
    logger.info("=" * 60)
    
    device = cfg.basic.device
    best_val_loss = float('inf')
    
    # 冻结 LLM
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False
    
    for epoch in range(cfg.stage.stage2_epochs):
        flow.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.stage.stage2_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Forward
            loss = compute_flow_loss(flow, batch, device)
            
            # Backward
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg.optim.accum_grad == 0:
                clip_grad_norm(flow.parameters(), cfg.optim.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss.update(loss.item(), n=len(batch['utts']))
            
            # Log
            if (batch_idx + 1) % cfg.basic.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{train_loss.avg:.4f}',
                    'lr': f'{get_lr(optimizer):.2e}'
                })
        
        # Validation
        val_loss = validate(flow, val_loader, compute_flow_loss, device, "Flow")
        
        logger.info(f"Epoch {epoch+1}: train_loss={train_loss.avg:.4f}, val_loss={val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(
                flow, optimizer, scheduler, epoch, best_val_loss,
                os.path.join(cfg.save.ckpt_dir, "flow_best.pt")
            )
            logger.info(f"保存最佳 Flow checkpoint (val_loss={val_loss:.4f})")
        
        if (epoch + 1) % cfg.save.save_interval == 0:
            save_ckpt(
                flow, optimizer, scheduler, epoch, best_val_loss,
                os.path.join(cfg.save.ckpt_dir, f"flow_epoch_{epoch+1}.pt")
            )
    
    return flow, best_val_loss


def train_llm_flow(cfg):
    """主训练函数
    
    Args:
        cfg: 训练配置
        
    Returns:
        (llm, flow): 训练后的模型
    """
    # 设置随机种子
    set_seed(cfg.basic.seed)
    
    # 设置日志
    setup_logging(cfg.save.ckpt_dir)
    
    logger.info("=" * 60)
    logger.info("Step-Audio-EditX LLM + Flow 微调训练")
    logger.info(f"训练模式: {cfg.basic.train_mode.upper()}")
    logger.info("=" * 60)
    
    # 创建保存目录
    os.makedirs(cfg.save.ckpt_dir, exist_ok=True)
    
    # 初始化模型
    logger.info("初始化模型...")
    llm, flow, cosy_model = prepare_models_for_training(cfg)
    
    # 初始化 tokenizer
    tokenizer = StepAudioTokenizer(
        encoder_path=cfg.model_llm.model_path,
        model_source=ModelSource.LOCAL
    ).tokenizer  # 获取 Qwen tokenizer
    
    # 构建数据加载器
    logger.info("构建数据加载器...")
    train_loader = build_dataloader(cfg, tokenizer, train=True)
    val_loader = build_dataloader(cfg, tokenizer, train=False)
    
    # 根据训练模式选择
    train_mode = cfg.basic.train_mode.lower()
    
    if train_mode == 'llm':
        # 仅训练 LLM
        optimizer = optim.AdamW(llm.parameters(), lr=cfg.optim.lr_llm, weight_decay=cfg.optim.weight_decay)
        scheduler = WarmupLR(optimizer, cfg.optim.warmup_steps)
        llm, _ = train_llm_only(llm, train_loader, val_loader, optimizer, scheduler, cfg)
        
    elif train_mode == 'flow':
        # 仅训练 Flow
        optimizer = optim.AdamW(flow.parameters(), lr=cfg.optim.lr_flow, weight_decay=cfg.optim.weight_decay)
        scheduler = WarmupLR(optimizer, cfg.optim.warmup_steps)
        flow, _ = train_flow_only(flow, llm, train_loader, val_loader, optimizer, scheduler, cfg)
        
    else:  # both
        # 分阶段训练
        # Stage 1: LLM
        optimizer_llm = optim.AdamW(llm.parameters(), lr=cfg.optim.lr_llm, weight_decay=cfg.optim.weight_decay)
        scheduler_llm = WarmupLR(optimizer_llm, cfg.optim.warmup_steps)
        llm, _ = train_llm_only(llm, train_loader, val_loader, optimizer_llm, scheduler_llm, cfg)
        
        # Stage 2: Flow
        optimizer_flow = optim.AdamW(flow.parameters(), lr=cfg.optim.lr_flow, weight_decay=cfg.optim.weight_decay)
        scheduler_flow = WarmupLR(optimizer_flow, cfg.optim.warmup_steps)
        flow, _ = train_flow_only(flow, llm, train_loader, val_loader, optimizer_flow, scheduler_flow, cfg)
    
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)
    
    return llm, flow
