"""
训练工具函数
包含权重保存/加载、梯度裁剪、EMA等功能
复用 CosyVoice 训练逻辑
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    设置随机种子，确保训练可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子设置为: {seed}")


def save_ckpt(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    ckpt_dir: str,
    model_name: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    is_best: bool = False
) -> str:
    """
    保存训练 checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        val_loss: 验证集损失
        ckpt_dir: 保存目录
        model_name: 模型名称 (llm/flow)
        scheduler: 学习率调度器
        is_best: 是否为最优模型
        
    Returns:
        保存的 checkpoint 路径
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存普通 checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_epoch_{epoch}.pt")
    torch.save(ckpt, ckpt_path)
    logger.info(f"保存 checkpoint: {ckpt_path}")
    
    # 如果是最优模型，额外保存一份
    if is_best:
        best_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
        torch.save(ckpt, best_path)
        logger.info(f"保存最优模型: {best_path}")
    
    return ckpt_path


def load_ckpt(
    model: nn.Module,
    ckpt_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    加载训练 checkpoint
    
    Args:
        model: 模型
        ckpt_path: checkpoint 路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
        
    Returns:
        checkpoint 信息字典
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 加载模型权重
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    logger.info(f"加载模型权重: {ckpt_path}")
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        logger.info("加载优化器状态")
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info("加载调度器状态")
    
    return {
        'epoch': ckpt.get('epoch', 0),
        'val_loss': ckpt.get('val_loss', float('inf')),
    }


def clip_grad_norm(
    parameters,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    梯度裁剪
    
    Args:
        parameters: 模型参数
        max_norm: 最大梯度范数
        norm_type: 范数类型
        
    Returns:
        裁剪前的梯度范数
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return total_norm.item()


class EMA:
    """
    指数移动平均 (Exponential Moving Average)
    用于平滑模型权重，提高泛化能力
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: 模型
            decay: 衰减系数
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """更新影子权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self) -> None:
        """应用影子权重到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self) -> None:
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量
    
    Args:
        model: 模型
        trainable_only: 是否只统计可训练参数
        
    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    打印模型信息
    
    Args:
        model: 模型
        model_name: 模型名称
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\n{'='*50}")
    print(f"{model_name} 信息:")
    print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  冻结参数量: {total_params - trainable_params:,}")
    print(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
    print(f"{'='*50}\n")


class AverageMeter:
    """
    计算并存储平均值和当前值
    用于训练过程中的损失/指标追踪
    """
    
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


def get_lr(optimizer: torch.optim.Optimizer) -> List[float]:
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        各参数组的学习率列表
    """
    return [param_group['lr'] for param_group in optimizer.param_groups]


def setup_logging(log_dir: str, log_name: str = "train") -> None:
    """
    设置日志
    
    Args:
        log_dir: 日志目录
        log_name: 日志文件名
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"日志保存到: {log_path}")
