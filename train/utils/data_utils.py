"""
数据处理工具函数
包含 mel 谱计算、token 编码、speaker embedding 提取等
复用 CosyVoice 和 Step-Audio-EditX 的数据处理逻辑
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random


def load_audio(
    audio_path: str,
    target_sr: int = 24000
) -> Tuple[torch.Tensor, int]:
    """
    加载音频文件并重采样
    
    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率
        
    Returns:
        (音频张量, 采样率)
    """
    audio, sr = torchaudio.load(audio_path)
    
    # 转为单通道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    return audio, target_sr


def normalize_audio(
    audio: torch.Tensor,
    max_amplitude: float = 0.6
) -> torch.Tensor:
    """
    音频归一化，避免削波
    
    Args:
        audio: 音频张量
        max_amplitude: 最大振幅
        
    Returns:
        归一化后的音频
    """
    norm = torch.max(torch.abs(audio), dim=1, keepdim=True)[0]
    if norm > max_amplitude:
        audio = audio / norm * max_amplitude
    return audio


def trim_audio(
    audio: torch.Tensor,
    max_len_sec: float,
    sample_rate: int
) -> torch.Tensor:
    """
    裁剪音频到最大长度
    
    Args:
        audio: 音频张量 (1, T)
        max_len_sec: 最大长度（秒）
        sample_rate: 采样率
        
    Returns:
        裁剪后的音频
    """
    max_samples = int(max_len_sec * sample_rate)
    if audio.shape[1] > max_samples:
        # 随机裁剪起始位置
        start = random.randint(0, audio.shape[1] - max_samples)
        audio = audio[:, start:start + max_samples]
    return audio


def spec_augment(
    mel: torch.Tensor,
    freq_mask_num: int = 2,
    freq_mask_width: int = 10,
    time_mask_num: int = 2,
    time_mask_width: int = 50
) -> torch.Tensor:
    """
    SpecAugment 数据增强
    对 mel 频谱应用频率掩码和时间掩码
    
    Args:
        mel: mel 频谱张量 (1, F, T) 或 (F, T)
        freq_mask_num: 频率掩码数量
        freq_mask_width: 频率掩码最大宽度
        time_mask_num: 时间掩码数量
        time_mask_width: 时间掩码最大宽度
        
    Returns:
        增强后的 mel 频谱
    """
    mel = mel.clone()
    
    # 处理维度
    squeeze = False
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
        squeeze = True
    
    _, n_freq, n_time = mel.shape
    
    # 频率掩码
    for _ in range(freq_mask_num):
        f_width = random.randint(1, min(freq_mask_width, n_freq // 2))
        f_start = random.randint(0, n_freq - f_width)
        mel[:, f_start:f_start + f_width, :] = 0
    
    # 时间掩码
    for _ in range(time_mask_num):
        t_width = random.randint(1, min(time_mask_width, n_time // 2))
        t_start = random.randint(0, n_time - t_width)
        mel[:, :, t_start:t_start + t_width] = 0
    
    if squeeze:
        mel = mel.squeeze(0)
    
    return mel


def pad_sequence(
    sequences: list,
    padding_value: float = 0.0,
    max_len: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对序列进行 padding
    
    Args:
        sequences: 序列列表
        padding_value: 填充值
        max_len: 最大长度（None 表示使用批次内最大长度）
        
    Returns:
        (padded 张量, 长度张量)
    """
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    if max_len is None:
        max_len = max(lengths).item()
    
    batch_size = len(sequences)
    
    # 确定序列维度
    if isinstance(sequences[0], torch.Tensor):
        if sequences[0].dim() == 1:
            out = torch.full((batch_size, max_len), padding_value)
        else:
            feat_dim = sequences[0].shape[-1]
            out = torch.full((batch_size, max_len, feat_dim), padding_value)
    else:
        out = torch.full((batch_size, max_len), padding_value)
    
    for i, seq in enumerate(sequences):
        length = len(seq)
        if isinstance(seq, torch.Tensor):
            out[i, :length] = seq
        else:
            out[i, :length] = torch.tensor(seq)
    
    return out, lengths


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    生成 padding mask
    
    Args:
        lengths: 长度张量 (B,)
        max_len: 最大长度
        
    Returns:
        padding mask (B, T)，True 表示 padding 位置
    """
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)
    lengths = lengths.unsqueeze(1).expand(batch_size, max_len)
    
    return seq_range >= lengths


class MelSpectrogramExtractor:
    """
    Mel 频谱提取器
    与 Step-Audio-EditX 的 CosyVoice frontend 兼容
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 2048,
        hop_length: int = 480,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            center=False,
            norm='slaney',
            mel_scale='slaney'
        )
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取 mel 频谱
        
        Args:
            audio: 音频张量 (1, T) 或 (T,)
            
        Returns:
            mel 频谱 (1, n_mels, T')
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        mel = self.mel_transform(audio)
        
        # 对数变换
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mel


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    DataLoader collate 函数
    处理不同长度的样本，进行 padding
    
    Args:
        batch: 样本列表
        
    Returns:
        批次字典
    """
    # 分离各字段
    prompt_tokens = [item['prompt_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    prompt_mels = [item['prompt_mel'] for item in batch]
    target_mels = [item['target_mel'] for item in batch]
    speaker_embeddings = [item['speaker_embedding'] for item in batch]
    
    # Padding tokens
    prompt_tokens_padded, prompt_token_lens = pad_sequence(prompt_tokens)
    target_tokens_padded, target_token_lens = pad_sequence(target_tokens)
    
    # Padding mels (需要处理多维)
    max_prompt_mel_len = max(m.shape[-1] for m in prompt_mels)
    max_target_mel_len = max(m.shape[-1] for m in target_mels)
    
    n_mels = prompt_mels[0].shape[-2]
    batch_size = len(batch)
    
    prompt_mels_padded = torch.zeros(batch_size, n_mels, max_prompt_mel_len)
    target_mels_padded = torch.zeros(batch_size, n_mels, max_target_mel_len)
    prompt_mel_lens = torch.zeros(batch_size, dtype=torch.long)
    target_mel_lens = torch.zeros(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        p_len = prompt_mels[i].shape[-1]
        t_len = target_mels[i].shape[-1]
        prompt_mels_padded[i, :, :p_len] = prompt_mels[i].squeeze(0)
        target_mels_padded[i, :, :t_len] = target_mels[i].squeeze(0)
        prompt_mel_lens[i] = p_len
        target_mel_lens[i] = t_len
    
    # Stack embeddings
    speaker_embeddings = torch.stack(speaker_embeddings)
    
    return {
        'prompt_tokens': prompt_tokens_padded.long(),
        'prompt_token_len': prompt_token_lens,
        'target_tokens': target_tokens_padded.long(),
        'target_token_len': target_token_lens,
        'prompt_mel': prompt_mels_padded,
        'prompt_mel_len': prompt_mel_lens,
        'target_mel': target_mels_padded,
        'target_mel_len': target_mel_lens,
        'speaker_embedding': speaker_embeddings,
    }
