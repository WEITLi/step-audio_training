"""
音频编辑/克隆任务数据集
适配 Step-Audio-EditX 的语音克隆和编辑任务
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train.utils.data_utils import (
    load_audio, normalize_audio, trim_audio, spec_augment,
    MelSpectrogramExtractor, collate_fn
)


class AudioEditDataset(Dataset):
    """
    音频编辑/克隆任务数据集
    
    数据格式 (JSONL):
    {
        "prompt_audio_path": "data/prompt/xxx.wav",
        "prompt_text": "原文本内容",
        "target_text": "编辑后的文本内容",
        "target_audio_path": "data/target/xxx.wav",
        "edit_type": "emotion/pronunciation/speed",
        "speaker_id": "spk_001"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        sample_rate: int = 24000,
        max_audio_len: float = 10.0,
        max_text_len: int = 200,
        tokenizer_path: Optional[str] = None,
        use_augment: bool = True,
        cache_tokens: bool = False
    ):
        """
        Args:
            data_path: JSONL 数据集路径
            sample_rate: 目标采样率
            max_audio_len: 最大音频长度（秒）
            max_text_len: 最大文本长度（token 数）
            tokenizer_path: tokenizer 路径（用于提取 speech token）
            use_augment: 是否使用数据增强
            cache_tokens: 是否缓存 token（节省重复计算）
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.use_augment = use_augment
        self.cache_tokens = cache_tokens
        
        # 加载数据
        self.samples = self._load_data()
        
        # 初始化 mel 提取器
        self.mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
        
        # 初始化 tokenizer（延迟加载）
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        
        # Token 缓存
        self._token_cache = {}
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载 JSONL 数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.data_path}")
        
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    # 验证必需字段
                    if self._validate_sample(sample):
                        samples.append(sample)
        
        print(f"加载 {len(samples)} 个样本")
        return samples
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """验证样本格式"""
        required_fields = ['prompt_audio_path', 'target_audio_path']
        for field in required_fields:
            if field not in sample:
                return False
            if not os.path.exists(sample[field]):
                return False
        return True
    
    def _init_tokenizer(self):
        """延迟初始化 tokenizer"""
        if self.tokenizer is None and self.tokenizer_path:
            try:
                from tokenizer import StepAudioTokenizer
                from model_loader import ModelSource
                self.tokenizer = StepAudioTokenizer(
                    encoder_path=self.tokenizer_path,
                    model_source=ModelSource.LOCAL
                )
            except Exception as e:
                print(f"警告: 无法加载 tokenizer: {e}")
                self.tokenizer = None
    
    def _extract_tokens(
        self, 
        audio: torch.Tensor, 
        audio_path: str
    ) -> List[int]:
        """提取 speech tokens"""
        # 检查缓存
        if self.cache_tokens and audio_path in self._token_cache:
            return self._token_cache[audio_path]
        
        self._init_tokenizer()
        
        if self.tokenizer is not None:
            try:
                vq0206_codes, _, _ = self.tokenizer.wav2token(
                    audio, self.sample_rate
                )
                tokens = vq0206_codes
            except Exception as e:
                print(f"Token 提取失败: {e}")
                tokens = []
        else:
            # 如果没有 tokenizer，返回空 token
            tokens = []
        
        # 缓存
        if self.cache_tokens:
            self._token_cache[audio_path] = tokens
        
        return tokens
    
    def _extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取 speaker embedding
        复用 Step-Audio-EditX 的 speaker embedding 提取逻辑
        """
        # 返回一个占位符 embedding（实际使用时从模型中提取）
        # 这里使用 192 维的随机向量作为占位符
        embedding = torch.randn(192)
        return embedding
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            包含以下字段的字典:
            - prompt_tokens: 参考音频 token
            - target_tokens: 目标音频 token
            - prompt_mel: 参考音频 mel 频谱
            - target_mel: 目标音频 mel 频谱
            - speaker_embedding: 说话人嵌入
            - prompt_text: 参考文本
            - target_text: 目标文本
            - edit_type: 编辑类型
        """
        sample = self.samples[idx]
        
        # 加载音频
        prompt_audio, _ = load_audio(sample['prompt_audio_path'], self.sample_rate)
        target_audio, _ = load_audio(sample['target_audio_path'], self.sample_rate)
        
        # 归一化
        prompt_audio = normalize_audio(prompt_audio)
        target_audio = normalize_audio(target_audio)
        
        # 裁剪到最大长度
        prompt_audio = trim_audio(prompt_audio, self.max_audio_len, self.sample_rate)
        target_audio = trim_audio(target_audio, self.max_audio_len, self.sample_rate)
        
        # 提取 mel 频谱
        prompt_mel = self.mel_extractor(prompt_audio)
        target_mel = self.mel_extractor(target_audio)
        
        # 数据增强（仅对训练集）
        if self.use_augment:
            # 对目标 mel 应用 SpecAugment
            target_mel = spec_augment(target_mel)
        
        # 提取 tokens
        prompt_tokens = self._extract_tokens(prompt_audio, sample['prompt_audio_path'])
        target_tokens = self._extract_tokens(target_audio, sample['target_audio_path'])
        
        # 提取 speaker embedding
        speaker_embedding = self._extract_speaker_embedding(prompt_audio)
        
        # 转换为张量
        if not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
        if not isinstance(target_tokens, torch.Tensor):
            target_tokens = torch.tensor(target_tokens, dtype=torch.long)
        
        return {
            'prompt_tokens': prompt_tokens,
            'target_tokens': target_tokens,
            'prompt_mel': prompt_mel,
            'target_mel': target_mel,
            'speaker_embedding': speaker_embedding,
            'prompt_text': sample.get('prompt_text', ''),
            'target_text': sample.get('target_text', ''),
            'edit_type': sample.get('edit_type', 'clone'),
        }


def build_dataloader(
    data_path: str,
    batch_size: int,
    sample_rate: int = 24000,
    max_audio_len: float = 10.0,
    max_text_len: int = 200,
    tokenizer_path: Optional[str] = None,
    num_workers: int = 4,
    train: bool = True
) -> DataLoader:
    """
    构建 DataLoader
    
    Args:
        data_path: 数据集路径
        batch_size: 批量大小
        sample_rate: 采样率
        max_audio_len: 最大音频长度
        max_text_len: 最大文本长度
        tokenizer_path: tokenizer 路径
        num_workers: 数据加载线程数
        train: 是否为训练集
        
    Returns:
        DataLoader 实例
    """
    dataset = AudioEditDataset(
        data_path=data_path,
        sample_rate=sample_rate,
        max_audio_len=max_audio_len,
        max_text_len=max_text_len,
        tokenizer_path=tokenizer_path,
        use_augment=train,
        cache_tokens=not train  # 验证集可以缓存
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader
