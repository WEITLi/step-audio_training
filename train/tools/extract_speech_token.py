#!/usr/bin/env python3
# Copyright (c) 2024 Stepfun AI
# 
# 提取 Speech Token 脚本
# 使用 Step-Audio-Tokenizer 提取离散语音 token

import argparse
import logging
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from tokenizer import StepAudioTokenizer
from model_loader import ModelSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_wav_scp(wav_scp_path):
    """加载 wav.scp 文件
    
    格式: utt_id audio_path
    例如:
        utt001 /path/to/audio1.wav
        utt002 /path/to/audio2.wav
    """
    utt2wav = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logger.warning(f"Invalid line in wav.scp: {line}")
                continue
            utt_id, wav_path = parts
            utt2wav[utt_id] = wav_path
    
    logger.info(f"Loaded {len(utt2wav)} utterances from {wav_scp_path}")
    return utt2wav


def extract_tokens(tokenizer, audio_path, target_sr=24000):
    """提取单个音频文件的 speech token
    
    Args:
        tokenizer: Step-Audio-Tokenizer 实例
        audio_path: 音频文件路径
        target_sr: 目标采样率
        
    Returns:
        speech_token: tensor of shape [token_len]
    """
    try:
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        
        # 归一化
        max_val = waveform.abs().max()
        if max_val > 1:
            waveform = waveform / max_val
        
        # 提取 token
        # Step-Audio-Tokenizer 使用双 codebook (VQ02 + VQ06)
        # 合并后的 token 范围: [0, 65536)
        with torch.no_grad():
            # 这里需要使用 Step-Audio-Tokenizer 的编码方法
            # 具体实现参考 tokenizer.py 中的 encode_audio 方法
            vq02_codes, vq06_codes = tokenizer.encode_audio(waveform, target_sr)
            
            # 合并 VQ02 和 VQ06 codes
            # CosyVoice 使用的格式是合并的 token
            speech_token = tokenizer.merge_vq0206_codes(vq02_codes, vq06_codes)
        
        return speech_token.squeeze()
        
    except Exception as e:
        logger.error(f"Failed to extract tokens from {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract speech tokens using Step-Audio-Tokenizer'
    )
    parser.add_argument(
        '--wav_scp',
        type=str,
        required=True,
        help='Path to wav.scp file (format: utt_id wav_path)'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='Path to Step-Audio-Tokenizer model'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for utt2speech_token.pt'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=24000,
        help='Target sample rate (default: 24000)'
    )
    parser.add_argument(
        '--model_source',
        type=str,
        default='local',
        choices=['local', 'hf', 'modelscope'],
        help='Model source (default: local)'
    )
    args = parser.parse_args()
    
    # 加载 tokenizer
    logger.info(f"Loading Step-Audio-Tokenizer from {args.tokenizer_path}")
    
    model_source_map = {
        'local': ModelSource.LOCAL,
        'hf': ModelSource.HF,
        'modelscope': ModelSource.MODELSCOPE
    }
    
    tokenizer = StepAudioTokenizer(
        encoder_path=args.tokenizer_path,
        model_source=model_source_map[args.model_source]
    )
    
    # 加载 wav.scp
    utt2wav = load_wav_scp(args.wav_scp)
    
    # 提取 token
    utt2speech_token = {}
    failed_utts = []
    
    logger.info(f"Extracting speech tokens for {len(utt2wav)} utterances...")
    
    for utt_id, wav_path in tqdm(utt2wav.items(), desc="Extracting tokens"):
        speech_token = extract_tokens(tokenizer, wav_path, args.sample_rate)
        
        if speech_token is not None:
            utt2speech_token[utt_id] = speech_token.cpu()
        else:
            failed_utts.append(utt_id)
    
    logger.info(f"Successfully extracted tokens for {len(utt2speech_token)} utterances")
    
    if failed_utts:
        logger.warning(f"Failed to extract tokens for {len(failed_utts)} utterances")
        logger.warning(f"Failed utterances: {failed_utts[:10]}...")  # 显示前10个
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(utt2speech_token, str(output_path))
    logger.info(f"Saved speech tokens to {args.output}")
    
    # 打印统计信息
    token_lens = [len(tokens) for tokens in utt2speech_token.values()]
    logger.info(f"Token length statistics:")
    logger.info(f"  Min: {min(token_lens)}")
    logger.info(f"  Max: {max(token_lens)}")
    logger.info(f"  Mean: {sum(token_lens) / len(token_lens):.1f}")


if __name__ == '__main__':
    main()
