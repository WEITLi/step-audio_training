#!/usr/bin/env python3
# Copyright (c) 2024 Stepfun AI
#
# 提取 Speaker Embedding 脚本
# 使用 FunASR Campplus 提取说话人嵌入

import argparse
import logging
import torch
import torchaudio
import onnxruntime
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_wav_scp(wav_scp_path):
    """加载 wav.scp 文件"""
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


def load_utt2spk(utt2spk_path):
    """加载 utt2spk 文件"""
    utt2spk = {}
    with open(utt2spk_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                logger.warning(f"Invalid line in utt2spk: {line}")
                continue
            utt_id, spk_id = parts
            utt2spk[utt_id] = spk_id
    
    logger.info(f"Loaded {len(utt2spk)} utt2spk mappings from {utt2spk_path}")
    return utt2spk


def extract_embedding(onnx_session, audio_path, target_sr=16000):
    """使用 Campplus ONNX 模型提取 speaker embedding
    
    Args:
        onnx_session: ONNX Runtime session
        audio_path: 音频文件路径
        target_sr: 目标采样率（Campplus 使用 16kHz）
        
    Returns:
        embedding: numpy array of shape [192]
    """
    try:
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样到 16kHz (Campplus 要求)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        
        # 归一化
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        
        # 准备输入
        audio_np = waveform.squeeze().numpy().astype(np.float32)
        
        # ONNX 推理
        inputs = {onnx_session.get_inputs()[0].name: audio_np[np.newaxis, :]}
        embedding = onnx_session.run(None, inputs)[0]
        
        # 返回 192-dim embedding
        return embedding.squeeze()
        
    except Exception as e:
        logger.error(f"Failed to extract embedding from {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract speaker embeddings using FunASR Campplus'
    )
    parser.add_argument(
        '--wav_scp',
        type=str,
        required=True,
        help='Path to wav.scp file'
    )
    parser.add_argument(
        '--utt2spk',
        type=str,
        required=True,
        help='Path to utt2spk file'
    )
    parser.add_argument(
        '--onnx_path',
        type=str,
        required=True,
        help='Path to Campplus ONNX model (campplus.onnx)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for utt2embedding.pt and spk2embedding.pt'
    )
    args = parser.parse_args()
    
    # 加载 ONNX 模型
    logger.info(f"Loading Campplus model from {args.onnx_path}")
    onnx_session = onnxruntime.InferenceSession(
        args.onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # 加载数据
    utt2wav = load_wav_scp(args.wav_scp)
    utt2spk = load_utt2spk(args.utt2spk)
    
    # 提取 utterance-level embedding
    utt2embedding = {}
    failed_utts = []
    
    logger.info(f"Extracting utterance embeddings for {len(utt2wav)} utterances...")
    
    for utt_id, wav_path in tqdm(utt2wav.items(), desc="Extracting utt embeddings"):
        embedding = extract_embedding(onnx_session, wav_path)
        
        if embedding is not None:
            utt2embedding[utt_id] = torch.from_numpy(embedding)
        else:
            failed_utts.append(utt_id)
    
    logger.info(f"Successfully extracted embeddings for {len(utt2embedding)} utterances")
    
    if failed_utts:
        logger.warning(f"Failed to extract embeddings for {len(failed_utts)} utterances")
    
    # 计算 speaker-level embedding (平均所有 utterance embedding)
    spk2utts = defaultdict(list)
    for utt_id, spk_id in utt2spk.items():
        if utt_id in utt2embedding:
            spk2utts[spk_id].append(utt_id)
    
    spk2embedding = {}
    logger.info(f"Computing speaker embeddings for {len(spk2utts)} speakers...")
    
    for spk_id, utt_ids in tqdm(spk2utts.items(), desc="Computing spk embeddings"):
        embeddings = [utt2embedding[utt_id] for utt_id in utt_ids]
        # 平均所有 utterance embedding
        spk_embedding = torch.stack(embeddings).mean(dim=0)
        spk2embedding[spk_id] = spk_embedding
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    utt2emb_path = output_dir / 'utt2embedding.pt'
    spk2emb_path = output_dir / 'spk2embedding.pt'
    
    torch.save(utt2embedding, str(utt2emb_path))
    torch.save(spk2embedding, str(spk2emb_path))
    
    logger.info(f"Saved utterance embeddings to {utt2emb_path}")
    logger.info(f"Saved speaker embeddings to {spk2emb_path}")
    
    # 打印统计信息
    logger.info(f"Statistics:")
    logger.info(f"  Total utterances: {len(utt2embedding)}")
    logger.info(f"  Total speakers: {len(spk2embedding)}")
    logger.info(f"  Avg utts per speaker: {len(utt2embedding) / len(spk2embedding):.1f}")


if __name__ == '__main__':
    main()
