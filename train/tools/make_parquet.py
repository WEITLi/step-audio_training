#!/usr/bin/env python3
# Copyright (c) 2024 Stepfun AI
#
# 打包数据为 Parquet 格式
# 将所有预处理结果打包成 Parquet 文件

import argparse
import logging
import json
import torch
import pandas as pd
import torchaudio
from pathlib import Path
from tqdm import tqdm
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_wav_scp(wav_scp_path):
    """加载 wav.scp"""
    utt2wav = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2wav[parts[0]] = parts[1]
    return utt2wav


def load_text(text_path):
    """加载 text 文件"""
    utt2text = {}
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1]
    return utt2text


def load_utt2spk(utt2spk_path):
    """加载 utt2spk"""
    utt2spk = {}
    with open(utt2spk_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                utt2spk[parts[0]] = parts[1]
    return utt2spk


def create_parquet_chunk(utt_list, utt2wav, utt2text, utt2spk,
                        utt2embedding, spk2embedding, utt2speech_token,
                        parquet_path):
    """创建单个 parquet 文件
    
    Args:
        utt_list: 该 chunk 包含的 utterance ID 列表
        utt2wav, utt2text, etc: 各种映射字典
        parquet_path: 输出路径
    """
    data_list = []
    
    for utt_id in tqdm(utt_list, desc=f"Processing {parquet_path.name}", leave=False):
        try:
            # 加载音频二进制数据
            wav_path = utt2wav[utt_id]
            with open(wav_path, 'rb') as f:
                audio_data = f.read()
            
            # 获取其他数据
            text = utt2text.get(utt_id, "")
            spk_id = utt2spk.get(utt_id, "unknown")
            utt_emb = utt2embedding.get(utt_id, torch.zeros(192))
            spk_emb = spk2embedding.get(spk_id, torch.zeros(192))
            speech_token = utt2speech_token.get(utt_id, torch.tensor([], dtype=torch.long))
            
            # 构建样本
            sample = {
                'utt': utt_id,
                'wav': wav_path,
                'audio_data': audio_data,
                'text': text,
                'spk': spk_id,
                'utt_embedding': utt_emb.numpy().tolist(),
                'spk_embedding': spk_emb.numpy().tolist(),
                'speech_token': speech_token.numpy().tolist(),
            }
            
            data_list.append(sample)
            
        except Exception as e:
            logger.error(f"Failed to process {utt_id}: {e}")
            continue
    
    # 保存为 parquet
    df = pd.DataFrame(data_list)
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    
    return len(data_list)


def main():
    parser = argparse.ArgumentParser(
        description='Pack preprocessed data into Parquet format'
    )
    parser.add_argument(
        '--src_dir',
        type=str,
        required=True,
        help='Source directory containing wav.scp, text, utt2spk, etc.'
    )
    parser.add_argument(
        '--des_dir',
        type=str,
        required=True,
        help='Destination directory for parquet files'
    )
    parser.add_argument(
        '--num_utts_per_parquet',
        type=int,
        default=1000,
        help='Number of utterances per parquet file (default: 1000)'
    )
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir)
    des_dir = Path(args.des_dir)
    des_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有数据
    logger.info("Loading metadata files...")
    utt2wav = load_wav_scp(src_dir / 'wav.scp')
    utt2text = load_text(src_dir / 'text')
    utt2spk = load_utt2spk(src_dir / 'utt2spk')
    
    logger.info("Loading preprocessed features...")
    utt2embedding = torch.load(src_dir / 'utt2embedding.pt')
    spk2embedding = torch.load(src_dir / 'spk2embedding.pt')
    utt2speech_token = torch.load(src_dir / 'utt2speech_token.pt')
    
    # 检查数据完整性
    utts = sorted(utt2wav.keys())
    logger.info(f"Total utterances: {len(utts)}")
    
    missing_data = []
    for utt_id in utts:
        if utt_id not in utt2text:
            missing_data.append(f"{utt_id}: missing text")
        if utt_id not in utt2embedding:
            missing_data.append(f"{utt_id}: missing embedding")
        if utt_id not in utt2speech_token:
            missing_data.append(f"{utt_id}: missing speech_token")
    
    if missing_data:
        logger.warning(f"Found {len(missing_data)} utterances with missing data")
        logger.warning(f"Examples: {missing_data[:5]}")
    
    # 分割成 chunks 并创建 parquet 文件
    num_chunks = (len(utts) + args.num_utts_per_parquet - 1) // args.num_utts_per_parquet
    logger.info(f"Creating {num_chunks} parquet files...")
    
    parquet_files = []
    
    for i in range(num_chunks):
        start_idx = i * args.num_utts_per_parquet
        end_idx = min((i + 1) * args.num_utts_per_parquet, len(utts))
        utt_chunk = utts[start_idx:end_idx]
        
        parquet_path = des_dir / f'parquet_{i:09d}.parquet'
        
        num_samples = create_parquet_chunk(
            utt_chunk, utt2wav, utt2text, utt2spk,
            utt2embedding, spk2embedding, utt2speech_token,
            parquet_path
        )
        
        if num_samples > 0:
            parquet_files.append(str(parquet_path))
            logger.info(f"Created {parquet_path.name} with {num_samples} samples")
    
    # 创建 data.list
    data_list_path = des_dir / 'data.list'
    with open(data_list_path, 'w', encoding='utf-8') as f:
        for parquet_file in parquet_files:
            f.write(parquet_file + '\n')
    
    logger.info(f"Created data.list with {len(parquet_files)} parquet files")
    logger.info(f"Output directory: {des_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
