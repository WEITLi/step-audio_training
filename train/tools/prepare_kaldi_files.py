#!/usr/bin/env python3
# Copyright (c) 2024 Stepfun AI
# Adapted from CosyVoice
#
# 准备 Kaldi 格式数据文件
# 从音频文件目录生成 wav.scp, text, utt2spk, spk2utt

import argparse
import logging
import os
import glob
from pathlib import Path
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_from_directory(audio_dir, output_dir, audio_ext='wav'):
    """从目录结构自动生成 Kaldi 格式文件
    
    预期目录结构:
        audio_dir/
        ├── speaker1/
        │   ├── audio1.wav (对应 audio1.txt)
        │   ├── audio2.wav (对应 audio2.txt)
        │   └── ...
        ├── speaker2/
        │   └── ...
        
    或者扁平结构:
        audio_dir/
        ├── audio1.wav (对应 audio1.txt)
        ├── audio2.wav (对应 audio2.txt)
        └── ...
    
    Args:
        audio_dir: 音频文件目录
        output_dir: 输出目录
        audio_ext: 音频文件扩展名
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有音频文件
    audio_pattern = f'**/*.{audio_ext}'
    audio_files = list(audio_dir.glob(audio_pattern))
    
    if not audio_files:
        raise ValueError(f"No {audio_ext} files found in {audio_dir}")
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    utt2wav = {}
    utt2text = {}
    utt2spk = {}
    spk2utt = {}
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        # 生成 utterance ID
        # 如果在子目录中，使用 "speaker_filename" 作为 utt_id
        # 如果在根目录，使用 "filename" 作为 utt_id
        rel_path = audio_path.relative_to(audio_dir)
        
        if len(rel_path.parts) > 1:
            # 在子目录中，第一层目录作为 speaker
            speaker = rel_path.parts[0]
            filename = rel_path.stem
            utt_id = f"{speaker}_{filename}"
        else:
            # 在根目录，从文件名推断 speaker 或使用默认
            filename = rel_path.stem
            # 尝试从文件名提取 speaker (假设格式: speakerXXX_uttYYY)
            if '_' in filename:
                speaker = filename.split('_')[0]
            else:
                speaker = 'default_speaker'
            utt_id = filename
        
        # 查找对应的文本文件
        txt_path = audio_path.with_suffix('.txt')
        if not txt_path.exists():
            # 尝试其他可能的文本文件扩展名
            for ext in ['.lab', '.trans', '.text']:
                alt_txt_path = audio_path.with_suffix(ext)
                if alt_txt_path.exists():
                    txt_path = alt_txt_path
                    break
        
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            logger.warning(f"No text file for {audio_path}, skipping")
            continue
        
        # 保存映射
        utt2wav[utt_id] = str(audio_path.absolute())
        utt2text[utt_id] = text
        utt2spk[utt_id] = speaker
        
        if speaker not in spk2utt:
            spk2utt[speaker] = []
        spk2utt[speaker].append(utt_id)
    
    # 写入文件
    logger.info(f"Writing Kaldi format files to {output_dir}")
    
    with open(output_dir / 'wav.scp', 'w', encoding='utf-8') as f:
        for utt_id, wav_path in sorted(utt2wav.items()):
            f.write(f"{utt_id} {wav_path}\n")
    
    with open(output_dir / 'text', 'w', encoding='utf-8') as f:
        for utt_id, text in sorted(utt2text.items()):
            f.write(f"{utt_id} {text}\n")
    
    with open(output_dir / 'utt2spk', 'w', encoding='utf-8') as f:
        for utt_id, speaker in sorted(utt2spk.items()):
            f.write(f"{utt_id} {speaker}\n")
    
    with open(output_dir / 'spk2utt', 'w', encoding='utf-8') as f:
        for speaker, utt_ids in sorted(spk2utt.items()):
            f.write(f"{speaker} {' '.join(sorted(utt_ids))}\n")
    
    logger.info(f"Generated files:")
    logger.info(f"  - wav.scp: {len(utt2wav)} utterances")
    logger.info(f"  - text: {len(utt2text)} utterances")
    logger.info(f"  - utt2spk: {len(utt2spk)} mappings")
    logger.info(f"  - spk2utt: {len(spk2utt)} speakers")


def prepare_from_jsonl(jsonl_path, output_dir):
    """从 JSONL 文件生成 Kaldi 格式文件
    
    JSONL 格式:
    {
        "audio_path": "/path/to/audio.wav",
        "text": "文本内容",
        "speaker_id": "spk001",  # 可选
        "utt_id": "utt001"       # 可选
    }
    
    Args:
        jsonl_path: JSONL 文件路径
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    utt2wav = {}
    utt2text = {}
    utt2spk = {}
    spk2utt = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="Processing JSONL")):
            try:
                data = json.loads(line.strip())
                
                # 提取字段
                audio_path = data.get('audio_path') or data.get('wav') or data.get('path')
                text = data.get('text') or data.get('transcript')
                speaker_id = data.get('speaker_id') or data.get('spk') or 'default_speaker'
                utt_id = data.get('utt_id') or data.get('utt') or f"utt_{idx:06d}"
                
                if not audio_path or not text:
                    logger.warning(f"Line {idx}: missing audio_path or text, skipping")
                    continue
                
                # 检查音频文件存在
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found: {audio_path}, skipping")
                    continue
                
                # 保存映射
                utt2wav[utt_id] = str(Path(audio_path).absolute())
                utt2text[utt_id] = text
                utt2spk[utt_id] = speaker_id
                
                if speaker_id not in spk2utt:
                    spk2utt[speaker_id] = []
                spk2utt[speaker_id].append(utt_id)
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {idx}: JSON decode error: {e}")
                continue
    
    # 写入文件（同上）
    logger.info(f"Writing Kaldi format files to {output_dir}")
    
    with open(output_dir / 'wav.scp', 'w', encoding='utf-8') as f:
        for utt_id, wav_path in sorted(utt2wav.items()):
            f.write(f"{utt_id} {wav_path}\n")
    
    with open(output_dir / 'text', 'w', encoding='utf-8') as f:
        for utt_id, text in sorted(utt2text.items()):
            f.write(f"{utt_id} {text}\n")
    
    with open(output_dir / 'utt2spk', 'w', encoding='utf-8') as f:
        for utt_id, speaker in sorted(utt2spk.items()):
            f.write(f"{utt_id} {speaker}\n")
    
    with open(output_dir / 'spk2utt', 'w', encoding='utf-8') as f:
        for speaker, utt_ids in sorted(spk2utt.items()):
            f.write(f"{speaker} {' '.join(sorted(utt_ids))}\n")
    
    logger.info(f"Generated files:")
    logger.info(f"  - wav.scp: {len(utt2wav)} utterances")
    logger.info(f"  - text: {len(utt2text)} utterances")
    logger.info(f"  - utt2spk: {len(utt2spk)} mappings")
    logger.info(f"  - spk2utt: {len(spk2utt)} speakers")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Kaldi format data files (wav.scp, text, utt2spk, spk2utt)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['directory', 'jsonl'],
        help='Data preparation mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory (for directory mode) or JSONL file (for jsonl mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for Kaldi format files'
    )
    parser.add_argument(
        '--audio_ext',
        type=str,
        default='wav',
        help='Audio file extension (default: wav)'
    )
    args = parser.parse_args()
    
    if args.mode == 'directory':
        prepare_from_directory(args.input, args.output, args.audio_ext)
    elif args.mode == 'jsonl':
        prepare_from_jsonl(args.input, args.output)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
