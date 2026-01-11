#!/usr/bin/env python3
"""
Extract Step-Audio dual-codebook tokens from audio files
Adapted for Step-Audio-EditX training pipeline
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer import StepAudioTokenizer
from model_loader import ModelSource


def extract_tokens(args):
    """Extract speech tokens from audio files"""
    
    print(f"Initializing Step-Audio Tokenizer from {args.tokenizer_path}")
    tokenizer = StepAudioTokenizer(
        encoder_path=args.tokenizer_path,
        model_source=ModelSource.LOCAL
    )
    
    # Read file list
    wav_scp = os.path.join(args.data_dir, 'wav.scp')
    if not os.path.exists(wav_scp):
        raise FileNotFoundError(f"wav.scp not found in {args.data_dir}")
    
    with open(wav_scp, 'r') as f:
        wav_list = [line.strip().split(maxsplit=1) for line in f]
    
    # Prepare output
    output_file = os.path.join(args.data_dir, 'utt2speech_token.pt')
    utt2token = {}
    
    print(f"Processing {len(wav_list)} audio files...")
    for utt_id, wav_path in tqdm(wav_list):
        try:
            # Load audio
            audio, sr = torchaudio.load(wav_path)
            
            # Extract tokens
            vq0206_codes, vq02_codes, vq06_codes = tokenizer.wav2token(
                audio, sr, 
                enable_trim=True,
                energy_norm=True
            )
            
            # Store both formats
            utt2token[utt_id] = {
                'vq0206': vq0206_codes,  # Mixed sequence
                'vq02': vq02_codes,      # VQ02 codes
                'vq06': vq06_codes,      # VQ06 codes
            }
            
        except Exception as e:
            print(f"Error processing {utt_id}: {e}")
            continue
    
    # Save tokens
    torch.save(utt2token, output_file)
    print(f"Saved {len(utt2token)} utterances to {output_file}")
    
    # Save statistics
    stats = {
        'total_utterances': len(utt2token),
        'avg_vq02_length': sum(len(v['vq02']) for v in utt2token.values()) / len(utt2token),
        'avg_vq06_length': sum(len(v['vq06']) for v in utt2token.values()) / len(utt2token),
        'avg_vq0206_length': sum(len(v['vq0206']) for v in utt2token.values()) / len(utt2token),
    }
    
    stats_file = os.path.join(args.data_dir, 'token_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract Step-Audio tokens from audio files'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directory containing wav.scp'
    )
    parser.add_argument(
        '--tokenizer_path',
        required=True,
        help='Path to Step-Audio-Tokenizer model'
    )
    
    args = parser.parse_args()
    extract_tokens(args)


if __name__ == '__main__':
    main()
