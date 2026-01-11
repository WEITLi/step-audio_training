#!/usr/bin/env python3
"""
Convert processed data to parquet format for training
Compatible with CosyVoice training pipeline
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torchaudio
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def process_utterance(utt_info, data_dir, utt2token, utt2embedding):
    """Process a single utterance and return parquet row"""
    utt_id, wav_path = utt_info
    
    try:
        # Load audio
        audio, sr = torchaudio.load(wav_path)
        audio = audio.squeeze().numpy()
        
        # Get tokens
        token_data = utt2token[utt_id]
        vq0206 = token_data['vq0206']
        vq02 = token_data['vq02']
        vq06 = token_data['vq06']
        
        # Get embedding
        embedding = utt2embedding[utt_id].numpy()
        
        # Read text if available
        text_file = os.path.join(data_dir, 'text')
        text = ""
        if os.path.exists(text_file):
            with open(text_file, 'r') as f:
                for line in f:
                    if line.startswith(utt_id):
                        text = line.strip().split(maxsplit=1)[1]
                        break
        
        return {
            'utt_id': utt_id,
            'audio': audio.tobytes(),
            'audio_shape': audio.shape,
            'sample_rate': sr,
            'speech_token': vq0206,  # Mixed sequence for training
            'vq02_codes': vq02,
            'vq06_codes': vq06,
            'embedding': embedding.tobytes(),
            'embedding_shape': embedding.shape,
            'text': text,
        }
    except Exception as e:
        print(f"Error processing {utt_id}: {e}")
        return None


def make_parquet(args):
    """Convert data to parquet format"""
    
    # Load tokens and embeddings
    token_file = os.path.join(args.src_dir, 'utt2speech_token.pt')
    embedding_file = os.path.join(args.src_dir, 'utt2embedding.pt')
    
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Token file not found: {token_file}")
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    
    print(f"Loading tokens from {token_file}")
    utt2token = torch.load(token_file)
    
    print(f"Loading embeddings from {embedding_file}")
    utt2embedding = torch.load(embedding_file)
    
    # Read wav.scp
    wav_scp = os.path.join(args.src_dir, 'wav.scp')
    with open(wav_scp, 'r') as f:
        wav_list = [line.strip().split(maxsplit=1) for line in f]
    
    print(f"Processing {len(wav_list)} utterances...")
    
    # Process in parallel
    process_fn = partial(
        process_utterance,
        data_dir=args.src_dir,
        utt2token=utt2token,
        utt2embedding=utt2embedding
    )
    
    with mp.Pool(args.num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_fn, wav_list),
            total=len(wav_list)
        ))
    
    # Filter out failed utterances
    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)} utterances")
    
    # Split into chunks
    num_parquets = (len(results) + args.num_utts_per_parquet - 1) // args.num_utts_per_parquet
    
    os.makedirs(args.des_dir, exist_ok=True)
    parquet_files = []
    
    for i in range(num_parquets):
        start_idx = i * args.num_utts_per_parquet
        end_idx = min((i + 1) * args.num_utts_per_parquet, len(results))
        chunk = results[start_idx:end_idx]
        
        # Create parquet file
        parquet_file = os.path.join(args.des_dir, f'data_{i:04d}.parquet')
        
        # Define schema
        schema = pa.schema([
            ('utt_id', pa.string()),
            ('audio', pa.binary()),
            ('audio_shape', pa.list_(pa.int64())),
            ('sample_rate', pa.int32()),
            ('speech_token', pa.list_(pa.int32())),
            ('vq02_codes', pa.list_(pa.int32())),
            ('vq06_codes', pa.list_(pa.int32())),
            ('embedding', pa.binary()),
            ('embedding_shape', pa.list_(pa.int64())),
            ('text', pa.string()),
        ])
        
        # Convert to pyarrow table
        table = pa.Table.from_pylist(chunk, schema=schema)
        
        # Write parquet
        pq.write_table(table, parquet_file)
        parquet_files.append(parquet_file)
        print(f"Saved {parquet_file} with {len(chunk)} utterances")
    
    # Create data.list
    data_list = os.path.join(args.des_dir, 'data.list')
    with open(data_list, 'w') as f:
        for pf in parquet_files:
            f.write(f"{pf}\n")
    
    print(f"\nCreated {len(parquet_files)} parquet files")
    print(f"Data list saved to {data_list}")


def main():
    parser = argparse.ArgumentParser(
        description='Create parquet files for training'
    )
    parser.add_argument(
        '--src_dir',
        required=True,
        help='Source directory with wav.scp, tokens, embeddings'
    )
    parser.add_argument(
        '--des_dir',
        required=True,
        help='Destination directory for parquet files'
    )
    parser.add_argument(
        '--num_utts_per_parquet',
        type=int,
        default=1000,
        help='Number of utterances per parquet file'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=8,
        help='Number of parallel processes'
    )
    
    args = parser.parse_args()
    make_parquet(args)


if __name__ == '__main__':
    main()
