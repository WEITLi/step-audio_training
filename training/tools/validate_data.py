#!/usr/bin/env python3
"""
Validate prepared training data
Checks data format, token distribution, and potential issues
"""

import argparse
import sys
from pathlib import Path
import torch
import pyarrow.parquet as pq
import numpy as np
from collections import Counter
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def validate_parquet_file(parquet_file):
    """Validate a single parquet file"""
    print(f"\nValidating {parquet_file}...")
    
    try:
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        print(f"  ✓ Loaded {len(df)} samples")
        
        # Check required columns
        required_cols = ['utt_id', 'audio', 'speech_token', 'embedding']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return False
        print(f"  ✓ All required columns present")
        
        # Validate tokens
        token_lengths = []
        token_ranges = []
        
        for idx, row in df.iterrows():
            tokens = row['speech_token']
            token_lengths.append(len(tokens))
            token_ranges.extend(tokens)
        
        print(f"  ✓ Token statistics:")
        print(f"    - Avg length: {np.mean(token_lengths):.1f}")
        print(f"    - Min length: {np.min(token_lengths)}")
        print(f"    - Max length: {np.max(token_lengths)}")
        print(f"    - Token range: [{np.min(token_ranges)}, {np.max(token_ranges)}]")
        
        # Check token distribution
        vq02_count = sum(1 for t in token_ranges if t < 1024)
        vq06_count = sum(1 for t in token_ranges if t >= 1024)
        ratio = vq02_count / vq06_count if vq06_count > 0 else 0
        
        print(f"    - VQ02 tokens: {vq02_count} ({vq02_count/len(token_ranges)*100:.1f}%)")
        print(f"    - VQ06 tokens: {vq06_count} ({vq06_count/len(token_ranges)*100:.1f}%)")
        print(f"    - Ratio (VQ02/VQ06): {ratio:.2f} (expected: ~0.67)")
        
        if abs(ratio - 0.67) > 0.1:
            print(f"  ⚠ Warning: Token ratio deviates from expected")
        
        # Validate embeddings
        emb_shapes = [len(np.frombuffer(row['embedding'], dtype=np.float32)) 
                      for _, row in df.iterrows()]
        if len(set(emb_shapes)) > 1:
            print(f"  ✗ Inconsistent embedding shapes: {set(emb_shapes)}")
            return False
        print(f"  ✓ Embedding shape consistent: {emb_shapes[0]}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_data_list(data_list_file):
    """Validate all parquet files in data list"""
    print(f"Validating data list: {data_list_file}")
    
    with open(data_list_file, 'r') as f:
        parquet_files = [line.strip() for line in f]
    
    print(f"Found {len(parquet_files)} parquet files")
    
    results = []
    for pf in parquet_files:
        if not Path(pf).exists():
            print(f"  ✗ File not found: {pf}")
            results.append(False)
        else:
            results.append(validate_parquet_file(pf))
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n{'='*60}")
    print(f"Validation Summary: {sum(results)}/{len(results)} files passed ({success_rate:.1f}%)")
    print(f"{'='*60}")
    
    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description='Validate training data format'
    )
    parser.add_argument(
        '--data_list',
        required=True,
        help='Path to data.list file'
    )
    
    args = parser.parse_args()
    
    success = validate_data_list(args.data_list)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
