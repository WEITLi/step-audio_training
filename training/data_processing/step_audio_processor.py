"""
Custom data processor for Step-Audio-EditX dual-codebook tokens
Extends CosyVoice data processing pipeline
"""

import torch
import numpy as np


def process_step_audio_tokens(sample):
    """
    Process Step-Audio dual-codebook tokens for training
    
    Converts dual-codebook format to training format:
    Input: vq02 codes, vq06 codes
    Output: interleaved token sequence for flow model
    """
    if 'vq02_codes' in sample and 'vq06_codes' in sample:
        vq02 = sample['vq02_codes']
        vq06 = sample['vq06_codes']
        
        # Interleave tokens: [02, 02, 06, 06, 06] pattern
        # This matches the Step-Audio-EditX token format
        speech_token = []
        i, j = 0, 0
        
        while i < len(vq02) - 1 and j < len(vq06) - 2:
            # Add 2 VQ02 tokens
            speech_token.extend(vq02[i:i+2])
            # Add 3 VQ06 tokens (offset by 1024)
            speech_token.extend([t + 1024 for t in vq06[j:j+3]])
            i += 2
            j += 3
        
        sample['speech_token'] = speech_token
        sample['speech_token_len'] = len(speech_token)
    
    return sample


def reshape_tokens_for_flow(tokens):
    """
    Reshape mixed token sequence for flow model input
    
    Converts: [02, 02, 06, 06, 06] -> [[02, 02, PAD], [06, 06, 06]]
    This is the format expected by CausalMaskedDiffWithXvec
    """
    if len(tokens) % 5 > 0:
        # Pad to multiple of 5
        pad_len = 5 - (len(tokens) % 5)
        tokens = tokens + [0] * pad_len
    
    num_groups = len(tokens) // 5
    vq02_tokens = []
    vq06_tokens = []
    
    for i in range(num_groups):
        group = tokens[i*5:(i+1)*5]
        # First 2 are VQ02
        vq02_tokens.extend(group[:2])
        vq02_tokens.append(1024)  # Padding token
        # Last 3 are VQ06 (subtract offset)
        vq06_tokens.extend([t - 1024 if t >= 1024 else t for t in group[2:5]])
    
    # Stack into dual-codebook format
    result = torch.stack([
        torch.tensor(vq02_tokens, dtype=torch.long),
        torch.tensor(vq06_tokens, dtype=torch.long),
    ], dim=1)
    
    return result


def compute_token_statistics(sample):
    """Compute statistics for tokens (for debugging)"""
    if 'speech_token' in sample:
        tokens = sample['speech_token']
        sample['token_stats'] = {
            'length': len(tokens),
            'unique_tokens': len(set(tokens)),
            'vq02_count': sum(1 for t in tokens if t < 1024),
            'vq06_count': sum(1 for t in tokens if t >= 1024),
        }
    return sample


def validate_token_format(sample):
    """
    Validate that tokens are in correct format
    Raises ValueError if format is invalid
    """
    if 'speech_token' not in sample:
        raise ValueError("Missing 'speech_token' in sample")
    
    tokens = sample['speech_token']
    
    # Check token range
    for t in tokens:
        if t < 0 or t >= 2048:
            raise ValueError(f"Token {t} out of range [0, 2048)")
    
    # Check pattern (should be roughly 2:3 ratio of VQ02:VQ06)
    vq02_count = sum(1 for t in tokens if t < 1024)
    vq06_count = sum(1 for t in tokens if t >= 1024)
    
    expected_ratio = 2.0 / 3.0
    actual_ratio = vq02_count / vq06_count if vq06_count > 0 else 0
    
    if abs(actual_ratio - expected_ratio) > 0.2:
        print(f"Warning: Token ratio {actual_ratio:.2f} deviates from expected {expected_ratio:.2f}")
    
    return sample


# Export functions for use in YAML config
__all__ = [
    'process_step_audio_tokens',
    'reshape_tokens_for_flow',
    'compute_token_statistics',
    'validate_token_format',
]
